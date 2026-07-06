"""Hidden-state readout-swap utilities for contrastive (y+/y-) tasks.

Given a list of contrastive ``Example``s, this module provides:

  - ``extract_final_hidden(model, examples, device)`` — run the model and
    capture the residual stream input to the readout module at the final
    prompt position. Returns (N, d) on CPU.

  - ``compute_margin(h, W_U, y_plus, y_minus)`` — the contrastive margin
    h · W_U[y+] - h · W_U[y-].

  - ``align_readout(W_s, W_t, mode)`` — gauge-alignment ladder
    {none, mean, scale, row_norm, procrustes} matched to ``run_aligned_swap_grid``.

  - ``swap_grid_metrics(...)`` — accuracy / mean_margin / delta_* over a
    (h_step, s_step) grid.

Hidden-state extraction matches ``temporal_patch_metrics._cache_hidden_generic``:
forward-pre-hook on the readout module (Pythia: embed_out, OLMo/LLaMA: lm_head).
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------
def _resolve_readout_module(model):
    if hasattr(model, "embed_out"):  # Pythia / GPT-NeoX
        return model.embed_out
    if hasattr(model, "lm_head"):  # OLMo / LLaMA-style
        return model.lm_head
    raise AttributeError(f"{type(model).__name__} has neither .embed_out nor .lm_head")


@torch.no_grad()
def extract_final_hidden(
    model,
    prompt_ids_list: list[list[int]],
    *,
    device: str = "cuda",
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Capture the residual-stream input to the readout at the FINAL prompt token.

    Returns (N, d_model) on CPU, fp32. Each prompt is run on its own as a
    batch of 1 (no padding or batching), so the final token of every example
    is at position -1. ``pad_token_id`` is accepted for API symmetry but unused.
    """
    if pad_token_id is None:
        pad_token_id = 0  # Pythia/OLMo have no pad; safe filler since we mask by length.

    captured: list[torch.Tensor] = []

    def hook(module, inputs):
        captured.append(inputs[0].detach())

    handle = _resolve_readout_module(model).register_forward_pre_hook(hook)
    out_rows: list[torch.Tensor] = []
    try:
        for ids in prompt_ids_list:
            captured.clear()
            t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            _ = model(t)
            h = captured[0][:, -1, :]  # (1, d_model)
            out_rows.append(h.detach().to("cpu", torch.float32))
    finally:
        handle.remove()
    return torch.cat(out_rows, dim=0)  # (N, d)


# ---------------------------------------------------------------------------
# Margin / accuracy
# ---------------------------------------------------------------------------
def compute_margin(
    h: torch.Tensor,  # (N, d)
    W_U: torch.Tensor,  # (V, d)
    y_plus: torch.Tensor,  # (N,)
    y_minus: torch.Tensor,  # (N,)
) -> torch.Tensor:
    """Per-example contrastive margin = h · W_U[y+] - h · W_U[y-]. (N,) fp32."""
    Wp = W_U[y_plus.long()].to(h.device).float()
    Wm = W_U[y_minus.long()].to(h.device).float()
    return ((h.float() * Wp).sum(-1) - (h.float() * Wm).sum(-1)).cpu()


def margin_summary(margins: torch.Tensor) -> dict:
    m = margins.float().cpu().numpy()
    return {
        "n": int(m.size),
        "mean_margin": float(m.mean()) if m.size else float("nan"),
        "median_margin": float(np.median(m)) if m.size else float("nan"),
        "std_margin": float(m.std(ddof=1)) if m.size > 1 else float("nan"),
        "accuracy": float((m > 0).mean()) if m.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Gauge-alignment ladder (mirrors run_aligned_swap_grid.align)
# ---------------------------------------------------------------------------
ALIGN_MODES = ("none", "mean", "scale", "row_norm", "procrustes")


def align_readout(W_s: torch.Tensor, W_t: torch.Tensor, mode: str) -> torch.Tensor:
    """Return ``W_s`` aligned to ``W_t``'s gauge under ``mode``. fp32 in/out, (V, d)."""
    Ws = W_s.float()
    Wt = W_t.float()
    if mode == "none":
        return Ws
    mu_s = Ws.mean(dim=0, keepdim=True)
    mu_t = Wt.mean(dim=0, keepdim=True)
    if mode == "mean":
        return Ws - mu_s + mu_t
    if mode == "scale":
        Ws_c = Ws - mu_s
        Wt_c = Wt - mu_t
        ratio = Wt_c.norm() / Ws_c.norm().clamp_min(1e-12)
        return Ws_c * ratio + mu_t
    if mode == "row_norm":
        ns = Ws.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        nt = Wt.norm(dim=-1, keepdim=True)
        return Ws * (nt / ns)
    if mode == "procrustes":
        Ws_c = Ws - mu_s
        Wt_c = Wt - mu_t
        M = Ws_c.T @ Wt_c
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh
        return Ws_c @ R + mu_t
    raise ValueError(f"unknown alignment mode {mode!r}")


# ---------------------------------------------------------------------------
# Heatmap evaluation
# ---------------------------------------------------------------------------
def evaluate_swap_cell(
    h: torch.Tensor,
    W_s: torch.Tensor,
    W_t: torch.Tensor,
    y_plus: torch.Tensor,
    y_minus: torch.Tensor,
    *,
    alignment: str = "none",
    device: str = "cpu",
    return_per_example: bool = False,
) -> dict | tuple[dict, dict]:
    """Score a single (h, W_s) cell relative to native (h, W_t).

    Returns {n, mean_margin, accuracy, mean_margin_native, accuracy_native,
             delta_margin, delta_accuracy}.

    If ``return_per_example=True``, returns ``(aggregate, per_example)`` where
    ``per_example`` contains CPU fp32 tensors ``margins``, ``margins_native``,
    plus ``y_plus`` / ``y_minus`` for downstream held-out / bootstrap analysis.
    """
    W_s_aligned = align_readout(W_s, W_t, alignment).to(device)
    h_dev = h.to(device).float()
    m_swap = compute_margin(h_dev, W_s_aligned, y_plus, y_minus)
    m_native = compute_margin(h_dev, W_t.to(device).float(), y_plus, y_minus)
    s_swap = margin_summary(m_swap)
    s_native = margin_summary(m_native)
    agg = {
        "n": s_swap["n"],
        "mean_margin": s_swap["mean_margin"],
        "accuracy": s_swap["accuracy"],
        "mean_margin_native": s_native["mean_margin"],
        "accuracy_native": s_native["accuracy"],
        "delta_margin": s_swap["mean_margin"] - s_native["mean_margin"],
        "delta_accuracy": s_swap["accuracy"] - s_native["accuracy"],
    }
    if not return_per_example:
        return agg
    per_ex = {
        "margins": m_swap.detach().cpu().float(),
        "margins_native": m_native.detach().cpu().float(),
        "y_plus": y_plus.detach().cpu().long(),
        "y_minus": y_minus.detach().cpu().long(),
    }
    return agg, per_ex
