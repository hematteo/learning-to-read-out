"""
Data extraction and preprocessing for proto-token dictionary learning.
=====================================================================
Single source of truth for W_U extraction, centering, SVD removal,
and SVD baseline computation.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch


def write_csv(path: Path, rows: list[dict], *, require_rows: bool = False) -> None:
    """Write ``rows`` to ``path`` as CSV (header from the first row's keys).

    Empty ``rows`` is a no-op by default; pass ``require_rows=True`` to raise
    instead (for callers where zero rows means the upstream data is missing).
    Parent directories are created if missing.
    """
    if not rows:
        if require_rows:
            raise ValueError(f"no rows for {path}")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def extract_wu_from_model(model) -> torch.Tensor:
    """Extract W_U from an already-loaded HuggingFace model (no reload)."""
    if hasattr(model, "embed_out"):
        W_U = model.embed_out.weight.detach().clone()
    elif hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().clone()
    else:
        raise ValueError(f"Cannot locate unembedding matrix in {type(model).__name__}")
    return W_U.float()


def extract_wu(model_name: str, device: str = "cpu", **load_kwargs):
    """Extract the unembedding matrix W_U and return (W_U, tokenizer).

    Supports GPT-2 (lm_head), Pythia/GPT-NeoX (embed_out), and
    Llama-style models including Qwen and Gemma (lm_head).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )

    W_U = extract_wu_from_model(model)
    print(f"Extracted W_U: {W_U.shape} from {model_name}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return W_U.to(device), tokenizer


def extract_we(model_name: str, device: str = "cpu", **load_kwargs):
    """Extract the input embedding matrix W_E and return (W_E, tokenizer).

    Mirrors ``extract_wu``'s model coverage: GPT-2 (transformer.wte),
    Pythia/GPT-NeoX (gpt_neox.embed_in), and Llama-style models including
    Qwen, Gemma, and OLMo (model.embed_tokens).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )

    if hasattr(model, "gpt_neox"):
        W_E = model.gpt_neox.embed_in.weight.detach().clone()
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        W_E = model.transformer.wte.weight.detach().clone()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        # Llama/Qwen/Gemma/OLMo-style decoder stacks.
        W_E = model.model.embed_tokens.weight.detach().clone()
    else:
        raise ValueError(f"Cannot locate embedding matrix in {model_name}")

    print(f"Extracted W_E: {W_E.shape} from {model_name}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return W_E.float().to(device), tokenizer


def center_and_project(
    W_U: torch.Tensor,
    svd_remove: int = 0,
    device=None,
    verbose: bool = True,
    normalize_rows: bool = False,
):
    if device is None:
        device = W_U.device
    data = W_U.to(device)
    data_mean = data.mean(dim=0)
    data = data - data_mean

    svd_components = None
    if svd_remove > 0:
        U, S, Vt = torch.linalg.svd(data, full_matrices=False)
        svd_components = Vt[:svd_remove].clone()
        proj = data @ svd_components.T
        data = data - proj @ svd_components
        if verbose:
            removed_var = S[:svd_remove].pow(2).sum() / S.pow(2).sum()
            print(
                f"  Removed top {svd_remove} SVD components "
                f"(variance: {removed_var:.1%}, "
                f"singular values: {', '.join(f'{s:.2f}' for s in S[:svd_remove])})"
            )

    centering: dict = {
        "mean": data_mean,
        "svd_components": svd_components,
        "svd_S": S.clone() if svd_remove > 0 else None,
    }

    if normalize_rows:
        row_norms = data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        centering["row_norms"] = row_norms.squeeze(1)
        data = data / row_norms

    return data, centering


def adaptive_center_and_project(W: torch.Tensor, svd_remove: int = 2, device: str = "cpu") -> tuple[torch.Tensor, dict]:
    return center_and_project(W, svd_remove=svd_remove, device=device)


def apply_centering(W_U: torch.Tensor, centering: dict) -> torch.Tensor:
    """Apply stored centering transform to raw W_U (for inference)."""
    data = W_U - centering["mean"].to(W_U.device)
    if centering.get("svd_components") is not None:
        comps = centering["svd_components"].to(W_U.device)
        data = data - (data @ comps.T) @ comps
    if centering.get("row_norms") is not None:
        row_norms = centering["row_norms"].to(data.device).unsqueeze(1).clamp(min=1e-8)
        data = data / row_norms
    return data


def center_scale_stats(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-snapshot center_scale stats (mean over rows, scale = sqrt(E[||x-μ||²]/d)).

    Matches wu_adapter.preprocess_snapshots(..., mode="center_scale") for one (V, d).
    Returns (mean (1, d), scale scalar tensor).
    """
    mean = W.mean(dim=0, keepdim=True)
    centered = W - mean
    d = W.shape[-1]
    mean_sq = centered.pow(2).sum(dim=-1).mean()
    scale = (mean_sq / d).clamp_min(1e-8).sqrt()
    return mean, scale


def explained_variance(W_true: torch.Tensor, W_hat: torch.Tensor) -> float:
    var_true = W_true.var().item()
    var_res = (W_true - W_hat).var().item()
    return 1.0 - (var_res / max(var_true, 1e-12))


_svd_cache: dict = {}


def _svd_fingerprint(data: torch.Tensor) -> tuple:
    """Content-based cache key for SVD.

    Keyed on shape + dtype + a hash of the full tensor bytes, so distinct
    matrices never collide (the previous 3x3-corner key let different matrices
    sharing a corner alias to a stale cached SVD).
    """
    contig = data.detach().cpu().contiguous()
    digest = hashlib.sha1(contig.numpy().tobytes()).hexdigest()
    return (
        tuple(contig.shape),
        str(contig.dtype),
        digest,
    )


def svd_baseline(data: torch.Tensor, rank: int) -> float:
    """Reconstruction MSE of rank-k SVD (for comparison with SAE).

    Caches the SVD decomposition so repeated calls with different ranks
    don't recompute the full SVD each time.
    """
    fp = _svd_fingerprint(data)
    if _svd_cache.get("fingerprint") != fp:
        data_cpu = data.detach().cpu()
        U, S, Vt = torch.linalg.svd(data_cpu, full_matrices=False)
        _svd_cache.clear()
        _svd_cache.update({"fingerprint": fp, "U": U, "S": S, "Vt": Vt, "data_cpu": data_cpu})
    U, S, Vt, data_cpu = (
        _svd_cache["U"],
        _svd_cache["S"],
        _svd_cache["Vt"],
        _svd_cache["data_cpu"],
    )
    recon = (U[:, :rank] * S[:rank]) @ Vt[:rank]
    return (data_cpu - recon).pow(2).sum(dim=1).mean().item()


def svd_baseline_from_centering(centering: dict, rank: int, n_rows: int) -> float | None:
    """Estimate rank-k SVD baseline MSE from stored singular values.

    WARNING: This is an approximation. The stored singular values are from
    the pre-removal SVD. After projecting out the top components, the true
    singular values of the residual are NOT simply S[svd_remove:].
    For exact baselines, use svd_baseline() on the actual data tensor.

    Returns None if centering lacks stored SVD info (e.g. svd_remove=0).
    """
    S = centering.get("svd_S")
    if S is None:
        return None
    svd_remove = centering["svd_components"].shape[0] if centering.get("svd_components") is not None else 0
    offset = svd_remove + rank
    if offset >= S.shape[0]:
        return 0.0
    return S[offset:].pow(2).sum().item() / n_rows


def get_device() -> str:
    """Return best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
