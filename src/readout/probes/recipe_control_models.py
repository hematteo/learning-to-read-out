"""Loaders for the 31M recipe-control checkpoints.

The trainer (``experiments/ablations/pretraining_recipe_control/scripts/train_control.py``)
writes one run directory per arm::

    <ckpt_root>/<condition>/config.json
    <ckpt_root>/<condition>/metrics.csv
    <ckpt_root>/<condition>/ckpts/step<N>/model_fp16.pt
    <ckpt_root>/<condition>/ckpts/step<N>/metrics.json

and the released arms (``hf.co/hematteo/readout-recipe-control``) download into the
same layout under ``${UM_SSD_ROOT}/runs/<condition>/`` (:func:`default_runs_root`).
Every recipe-control analysis (crosscoder fits, lifecycle statistics, readout swaps,
probes, gauge landscape) reads the checkpoints through this module, which is the
single place that knows how to rebuild the trainer's model so a saved ``state_dict``
loads exactly (parallel residual, rotary 0.25, untied W_U, gelu, LayerNorm eps 1e-5).

These models are deliberately not registered in ``readout.core.model_specs``: that
registry is for the published Pythia/OLMo checkpoints.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from readout.core.paths import ssd_path

# Mirror of train_control.py MODEL_SIZES (HF GPTNeoXConfig field names).
MODEL_SIZES: dict[str, dict[str, int]] = {
    "14M": dict(num_hidden_layers=6, hidden_size=128, num_attention_heads=4, intermediate_size=512),
    "31M": dict(num_hidden_layers=6, hidden_size=256, num_attention_heads=8, intermediate_size=1024),
    "70M": dict(num_hidden_layers=6, hidden_size=512, num_attention_heads=8, intermediate_size=2048),
}
VOCAB = 50304  # Pythia padded vocab (GPT-NeoX-20B tokenizer); the control uses the same
SEQLEN = 2048

# The four arms the paper reports, in readout LR order with the warmup arm last;
# `warmup_long` is a fifth released arm that the paper does not report.
PAPER_CONDITIONS = ("wu_lr_0p25", "baseline", "wu_lr_4x", "warmup_short")
RELEASED_CONDITIONS = PAPER_CONDITIONS + ("warmup_long",)

WU_KEY = "embed_out.weight"
LNF_WEIGHT_KEY = "gpt_neox.final_layer_norm.weight"
LNF_BIAS_KEY = "gpt_neox.final_layer_norm.bias"


def default_runs_root() -> Path:
    """``${UM_SSD_ROOT}/runs``: where ``hf download hematteo/readout-recipe-control`` lands."""
    return ssd_path("runs")


def build_control_model(model_size: str | dict[str, int], seq_len: int = SEQLEN, vocab_size: int = VOCAB):
    """Reconstruct the trainer's ``GPTNeoXForCausalLM`` so a control ``state_dict`` loads cleanly.

    ``model_size`` is a key of :data:`MODEL_SIZES` or an explicit dict with the same
    four fields (tests use a tiny one).
    """
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    spec = MODEL_SIZES[model_size] if isinstance(model_size, str) else model_size
    cfg = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=spec["hidden_size"],
        num_hidden_layers=spec["num_hidden_layers"],
        num_attention_heads=spec["num_attention_heads"],
        intermediate_size=spec["intermediate_size"],
        max_position_embeddings=seq_len,
        rotary_pct=0.25,
        rotary_emb_base=10000,
        use_parallel_residual=True,
        tie_word_embeddings=False,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        use_cache=False,
        attn_implementation="sdpa",
    )
    return GPTNeoXForCausalLM(cfg)


def condition_dir(ckpt_root: Path, cond: str) -> Path:
    return Path(ckpt_root) / cond


def condition_config(ckpt_root: Path, cond: str) -> dict:
    """The trainer's ``config.json`` (``model_size``, ``readout_lr_mult``, ``warmup_steps``, ...)."""
    return json.loads((condition_dir(ckpt_root, cond) / "config.json").read_text())


def condition_steps(ckpt_root: Path, cond: str) -> list[int]:
    """Sorted checkpoint steps present under ``<cond>/ckpts/step*/`` (never hardcoded).

    The step near 2480 (the first job-slot boundary) differs by a few steps
    across arms, so callers discover steps by globbing and map requested steps
    with :func:`resolve_step`.
    """
    cdir = condition_dir(ckpt_root, cond) / "ckpts"
    steps = sorted(int(p.name[4:]) for p in cdir.glob("step*") if p.name[4:].isdigit())
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {cdir}")
    return steps


def resolve_step(ckpt_root: Path, cond: str, step: int | str) -> int:
    """Map ``'final'`` / an int to an available step (nearest on a miss)."""
    steps = condition_steps(ckpt_root, cond)
    if step == "final":
        return steps[-1]
    step = int(step)
    if step in steps:
        return step
    return min(steps, key=lambda s: abs(s - step))


def state_dict_path(ckpt_root: Path, cond: str, step: int) -> Path:
    return condition_dir(ckpt_root, cond) / "ckpts" / f"step{step}" / "model_fp16.pt"


def load_state_dict(ckpt_root: Path, cond: str, step: int) -> dict[str, torch.Tensor]:
    """The fp16 ``state_dict`` of one checkpoint, on CPU."""
    return torch.load(state_dict_path(ckpt_root, cond, step), map_location="cpu")


def load_body(
    ckpt_root: Path,
    cond: str,
    step: int,
    model_size: str | dict[str, int] | None = None,
    seq_len: int = SEQLEN,
    device: str = "cpu",
):
    """A condition's full model at ``step`` (fp32, eval mode) on ``device``.

    ``model_size`` defaults to the run's ``config.json``. ``strict=False`` covers
    only recomputed rotary / causal buffers: every trained parameter must be in
    the checkpoint, or this raises.
    """
    if model_size is None:
        model_size = condition_config(ckpt_root, cond)["model_size"]
    sd = load_state_dict(ckpt_root, cond, step)
    model = build_control_model(model_size, seq_len, vocab_size=sd[WU_KEY].shape[0])
    model.load_state_dict(sd, strict=False)
    missing = {n for n, _ in model.named_parameters()} - set(sd)
    if missing:
        raise RuntimeError(f"{cond} step{step}: trained params absent from checkpoint: {sorted(missing)[:6]}")
    return model.float().to(device).eval()


def load_readout(ckpt_root: Path, cond: str, step: int, device: str = "cpu") -> dict[str, torch.Tensor]:
    """``{wu, ln_w, ln_b}`` (fp32) for one condition/step, without a body forward."""
    sd = load_state_dict(ckpt_root, cond, step)
    return dict(
        wu=sd[WU_KEY].float().to(device),  # (V, d)
        ln_w=sd[LNF_WEIGHT_KEY].float().to(device),  # (d,)
        ln_b=sd[LNF_BIAS_KEY].float().to(device),  # (d,)
    )


def load_condition_wu(
    ckpt_root: Path,
    cond: str,
    dtype: torch.dtype = torch.float32,
    steps: list[int] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Stack ``embed_out.weight`` across the condition's checkpoints -> ``(K, V, d)``, steps.

    This is the trajectory-crosscoder input for one arm; ``steps`` defaults to every
    checkpoint on disk.
    """
    steps = list(steps) if steps is not None else condition_steps(ckpt_root, cond)
    snaps = []
    vocab = None
    for st in steps:
        wu = load_state_dict(ckpt_root, cond, st)[WU_KEY].to(dtype)  # (V, d)
        if vocab is None:
            vocab = wu.shape[0]
        elif wu.shape[0] != vocab:
            raise ValueError(f"{cond} step{st}: W_U rows {wu.shape[0]} != {vocab}")
        snaps.append(wu)
    return torch.stack(snaps, dim=0), steps  # (K, V, d)


def val_loss_at(ckpt_root: Path, cond: str, step: int) -> float | None:
    """Held-out validation loss (nats) the trainer logged at ``step``.

    Reads ``ckpts/step<N>/metrics.json``; falls back to the matching ``metrics.csv``
    row; ``None`` if neither records it.
    """
    mj = condition_dir(ckpt_root, cond) / "ckpts" / f"step{step}" / "metrics.json"
    if mj.is_file():
        val = json.loads(mj.read_text()).get("val_loss")
        if val is not None:
            return float(val)
    mc = condition_dir(ckpt_root, cond) / "metrics.csv"
    if mc.is_file():
        with mc.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if int(row["step"]) == step and row.get("val_loss") not in (None, ""):
                    return float(row["val_loss"])
    return None
