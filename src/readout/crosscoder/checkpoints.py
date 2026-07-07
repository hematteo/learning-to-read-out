"""Canonical loader for W_U / W_E crosscoder checkpoints.

Subsumes the seven near-duplicate ``load_crosscoder`` / ``parse_ckpt`` /
``load_crosscoder_state`` helpers that were sprinkled across experiments.

Two on-disk formats are supported and returned through the same dataclass:

* legacy ``.pt`` — torch-pickled dict with ``state_dict`` / ``training`` /
  ``steps`` / ``model_name`` / ``quality`` / ``preprocess_stats`` keys.
* HF release ``.safetensors`` — flat tensor file with a sibling
  ``<name>.config.json`` carrying the metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

# Checkpoints trained 2026-07-06 embed torch.__version__ as a raw TorchVersion
# in their provenance dict, which weights_only=True otherwise rejects.
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


@dataclass(frozen=True)
class Checkpoint:
    """One crosscoder checkpoint, format-agnostic.

    ``raw`` is the original ``.pt`` payload (or ``None`` for safetensors).
    Avoid reading from it in new code; use the typed fields instead.
    """

    path: Path
    state_dict: dict[str, torch.Tensor]
    training: dict
    steps: list[int] | None
    model_name: str | None
    quality: dict | None
    preprocess_stats: dict | None
    raw: dict | None


def unwrap_ckpt(ck: dict) -> dict:
    """Normalize new-format ``.pt`` checkpoints (nested under ``config``) to flat layout.

    Newer 1B checkpoints (d=16384, d=24576) store metadata as
    ``ck = {'state_dict': ..., 'config': {steps, model_name, preprocess_mode, ...}}``.
    Older checkpoints (160M Run 3, 1B d=8192) store the same fields at the top
    level and ``ck['config']`` is the inner crosscoder config (sae_type,
    d_model, etc.). The new-format wrapper is detected by the presence of
    metadata keys (``steps`` / ``model_name``) inside ``ck['config']`` AND
    their absence at the top level.
    """
    cfg = ck.get("config")
    if not isinstance(cfg, dict):
        return ck
    is_wrapper = "steps" in cfg and "model_name" in cfg and "steps" not in ck and "model_name" not in ck
    if not is_wrapper:
        return ck
    # Keep every other top-level key (quality, preprocess_stats, ...) — the
    # wrapper detection only guarantees steps/model_name live inside cfg.
    merged = {**cfg, **{k: v for k, v in ck.items() if k != "config"}}
    merged["config"] = cfg.get("config", cfg)
    return merged


def load_checkpoint(path: Path | str) -> Checkpoint:
    """Load a crosscoder checkpoint from ``.pt`` or ``.safetensors``."""
    p = Path(path)
    if p.suffix == ".pt":
        raw = torch.load(p, map_location="cpu", weights_only=True)
        # Wrapper checkpoints nest model_name/quality/preprocess_stats under
        # "config"; unwrap so both formats read through the same keys.
        ck = unwrap_ckpt(raw)
        sd = ck.get("state_dict", ck)
        # 1B aggregates-sidecar convention nests training under "config"
        training = ck.get("training", ck.get("config", {}).get("training", {}))
        return Checkpoint(
            path=p,
            state_dict=sd,
            training=training,
            steps=ck.get("steps") or ck.get("config", {}).get("steps"),
            model_name=ck.get("model") or ck.get("model_name"),
            quality=ck.get("quality"),
            preprocess_stats=ck.get("preprocess_stats"),
            raw=raw if isinstance(raw, dict) else None,
        )
    if p.suffix == ".safetensors":
        cfg_path = p.with_suffix(".config.json")
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        sd = load_safetensors(str(p))
        return Checkpoint(
            path=p,
            state_dict=sd,
            training=cfg.get("training", {}),
            steps=cfg.get("steps"),
            model_name=cfg.get("model_name"),
            quality=cfg.get("quality"),
            preprocess_stats=cfg.get("preprocess_stats"),
            raw=None,
        )
    raise ValueError(f"Unknown ckpt format: {p}")
