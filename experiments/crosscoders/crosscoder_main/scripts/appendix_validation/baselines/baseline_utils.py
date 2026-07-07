"""Loader / IO helpers specific to the Appendix-B baseline experiments.

Generic model specs, step schedules, and snapshot helpers live in
``readout.core.model_specs``; this module only carries the appendix-specific
output paths and the eval-corpus token-count loader.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import torch

from readout.core.data import get_device
from readout.core.model_specs import (  # noqa: F401  (re-exports for callers)
    DEFAULT_STEPS_32,
    OLMO_2_7B,
    PYTHIA_1B,
    PYTHIA_6_9B,
    PYTHIA_160M,
    SPECS,
    ModelSpec,
    auto_steps_for,
    iter_snapshots,
    snap_path_for,
)
from readout.core.paths import repo_root, ssd_path

REPO = repo_root()
DEFAULT_SNAP_DIR = Path(os.environ.get("WU_SNAP_DIR", str(ssd_path("snapshots"))))
# Canonical released Pythia eval corpus — same resolution as
# readout.dynamics.temporal_patch.CORPUS_TOKENS_PYTHIA (see docs/DATA.md).
# Override with the READOUT_EVAL_TOKENS env var or a per-call path.
EVAL_TOKENS_PT = Path(
    os.environ.get(
        "READOUT_EVAL_TOKENS",
        str(ssd_path("hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt")),
    )
)
FIG_DIR = REPO / "figures/crosscoder_main/appendix_validation"
RAW_DIR = REPO / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/baselines/raw"


def device() -> torch.device:
    """Best available device (cuda > mps > cpu)."""
    return torch.device(get_device())


def load_snapshot(p: Path) -> torch.Tensor:
    from readout.crosscoder.snapshots import load_snapshot_at

    return load_snapshot_at(p)


def load_token_corpus_counts(vocab: int, eval_tokens: Path | None = None) -> torch.Tensor:
    """Eval-corpus token counts in shape ``(vocab,)``, int64.

    Source: the canonical released eval corpus (``EVAL_TOKENS_PT``). Reflects
    eval-corpus frequency, not Pile training frequency; label plots accordingly.
    """
    path = Path(eval_tokens) if eval_tokens is not None else EVAL_TOKENS_PT
    if not path.exists():
        raise FileNotFoundError(
            f"eval-tokens corpus not found at {path}. Point READOUT_EVAL_TOKENS "
            "(or the eval_tokens argument) at the released corpus: "
            "<UM_SSD_ROOT>/hf_release/parameter-trajectory-crosscoders/"
            "evaluation/eval-corpus/eval_tokens.pt (see docs/DATA.md); "
            "UM_SSD_ROOT selects the SSD root."
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    ids = data["ids"].tolist()
    counts = Counter(ids)
    out = torch.zeros(vocab, dtype=torch.int64)
    for tid, c in counts.items():
        if 0 <= tid < vocab:
            out[tid] = c
    return out


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
