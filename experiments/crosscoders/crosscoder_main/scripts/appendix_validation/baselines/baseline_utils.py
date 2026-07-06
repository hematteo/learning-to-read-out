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

from readout.core.model_specs import (  # noqa: F401  (re-exports for callers)
    DEFAULT_STEPS_32,
    OLMO_2_7B,
    PYTHIA_1B,
    PYTHIA_6_9B,
    PYTHIA_160M,
    SPECS,
    ModelSpec,
    auto_steps_for,
    device,
    iter_snapshots,
    snap_path_for,
)
from readout.core.paths import ssd_path

REPO = Path(__file__).resolve().parents[6]
DEFAULT_SNAP_DIR = Path(os.environ.get("WU_SNAP_DIR", str(ssd_path("snapshots"))))
EVAL_TOKENS_PT = REPO / "_archive/legacy_crosscoder_160m/intervention/eval_tokens.pt"
FIG_DIR = REPO / "figures/run5/a_instrument_validation"
RAW_DIR = (
    REPO
    / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/baselines/raw"
)


def load_snapshot(p: Path) -> torch.Tensor:
    from readout.crosscoder.snapshots import load_snapshot_at

    return load_snapshot_at(p)


def load_token_corpus_counts(vocab: int) -> torch.Tensor:
    """Eval-corpus token counts in shape ``(vocab,)``, int64.

    Source: the non-shipped legacy eval corpus
    ``_archive/legacy_crosscoder_160m/intervention/eval_tokens.pt``. Reflects
    eval-corpus frequency, not Pile training frequency; label plots accordingly.
    """
    data = torch.load(EVAL_TOKENS_PT, map_location="cpu", weights_only=False)
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
