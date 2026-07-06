"""Canonical model and snapshot-step registry for trajectory crosscoder code.

Centralises the per-model HF names, ``ModelSpec`` records, and the
cross-snap-32 step schedules so analysis scripts don't carry their own
copies. Step lists were previously duplicated across
``experiments/baselines/dense_readout_diagnostics/``,
``experiments/lifecycle/feature_lifecycle_trajectories/``, and
``experiments/crosscoders/crosscoder_main/scripts/appendix_validation/baselines/``; the
canonical source is now this module.

This module is THE single source of truth for snapshot-step schedules: other
modules (and scripts) must import ``DEFAULT_STEPS_32`` from here rather than
re-typing the step list. The training-time copy
``readout.crosscoder.wu_adapter.DEFAULT_STEPS`` is the same Pythia schedule but lives
in a training module that pulls heavy deps; analysis code should import
``DEFAULT_STEPS_32`` from here.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from readout.core.paths import snapshot_path as _snapshot_path
from readout.core.paths import ssd_root

# ---- step schedules ---------------------------------------------------------

# Ge et al. 2026 (arxiv 2509.17196) §4: 32-snapshot schedule used by every
# Pythia cross-snapshot-32 crosscoder in the paper.
DEFAULT_STEPS_32: list[int] = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    14000, 21000, 27000, 34000,
    47000, 61000, 75000, 89000, 102000, 116000, 130000, 143000,
]  # fmt: skip

# OLMo-2-7B 32-snapshot schedule (different early/late grids than Pythia).
OLMO_STEPS_32: list[int] = [
    150, 600, 700, 850, 900,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    14000, 21000, 27000, 34000, 47000,
    110000, 173000, 236000, 299000, 362000, 425000, 488000,
    614000, 677000, 740000, 803000, 866000, 928000,
]  # fmt: skip


# ---- model specs ------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Per-model identity used by snapshot loaders and baselines."""

    name: str  # full HF id, e.g. "EleutherAI/pythia-1b"
    label: str  # short label, e.g. "pythia-1b"
    d_model: int
    vocab: int


PYTHIA_160M = ModelSpec("EleutherAI/pythia-160m", "pythia-160m", 768, 50304)
PYTHIA_1B = ModelSpec("EleutherAI/pythia-1b", "pythia-1b", 2048, 50304)
PYTHIA_6_9B = ModelSpec("EleutherAI/pythia-6.9b", "pythia-6.9b", 4096, 50432)
OLMO_2_7B = ModelSpec("allenai/OLMo-2-1124-7B", "olmo-2-7b", 4096, 100352)

SPECS: dict[str, ModelSpec] = {s.label: s for s in (PYTHIA_160M, PYTHIA_1B, PYTHIA_6_9B, OLMO_2_7B)}

# label -> HF name; useful for scripts keyed on short labels.
MODEL_HF_NAMES: dict[str, str] = {label: spec.name for label, spec in SPECS.items()}

# Default cross-snap-32 schedule per model label.
DEFAULT_STEPS_BY_MODEL: dict[str, list[int]] = {
    "pythia-160m": DEFAULT_STEPS_32,
    "pythia-1b": DEFAULT_STEPS_32,
    "pythia-6.9b": DEFAULT_STEPS_32,
    "olmo-2-7b": OLMO_STEPS_32,
}


# ---- snapshot path / load helpers ------------------------------------------


def default_snap_dir() -> Path:
    """Snapshot root on disk. ``WU_SNAP_DIR`` overrides ``UM_SSD_ROOT/snapshots``."""
    env = os.environ.get("WU_SNAP_DIR")
    if env:
        return Path(env)
    return ssd_root() / "snapshots"


def snap_path_for(
    model_name: str,
    step: int,
    *,
    matrix: str = "W_U",
    snap_dir: Path | None = None,
) -> Path:
    """Resolve a snapshot path, with a per-model subdir fallback to a flat dir.

    Wraps ``readout.core.paths.snapshot_path`` for the canonical layout, and falls
    back to a flat ``<snap_dir>/<slug>_step<N>_<wu|we>.pt`` layout used by some
    older caches.
    """
    suffix = "wu" if matrix == "W_U" else "we"
    if snap_dir is None:
        p = _snapshot_path(model_name, step, kind=suffix)
        if p.exists():
            return p
        snap_dir = default_snap_dir()
    base = model_name.split("/")[-1]
    slug = model_name.replace("/", "_")
    candidates = [
        snap_dir / base / f"{slug}_step{step}_{suffix}.pt",
        snap_dir / f"{slug}_step{step}_{suffix}.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No snapshot for {model_name} step={step} ({suffix})")


def iter_snapshots(
    spec: ModelSpec,
    steps: Iterable[int] = DEFAULT_STEPS_32,
    *,
    snap_dir: Path | None = None,
    matrix: str = "W_U",
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ``(step, W_t)`` pairs from disk one at a time. CPU tensors, fp32."""
    from readout.crosscoder.snapshots import load_snapshot_at

    for s in steps:
        yield (
            s,
            load_snapshot_at(snap_path_for(spec.name, s, matrix=matrix, snap_dir=snap_dir)),
        )


def auto_steps_for(
    spec: ModelSpec,
    *,
    snap_dir: Path | None = None,
    matrix: str = "W_U",
) -> list[int]:
    """All available step indices on disk for ``spec``, sorted ascending.

    Discovers files matching ``{slug}_step{N}_{wu|we}.pt`` under either
    ``<snap_dir>/<base>/`` or ``<snap_dir>/`` (the legacy flat layout).
    """
    base = spec.name.split("/")[-1]
    slug = spec.name.replace("/", "_")
    suffix = "wu" if matrix == "W_U" else "we"
    if snap_dir is None:
        snap_dir = default_snap_dir()
    candidates: list[int] = []
    for d in (snap_dir / base, snap_dir):
        if not d.exists():
            continue
        for p in d.glob(f"{slug}_step*_{suffix}.pt"):
            if p.name.startswith("._"):
                continue
            m = re.search(r"_step(\d+)_", p.name)
            if m:
                candidates.append(int(m.group(1)))
    return sorted(set(candidates))
