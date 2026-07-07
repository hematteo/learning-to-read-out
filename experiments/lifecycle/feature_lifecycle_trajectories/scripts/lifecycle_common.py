"""Shared run table plumbing for the lifecycle metric scripts in this directory.

Not an entry point. Holds the `Run` dataclass, the rates/decoder-norm loaders,
and the decoder-geometry cache machinery that `find_reorganization_steps.py`
and `plot_selected_wishbone.py` both use, so the entry scripts never import
each other (same pattern as crosscoder_we/scripts/we_common.py).

Cache layout (all under `CACHE`, gitignored, regenerated on demand):
  - `<key>_decoder_norms.npy`               (K, D) — written by
    plot_normalized_trajectories.py; run that script first.
  - `<geometry key>_adjacent_rotation_radians.npy` (K-1, D) and
    `<geometry key>_cos_to_terminal.npy`    (K, D) — derived on first miss by
    `ensure_geometry_cache` from the released crosscoder checkpoints
    (decoder tensor W_D), using the canonical cosine conventions in
    readout.dynamics.metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from readout.core.model_specs import DEFAULT_STEPS_32 as _PYTHIA_STEPS_LIST
from readout.core.model_specs import OLMO_STEPS_32 as _OLMO_STEPS_LIST
from readout.core.paths import release_root, repo_root
from readout.dynamics import metrics as dynamics_metrics

REPO = repo_root()
CACHE = REPO / "figures/feature_lifecycle_trajectories/section52_lifecycle/cache"

PYTHIA_STEPS_32 = np.asarray(_PYTHIA_STEPS_LIST, dtype=np.int64)
OLMO_STEPS_32 = np.asarray(_OLMO_STEPS_LIST, dtype=np.int64)

# Released crosscoder checkpoints carrying the (K, D, d) decoder tensor the
# geometry caches are derived from, keyed by Run.key (same instruments as
# plot_normalized_trajectories.SELECTED_RUNS / the README input list).
GEOMETRY_CHECKPOINTS: dict[str, Path] = {
    "pythia160m_d24576": release_root() / "pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors",
    "pythia1b_d24576": release_root() / "pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors",
    "pythia69b_d32768": release_root() / "pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors",
    "olmo2_7b_d32768": release_root() / "olmo-2-7b/W_U/cross-snapshot-32/d32768/seed0.safetensors",
}


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    rates_path: Path
    steps: np.ndarray
    xticks: tuple[int, ...]
    slug: str | None = None  # output stem for wishbone sidecars; defaults to key
    rates_key: str | None = None  # entry in a rates_per_seed blob; None -> first
    geometry_key: str | None = None  # geometry cache key; None -> key

    def __post_init__(self) -> None:
        if self.slug is None:
            object.__setattr__(self, "slug", self.key)


def geometry_cache_key(run: Run) -> str:
    return run.geometry_key or run.key


def load_rates(run: Run, *, dtype: np.dtype | type = np.float32) -> np.ndarray:
    """Load the (K, D) firing-rate matrix for a run from its rates blob."""
    if not run.rates_path.exists():
        raise FileNotFoundError(run.rates_path)
    blob = torch.load(run.rates_path, map_location="cpu", weights_only=False)
    if "rates" in blob:
        rates = blob["rates"]
    elif "rates_per_seed" in blob:
        if run.rates_key is not None:
            rates = blob["rates_per_seed"][run.rates_key]
        else:
            rates = next(iter(blob["rates_per_seed"].values()))
    else:
        raise KeyError(f"{run.rates_path} has no rates or rates_per_seed")
    if isinstance(rates, torch.Tensor):
        rates = rates.detach().cpu().numpy()
    rates = rates.astype(dtype)
    if rates.shape[0] != run.steps.size:
        raise ValueError(f"{run.key}: rates shape {rates.shape}, steps {run.steps.shape}")
    return rates


def decoder_norms_path(run: Run) -> Path:
    return CACHE / f"{run.key}_decoder_norms.npy"


def load_decoder_norms(run: Run, *, dtype: np.dtype | type = np.float32) -> np.ndarray:
    path = decoder_norms_path(run)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run plot_normalized_trajectories.py first to build the decoder-norm cache"
        )
    return np.load(path).astype(dtype)


def _geometry_from_decoder(W_D: torch.Tensor, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """(K, D, d) decoder -> (adjacent rotation in radians (K-1, D), cos-to-terminal (K, D)).

    Chunked over features so the float64 intermediate stays small for the
    d=32768 x d_model=4096 instruments; cosine conventions come from
    readout.dynamics.metrics (adjacent_rotation / terminal_direction_distance
    both return 1 - cos).
    """
    K, D, _ = W_D.shape
    rotation = np.empty((K - 1, D), dtype=np.float32)
    cos_terminal = np.empty((K, D), dtype=np.float32)
    for start in range(0, D, chunk):
        end = min(start + chunk, D)
        block = W_D[:, start:end, :]  # (K, C, d)
        one_minus_cos_adj = dynamics_metrics.adjacent_rotation(block)  # (K-1, C)
        rotation[:, start:end] = np.arccos(np.clip(1.0 - one_minus_cos_adj, -1.0, 1.0))
        one_minus_cos_term = dynamics_metrics.terminal_direction_distance(block)  # (K, C)
        cos_terminal[:, start:end] = np.clip(1.0 - one_minus_cos_term, -1.0, 1.0)
    return rotation, cos_terminal


def ensure_geometry_cache(run: Run) -> tuple[Path, Path]:
    """Return (rotation_path, cos_terminal_path), deriving both on cache miss."""
    key = geometry_cache_key(run)
    rotation_path = CACHE / f"{key}_adjacent_rotation_radians.npy"
    cos_terminal_path = CACHE / f"{key}_cos_to_terminal.npy"
    if rotation_path.exists() and cos_terminal_path.exists():
        return rotation_path, cos_terminal_path

    ckpt_path = GEOMETRY_CHECKPOINTS.get(run.key)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"{rotation_path.name} / {cos_terminal_path.name} missing and the released "
            f"crosscoder checkpoint to derive them from is unavailable ({ckpt_path}); "
            "see the experiment README Inputs section"
        )
    from readout.crosscoder.checkpoints import load_checkpoint

    print(f"[{run.key}] deriving decoder-geometry cache from {ckpt_path}")
    W_D = load_checkpoint(ckpt_path).state_dict["W_D"].float()  # (K, D, d)
    if W_D.shape[0] != run.steps.size:
        raise ValueError(f"{run.key}: checkpoint has K={W_D.shape[0]} heads, steps {run.steps.size}")
    rotation, cos_terminal = _geometry_from_decoder(W_D)
    del W_D
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(rotation_path, rotation)
    np.save(cos_terminal_path, cos_terminal)
    print(f"[{run.key}] wrote {rotation_path.name} and {cos_terminal_path.name}")
    return rotation_path, cos_terminal_path
