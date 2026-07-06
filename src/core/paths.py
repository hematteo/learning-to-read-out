"""Canonical SSD path resolution for the W_U / W_E crosscoder project.

Honours ``UM_SSD_ROOT`` (default: ``<repo>/local_snapshots``) so the same code
runs on a laptop or a cluster after ``export UM_SSD_ROOT=...``.

Data layout under ``UM_SSD_ROOT``:

    <ssd>/snapshots/<short>/<slug>_step<N>_wu.pt
    <ssd>/hf_release/parameter-trajectory-crosscoders/<model>/<kind>/cross-snapshot-32/d<N>/seed<S>.safetensors
    <ssd>/derived/aggregates/aggregates_<run-id>.pt

Use:

    from src.core.paths import snapshot_path, release_path
    p = snapshot_path("EleutherAI/pythia-160m", step=1000)
    cc = release_path("pythia-1b", kind="W_U", dim=24576, seed=0)

The legacy ``${UM_SSD_ROOT}/wu_crosscoder/...`` symlink farm is
intentionally not exposed here. New code should resolve through these helpers;
old code that still hardcodes ``wu_crosscoder/`` is being migrated.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SSD_ROOT = PROJECT / "local_snapshots"


def repo_root() -> Path:
    """Repo root, robust to scripts moving up/down the tree.

    Walks up from this module to the first dir containing ``.git`` or
    ``pyproject.toml``. Use instead of ``Path(__file__).resolve().parents[N]``
    in experiment scripts so path resolution survives reorgs.
    """
    for p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    raise RuntimeError(
        "could not find repo root from src/core/paths.py "
        "(no .git or pyproject.toml in any parent dir)"
    )


_MODEL_SHORT = {
    "EleutherAI/pythia-160m": "pythia-160m",
    "EleutherAI/pythia-1b": "pythia-1b",
    "EleutherAI/pythia-6.9b": "pythia-6.9b",
    "allenai/OLMo-2-1124-7B": "OLMo-2-1124-7B",
}


def ssd_root() -> Path:
    """Root of the canonical project SSD.

    Override with ``UM_SSD_ROOT``; defaults to ``<repo>/local_snapshots``.
    """
    return Path(os.environ.get("UM_SSD_ROOT", str(DEFAULT_SSD_ROOT)))


def ssd_path(*parts: str) -> Path:
    """Join ``parts`` under :func:`ssd_root` (replaces hardcoded SSD literals)."""
    return ssd_root().joinpath(*parts)


def model_short(model_name: str) -> str:
    """Per-model dir name on the SSD (raises on unknown model)."""
    if model_name in _MODEL_SHORT:
        return _MODEL_SHORT[model_name]
    raise KeyError(
        f"Unknown model_name {model_name!r}. Add it to src.core.paths._MODEL_SHORT."
    )


def model_slug(model_name: str) -> str:
    """Filename-safe slug used in snapshot filenames (HF '/' -> '_')."""
    return model_name.replace("/", "_")


def snapshot_dir(model_name: str) -> Path:
    """Per-model snapshot directory.

    e.g. ``EleutherAI/pythia-160m`` -> ``<ssd>/snapshots/pythia-160m/``
    """
    return ssd_root() / "snapshots" / model_short(model_name)


def snapshot_path(model_name: str, step: int, *, kind: str = "wu") -> Path:
    """Path to a single snapshot .pt file.

    kind="wu" -> unembedding row matrix; kind="we" -> input-embedding rows.
    """
    if kind not in ("wu", "we"):
        raise ValueError(f"kind must be 'wu' or 'we', got {kind!r}")
    return snapshot_dir(model_name) / f"{model_slug(model_name)}_step{step}_{kind}.pt"


def release_root() -> Path:
    """Root of the published HF crosscoder release tree."""
    return ssd_root() / "hf_release" / "parameter-trajectory-crosscoders"


def release_path(
    model_short_name: str,
    kind: str = "W_U",
    *,
    dim: int,
    seed: int,
    snapshots: int = 32,
) -> Path:
    """Path to a published crosscoder safetensors file.

    e.g. ``release_path("pythia-1b", "W_U", dim=24576, seed=0)``
    -> ``<ssd>/hf_release/.../pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors``
    """
    return (
        release_root()
        / model_short_name
        / kind
        / f"cross-snapshot-{snapshots}"
        / f"d{dim}"
        / f"seed{seed}.safetensors"
    )


def aggregate_path(run_id: str) -> Path:
    """Per-(model, d_sae, seed) aggregate tensor under ``derived/aggregates/``."""
    return ssd_root() / "derived" / "aggregates" / f"aggregates_{run_id}.pt"


def crosscoder_main_derived_dir() -> Path:
    """In-repo dir holding per-run aggregates produced by the main analysis.

    Currently ``experiments/crosscoders/crosscoder_main/derived/``; centralised
    here so call sites don't hardcode the path.
    """
    return PROJECT / "experiments" / "crosscoders" / "crosscoder_main" / "derived"

