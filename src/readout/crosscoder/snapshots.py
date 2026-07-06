"""Load already-extracted W_U / W_E snapshots from the canonical SSD layout.

This module is the canonical snapshot/W_U loading entry point: it centralizes
the ``{"W_U": ...}``-style dict-unwrapping and SSD path resolution, and other
modules should route their snapshot reads through it.

Use this for analysis. For training (which extracts on cache miss),
``readout.crosscoder.wu_adapter.load_wu_snapshot`` / ``load_snapshots`` is the right entry.

Files are resolved through ``readout.core.paths.snapshot_path`` so all callers go
through the same SSD layout (``UM_SSD_ROOT``-aware).
"""

from __future__ import annotations

from pathlib import Path

import torch

from readout.core.paths import snapshot_path


def load_snapshot_at(p: Path | str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Load a snapshot tensor from a known path. Unwraps ``{"W_U": ...}`` style dicts."""
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(p)
    x = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(x, dict):
        for k in ("W_U", "w_u", "W_E", "w_e"):
            if k in x:
                x = x[k]
                break
    return x.to(dtype) if dtype is not None else x.float()


def load_snapshot(
    model_name: str,
    step: int,
    *,
    kind: str = "wu",
    dtype: torch.dtype | None = None,
    dir_override: Path | str | None = None,
) -> torch.Tensor:
    """Load one snapshot at the canonical SSD path. Returns ``(V, d_model)``.

    ``dir_override`` resolves the canonical filename inside a different
    directory (e.g. a non-default snapshot cache) instead of
    ``snapshot_dir(model_name)``.
    """
    p = snapshot_path(model_name, step, kind=kind)
    if dir_override is not None:
        p = Path(dir_override) / p.name
    return load_snapshot_at(p, dtype=dtype)


def load_snapshots(
    model_name: str,
    steps: list[int],
    *,
    kind: str = "wu",
    dtype: torch.dtype | None = None,
    dir_override: Path | str | None = None,
) -> torch.Tensor:
    """Stack snapshots in the given step order. Returns ``(K, V, d_model)``."""
    return torch.stack(
        [load_snapshot(model_name, s, kind=kind, dtype=dtype, dir_override=dir_override) for s in steps],
        dim=0,
    )
