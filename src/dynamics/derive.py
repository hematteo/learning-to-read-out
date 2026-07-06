"""Derive and persist per-row artifacts for the analysis table.

For each (run_id, seed) row, compute (or skip-if-cached) the artifacts named
in plan §2:
  - rates_norm:           (K, D)   relative-to-peak normalization
  - rotation:             (K-1, D) adjacent decoder rotation 1 - cos(d_k, d_{k+1})
  - rotation_weight:      (K-1, D) sqrt(||d_k|| * ||d_{k+1}||)
  - direction_to_terminal:(K, D)   1 - cos(d_k, d_K)
  - lifecycle.npz:        per-feature first_active, peak_step, lifetime,
                          stabilized_step, drift_score, never_active_mask
  - cusum.npz:            per-feature CUSUM-max on rates / norms / rotation
  - cusum_null.pt:        signed-CUSUM permutation null on rates (existing
                          100k caches preferred; otherwise 10k computed here)

Artifacts live under <DERIVED_ROOT>/<run_id>__seed{S}/. Each file is
content-addressable enough for a downstream reader: present means computed
under the current metric_version. Bumping metric_version invalidates the
cache (callers should pass --force-rederive or clear the directory).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.core.paths import crosscoder_main_derived_dir, ssd_path
from src.crosscoder.checkpoint_loaders import load_run

from . import metrics
from .discovery import RunRow


def default_derived_root() -> Path:
    env = os.environ.get("DYNAMICS_DERIVED_ROOT")
    if env:
        return Path(env)
    return crosscoder_main_derived_dir()


@dataclass
class DerivedPaths:
    """Filesystem paths to per-row derived artifacts (None if not present)."""

    rates_norm: str | None = None
    rotation: str | None = None
    rotation_weight: str | None = None
    direction_to_terminal: str | None = None
    lifecycle: str | None = None
    cusum: str | None = None
    cusum_null: str | None = None

    def as_dict(self) -> dict:
        return {
            "rates_norm_path": self.rates_norm,
            "rotation_path": self.rotation,
            "rotation_weight_path": self.rotation_weight,
            "direction_to_terminal_path": self.direction_to_terminal,
            "lifecycle_path": self.lifecycle,
            "cusum_path": self.cusum,
            "cusum_null_path": self.cusum_null,
        }


# Locked permutation-null caches at n_perms=10^5 for runs that already have
# them. Keyed by run_id; the value is a callable taking the seed and
# returning a Path or None. The analysis table prefers these over freshly-computed 10^4.
_KNOWN_NULL_CACHES: dict[str, Callable[[int], Path | None]] = {
    "run3_wu_160m_d8192": lambda seed: Path(
        str(
            ssd_path(
                "wu_crosscoder",
                "run3_ge_exact_results",
                "perm_results_100k",
                "perm_100k_canonical.pt",
            )
        )
    ),
}


def known_null_cache(row: RunRow) -> Path | None:
    """Return the existing 100k null cache for a row, if one is known to exist."""
    factory = _KNOWN_NULL_CACHES.get(row.run_id)
    if factory is None:
        return None
    p = factory(row.seed)
    return p if p is not None and Path(p).exists() else None


def _row_dir(row: RunRow, root: Path) -> Path:
    sub = f"{row.run_id}__seed{row.seed}"
    return root / sub


def existing_paths(row: RunRow, root: Path | None = None) -> DerivedPaths:
    """Probe disk for already-cached artifacts for this row.

    Used by build_analysis_table.py to populate path fields without re-running
    the (potentially slow) derivation.
    """
    root = root if root is not None else default_derived_root()
    d = _row_dir(row, root)

    def _maybe(p: Path) -> str | None:
        return str(p) if p.exists() else None

    paths = DerivedPaths(
        rates_norm=_maybe(d / "rates_norm.npy"),
        rotation=_maybe(d / "rotation.npy"),
        rotation_weight=_maybe(d / "rotation_weight.npy"),
        direction_to_terminal=_maybe(d / "direction_to_terminal.npy"),
        lifecycle=_maybe(d / "lifecycle.npz"),
        cusum=_maybe(d / "cusum.npz"),
        cusum_null=_maybe(d / "cusum_null.pt"),
    )
    if paths.cusum_null is None:
        kn = known_null_cache(row)
        if kn is not None:
            paths.cusum_null = str(kn)
    return paths


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32, copy=False))


def derive_row(
    row: RunRow,
    *,
    root: Path | None = None,
    derive_rotation: bool = True,
    derive_cusum_null: bool = False,
    n_perms: int = 10_000,
    force: bool = False,
    device: str = "cpu",
) -> DerivedPaths:
    """Compute and cache derived artifacts for one (run, seed) row.

    Skip-existing semantics: artifacts already on disk are kept unless
    `force=True`. Rotation and direction-to-terminal require the full
    decoder weights tensor (K, D, d_model) loaded from the checkpoint;
    pass `derive_rotation=False` on machines where that would OOM.
    """
    root = root if root is not None else default_derived_root()
    out_dir = _row_dir(row, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = existing_paths(row, root)

    need_rotation = derive_rotation and (
        force or paths.rotation is None or paths.rotation_weight is None or paths.direction_to_terminal is None
    )
    need_lifecycle = force or paths.lifecycle is None
    need_cusum = force or paths.cusum is None
    need_rates_norm = force or paths.rates_norm is None
    # cusum_null may be served by a known 100k cache (existing_paths handled
    # that). Only compute fresh when the user opted in AND no usable cache
    # exists, OR when force=True.
    need_cusum_null = derive_cusum_null and (force or paths.cusum_null is None)

    need_rates_arr = need_rotation or need_lifecycle or need_cusum or need_rates_norm or need_cusum_null
    if not need_rates_arr:
        return paths

    arr = load_run(
        run_id=row.run_id,
        seed=row.seed,
        d_sae=row.d_sae,
        ckpt_path=row.ckpt_path,
        rates_cache=row.rates_path,
        norms_cache=row.norms_path,
        seed_index_in_cache=row.seed_index_in_cache,
        keep_decoder_weights=need_rotation,
        steps_fallback=row.steps,
        allow_missing_rates=True,
    )
    rates = arr.rates  # (K, D) or None for decoder-only rows
    norms = arr.decoder_norms  # (K, D)
    D = norms.shape[1]
    have_rates = rates is not None

    if need_rates_norm and have_rates:
        peak = rates.max(axis=0)  # (D,)
        rates_norm = np.zeros_like(rates, dtype=np.float32)
        nz = peak > 0
        rates_norm[:, nz] = (rates[:, nz] / peak[None, nz]).astype(np.float32)
        _save_npy(out_dir / "rates_norm.npy", rates_norm)
        paths.rates_norm = str(out_dir / "rates_norm.npy")

    rotation = None
    if need_rotation:
        if arr.decoder_weights is None:
            raise RuntimeError(f"derive_rotation=True but decoder weights unavailable for {row.run_id} seed {row.seed}")
        rotation = metrics.adjacent_rotation(arr.decoder_weights).astype(np.float32)
        _save_npy(out_dir / "rotation.npy", rotation)
        rot_w = np.sqrt(norms[:-1] * norms[1:]).astype(np.float32)
        _save_npy(out_dir / "rotation_weight.npy", rot_w)
        d_term = metrics.terminal_direction_distance(arr.decoder_weights).astype(np.float32)
        _save_npy(out_dir / "direction_to_terminal.npy", d_term)
        # Drop the heavy decoder tensor; lifecycle only needs scalar columns.
        arr.decoder_weights = None
        paths.rotation = str(out_dir / "rotation.npy")
        paths.rotation_weight = str(out_dir / "rotation_weight.npy")
        paths.direction_to_terminal = str(out_dir / "direction_to_terminal.npy")
    elif paths.rotation is not None:
        rotation = np.load(paths.rotation)

    if need_lifecycle and have_rates:
        d_term_arr = np.load(paths.direction_to_terminal) if paths.direction_to_terminal is not None else None
        lc = metrics.lifecycle(
            rates,
            decoder_norms=norms,
            rotation=rotation,
            direction_to_terminal=d_term_arr,
        )
        np.savez_compressed(
            out_dir / "lifecycle.npz",
            first_active=lc.first_active.astype(np.int32),
            peak_step=lc.peak_step.astype(np.int32),
            lifetime=lc.lifetime.astype(np.float32),
            stabilized_step=lc.stabilized_step.astype(np.int32),
            drift_score=lc.drift_score.astype(np.float32),
            never_active_mask=lc.never_active_mask,
        )
        paths.lifecycle = str(out_dir / "lifecycle.npz")

    if need_cusum:
        if have_rates:
            cusum_rates = metrics.cusum_max(rates).astype(np.float32)
        else:
            cusum_rates = np.full(D, np.nan, dtype=np.float32)
        cusum_norms = metrics.cusum_max(norms).astype(np.float32)
        if rotation is not None:
            cusum_rotation = metrics.cusum_max(rotation).astype(np.float32)
        else:
            cusum_rotation = np.full(D, np.nan, dtype=np.float32)
        np.savez_compressed(
            out_dir / "cusum.npz",
            cusum_rates=cusum_rates,
            cusum_norms=cusum_norms,
            cusum_rotation=cusum_rotation,
        )
        paths.cusum = str(out_dir / "cusum.npz")

    if need_cusum_null:
        kn = known_null_cache(row)
        if kn is not None and not force:
            paths.cusum_null = str(kn)
        elif not have_rates:
            # The signed-CUSUM null is defined on firing rates; without a rate
            # matrix we cannot compute it. Leave cusum_null_path unset.
            pass
        else:
            from src.baselines.permutation_test_fast import (  # noqa: E402
                permutation_test_vectorized,
            )

            obs, p, null, obs_z, eff_b = permutation_test_vectorized(
                torch.from_numpy(rates),
                n_permutations=n_perms,
                batch=1000,
                seed=row.seed,
                device=device,
            )
            null_path = out_dir / "cusum_null.pt"
            torch.save(
                {
                    "observed": float(obs),
                    "p_value": float(p),
                    "n_permutations": int(n_perms),
                    "effective_batch": int(eff_b),
                    "null_mean": float(null.mean()),
                    "null_std": float(null.std()),
                    "null_max": float(null.max()),
                    "null_p95": float(np.quantile(null, 0.95)),
                    "null_p99": float(np.quantile(null, 0.99)),
                    "obs_in_null_sigma_units": float(obs_z),
                    "n_null_geq_obs": int((null >= obs).sum()),
                    "stat_kind": "signed_cusum_max_feature",
                },
                null_path,
            )
            paths.cusum_null = str(null_path)

    return paths
