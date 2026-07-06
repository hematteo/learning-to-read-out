"""Lifecycle, rotation, and CUSUM metrics for the developmental-dynamics paper.

Operational definitions are locked to a fixed `metric_version` (see `provenance.py`).
This module is the single source of truth for them; phase scripts call into
here and never re-derive metrics ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Defaults from plan §3. Overridden in phase scripts via SI sensitivity sweeps.
ALPHA_REL = 0.10  # relative-to-peak threshold
ABS_FLOOR = 1e-4  # absolute floor; tiebreaker for sparse features
M_CONSEC = 2  # consecutive-snapshot requirement for first-active
EPS_RATE = 0.20  # stabilization rate tolerance, fraction of terminal median
DELTA_DIR = 0.05  # stabilization direction tolerance, 1 - cos
DEAD_NORM_FRAC = 0.10  # active-mask: decoder norm must be >= this * snapshot median


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


@dataclass
class Lifecycle:
    """Per-feature lifecycle table.

    All arrays are length D (number of features). Times are stored as
    snapshot indices in [0, K-1], with K the number of snapshots; convert
    to step numbers via the run's `steps` list outside this module.
    """

    first_active: np.ndarray  # int, K means never active
    peak_step: np.ndarray  # int in [0, K-1]
    lifetime: np.ndarray  # float in [0, 1], fraction of snapshots above alpha*peak
    stabilized_step: np.ndarray  # int, K means drifting / never stabilized
    drift_score: np.ndarray  # float, mean rotation post-first-active (norm-weighted)
    never_active_mask: np.ndarray  # bool, peak rate below ABS_FLOOR


def lifecycle(
    rates: np.ndarray,
    decoder_norms: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    direction_to_terminal: np.ndarray | None = None,
    *,
    alpha: float = ALPHA_REL,
    m: int = M_CONSEC,
    abs_floor: float = ABS_FLOOR,
    eps: float = EPS_RATE,
    delta: float = DELTA_DIR,
) -> Lifecycle:
    """Compute per-feature lifecycle table from (K, D) rates.

    Definitions (plan §3):
      - first_active: earliest k with rate > max(abs_floor, alpha*peak) for m consecutive snapshots
      - peak_step:    argmax_k rates_kj, ties broken by larger k
      - lifetime:     fraction of snapshots where rate > alpha*peak
      - stabilized:   earliest k* such that for all k' >= k*:
                      |rate - tilde_r| < eps*tilde_r AND 1 - cos(d_k', d_K) < delta
                      where tilde_r is the median of the last quartile
      - drift_score:  mean adjacent rotation (1 - cos) over post-first-active snapshots,
                      norm-weighted by sqrt(||d_k|| * ||d_{k+1}||) to suppress dead-feature noise
    """
    rates = _to_np(rates).astype(np.float64)  # (K, D)
    K, D = rates.shape
    peak = rates.max(axis=0)  # (D,)

    # Peak step with later-wins tiebreak: argmax over flipped axis, then map back.
    peak_step = (K - 1) - np.argmax(rates[::-1], axis=0)
    peak_step[peak == 0] = K  # sentinel for never-active features

    never_active = peak < abs_floor

    # Active mask: rate > max(abs_floor, alpha * peak), shape (K, D).
    thr = np.maximum(abs_floor, alpha * peak)  # (D,)
    active = rates > thr[None, :]

    # First-active: earliest k where active is true for m consecutive snapshots.
    first_active = np.full(D, K, dtype=np.int64)
    if m == 1:
        idx = np.argmax(active, axis=0)
        has = active.any(axis=0)
        first_active[has] = idx[has]
    else:
        # Rolling AND window of size m.
        # window[k, j] is True iff active[k:k+m, j].all().
        win_count = K - m + 1
        if win_count > 0:
            # Slow but readable; D ~ 8k--100k, K ~ 32, so this is fine.
            window = np.ones((win_count, D), dtype=bool)
            for off in range(m):
                window &= active[off : off + win_count]
            idx = np.argmax(window, axis=0)
            has = window.any(axis=0)
            first_active[has] = idx[has]

    first_active[never_active] = K

    # Lifetime.
    lifetime = active.mean(axis=0)  # (D,)
    lifetime[never_active] = 0.0

    # Stabilization: rate within eps of last-quartile median and direction
    # within delta of terminal.
    q_start = max(1, K - K // 4)
    tilde_r = np.median(rates[q_start:], axis=0)  # (D,)

    rate_close = np.abs(rates - tilde_r[None, :]) < (eps * np.abs(tilde_r[None, :]) + 1e-12)
    if direction_to_terminal is not None:
        dterm = _to_np(direction_to_terminal).astype(np.float64)
        if dterm.shape != (K, D):
            raise ValueError(f"direction_to_terminal must be {(K, D)}, got {dterm.shape}")
        dir_close = dterm < delta
    elif rotation is not None:
        rot = _to_np(rotation).astype(np.float64)  # (K-1, D), 1 - cos(d_k, d_{k+1})
        if rot.shape != (K - 1, D):
            raise ValueError(f"rotation must be {(K - 1, D)}, got {rot.shape}")
        # Backward-compatible fallback for older scripts. Paper-facing analyses
        # should pass direction_to_terminal from terminal_direction_distance().
        dir_close = np.ones_like(rate_close, dtype=bool)
        dir_close[:-1] = rot < delta
    else:
        dir_close = np.ones_like(rate_close, dtype=bool)

    close = rate_close & dir_close  # (K, D)
    # Stabilized step: smallest k such that close[k:, :] is all True.
    stabilized_step = np.full(D, K, dtype=np.int64)
    # Walk backwards to find the last "non-close" snapshot per feature.
    not_close_idx = np.where(~close)
    last_bad = np.full(D, -1, dtype=np.int64)
    if not_close_idx[0].size:
        # For each feature, find the max snapshot where close is False.
        np.maximum.at(last_bad, not_close_idx[1], not_close_idx[0])
    stabilized_step = (last_bad + 1).clip(max=K)
    stabilized_step[never_active] = K

    # Drift score: mean adjacent rotation post-first-active, norm-weighted.
    if rotation is not None and decoder_norms is not None:
        rot = _to_np(rotation).astype(np.float64)
        norms = _to_np(decoder_norms).astype(np.float64)
        # Pair weight: sqrt(||d_k|| * ||d_{k+1}||).
        pair_w = np.sqrt(norms[:-1] * norms[1:])  # (K-1, D)
        drift = np.zeros(D, dtype=np.float64)
        for j in range(D):
            fa = first_active[j]
            if fa >= K - 1:
                drift[j] = 0.0
                continue
            w = pair_w[fa:, j]
            r = rot[fa:, j]
            denom = w.sum()
            drift[j] = float((w * r).sum() / (denom + 1e-12))
        drift[never_active] = 0.0
    else:
        drift = np.zeros(D, dtype=np.float64)

    return Lifecycle(
        first_active=first_active,
        peak_step=peak_step,
        lifetime=lifetime,
        stabilized_step=stabilized_step,
        drift_score=drift,
        never_active_mask=never_active,
    )


def adjacent_rotation(decoder_weights: np.ndarray | torch.Tensor) -> np.ndarray:
    """1 - cos between adjacent decoder columns.

    Input: (K, D, d) decoder tensor (W_D in checkpoint).
    Output: (K-1, D) with values in [0, 2].
    """
    W = _to_np(decoder_weights).astype(np.float64)
    if W.ndim != 3:
        raise ValueError(f"decoder_weights must be (K, D, d), got {W.shape}")
    a = W[:-1]
    b = W[1:]
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return 1.0 - num / den


def terminal_direction_distance(
    decoder_weights: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """1 - cos between each decoder column and its terminal-snapshot direction.

    Input: (K, D, d) decoder tensor (W_D in checkpoint).
    Output: (K, D), where the final row is approximately zero for nonzero
    terminal columns. This is the direction term used by lifecycle()
    stabilization in the dynamics plan.
    """
    W = _to_np(decoder_weights).astype(np.float64)
    if W.ndim != 3:
        raise ValueError(f"decoder_weights must be (K, D, d), got {W.shape}")
    terminal = W[-1:]  # (1, D, d)
    num = (W * terminal).sum(axis=-1)
    den = np.linalg.norm(W, axis=-1) * np.linalg.norm(terminal, axis=-1) + 1e-12
    return 1.0 - num / den


def cusum_max(values: np.ndarray) -> np.ndarray:
    """Max-feature CUSUM statistic (matches transition_metric_suite convention).

    Input: (K, D), per-snapshot per-feature scalar (rate / norm / rotation).
    Output: (D,), max over snapshots of the centered cumulative sum.

    The temporal-permutation null is computed by phase scripts that need it;
    permutation_test_fast.py in src/ already implements the production version
    at 100k perms.
    """
    x = _to_np(values).astype(np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    cs = np.cumsum(centered, axis=0)
    return np.abs(cs).max(axis=0)


def active_mask(
    rates: np.ndarray,
    decoder_norms: np.ndarray,
    snapshot_idx: int,
    *,
    abs_floor: float = ABS_FLOOR,
    norm_frac: float = DEAD_NORM_FRAC,
) -> np.ndarray:
    """Per-feature boolean mask: feature is active at snapshot_idx.

    Active iff:
      rate > abs_floor AND decoder_norm > norm_frac * (median active-norm at snapshot)

    The norm filter prevents averages from being dominated by tiny-norm columns
    that never carried signal. See plan §3 (Active-feature mask).
    """
    r = _to_np(rates).astype(np.float64)
    n = _to_np(decoder_norms).astype(np.float64)
    rate_ok = r[snapshot_idx] > abs_floor
    if not rate_ok.any():
        return rate_ok
    median_active_norm = np.median(n[snapshot_idx][rate_ok])
    norm_ok = n[snapshot_idx] > norm_frac * median_active_norm
    return rate_ok & norm_ok
