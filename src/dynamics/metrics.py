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


def prospective_first_active(
    rates: np.ndarray,
    steps: list[int] | np.ndarray,
    *,
    abs_floor: float = ABS_FLOOR,
    m: int = M_CONSEC,
    baseline_step: int = 128,
) -> np.ndarray:
    """First-active step under the prospective rule from plan §3.

    Earliest k such that for m consecutive snapshots,
        rate > max(abs_floor, mu_base + 3 sigma_base),
    where (mu_base, sigma_base) come from the baseline window
    (snapshots with step <= baseline_step). If fewer than 2 baseline snapshots
    exist (coarse grid), use the earliest quartile of snapshots instead.

    Returns an int array of length D, with K (sentinel) for never-active.
    """
    rates = _to_np(rates).astype(np.float64)
    K, D = rates.shape
    steps_arr = np.asarray(steps)

    base_mask = steps_arr <= baseline_step
    if base_mask.sum() < 2:
        # Coarse grid fallback — earliest quartile.
        base_mask = np.zeros(K, dtype=bool)
        base_mask[: max(2, K // 4)] = True
    mu = rates[base_mask].mean(axis=0)
    sd = rates[base_mask].std(axis=0)
    thr = np.maximum(abs_floor, mu + 3.0 * sd)  # (D,)
    active = rates > thr[None, :]

    out = np.full(D, K, dtype=np.int64)
    if m == 1:
        idx = np.argmax(active, axis=0)
        out[active.any(axis=0)] = idx[active.any(axis=0)]
        return out
    win_count = K - m + 1
    if win_count <= 0:
        return out
    window = np.ones((win_count, D), dtype=bool)
    for off in range(m):
        window &= active[off : off + win_count]
    has = window.any(axis=0)
    out[has] = np.argmax(window, axis=0)[has]
    return out


def local_stabilization(
    rates: np.ndarray,
    rotation: np.ndarray,
    decoder_norms: np.ndarray | None = None,
    *,
    eps: float = EPS_RATE,
    delta: float = DELTA_DIR,
    w: int = 3,
    abs_floor: float = ABS_FLOOR,
    norm_frac: float = DEAD_NORM_FRAC,
) -> np.ndarray:
    """Local stabilization step from plan §3 (control for terminal-direction rule).

    Earliest k such that the next w available snapshots all have:
      adjacent rotation < delta AND
      |rate change| < eps * rolling median active rate at the snapshot pair.

    Inputs: rates (K, D), rotation (K-1, D), optional decoder_norms (K, D).
    Returns int array of length D with K sentinel for "never locally stable".
    """
    rates = _to_np(rates).astype(np.float64)
    rot = _to_np(rotation).astype(np.float64)
    K, D = rates.shape
    if rot.shape != (K - 1, D):
        raise ValueError(f"rotation must be {(K - 1, D)}, got {rot.shape}")

    # Rolling median active rate per snapshot pair: median over features that
    # are active at snapshot k (rate > floor + decoder norm above 0.1*median
    # active norm if norms supplied). Used as the denominator for "rate change".
    rate_change = np.abs(rates[1:] - rates[:-1])  # (K-1, D)
    pair_med = np.zeros(K - 1)
    for k in range(K - 1):
        rate_ok = rates[k] > abs_floor
        if decoder_norms is not None and rate_ok.any():
            n = _to_np(decoder_norms).astype(np.float64)
            med_nrm = np.median(n[k][rate_ok])
            mask = rate_ok & (n[k] > norm_frac * med_nrm)
        else:
            mask = rate_ok
        pair_med[k] = float(np.median(rates[k][mask])) if mask.any() else 0.0

    rate_close = rate_change < (eps * pair_med[:, None] + 1e-12)  # (K-1, D)
    rot_close = rot < delta  # (K-1, D)
    pair_close = rate_close & rot_close  # (K-1, D)

    out = np.full(D, K, dtype=np.int64)
    win_count = (K - 1) - w + 1  # number of valid windows of w pairs
    if win_count <= 0:
        return out
    window = np.ones((win_count, D), dtype=bool)
    for off in range(w):
        window &= pair_close[off : off + win_count]
    has = window.any(axis=0)
    out[has] = np.argmax(window, axis=0)[has]
    return out


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


def population_curves(
    rates: np.ndarray,
    decoder_norms: np.ndarray,
    rotation: np.ndarray,
    *,
    drift_thr: float = 0.05,
    stab_window: int = 2,
) -> dict:
    """Per-snapshot counts of {new, active, stabilizing, drifting, retired} features.

    Inputs:
      rates:          (K, D)
      decoder_norms:  (K, D)
      rotation:       (K-1, D), adjacent decoder rotation 1 - cos
    """
    K, D = rates.shape
    lc = lifecycle(rates, decoder_norms=decoder_norms, rotation=rotation)
    curves = {
        "new": np.zeros(K, dtype=np.int64),
        "active": np.zeros(K, dtype=np.int64),
        "stabilizing": np.zeros(K, dtype=np.int64),
        "drifting": np.zeros(K, dtype=np.int64),
        "retired": np.zeros(K, dtype=np.int64),
    }
    for k in range(K):
        am = active_mask(rates, decoder_norms, k)
        curves["active"][k] = int(am.sum())
        new_at_k = (lc.first_active == k) & (~lc.never_active_mask)
        curves["new"][k] = int(new_at_k.sum())
        # Retired: was active earlier, not active now.
        was_active = (lc.first_active < k) & (~lc.never_active_mask)
        curves["retired"][k] = int((was_active & (~am)).sum())
    # Stabilizing/drifting use rotation; rotation has K-1 entries between snapshots.
    for k in range(K - 1):
        am = active_mask(rates, decoder_norms, k)
        # Stabilizing: rotation in next stab_window snapshots is below threshold.
        end = min(K - 1, k + stab_window)
        win = rotation[k:end]
        stab = (win < drift_thr).all(axis=0) if win.size else np.zeros(D, dtype=bool)
        curves["stabilizing"][k] = int((am & stab).sum())
        curves["drifting"][k] = int((am & (~stab)).sum())
    return curves


# ---------------------------------------------------------------------------
# Lifecycle helpers: HDBSCAN-free lifecycle class, CUSUM peak step, distributional
# distances. These are the building blocks for the cross-axis robustness grid.
# ---------------------------------------------------------------------------


def lifecycle_class(
    rates: np.ndarray,
    steps: list[int] | np.ndarray,
    *,
    boundary_step: int = 1000,
    abs_floor: float = ABS_FLOOR,
) -> np.ndarray:
    """HDBSCAN-free lifecycle class label for every feature.

    Classes are derived from peak-step relative to a developmental boundary,
    not from clustering. This gives a class definition that survives at every
    scale (HDBSCAN's noise rate climbed to 78% at 1B d=16384, which is why
    clustering was abandoned for this label).

    Returns int array of length D with values:
      0 = never_active     (peak rate < abs_floor)
      1 = decay            (peak_step at a snapshot whose step <= boundary_step)
      2 = rise             (peak_step at a snapshot whose step > boundary_step)

    Plan §5 row "lifecycle-class proportions" uses Earth-mover distance on the
    {0, 1, 2} histogram across runs.
    """
    rates = _to_np(rates).astype(np.float64)
    K, D = rates.shape
    steps_arr = np.asarray(steps)
    if len(steps_arr) != K:
        raise ValueError(f"steps length {len(steps_arr)} != K={K} from rates shape")

    peak = rates.max(axis=0)
    never_active = peak < abs_floor
    peak_step_idx = (K - 1) - np.argmax(rates[::-1], axis=0)  # later-wins tiebreak
    peak_actual_step = steps_arr[peak_step_idx]

    cls = np.where(peak_actual_step <= boundary_step, 1, 2).astype(np.int64)
    cls[never_active] = 0
    return cls


def cusum_peak_step(values: np.ndarray) -> int:
    """Snapshot index of the most extreme cumulative-sum deviation.

    Input: (K, D), per-snapshot per-feature scalar (rate / norm / rotation).
    Output: scalar int in [0, K-1], the snapshot at which the population CUSUM
    reaches its peak absolute value.

    Aggregation choice: we sum centered values across features first, so the
    statistic answers "at which snapshot does the population mean drift away
    from its run-average the most?". This is the locus, not the magnitude.
    """
    x = _to_np(values).astype(np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    population_drift = centered.sum(axis=1)  # (K,)
    cs = np.cumsum(population_drift)
    return int(np.argmax(np.abs(cs)))


def em_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-Wasserstein (Earth-mover) distance between two 1D samples.

    Used by the lifecycle grid for first_active and peak_step distributions
    across runs. Thin wrapper around scipy so callers don't have to import it.
    """
    from scipy.stats import wasserstein_distance

    a = _to_np(a).astype(np.float64).ravel()
    b = _to_np(b).astype(np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def mmd_rbf(a: np.ndarray, b: np.ndarray, *, sigma: float | None = None) -> float:
    """Unbiased squared MMD between two 1D samples under a Gaussian RBF kernel.

    Bandwidth defaults to the median pairwise distance over the pooled sample
    (median heuristic). Returns 0 (modulo float noise) when distributions match;
    positive otherwise. Negative values are clipped to 0 to avoid reporting
    sub-zero MMD as artefactual signal.
    """
    a = _to_np(a).astype(np.float64).ravel()
    b = _to_np(b).astype(np.float64).ravel()
    n, m = a.size, b.size
    if n < 2 or m < 2:
        return float("nan")

    if sigma is None:
        pooled = np.concatenate([a, b])
        # Median of pairwise distances on a downsample for cheapness.
        rng = np.random.default_rng(0)
        idx = rng.choice(pooled.size, size=min(pooled.size, 512), replace=False)
        d = np.abs(pooled[idx, None] - pooled[None, idx])
        med = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
        sigma = max(med, 1e-6)

    gamma = 1.0 / (2.0 * sigma * sigma)

    def k(x, y):
        return np.exp(-gamma * (x[:, None] - y[None, :]) ** 2)

    Kaa = k(a, a)
    Kbb = k(b, b)
    Kab = k(a, b)
    # Unbiased estimator: drop the diagonal in the same-sample terms.
    np.fill_diagonal(Kaa, 0.0)
    np.fill_diagonal(Kbb, 0.0)
    mmd2 = Kaa.sum() / (n * (n - 1)) + Kbb.sum() / (m * (m - 1)) - 2.0 * Kab.mean()
    return float(max(mmd2, 0.0))


def bootstrap_peak_step_ci(
    per_feature_curve: np.ndarray,
    *,
    n_boot: int = 1000,
    ci: tuple[float, float] = (2.5, 97.5),
    rng_seed: int = 0,
) -> tuple[int, int, int, np.ndarray]:
    """Bootstrap CI on argmax(population-mean curve) by resampling features.

    Input: (K, D) per-(snapshot, feature) values. NaN cells are excluded from
    the per-snapshot mean (matches the active-mask convention used upstream).
    Resamples columns with replacement, recomputes the per-snapshot nanmean,
    takes the argmax — repeated n_boot times.

    Returns (point_idx, ci_lo_idx, ci_hi_idx, boot_peaks).
    `boot_peaks` is the full bootstrap distribution so callers can compute
    extra summaries (e.g. probability mass on a target index).
    """
    x = _to_np(per_feature_curve).astype(np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (K, D), got {x.shape}")
    K, D = x.shape
    if D == 0:
        return 0, 0, 0, np.zeros(0, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        point = int(np.nanargmax(np.nanmean(x, axis=1)))

    rng = np.random.default_rng(rng_seed)
    boot_peaks = np.empty(n_boot, dtype=np.int64)
    for b in range(n_boot):
        idx = rng.integers(0, D, size=D)
        sample = x[:, idx]
        with np.errstate(invalid="ignore"):
            boot_peaks[b] = int(np.nanargmax(np.nanmean(sample, axis=1)))
    lo, hi = np.percentile(boot_peaks, ci)
    return point, int(lo), int(hi), boot_peaks


def peak_effect_size(curve: np.ndarray, peak_idx: int) -> dict:
    """How prominent is the peak vs the rest of the curve.

    Inputs:
      curve:    (K,) population-mean curve (already aggregated across features)
      peak_idx: index of the peak (typically argmax(curve))

    Returns a dict with:
      peak_value:        curve[peak_idx]
      peak_to_baseline:  |peak| / |median of off-peak window|, off-peak = curve
                         excluding peak ± 1
      peak_to_neighbor:  |peak| / |mean of immediate neighbours|; NaN at the
                         endpoints since only one neighbour exists
      peak_z:            (peak - off-peak mean) / off-peak std
    """
    c = _to_np(curve).astype(np.float64)
    if c.ndim != 1:
        raise ValueError(f"expected 1D curve, got {c.shape}")
    K = c.size
    peak_val = float(c[peak_idx])
    mask = np.ones(K, dtype=bool)
    mask[max(0, peak_idx - 1) : min(K, peak_idx + 2)] = False
    off = c[mask]
    off_med = float(np.median(off)) if off.size else float("nan")
    off_mean = float(off.mean()) if off.size else float("nan")
    off_std = float(off.std(ddof=0)) if off.size else float("nan")

    if 0 < peak_idx < K - 1:
        neigh = float(0.5 * (c[peak_idx - 1] + c[peak_idx + 1]))
    elif peak_idx > 0:
        neigh = float(c[peak_idx - 1])
    elif peak_idx < K - 1:
        neigh = float(c[peak_idx + 1])
    else:
        neigh = float("nan")

    return {
        "peak_value": peak_val,
        "peak_to_baseline": float(abs(peak_val) / max(abs(off_med), 1e-12)),
        "peak_to_neighbor": float(abs(peak_val) / max(abs(neigh), 1e-12)) if not np.isnan(neigh) else float("nan"),
        "peak_z": float((peak_val - off_mean) / max(off_std, 1e-12)),
    }


def class_proportion_emd(a: np.ndarray, b: np.ndarray, n_classes: int = 3) -> float:
    """EM distance between two categorical class-proportion vectors.

    Treats classes as integers on a line {0, 1, ..., n_classes-1}; this matches
    how lifecycle_class is laid out (never_active=0 < decay=1 < rise=2). For a
    truly nominal categorisation, use total-variation distance instead.
    """
    from scipy.stats import wasserstein_distance

    a = _to_np(a).astype(np.int64).ravel()
    b = _to_np(b).astype(np.int64).ravel()
    bins = np.arange(n_classes + 1)
    pa = np.histogram(a, bins=bins)[0].astype(np.float64)
    pb = np.histogram(b, bins=bins)[0].astype(np.float64)
    pa /= max(pa.sum(), 1.0)
    pb /= max(pb.sum(), 1.0)
    return float(wasserstein_distance(np.arange(n_classes), np.arange(n_classes), pa, pb))
