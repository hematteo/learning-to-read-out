"""Rule-based lifecycle profile taxonomy for crosscoder feature trajectories.

Single source of truth for the paper's profile taxonomy (Section 5.2), which
is also hand-typeset as ``tab:lifecycle-profile-rules``: the refined ruleset
implemented by :func:`classify_profiles_refined` assigns each active feature's
peak-normalized trajectory to one of the five profiles in
:data:`PROFILE_ORDER`. Both the lifecycle experiment
(``experiments/lifecycle/feature_lifecycle_trajectories``) and the OLMo
matched-checkpoint-window ablation
(``experiments/ablations/olmo_matched_checkpoint_window``) import from here;
do not re-derive these rules in experiment scripts.
"""

from __future__ import annotations

import numpy as np

# Trajectory-summary definitions shared by every ruleset.
ALIVE_THR = 0.50  # normalized level counting as "alive"
EARLY_TAU = 0.35  # tau <= EARLY_TAU is the early window
LATE_TAU = 0.65  # tau >= LATE_TAU is the late window

# tab:lifecycle-profile-rules thresholds (refined ruleset).
PERSISTENT_ALIVE_FRAC_THR = 0.70  # alive_fraction >= ... (with low range), or
PERSISTENT_RANGE_THR = 0.35  # trajectory_range <= ...
PERSISTENT_LEVEL_THR = 0.45  # min(early, mid_mean, late) >= ...
TRANSITIONAL_MID_PEAK_THR = 0.75  # mid_peak >= ...
TRANSITIONAL_PROMINENCE_THR = 0.35  # mid_peak - max(early, late) >= ...
TRANSITIONAL_LATE_LEVEL_THR = 0.30  # late_level <= ...
TRANSITIONAL_SPAN_THR = 0.35  # support_span_tau <= ...
DIRECTIONAL_LEVEL_THR = 0.50  # early (resp. late) level >= ...
DIRECTIONAL_CONTRAST_THR = 0.30  # early - late (resp. late - early) >= ...
EARLY_PEAK_TAU_THR = 0.45  # early-decaying: peak_tau <= ...
LATE_PEAK_TAU_THR = 0.55  # late-emerging: peak_tau >= ...

PROFILE_ORDER = (
    "early_decay",
    "late_emerge",
    "persistent",
    "transitional",
    "mixed",
)

PROFILE_LABELS = {
    "early_decay": "early-decaying",
    "late_emerge": "late-emerging",
    "persistent": "persistent",
    "transitional": "transitional",
    "mixed": "mixed/ambiguous",
}


def tau_from_steps(steps: np.ndarray) -> np.ndarray:
    """Map training steps to log-time tau in [0, 1]."""
    log_steps = np.log10(steps.astype(np.float64) + 1.0)
    return (log_steps - log_steps.min()) / np.clip(log_steps.max() - log_steps.min(), 1e-12, None)


def lifecycle_stats(
    traj: np.ndarray,
    steps: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-feature trajectory summaries. traj: (K, n_active), peak-normalized."""
    tau = tau_from_steps(steps)
    alive = traj >= ALIVE_THR
    peak_idx = traj.argmax(axis=0)
    first_alive = alive.argmax(axis=0)
    last_alive = alive.shape[0] - 1 - alive[::-1].argmax(axis=0)
    alive_frac = alive.mean(axis=0)
    support_span = tau[last_alive] - tau[first_alive]
    peak_tau = tau[peak_idx]
    birth_tau = tau[first_alive]
    death_tau = tau[last_alive]
    early_level = traj[tau <= EARLY_TAU].mean(axis=0)
    late_level = traj[tau >= LATE_TAU].mean(axis=0)
    mid_mask = (tau > EARLY_TAU) & (tau < LATE_TAU)
    mid_peak = traj[mid_mask].max(axis=0) if mid_mask.any() else np.zeros(traj.shape[1])
    mid_mean = traj[mid_mask].mean(axis=0) if mid_mask.any() else np.zeros(traj.shape[1])
    trajectory_range = traj.max(axis=0) - traj.min(axis=0)

    return {
        "peak_idx": peak_idx.astype(np.int16),
        "peak_step": steps[peak_idx].astype(np.int64),
        "birth_idx": first_alive.astype(np.int16),
        "birth_step": steps[first_alive].astype(np.int64),
        "death_idx": last_alive.astype(np.int16),
        "death_step": steps[last_alive].astype(np.int64),
        "peak_tau": peak_tau.astype(np.float32),
        "birth_tau": birth_tau.astype(np.float32),
        "death_tau": death_tau.astype(np.float32),
        "support_span_tau": support_span.astype(np.float32),
        "alive_fraction": alive_frac.astype(np.float32),
        "early_level": early_level.astype(np.float32),
        "late_level": late_level.astype(np.float32),
        "mid_peak": mid_peak.astype(np.float32),
        "mid_mean": mid_mean.astype(np.float32),
        "trajectory_range": trajectory_range.astype(np.float32),
    }


def classify_profiles_refined(
    traj: np.ndarray,
    steps: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assign each feature to a profile; rules applied in priority order
    (persistent > transitional > early_decay > late_emerge > mixed)."""
    stats = lifecycle_stats(traj, steps)
    early_level = stats["early_level"]
    late_level = stats["late_level"]
    mid_peak = stats["mid_peak"]
    mid_mean = stats["mid_mean"]
    peak_tau = stats["peak_tau"]
    alive_frac = stats["alive_fraction"]
    trajectory_range = stats["trajectory_range"]

    profile = np.full(traj.shape[1], "mixed", dtype=object)
    persistent = ((alive_frac >= PERSISTENT_ALIVE_FRAC_THR) & (trajectory_range <= PERSISTENT_RANGE_THR)) | (
        np.minimum(np.minimum(early_level, mid_mean), late_level) >= PERSISTENT_LEVEL_THR
    )
    profile[persistent] = "persistent"

    unmatched = profile == "mixed"
    # Transitional peaks must fall strictly inside the mid window (EARLY_TAU, LATE_TAU).
    transitional = (
        unmatched
        & (mid_peak >= TRANSITIONAL_MID_PEAK_THR)
        & ((mid_peak - np.maximum(early_level, late_level)) >= TRANSITIONAL_PROMINENCE_THR)
        & (late_level <= TRANSITIONAL_LATE_LEVEL_THR)
        & (stats["support_span_tau"] <= TRANSITIONAL_SPAN_THR)
        & (peak_tau > EARLY_TAU)
        & (peak_tau < LATE_TAU)
    )
    profile[transitional] = "transitional"

    unmatched = profile == "mixed"
    early_decay = (
        unmatched
        & (early_level >= DIRECTIONAL_LEVEL_THR)
        & ((early_level - late_level) >= DIRECTIONAL_CONTRAST_THR)
        & (peak_tau <= EARLY_PEAK_TAU_THR)
    )
    profile[early_decay] = "early_decay"

    unmatched = profile == "mixed"
    late_emerge = (
        unmatched
        & (late_level >= DIRECTIONAL_LEVEL_THR)
        & ((late_level - early_level) >= DIRECTIONAL_CONTRAST_THR)
        & (peak_tau >= LATE_PEAK_TAU_THR)
    )
    profile[late_emerge] = "late_emerge"
    return {"profile": profile, **stats}
