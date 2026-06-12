"""Build lifecycle profile composition and median-profile diagnostics.

This plot quantifies the qualitative lifecycle profiles discussed in Section
5.2.  It uses the same active-feature trajectories as the selected decoder-norm
trajectory figure and assigns each feature to a small rule-based profile:

  - persistent: sustained across training or low-range with broad support;
  - early-decaying: strong early-to-late contrast and early peak;
  - late-emerging: strong late-to-early contrast and late peak;
  - transitional/localized: rare, narrow mid-training peak with a low late tail;
  - mixed/ambiguous: everything else.

Outputs are written to results/experiments/lifecycle/feature_lifecycle_trajectories
with CSV and .pt sidecars for auditability.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "results/experiments/lifecycle/feature_lifecycle_trajectories"
SOURCE = OUT / "selected_decoder_norm_trajectories.pt"

ALIVE_THR = 0.50
PERSISTENT_ALIVE_FRAC_THR = 0.40
EARLY_TAU = 0.35
LATE_TAU = 0.65
EARLY_LEVEL_THR = 0.45
LATE_LEVEL_THR = 0.45
LOW_LEVEL_THR = 0.35
MID_PEAK_THR = 0.70
CURRENT_STEM = "selected_lifecycle_profile_composition"
REFINED_STEM = "selected_lifecycle_profile_composition_refined"


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    xticks: tuple[int, ...]


RUNS = (
    Run("pythia160m_d24576", "Pythia-160M", (1, 1000, 14000, 143000)),
    Run("pythia1b_d24576", "Pythia-1B", (1, 1000, 14000, 143000)),
    Run("pythia69b_d32768", "Pythia-6.9B", (1, 1000, 14000, 143000)),
    Run("olmo2_7b_d32768", "OLMo-2-7B", (150, 1000, 14000, 928000)),
)

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


def fmt_step(step: int) -> str:
    if step < 1000:
        return str(step)
    if step % 1000 == 0:
        return f"{step // 1000}k"
    return f"{step / 1000:.0f}k"


def tau_from_steps(steps: np.ndarray) -> np.ndarray:
    log_steps = np.log10(steps.astype(np.float64) + 1.0)
    return (log_steps - log_steps.min()) / np.clip(log_steps.max() - log_steps.min(), 1e-12, None)


def lifecycle_stats(
    traj: np.ndarray,
    steps: np.ndarray,
) -> dict[str, np.ndarray]:
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


def classify_profiles_current(
    traj: np.ndarray,
    steps: np.ndarray,
) -> dict[str, np.ndarray]:
    stats = lifecycle_stats(traj, steps)
    early_level = stats["early_level"]
    late_level = stats["late_level"]
    mid_peak = stats["mid_peak"]
    peak_tau = stats["peak_tau"]
    alive_frac = stats["alive_fraction"]
    profile = np.full(traj.shape[1], "mixed", dtype=object)
    persistent = (
        (early_level >= EARLY_LEVEL_THR) & (late_level >= LATE_LEVEL_THR) & (alive_frac >= PERSISTENT_ALIVE_FRAC_THR)
    )
    early_decay = (~persistent) & (early_level >= EARLY_LEVEL_THR) & (late_level <= LOW_LEVEL_THR) & (peak_tau <= 0.45)
    late_emerge = (~persistent) & (early_level <= LOW_LEVEL_THR) & (late_level >= LATE_LEVEL_THR) & (peak_tau >= 0.55)
    transitional = (
        (~persistent)
        & (~early_decay)
        & (~late_emerge)
        & (early_level <= LATE_LEVEL_THR)
        & (late_level <= LATE_LEVEL_THR)
        & (mid_peak >= MID_PEAK_THR)
    )

    profile[persistent] = "persistent"
    profile[early_decay] = "early_decay"
    profile[late_emerge] = "late_emerge"
    profile[transitional] = "transitional"
    return {"profile": profile, **stats}


def classify_profiles_refined(
    traj: np.ndarray,
    steps: np.ndarray,
) -> dict[str, np.ndarray]:
    stats = lifecycle_stats(traj, steps)
    early_level = stats["early_level"]
    late_level = stats["late_level"]
    mid_peak = stats["mid_peak"]
    mid_mean = stats["mid_mean"]
    peak_tau = stats["peak_tau"]
    alive_frac = stats["alive_fraction"]
    trajectory_range = stats["trajectory_range"]

    profile = np.full(traj.shape[1], "mixed", dtype=object)
    persistent = ((alive_frac >= 0.70) & (trajectory_range <= 0.35)) | (
        np.minimum(np.minimum(early_level, mid_mean), late_level) >= 0.45
    )
    profile[persistent] = "persistent"

    unmatched = profile == "mixed"
    transitional = (
        unmatched
        & (mid_peak >= 0.75)
        & ((mid_peak - np.maximum(early_level, late_level)) >= 0.35)
        & (late_level <= 0.30)
        & (stats["support_span_tau"] <= 0.35)
        & (peak_tau > 0.35)
        & (peak_tau < 0.65)
    )
    profile[transitional] = "transitional"

    unmatched = profile == "mixed"
    early_decay = unmatched & (early_level >= 0.50) & ((early_level - late_level) >= 0.30) & (peak_tau <= 0.45)
    profile[early_decay] = "early_decay"

    unmatched = profile == "mixed"
    late_emerge = unmatched & (late_level >= 0.50) & ((late_level - early_level) >= 0.30) & (peak_tau >= 0.55)
    profile[late_emerge] = "late_emerge"
    return {"profile": profile, **stats}


def summarize_profiles(model: str, profile: np.ndarray) -> list[dict[str, object]]:
    n = profile.size
    rows: list[dict[str, object]] = []
    for name in PROFILE_ORDER:
        count = int((profile == name).sum())
        rows.append(
            {
                "model": model,
                "profile": name,
                "label": PROFILE_LABELS[name],
                "count": count,
                "fraction": count / n,
            }
        )
    return rows


def feature_rows(
    model: str,
    feature_ids: np.ndarray,
    stats: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, feature_id in enumerate(feature_ids):
        rows.append(
            {
                "model": model,
                "feature": int(feature_id),
                "profile": str(stats["profile"][i]),
                "peak_step": int(stats["peak_step"][i]),
                "birth_step": int(stats["birth_step"][i]),
                "death_step": int(stats["death_step"][i]),
                "peak_tau": float(stats["peak_tau"][i]),
                "birth_tau": float(stats["birth_tau"][i]),
                "death_tau": float(stats["death_tau"][i]),
                "support_span_tau": float(stats["support_span_tau"][i]),
                "alive_fraction": float(stats["alive_fraction"][i]),
                "early_level": float(stats["early_level"][i]),
                "late_level": float(stats["late_level"][i]),
                "mid_peak": float(stats["mid_peak"][i]),
                "mid_mean": float(stats["mid_mean"][i]),
                "trajectory_range": float(stats["trajectory_range"][i]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


Classifier = Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]]


def build_outputs(
    source: dict[str, dict[str, object]],
    classifier: Classifier,
    stem: str,
    ruleset_name: str,
) -> None:
    summary_rows: list[dict[str, object]] = []
    all_feature_rows: list[dict[str, object]] = []
    sidecar: dict[str, dict[str, object]] = {}

    for run in RUNS:
        run_source = {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for key, value in source[run.key].items()
        }
        traj = np.asarray(run_source["traj"], dtype=np.float32)
        steps = np.asarray(run_source["steps"], dtype=np.int64)
        active_ids = np.asarray(run_source["active_feature_ids"], dtype=np.int64)
        stats = classifier(traj, steps)
        summary_rows.extend(summarize_profiles(run.key, stats["profile"]))
        all_feature_rows.extend(feature_rows(run.key, active_ids, stats))
        sidecar[run.key] = {
            "ruleset": ruleset_name,
            "steps": torch.as_tensor(steps),
            "active_feature_ids": torch.as_tensor(active_ids),
            "profile_order": PROFILE_ORDER,
            "profile": list(map(str, stats["profile"])),
            "peak_step": torch.as_tensor(stats["peak_step"]),
            "birth_step": torch.as_tensor(stats["birth_step"]),
            "death_step": torch.as_tensor(stats["death_step"]),
            "peak_tau": torch.as_tensor(stats["peak_tau"]),
            "birth_tau": torch.as_tensor(stats["birth_tau"]),
            "death_tau": torch.as_tensor(stats["death_tau"]),
            "support_span_tau": torch.as_tensor(stats["support_span_tau"]),
            "alive_fraction": torch.as_tensor(stats["alive_fraction"]),
            "early_level": torch.as_tensor(stats["early_level"]),
            "late_level": torch.as_tensor(stats["late_level"]),
            "mid_peak": torch.as_tensor(stats["mid_peak"]),
            "mid_mean": torch.as_tensor(stats["mid_mean"]),
            "trajectory_range": torch.as_tensor(stats["trajectory_range"]),
        }

    write_csv(OUT / f"{stem}_summary.csv", summary_rows)
    write_csv(OUT / f"{stem}_features.csv", all_feature_rows)
    torch.save(sidecar, OUT / f"{stem}.pt")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    build_outputs(source, classify_profiles_current, CURRENT_STEM, "current")
    build_outputs(source, classify_profiles_refined, REFINED_STEM, "refined")


if __name__ == "__main__":
    main()
