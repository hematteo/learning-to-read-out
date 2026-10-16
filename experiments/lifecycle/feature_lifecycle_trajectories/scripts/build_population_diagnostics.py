"""Population-level lifecycle diagnostics (fig:app-selected-population-lifecycle-diagnostics).

Separates when active features peak from where decoder-norm mass lives over
training, for the four selected dictionaries:

  - row A: fraction of active features whose decoder norm peaks at each step;
  - row B: total decoder-norm mass over active features per step, normalized
    by its maximum within each model.

It also writes the two statistics the Appendix E text quotes for the Pythia
dictionaries: the fraction of active features peaking inside the 512-1.6k
reorganization window, and the mean relative decoder norm rho_f(t) of the
features that peak by step 128.

Reads the `<key>_decoder_norms.npy` caches written by
plot_normalized_trajectories.py (run that first). Writes metrics only, to
results/experiments/lifecycle/feature_lifecycle_trajectories/:
selected_population_lifecycle_diagnostics_{peak_counts,norm_mass,window_peaks,early_peak_decay}.csv
and the matching .pt sidecar.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lifecycle_common import CACHE, OLMO_STEPS_32, PYTHIA_STEPS_32

from readout.core.data import write_csv
from readout.core.paths import repo_root
from readout.core.repro import log_run_provenance
from readout.dynamics.lifecycle import ACTIVE_NORM_THR

OUT = repo_root() / "results/experiments/lifecycle/feature_lifecycle_trajectories"
STEM = "selected_population_lifecycle_diagnostics"
PYTHIA_WINDOW = (512, 1600)  # reorganization window selected in all three Pythia models
EARLY_PEAK_MAX_STEP = 128


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    steps: np.ndarray


RUNS: tuple[Run, ...] = (
    Run("pythia160m_d24576", "Pythia-160M", PYTHIA_STEPS_32),
    Run("pythia1b_d24576", "Pythia-1B", PYTHIA_STEPS_32),
    Run("pythia69b_d32768", "Pythia-6.9B", PYTHIA_STEPS_32),
    Run("olmo2_7b_d32768", "OLMo-2-7B", OLMO_STEPS_32),
)


def load_norms(run: Run) -> np.ndarray:
    path = CACHE / f"{run.key}_decoder_norms.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run plot_normalized_trajectories.py first")
    norms = np.load(path).astype(np.float32)  # (K, D)
    if norms.shape[0] != run.steps.shape[0]:
        raise ValueError(f"{run.key}: norms shape {norms.shape}, steps {run.steps.shape}")
    if not np.isfinite(norms).all():
        raise ValueError(f"{run.key}: non-finite decoder norms")
    return norms


def summarize_run(run: Run, norms: np.ndarray) -> tuple[dict, list[dict], list[dict]]:
    """(sidecar payload, peak-count rows, norm-mass rows) for one dictionary. norms: (K, D)."""
    peak = norms.max(axis=0)
    active = peak > ACTIVE_NORM_THR
    active_ids = np.where(active)[0]
    peak_idx = norms[:, active].argmax(axis=0)
    peak_step = run.steps[peak_idx]
    unique_steps = np.unique(peak_step)
    count_fraction = np.array([(peak_step == step).mean() for step in unique_steps], dtype=np.float32)

    active_norms = norms[:, active]  # (K, n_active)
    total_mass = active_norms.sum(axis=1)
    mean_mass = active_norms.mean(axis=1)
    mass_over_time_norm = total_mass / np.clip(total_mass.max(), 1e-12, None)
    mass_over_time_share = total_mass / np.clip(total_mass.sum(), 1e-12, None)

    peak_rows = [
        {
            "model": run.key,
            "step": int(step),
            "feature_count_fraction": float(frac),
            "n_active_features": int(active_ids.size),
        }
        for step, frac in zip(unique_steps, count_fraction)
    ]
    mass_rows = [
        {
            "model": run.key,
            "step": int(step),
            "total_decoder_norm_mass": float(total),
            "mean_decoder_norm_active_feature": float(mean),
            "mass_normalized_to_model_max": float(normed),
            "mass_fraction_over_snapshots": float(share),
            "n_active_features": int(active_ids.size),
        }
        for step, total, mean, normed, share in zip(
            run.steps, total_mass, mean_mass, mass_over_time_norm, mass_over_time_share
        )
    ]
    payload = {
        "active_feature_ids": active_ids.astype(np.int32),
        "peak_step": peak_step.astype(np.int64),
        "unique_peak_steps": unique_steps.astype(np.int64),
        "peak_step_count_fraction": count_fraction,
        "steps": run.steps.astype(np.int64),
        "total_decoder_norm_mass": total_mass.astype(np.float32),
        "mean_decoder_norm_active_feature": mean_mass.astype(np.float32),
        "mass_normalized_to_model_max": mass_over_time_norm.astype(np.float32),
        "mass_fraction_over_snapshots": mass_over_time_share.astype(np.float32),
    }
    return payload, peak_rows, mass_rows


def early_peak_rows(run: Run, norms: np.ndarray) -> tuple[list[dict], list[dict]]:
    """(window-peak row, per-step rho rows) for one dictionary. norms: (K, D)."""
    active_norms = norms[:, norms.max(axis=0) > ACTIVE_NORM_THR]  # (K, n_active)
    rho = active_norms / active_norms.max(axis=0)  # (K, n_active), each feature's norm over its own max
    peak_step = run.steps[active_norms.argmax(axis=0)]
    lo, hi = PYTHIA_WINDOW
    window_rows = [
        {
            "model": run.key,
            "window_lo": lo,
            "window_hi": hi,
            "fraction_active_peaking_in_window": float(((peak_step >= lo) & (peak_step <= hi)).mean()),
            "n_active_features": int(peak_step.size),
        }
    ]
    early = peak_step <= EARLY_PEAK_MAX_STEP
    if not early.any():
        return window_rows, []
    mean_rho = rho[:, early].mean(axis=1)  # (K,)
    decay_rows = [
        {
            "model": run.key,
            "step": int(step),
            "mean_rho_early_peaking": float(value),
            "n_early_peaking": int(early.sum()),
        }
        for step, value in zip(run.steps, mean_rho)
    ]
    return window_rows, decay_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    log_run_provenance()
    payload: dict[str, dict] = {}
    peak_rows: list[dict] = []
    mass_rows: list[dict] = []
    window_rows: list[dict] = []
    decay_rows: list[dict] = []
    for run in RUNS:
        norms = load_norms(run)
        run_payload, run_peak_rows, run_mass_rows = summarize_run(run, norms)
        payload[run.key] = run_payload
        peak_rows.extend(run_peak_rows)
        mass_rows.extend(run_mass_rows)
        if run.steps is PYTHIA_STEPS_32:
            run_window_rows, run_decay_rows = early_peak_rows(run, norms)
            window_rows.extend(run_window_rows)
            decay_rows.extend(run_decay_rows)
    write_csv(args.out_dir / f"{STEM}_peak_counts.csv", peak_rows, require_rows=True)
    write_csv(args.out_dir / f"{STEM}_norm_mass.csv", mass_rows, require_rows=True)
    write_csv(args.out_dir / f"{STEM}_window_peaks.csv", window_rows, require_rows=True)
    write_csv(args.out_dir / f"{STEM}_early_peak_decay.csv", decay_rows, require_rows=True)
    torch.save(payload, args.out_dir / f"{STEM}.pt")
    print(f"wrote {args.out_dir / STEM}_{{peak_counts,norm_mass,window_peaks,early_peak_decay}}.csv and .pt")


if __name__ == "__main__":
    main()
