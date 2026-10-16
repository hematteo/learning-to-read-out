"""Late-start Pythia-1B vs OLMo-2-7B lifecycle comparison (Appendix D.3).

Compares the Pythia-1B "late-start" W_U crosscoder (trained on a 32-step
schedule starting at step 256) with the OLMo-2-7B W_U crosscoder. Tests whether
OLMo's missing early-decaying features are explainable as a checkpointing
artifact: if Pythia fitted on a similarly late-starting schedule shows the same
near-zero share, the absence in OLMo is plausibly about checkpoint coverage,
not the model.

Outputs, under results/lifecycle/olmo_window/:
  comparison_summary.csv          activation-rate shares (decayers, early-dominant)
  comparison_feature_scores.csv   per-feature peak step, early-minus-late score, final level
  late_start_profile_fractions.csv  rule-based profile shares of the selected and
                                    late-start Pythia-1B dictionaries
  manifest.json                   inputs, git commit, summaries, decay definitions
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from readout.core.model_specs import DEFAULT_STEPS_32
from readout.core.paths import repo_root
from readout.core.repro import git_commit
from readout.dynamics.lifecycle import profile_fractions

REPO = repo_root()
PYTHIA_LATE_START_STEPS = np.asarray(
    [
        256,
        512,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        13000,
        14000,
        21000,
        23000,
        27000,
        33000,
        34000,
        43000,
        47000,
        53000,
        61000,
        73000,
        83000,
        89000,
        93000,
        102000,
        113000,
        123000,
        130000,
        143000,
    ],
    dtype=np.int64,
)

ACTIVE_NORM_THR = 0.01
EARLY_CUTOFF_STEP = 1000
LATE_CUTOFF_STEP = 14000  # first "late" step for the early-minus-late contrast
DECAY_FRAC_THR = 0.5  # final < 0.5 * peak

# Written verbatim into the manifest.
DECAY_DEFINITION = {
    "decayer_mask": f"peak_idx <= last_index_with_step<={EARLY_CUTOFF_STEP} AND final_rel < {DECAY_FRAC_THR}",
    "early_dominant_mask": (
        f"(mean rate over steps<={EARLY_CUTOFF_STEP}) - (mean rate over steps>={LATE_CUTOFF_STEP}) > 0.2"
    ),
}

FEATURE_LIFECYCLE_CACHE = REPO / "results/lifecycle/cache"
# Selected full-schedule Pythia-1B dictionary, for the late-start profile comparison.
SELECTED_1B_NORMS = FEATURE_LIFECYCLE_CACHE / "pythia1b_d24576_decoder_norms.npy"


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    rates_path: Path
    norms_path: Path | None  # None means norms come from inside rates blob
    steps: np.ndarray


PYTHIA_1B_LS = Run(
    key="pythia1b_late_start",
    label="Pythia-1B late-start (this work)",
    rates_path=REPO / "results/lifecycle/olmo_window/"
    "pythia1b_wu_late_start32_dsae24576_seed0_rates.pt",
    norms_path=None,
    steps=PYTHIA_LATE_START_STEPS,
)

OLMO_2_7B = Run(
    key="olmo2_7b",
    label="OLMo-2-7B",
    rates_path=REPO / "results/validation/appendix_validation/large_evals/"
    "olmo27b_d32768_seed0.pt",
    norms_path=FEATURE_LIFECYCLE_CACHE / "olmo2_7b_d32768_decoder_norms.npy",
    steps=np.asarray([], dtype=np.int64),  # populated from blob
)


def _load_rates(run: Run) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    blob = torch.load(run.rates_path, map_location="cpu", weights_only=False)
    rates = blob["rates"]
    if isinstance(rates, torch.Tensor):
        rates = rates.numpy()
    rates = rates.astype(np.float64)
    norms = blob.get("norms")
    if isinstance(norms, torch.Tensor):
        norms = norms.numpy().astype(np.float64)
    if norms is None and run.norms_path is not None:
        norms = np.load(run.norms_path).astype(np.float64)
    if "steps" in blob and run.steps.size == 0:
        steps_in = blob["steps"]
        steps = (steps_in.numpy() if isinstance(steps_in, torch.Tensor) else np.asarray(steps_in)).astype(np.int64)
    else:
        steps = run.steps
    if norms is not None and norms.shape != rates.shape:
        raise ValueError(f"{run.key}: norms {norms.shape} do not match rates {rates.shape}")
    return rates, norms, steps


def _active_traj_relative(rates: np.ndarray, norms: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (rel_trajectory, active_mask). rates: (K, D). Returns (n_active, K)."""
    if norms is not None:
        active = norms.max(axis=0) > ACTIVE_NORM_THR
    else:
        # OLMo blob has no norms — use peak rate > 0 as the alive criterion.
        active = rates.max(axis=0) > 0.0
    traj = rates[:, active].T  # (n_active, K)
    peak = np.clip(traj.max(axis=1, keepdims=True), 1e-12, None)
    return traj / peak, active


def _summarize(traj_rel: np.ndarray, steps: np.ndarray) -> dict:
    """Activation-rate shares of Appendix D.3 (early-peaking decayers, early-dominant score)."""
    peak_idx = traj_rel.argmax(axis=1)
    final = traj_rel[:, -1]
    early_mask = steps <= EARLY_CUTOFF_STEP
    late_mask = steps >= LATE_CUTOFF_STEP
    early_last_idx = int(np.flatnonzero(early_mask)[-1])
    early_minus_late = traj_rel[:, early_mask].mean(axis=1) - traj_rel[:, late_mask].mean(axis=1)
    decayer_mask = (peak_idx <= early_last_idx) & (final < DECAY_FRAC_THR)
    return {
        "n_active": int(traj_rel.shape[0]),
        "decayer_count": int(decayer_mask.sum()),
        "decayer_frac": float(decayer_mask.mean()),
        "elm_gt_0p2_count": int((early_minus_late > 0.2).sum()),
        "elm_gt_0p2_frac": float((early_minus_late > 0.2).mean()),
        "median_final_over_peak": float(np.median(final)),
        "early_minus_late": early_minus_late,
        "peak_idx": peak_idx,
        "final_rel": final,
    }


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def compute_late_start_profile_fractions(
    late_norms: np.ndarray,
    late_steps: np.ndarray,
    out_dirs: tuple[Path, ...],
    selected_norms_path: Path = SELECTED_1B_NORMS,
) -> list[dict]:
    """Refined-profile fractions (sec:app-olmo-checkpoint-window-control) for the selected
    Pythia-1B dictionary, the same dictionary restricted to steps >= the late-start
    schedule's first step without refitting, and the late-start dictionary."""
    selected = np.load(selected_norms_path).astype(np.float32)  # (32, D)
    full_steps = np.asarray(DEFAULT_STEPS_32, dtype=np.int64)
    keep = full_steps >= late_steps[0]
    rows = [
        {"dictionary": "selected d24576", **profile_fractions(selected, full_steps)},
        {
            "dictionary": f"selected d24576, steps >= {int(late_steps[0])} (not refitted)",
            **profile_fractions(selected[keep], full_steps[keep]),
        },
        {
            "dictionary": f"late-start d24576 (fitted from step {int(late_steps[0])})",
            **profile_fractions(late_norms, late_steps),
        },
    ]
    for out_dir in out_dirs:
        path = out_dir / "late_start_profile_fractions.csv"
        _write_csv(path, list(rows[0]), [list(row.values()) for row in rows])
        print(f"  -> {path}")
    return rows


# ─── main ──────────────────────────────────────────────────────────────────


def main() -> int:
    fig_dir = REPO / "results/lifecycle/olmo_window"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs_loaded: list[tuple[Run, np.ndarray, np.ndarray]] = []  # (run, rates, steps)
    for run in (PYTHIA_1B_LS, OLMO_2_7B):
        rates, norms, steps = _load_rates(run)
        traj_rel, _ = _active_traj_relative(rates, norms)
        runs_loaded.append((run, traj_rel, steps))
        print(f"{run.key}: rates {rates.shape} → traj_rel {traj_rel.shape} steps[0..2]={steps[:3].tolist()}")

    summaries: list[tuple[Run, dict]] = []
    for run, traj_rel, steps in runs_loaded:
        s = _summarize(traj_rel, steps)
        summaries.append((run, s))
        print(
            f"[{run.key}] n_active={s['n_active']} decayers={s['decayer_count']} "
            f"({s['decayer_frac']:.4%}) early_dominant={s['elm_gt_0p2_count']} "
            f"({s['elm_gt_0p2_frac']:.4%}) median_final/peak={s['median_final_over_peak']:.3f}"
        )

    # CSV: side-by-side summary of the two runs
    _write_csv(
        fig_dir / "comparison_summary.csv",
        [
            "key",
            "label",
            "n_active",
            "decayer_count",
            "decayer_frac",
            "early_dominant_count",
            "early_dominant_frac",
            "median_final_over_peak",
        ],
        [
            [
                run.key,
                run.label,
                s["n_active"],
                s["decayer_count"],
                s["decayer_frac"],
                s["elm_gt_0p2_count"],
                s["elm_gt_0p2_frac"],
                s["median_final_over_peak"],
            ]
            for run, s in summaries
        ],
    )
    feature_score_rows = []
    for (run, _traj_rel, steps), (_, s) in zip(runs_loaded, summaries, strict=True):
        peak_idx = s["peak_idx"]
        for feature_row, (peak_i, score, final_rel) in enumerate(
            zip(peak_idx, s["early_minus_late"], s["final_rel"], strict=True)
        ):
            feature_score_rows.append(
                [
                    run.key,
                    feature_row,
                    int(peak_i),
                    int(steps[int(peak_i)]),
                    float(score),
                    float(final_rel),
                ]
            )
    _write_csv(
        fig_dir / "comparison_feature_scores.csv",
        [
            "key",
            "active_feature_row",
            "peak_idx",
            "peak_step",
            "early_minus_late",
            "final_rel",
        ],
        feature_score_rows,
    )

    _, late_norms, late_steps = _load_rates(PYTHIA_1B_LS)
    compute_late_start_profile_fractions(late_norms, late_steps, (fig_dir,))

    # Manifest
    manifest = {
        "git_commit": git_commit(),
        "runs": [{"key": run.key, "label": run.label, "rates_path": str(run.rates_path)} for run, _, _ in runs_loaded],
        "summaries": [
            {
                "key": run.key,
                **{k: v for k, v in s.items() if not isinstance(v, np.ndarray)},
            }
            for run, s in summaries
        ],
        "decay_definition": DECAY_DEFINITION,
    }
    (fig_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote CSVs + manifest to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
