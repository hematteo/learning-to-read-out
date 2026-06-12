"""Estimate model-specific sparse-readout reorganization windows.

This is an exploratory timing analysis for the four selected paper
instruments.  It improves on a raw adjacent-checkpoint peak search by:

  - weighting feature lifecycle turnover by local decoder-norm strength;
  - comparing decoder-norm turnover, activation-rate turnover, decoder
    geometry change, rank instability, and peak-time density;
  - weighting decoder rotation / direction velocity by pairwise feature strength;
  - keeping total decoder-norm mass growth separate from structural change;
  - integrating adjacent changes over fixed-width log-training windows;
  - selecting windows where the enriched metrics co-occur above a circular-shift null; and
  - bootstrapping features to report uncertainty over the selected window.

Outputs are written to results/experiments/lifecycle/feature_lifecycle_trajectories
with CSV and .pt sidecars so the figure can be audited without re-running the
full pipeline.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[4]))
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.core.data import write_csv
from src.core.paths import ssd_path

REPO = Path(__file__).resolve().parents[4]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.model_specs import DEFAULT_STEPS_32 as _PYTHIA_STEPS_LIST  # noqa: E402
from src.core.model_specs import OLMO_STEPS_32 as _OLMO_STEPS_LIST  # noqa: E402

OUT = REPO / "results/experiments/lifecycle/feature_lifecycle_trajectories"
CACHE = REPO / "figures/feature_lifecycle_trajectories/section52_lifecycle/cache"

ACTIVE_THR = 0.01
BOOTSTRAPS = 400
NULL_SHIFTS = 200
BOOTSTRAP_NULL_SHIFTS = 50
EPS = 1e-8
LOG10_WINDOW = 0.5
RNG_SEED = 20260503
TOPK_FRAC = 0.10

PYTHIA_STEPS_32 = np.asarray(_PYTHIA_STEPS_LIST, dtype=np.int64)
OLMO_STEPS_32 = np.asarray(_OLMO_STEPS_LIST, dtype=np.int64)


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    d_model: int
    rates_path: Path
    steps: np.ndarray
    xticks: tuple[int, ...]


RUNS = (
    Run(
        "pythia160m_d24576",
        "Pythia-160M",
        768,
        Path(str(ssd_path("derived", "aggregates", "aggregates_pythia-160m_d24576_seed0.pt"))),
        PYTHIA_STEPS_32,
        (1, 1000, 14000, 143000),
    ),
    Run(
        "pythia1b_d24576",
        "Pythia-1B",
        2048,
        Path(str(ssd_path("derived", "aggregates", "aggregates_pythia-1b_d24576_seed0.pt"))),
        PYTHIA_STEPS_32,
        (1, 1000, 14000, 143000),
    ),
    Run(
        "pythia69b_d32768",
        "Pythia-6.9B",
        4096,
        REPO
        / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/p69b_sparse_d32768_seed0.pt",
        PYTHIA_STEPS_32,
        (1, 1000, 14000, 143000),
    ),
    Run(
        "olmo2_7b_d32768",
        "OLMo-2-7B",
        4096,
        REPO
        / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/olmo27b_d32768_seed0.pt",
        OLMO_STEPS_32,
        (150, 1000, 14000, 928000),
    ),
)


def fmt_step(step: float) -> str:
    step_i = int(round(step))
    if step_i < 1000:
        return str(step_i)
    if step_i % 1000 == 0:
        return f"{step_i // 1000}k"
    if step_i < 10000:
        return f"{step_i / 1000:.1f}k"
    return f"{step_i / 1000:.0f}k"


def cache_key_for(run: Run, *, geometry: bool = False) -> str:
    if geometry and run.key == "pythia69b_d32768":
        return "pythia69b_sparse_d32768"
    return run.key


def load_rates(run: Run) -> np.ndarray:
    blob = torch.load(run.rates_path, map_location="cpu", weights_only=False)
    if "rates" in blob:
        rates = blob["rates"]
    else:
        rates = next(iter(blob["rates_per_seed"].values()))
    if isinstance(rates, torch.Tensor):
        rates = rates.detach().cpu().numpy()
    rates = rates.astype(np.float32)
    if rates.shape[0] != run.steps.size:
        raise ValueError(f"{run.key}: rates shape {rates.shape}, steps {run.steps.shape}")
    return rates


def load_arrays(run: Run) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norms = np.load(CACHE / f"{run.key}_decoder_norms.npy").astype(np.float32)
    rotation_key = cache_key_for(run, geometry=True)
    rotation = np.load(CACHE / f"{rotation_key}_adjacent_rotation_radians.npy").astype(np.float32)
    cos_terminal = np.load(CACHE / f"{rotation_key}_cos_to_terminal.npy").astype(np.float32)
    rates = load_rates(run)
    if norms.shape[0] != run.steps.size:
        raise ValueError(f"{run.key}: norms shape {norms.shape}, steps {run.steps.shape}")
    if rotation.shape[0] != run.steps.size - 1:
        raise ValueError(f"{run.key}: rotation shape {rotation.shape}")
    if cos_terminal.shape != norms.shape:
        raise ValueError(f"{run.key}: cos-terminal shape {cos_terminal.shape}, norms {norms.shape}")
    if rates.shape != norms.shape:
        raise ValueError(f"{run.key}: rates shape {rates.shape}, norms {norms.shape}")
    return norms, rotation, rates, cos_terminal


def minmax(x: np.ndarray) -> np.ndarray:
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    return (x - lo) / np.clip(hi - lo, EPS, None)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    clean_weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return (clean_values * clean_weights).sum(axis=1) / np.clip(clean_weights.sum(axis=1), EPS, None)


def topk_rank_instability(active_norms: np.ndarray) -> np.ndarray:
    n_features = active_norms.shape[1]
    top_k = int(np.clip(round(n_features * TOPK_FRAC), 256, n_features))
    out = np.zeros(active_norms.shape[0] - 1, dtype=np.float64)
    for k in range(active_norms.shape[0] - 1):
        left = np.argpartition(active_norms[k], -top_k)[-top_k:]
        right = np.argpartition(active_norms[k + 1], -top_k)[-top_k:]
        overlap = np.intersect1d(left, right, assume_unique=False).size
        union = 2 * top_k - overlap
        out[k] = 1.0 - (overlap / max(union, 1))
    return out


def pair_metrics(
    norms: np.ndarray,
    rotation: np.ndarray,
    rates: np.ndarray,
    cos_terminal: np.ndarray,
) -> dict[str, np.ndarray]:
    peak = norms.max(axis=0)
    active = peak > ACTIVE_THR
    active_norms = norms[:, active]
    active_rotation = rotation[:, active]
    active_rates = rates[:, active]
    active_cos_terminal = cos_terminal[:, active]

    normalized = active_norms / np.clip(active_norms.max(axis=0, keepdims=True), EPS, None)
    delta = np.abs(np.diff(normalized, axis=0))

    turnover_weight = np.maximum(active_norms[:-1], active_norms[1:])
    lifecycle_turnover = weighted_mean(delta, turnover_weight)

    normalized_rates = active_rates / np.clip(active_rates.max(axis=0, keepdims=True), EPS, None)
    activation_turnover = weighted_mean(np.abs(np.diff(normalized_rates, axis=0)), turnover_weight)

    rotation_weight = np.sqrt(np.clip(active_norms[:-1] * active_norms[1:], 0.0, None))
    weighted_rotation = weighted_mean(active_rotation, rotation_weight)
    unweighted_rotation = np.nanmean(np.nan_to_num(active_rotation, nan=0.0), axis=1)
    direction_velocity = weighted_mean(np.abs(np.diff(active_cos_terminal, axis=0)), rotation_weight)
    rank_instability = topk_rank_instability(active_norms)

    peak_idx = active_norms.argmax(axis=0)
    peak_mass = active_norms.max(axis=0)
    peak_density = np.zeros(active_norms.shape[0] - 1, dtype=np.float64)
    denom = np.clip(peak_mass.sum(), EPS, None)
    for k in range(active_norms.shape[0] - 1):
        peak_density[k] = peak_mass[peak_idx == k + 1].sum() / denom

    total_mass = active_norms.sum(axis=1)
    normalized_mass = total_mass / np.clip(total_mass.max(), EPS, None)
    signed_mass_change = np.diff(normalized_mass)

    return {
        "active_feature_count": np.array([active.sum()], dtype=np.int64),
        "lifecycle_turnover": lifecycle_turnover,
        "activation_turnover": activation_turnover,
        "weighted_rotation": weighted_rotation,
        "unweighted_rotation": unweighted_rotation,
        "direction_velocity": direction_velocity,
        "rank_instability": rank_instability,
        "peak_density": peak_density,
        "signed_mass_change": signed_mass_change,
        "abs_mass_change": np.abs(signed_mass_change),
    }


def candidate_windows(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_steps = np.log10(steps.astype(np.float64) + 1.0)
    starts = log_steps[:-1].copy()
    max_start = log_steps[-1] - LOG10_WINDOW
    starts = starts[starts <= max_start + EPS]
    if starts.size == 0:
        starts = np.array([log_steps[0]], dtype=np.float64)
    ends = starts + LOG10_WINDOW
    centers = 10 ** ((starts + ends) / 2.0) - 1.0
    return starts, ends, centers


def integrate_windows(steps: np.ndarray, metric: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    log_steps = np.log10(steps.astype(np.float64) + 1.0)
    pair_starts = log_steps[:-1]
    pair_ends = log_steps[1:]
    pair_width = np.clip(pair_ends - pair_starts, EPS, None)
    rates = metric / pair_width
    scores = np.zeros(starts.shape, dtype=np.float64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        overlap = np.maximum(0.0, np.minimum(pair_ends, end) - np.maximum(pair_starts, start))
        scores[i] = np.sum(rates * overlap)
    return scores


SCORE_COMPONENTS = (
    "lifecycle_turnover",
    "activation_turnover",
    "weighted_rotation",
    "direction_velocity",
    "rank_instability",
    "peak_density",
)


def enriched_score_from_windows(windows: dict[str, np.ndarray]) -> np.ndarray:
    scaled = [minmax(windows[f"{name}_window"]).astype(np.float64) for name in SCORE_COMPONENTS]
    stacked = np.vstack(scaled)
    return np.exp(np.mean(np.log(np.clip(stacked, EPS, None)), axis=0))


def component_windows(
    steps: np.ndarray,
    metrics: dict[str, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> dict[str, np.ndarray]:
    windows: dict[str, np.ndarray] = {
        "turnover_window": integrate_windows(steps, metrics["lifecycle_turnover"], starts, ends),
        "activation_turnover_window": integrate_windows(steps, metrics["activation_turnover"], starts, ends),
        "rotation_window": integrate_windows(steps, metrics["weighted_rotation"], starts, ends),
        "direction_velocity_window": integrate_windows(steps, metrics["direction_velocity"], starts, ends),
        "rank_instability_window": integrate_windows(steps, metrics["rank_instability"], starts, ends),
        "peak_density_window": integrate_windows(steps, metrics["peak_density"], starts, ends),
        "mass_growth_window": integrate_windows(steps, np.maximum(metrics["signed_mass_change"], 0.0), starts, ends),
        "mass_change_window": integrate_windows(steps, metrics["abs_mass_change"], starts, ends),
    }
    for name in SCORE_COMPONENTS:
        windows[f"{name}_window"] = windows[
            {
                "lifecycle_turnover": "turnover_window",
                "activation_turnover": "activation_turnover_window",
                "weighted_rotation": "rotation_window",
                "direction_velocity": "direction_velocity_window",
                "rank_instability": "rank_instability_window",
                "peak_density": "peak_density_window",
            }[name]
        ]
    return windows


def null_scores(
    steps: np.ndarray,
    metrics: dict[str, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
    rng: np.random.Generator,
    *,
    n_shifts: int,
) -> np.ndarray:
    n_pairs = len(metrics["lifecycle_turnover"])
    scores = np.empty((n_shifts, starts.size), dtype=np.float64)
    for i in range(n_shifts):
        shifted = dict(metrics)
        for name in SCORE_COMPONENTS:
            shift = int(rng.integers(1, n_pairs))
            shifted[name] = np.roll(metrics[name], shift)
        shifted_windows = component_windows(steps, shifted, starts, ends)
        scores[i] = enriched_score_from_windows(shifted_windows)
    return scores


def window_metrics(
    steps: np.ndarray,
    metrics: dict[str, np.ndarray],
    *,
    null_seed: int | None = None,
    null_shifts: int = NULL_SHIFTS,
) -> dict[str, np.ndarray]:
    starts, ends, centers = candidate_windows(steps)
    windows = {
        "window_start_log10": starts,
        "window_end_log10": ends,
        "window_center_step": centers,
        "window_start_step": 10**starts - 1.0,
        "window_end_step": 10**ends - 1.0,
    }
    windows.update(component_windows(steps, metrics, starts, ends))
    structural_score = enriched_score_from_windows(windows)
    windows["structural_score"] = structural_score
    if null_seed is not None:
        rng = np.random.default_rng(null_seed)
        null = null_scores(steps, metrics, starts, ends, rng, n_shifts=null_shifts)
        null_mean = null.mean(axis=0)
        null_std = null.std(axis=0)
        windows["structural_score_null_mean"] = null_mean
        windows["structural_score_null_std"] = null_std
        windows["structural_score_z"] = (structural_score - null_mean) / np.clip(null_std, EPS, None)
        windows["structural_score_null_p"] = ((null >= structural_score[None, :]).sum(axis=0) + 1.0) / (
            null_shifts + 1.0
        )
    return windows


def primary_score(windows: dict[str, np.ndarray]) -> np.ndarray:
    return windows.get("structural_score_z", windows["structural_score"])


def summarize_windows(run: Run, windows: dict[str, np.ndarray]) -> dict[str, object]:
    score = primary_score(windows)
    idx = int(np.nanargmax(score))
    turnover_idx = int(np.nanargmax(windows["turnover_window"]))
    rotation_idx = int(np.nanargmax(windows["rotation_window"]))
    mass_idx = int(np.nanargmax(windows["mass_growth_window"]))
    return {
        "model": run.key,
        "label": run.label,
        "active_feature_count": None,
        "primary_window_idx": idx,
        "primary_start_step": int(round(windows["window_start_step"][idx])),
        "primary_end_step": int(round(windows["window_end_step"][idx])),
        "primary_center_step": int(round(windows["window_center_step"][idx])),
        "primary_structural_score": float(score[idx]),
        "primary_raw_structural_score": float(windows["structural_score"][idx]),
        "primary_null_p": float(windows.get("structural_score_null_p", np.full_like(score, np.nan))[idx]),
        "turnover_peak_start_step": int(round(windows["window_start_step"][turnover_idx])),
        "turnover_peak_end_step": int(round(windows["window_end_step"][turnover_idx])),
        "rotation_peak_start_step": int(round(windows["window_start_step"][rotation_idx])),
        "rotation_peak_end_step": int(round(windows["window_end_step"][rotation_idx])),
        "mass_growth_peak_start_step": int(round(windows["window_start_step"][mass_idx])),
        "mass_growth_peak_end_step": int(round(windows["window_end_step"][mass_idx])),
    }


def bootstrap_windows(
    run: Run,
    norms: np.ndarray,
    rotation: np.ndarray,
    rates: np.ndarray,
    cos_terminal: np.ndarray,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED + sum(ord(c) for c in run.key))
    active = norms.max(axis=0) > ACTIVE_THR
    active_norms = norms[:, active]
    active_rotation = rotation[:, active]
    active_rates = rates[:, active]
    active_cos_terminal = cos_terminal[:, active]
    n_features = active_norms.shape[1]
    rows: list[dict[str, object]] = []
    for bootstrap_idx in range(BOOTSTRAPS):
        sample = rng.integers(0, n_features, size=n_features)
        metrics = pair_metrics(
            active_norms[:, sample],
            active_rotation[:, sample],
            active_rates[:, sample],
            active_cos_terminal[:, sample],
        )
        windows = window_metrics(
            run.steps,
            metrics,
            null_seed=RNG_SEED + sum(ord(c) for c in run.key) + bootstrap_idx,
            null_shifts=BOOTSTRAP_NULL_SHIFTS,
        )
        idx = int(np.nanargmax(primary_score(windows)))
        rows.append(
            {
                "model": run.key,
                "bootstrap_idx": bootstrap_idx,
                "window_idx": idx,
                "start_step": int(round(windows["window_start_step"][idx])),
                "end_step": int(round(windows["window_end_step"][idx])),
                "center_step": int(round(windows["window_center_step"][idx])),
                "primary_score": float(primary_score(windows)[idx]),
                "structural_score": float(windows["structural_score"][idx]),
                "null_p": float(windows["structural_score_null_p"][idx]),
            }
        )
    return rows


def rows_from_arrays(model: str, arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    n = len(next(iter(arrays.values())))
    rows: list[dict[str, object]] = []
    for i in range(n):
        row: dict[str, object] = {"model": model, "idx": i}
        for name, values in arrays.items():
            value = values[i]
            row[name] = int(value) if np.issubdtype(values.dtype, np.integer) else float(value)
        rows.append(row)
    return rows


def add_bootstrap_intervals(summary_rows: list[dict[str, object]], bootstrap_rows: list[dict[str, object]]) -> None:
    for row in summary_rows:
        centers = np.array(
            [r["center_step"] for r in bootstrap_rows if r["model"] == row["model"]],
            dtype=np.float64,
        )
        starts = np.array(
            [r["start_step"] for r in bootstrap_rows if r["model"] == row["model"]],
            dtype=np.float64,
        )
        ends = np.array(
            [r["end_step"] for r in bootstrap_rows if r["model"] == row["model"]],
            dtype=np.float64,
        )
        row["bootstrap_center_p05"] = int(round(np.percentile(centers, 5)))
        row["bootstrap_center_p50"] = int(round(np.percentile(centers, 50)))
        row["bootstrap_center_p95"] = int(round(np.percentile(centers, 95)))
        row["bootstrap_start_p05"] = int(round(np.percentile(starts, 5)))
        row["bootstrap_end_p95"] = int(round(np.percentile(ends, 95)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    tensor_sidecar: dict[str, dict[str, object]] = {}

    for run in RUNS:
        norms, rotation, rates, cos_terminal = load_arrays(run)
        metrics = pair_metrics(norms, rotation, rates, cos_terminal)
        active_count = int(metrics["active_feature_count"][0])
        log_steps = np.log10(run.steps.astype(np.float64) + 1.0)
        pair_arrays = {
            "from_step": run.steps[:-1],
            "to_step": run.steps[1:],
            "pair_start_log10": log_steps[:-1],
            "pair_end_log10": log_steps[1:],
            "log10_width": np.diff(log_steps),
            "lifecycle_turnover": metrics["lifecycle_turnover"],
            "activation_turnover": metrics["activation_turnover"],
            "weighted_rotation": metrics["weighted_rotation"],
            "unweighted_rotation": metrics["unweighted_rotation"],
            "direction_velocity": metrics["direction_velocity"],
            "rank_instability": metrics["rank_instability"],
            "peak_density": metrics["peak_density"],
            "signed_mass_change": metrics["signed_mass_change"],
            "abs_mass_change": metrics["abs_mass_change"],
        }
        run_pair_rows = rows_from_arrays(run.key, pair_arrays)
        for row in run_pair_rows:
            row["active_feature_count"] = active_count
        pair_rows.extend(run_pair_rows)

        windows = window_metrics(
            run.steps,
            metrics,
            null_seed=RNG_SEED + sum(ord(c) for c in run.key),
        )
        run_window_rows = rows_from_arrays(run.key, windows)
        window_rows.extend(run_window_rows)

        summary = summarize_windows(run, windows)
        summary["active_feature_count"] = active_count
        summary_rows.append(summary)

        run_bootstrap_rows = bootstrap_windows(run, norms, rotation, rates, cos_terminal)
        bootstrap_rows.extend(run_bootstrap_rows)

        tensor_sidecar[run.key] = {
            "steps": torch.as_tensor(run.steps),
            "active_feature_count": active_count,
            "pair_metrics": {k: torch.as_tensor(v) for k, v in pair_arrays.items()},
            "window_metrics": {k: torch.as_tensor(v) for k, v in windows.items()},
        }

    add_bootstrap_intervals(summary_rows, bootstrap_rows)
    write_csv(OUT / "reorganization_step_pair_metrics_selected.csv", pair_rows)
    write_csv(OUT / "reorganization_step_peaks_selected.csv", summary_rows)
    write_csv(OUT / "reorganization_window_metrics_selected.csv", window_rows)
    write_csv(OUT / "reorganization_window_summary_selected.csv", summary_rows)
    write_csv(OUT / "reorganization_bootstrap_selected.csv", bootstrap_rows)
    torch.save(tensor_sidecar, OUT / "reorganization_window_metrics_selected.pt")


if __name__ == "__main__":
    main()
