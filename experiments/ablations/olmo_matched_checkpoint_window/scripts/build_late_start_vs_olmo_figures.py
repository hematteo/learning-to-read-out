"""Compute late-start Pythia-1B vs OLMo-2-7B activation-rate diagnostics.

Computes and persists metrics comparing the new Pythia-1B "late-start" W_U
crosscoder (trained on a 32-step schedule starting at step 256) against the
canonical OLMo-2-7B W_U crosscoder. Tests whether OLMo's missing decayer
branch is explainable as a checkpointing artifact: if Pythia trained on a
similarly late-starting schedule shows the same near-zero decayer pattern,
then the absence in OLMo is plausibly about checkpoint coverage, not the model.

Outputs (CSV / JSON / .pt metric sidecars) under
figures/olmo_matched_checkpoint_window/.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from readout.core.paths import repo_root

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
LATE_CUTOFF_STEP = 3000
DECAY_FRAC_THR = 0.5  # final < 0.5 * peak

FEATURE_LIFECYCLE_CACHE = (
    REPO / "figures/feature_lifecycle_trajectories/section52_lifecycle/cache"
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
    rates_path=REPO / "results/experiments/olmo_matched_checkpoint_window/"
    "pythia1b_wu_late_start32_dsae24576_seed0_rates.pt",
    norms_path=None,
    steps=PYTHIA_LATE_START_STEPS,
)

OLMO_2_7B = Run(
    key="olmo2_7b",
    label="OLMo-2-7B",
    rates_path=REPO
    / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/"
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
        steps = (
            steps_in.numpy()
            if isinstance(steps_in, torch.Tensor)
            else np.asarray(steps_in)
        ).astype(np.int64)
    else:
        steps = run.steps
    if norms is not None and norms.shape != rates.shape:
        raise ValueError(
            f"{run.key}: norms {norms.shape} do not match rates {rates.shape}"
        )
    return rates, norms, steps


def _active_traj_relative(
    rates: np.ndarray, norms: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
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
    """Same metric definitions as analyze_late_start_decay.py / OLMo handoff numbers."""
    peak_idx = traj_rel.argmax(axis=1)
    final = traj_rel[:, -1]
    early_mask = steps <= EARLY_CUTOFF_STEP
    late_mask = steps >= 14000
    early_last_idx = int(np.flatnonzero(early_mask)[-1])
    early_minus_late = traj_rel[:, early_mask].mean(axis=1) - traj_rel[
        :, late_mask
    ].mean(axis=1)
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


def _short_label(label: str) -> str:
    return label.replace(" (this work)", "")


def _normalize_by_peak(values: np.ndarray) -> np.ndarray:
    peak = np.clip(np.nanmax(values, axis=0, keepdims=True), 1e-12, None)
    return values / peak


def _build_metric_payload(runs: tuple[Run, ...]) -> dict[str, dict]:
    metric_payload: dict[str, dict] = {}
    for run in runs:
        rates, norms, steps = _load_rates(run)
        if norms is None:
            raise ValueError(f"{run.key}: decoder norms are required")
        active = norms.max(axis=0) > ACTIVE_NORM_THR
        rates_active = rates[:, active]
        norms_active = norms[:, active]
        norm_rel = _normalize_by_peak(norms_active)
        rate_rel = _normalize_by_peak(rates_active)
        norm_peak_idx = norm_rel.argmax(axis=0)
        metric_payload[run.key] = {
            "label": _short_label(run.label),
            "steps": steps,
            "active_feature_idx": np.flatnonzero(active),
            "norms_active": norms_active,
            "rates_active": rates_active,
            "decoder_norm_peak_idx": norm_peak_idx,
            "decoder_norm_peak_step": steps[norm_peak_idx],
            "norm_rel": norm_rel,
            "rate_rel": rate_rel,
        }
    return metric_payload


def _write_metric_grid_summary(path: Path, payload: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "metric",
                "step",
                "n_features",
                "median",
                "q10",
                "q25",
                "q75",
                "q90",
            ]
        )
        for run_key, run_payload in payload.items():
            steps = run_payload["steps"]
            for metric in ("norm_rel", "rate_rel"):
                values = run_payload[metric]
                for i, step in enumerate(steps):
                    row = values[i]
                    writer.writerow(
                        [
                            run_key,
                            metric,
                            int(step),
                            int(np.isfinite(row).sum()),
                            float(np.nanmedian(row)),
                            float(np.nanquantile(row, 0.10)),
                            float(np.nanquantile(row, 0.25)),
                            float(np.nanquantile(row, 0.75)),
                            float(np.nanquantile(row, 0.90)),
                        ]
                    )


def _save_metric_grid_sidecars(base: Path, payload: dict[str, dict]) -> None:
    _write_metric_grid_summary(base.with_suffix(".csv"), payload)
    torch.save(
        {
            run_key: {
                "label": run_payload["label"],
                "steps": torch.as_tensor(run_payload["steps"], dtype=torch.int64),
                "active_feature_idx": torch.as_tensor(
                    run_payload["active_feature_idx"], dtype=torch.int64
                ),
                "decoder_norm_peak_idx": torch.as_tensor(
                    run_payload["decoder_norm_peak_idx"], dtype=torch.int64
                ),
                "decoder_norm_peak_step": torch.as_tensor(
                    run_payload["decoder_norm_peak_step"], dtype=torch.int64
                ),
                "norm_rel": torch.from_numpy(
                    run_payload["norm_rel"].astype(np.float32)
                ),
                "rate_rel": torch.from_numpy(
                    run_payload["rate_rel"].astype(np.float32)
                ),
            }
            for run_key, run_payload in payload.items()
        },
        base.with_suffix(".pt"),
    )


def compute_appendix_metric_grid(
    runs: tuple[Run, Run],
    out_dirs: tuple[Path, ...],
) -> None:
    """Appendix-style lifecycle grid metrics (selected_lifecycle_metric_diagnostics)."""
    metric_payload = _build_metric_payload(runs)

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / "olmo_matched_checkpoint_window_metric_grid"
        _save_metric_grid_sidecars(base, metric_payload)
        print(f"  -> {base.with_suffix('.csv')}")


def compute_population_lifecycle_diagnostics(
    runs: tuple[Run, Run],
    out_dirs: tuple[Path, ...],
) -> None:
    payload = _build_metric_payload(runs)
    peak_rows: list[list] = []
    mass_rows: list[list] = []
    sidecar: dict[str, dict] = {}

    for run in runs:
        run_payload = payload[run.key]
        steps = run_payload["steps"]
        peak_step = run_payload["decoder_norm_peak_step"]
        unique_steps, counts = np.unique(peak_step, return_counts=True)
        fractions = counts / counts.sum()
        norms_active = run_payload["norms_active"]
        total_mass = norms_active.sum(axis=1)
        mean_mass = norms_active.mean(axis=1)
        normed_mass = total_mass / np.clip(total_mass.max(), 1e-12, None)

        for step, fraction, count in zip(unique_steps, fractions, counts, strict=True):
            peak_rows.append(
                [run.key, int(step), int(count), float(fraction), int(counts.sum())]
            )
        for step, total, mean, mass in zip(
            steps, total_mass, mean_mass, normed_mass, strict=True
        ):
            mass_rows.append(
                [
                    run.key,
                    int(step),
                    float(total),
                    float(mean),
                    float(mass),
                    int(norms_active.shape[1]),
                ]
            )
        sidecar[run.key] = {
            "steps": torch.as_tensor(steps, dtype=torch.int64),
            "active_feature_idx": torch.as_tensor(
                run_payload["active_feature_idx"], dtype=torch.int64
            ),
            "decoder_norm_peak_step": torch.as_tensor(peak_step, dtype=torch.int64),
            "unique_peak_steps": torch.as_tensor(unique_steps, dtype=torch.int64),
            "peak_step_count_fraction": torch.from_numpy(fractions.astype(np.float32)),
            "total_decoder_norm_mass": torch.from_numpy(total_mass.astype(np.float32)),
            "mean_decoder_norm_active_feature": torch.from_numpy(
                mean_mass.astype(np.float32)
            ),
            "mass_normalized_to_model_max": torch.from_numpy(
                normed_mass.astype(np.float32)
            ),
        }

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = (
            out_dir / "olmo_matched_checkpoint_window_population_lifecycle_diagnostics"
        )
        _write_csv(
            base.with_name(base.name + "_peak_counts.csv"),
            [
                "run",
                "step",
                "feature_count",
                "feature_count_fraction",
                "n_active_features",
            ],
            peak_rows,
        )
        _write_csv(
            base.with_name(base.name + "_norm_mass.csv"),
            [
                "run",
                "step",
                "total_decoder_norm_mass",
                "mean_decoder_norm_active_feature",
                "mass_normalized_to_model_max",
                "n_active_features",
            ],
            mass_rows,
        )
        torch.save(sidecar, base.with_suffix(".pt"))
        print(f"  -> {base.with_suffix('.pt')}")


def _heatmap_coordinates(run_payload: dict) -> dict[str, np.ndarray]:
    norms = run_payload["norms_active"]
    steps = run_payload["steps"]
    log_steps = np.log1p(steps.astype(np.float64))
    denom = norms.sum(axis=0)
    centroid = (log_steps[:, None] * norms).sum(axis=0) / np.clip(denom, 1e-12, None)
    spread = np.sqrt(
        (((log_steps[:, None] - centroid[None, :]) ** 2) * norms).sum(axis=0)
        / np.clip(denom, 1e-12, None)
    )
    peak_idx = run_payload["decoder_norm_peak_idx"]
    order = np.lexsort((centroid, spread, peak_idx))
    return {
        "order": order.astype(np.int32),
        "centroid_step": np.expm1(centroid).astype(np.float32),
        "centroid_log_step": centroid.astype(np.float32),
        "spread_log_step": spread.astype(np.float32),
        "peak_idx": peak_idx.astype(np.int16),
        "peak_step": run_payload["steps"][peak_idx].astype(np.int64),
        "normalized_sorted": run_payload["norm_rel"][:, order].T.astype(np.float32),
    }


def compute_decoder_norm_heatmaps(
    runs: tuple[Run, Run],
    out_dirs: tuple[Path, ...],
) -> None:
    payload = _build_metric_payload(runs)
    heatmap_payload: dict[str, dict] = {}
    summary_rows: list[list] = []
    feature_rows: list[list] = []

    for run in runs:
        run_payload = payload[run.key]
        coords = _heatmap_coordinates(run_payload)
        heatmap_payload[run.key] = {
            "steps": torch.as_tensor(run_payload["steps"], dtype=torch.int64),
            "active_feature_idx": torch.as_tensor(
                run_payload["active_feature_idx"], dtype=torch.int64
            ),
            "order": torch.as_tensor(coords["order"], dtype=torch.int64),
            "centroid_step": torch.from_numpy(
                coords["centroid_step"].astype(np.float32)
            ),
            "centroid_log_step": torch.from_numpy(
                coords["centroid_log_step"].astype(np.float32)
            ),
            "spread_log_step": torch.from_numpy(
                coords["spread_log_step"].astype(np.float32)
            ),
            "peak_step": torch.as_tensor(coords["peak_step"], dtype=torch.int64),
            "peak_idx": torch.as_tensor(coords["peak_idx"], dtype=torch.int64),
        }
        active_count = int(run_payload["active_feature_idx"].size)
        summary_rows.append(
            [
                run.key,
                active_count,
                float(np.median(coords["centroid_step"])),
                float(np.quantile(coords["centroid_step"], 0.25)),
                float(np.quantile(coords["centroid_step"], 0.75)),
                float(np.median(coords["spread_log_step"])),
            ]
        )
        for rank, idx in enumerate(coords["order"]):
            feature = int(run_payload["active_feature_idx"][idx])
            feature_rows.append(
                [
                    run.key,
                    feature,
                    int(rank),
                    float(coords["centroid_step"][idx]),
                    float(coords["centroid_log_step"][idx]),
                    float(coords["spread_log_step"][idx]),
                    int(coords["peak_step"][idx]),
                    int(coords["peak_idx"][idx]),
                ]
            )

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / "olmo_matched_checkpoint_window_decoder_norm_heatmaps"
        _write_csv(
            base.with_name(base.name + "_summary.csv"),
            [
                "run",
                "n_active_features",
                "median_centroid_step",
                "q25_centroid_step",
                "q75_centroid_step",
                "median_spread_log_step",
            ],
            summary_rows,
        )
        _write_csv(
            base.with_name(base.name + "_rows.csv"),
            [
                "run",
                "feature",
                "heatmap_rank",
                "centroid_step",
                "centroid_log_step",
                "spread_log_step",
                "peak_step",
                "peak_idx",
            ],
            feature_rows,
        )
        torch.save(heatmap_payload, base.with_suffix(".pt"))
        print(f"  -> {base.with_suffix('.pt')}")


def _pca_scores(traj_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = traj_rel - traj_rel.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt.T
    explained = singular_values**2 / np.clip(np.sum(singular_values**2), 1e-12, None)
    return scores, explained


def compute_wishbone_density_grid(
    runs: tuple[Run, Run],
    out_dirs: tuple[Path, ...],
) -> None:
    payload = _build_metric_payload(runs)
    pca_payload: dict[str, dict] = {}

    for run in runs:
        run_payload = payload[run.key]
        scores, explained = _pca_scores(run_payload["rate_rel"].T)
        pca_payload[run.key] = {
            "scores": torch.from_numpy(scores.astype(np.float32)),
            "explained_variance_ratio": torch.from_numpy(explained.astype(np.float32)),
            "steps": torch.as_tensor(run_payload["steps"], dtype=torch.int64),
            "active_feature_idx": torch.as_tensor(
                run_payload["active_feature_idx"], dtype=torch.int64
            ),
        }

    summary_rows = [
        [
            run.key,
            payload[run.key]["label"],
            int(payload[run.key]["active_feature_idx"].size),
            float(pca_payload[run.key]["explained_variance_ratio"][0]),
            float(pca_payload[run.key]["explained_variance_ratio"][1]),
            float(pca_payload[run.key]["explained_variance_ratio"][2]),
        ]
        for run in runs
    ]
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        bases = [
            out_dir
            / "olmo_matched_checkpoint_window_wishbone_pca_density_grid_viridis",
            out_dir
            / "olmo_matched_checkpoint_window_wishbone_plain_hexbin_grid_viridis",
        ]
        _write_csv(
            bases[0].with_name(bases[0].name + "_summary.csv"),
            ["run", "label", "n_active", "pc1_var", "pc2_var", "pc3_var"],
            summary_rows,
        )
        torch.save(pca_payload, bases[0].with_suffix(".pt"))
        _write_csv(
            bases[1].with_name(bases[1].name + "_summary.csv"),
            ["run", "label", "n_active", "pc1_var", "pc2_var", "pc3_var"],
            summary_rows,
        )
        torch.save(pca_payload, bases[1].with_suffix(".pt"))
        print(f"  -> {bases[0].with_suffix('.pt')}")


def _tau_from_steps(steps: np.ndarray) -> np.ndarray:
    log_steps = np.log10(steps.astype(np.float64) + 1.0)
    return (log_steps - log_steps.min()) / np.clip(
        log_steps.max() - log_steps.min(), 1e-12, None
    )


def _lifecycle_stats(traj: np.ndarray, steps: np.ndarray) -> dict[str, np.ndarray]:
    tau = _tau_from_steps(steps)
    alive = traj >= 0.50
    peak_idx = traj.argmax(axis=0)
    first_alive = alive.argmax(axis=0)
    last_alive = alive.shape[0] - 1 - alive[::-1].argmax(axis=0)
    alive_frac = alive.mean(axis=0)
    support_span = tau[last_alive] - tau[first_alive]
    peak_tau = tau[peak_idx]
    early_level = traj[tau <= 0.35].mean(axis=0)
    late_level = traj[tau >= 0.65].mean(axis=0)
    mid_mask = (tau > 0.35) & (tau < 0.65)
    mid_peak = traj[mid_mask].max(axis=0) if mid_mask.any() else np.zeros(traj.shape[1])
    mid_mean = (
        traj[mid_mask].mean(axis=0) if mid_mask.any() else np.zeros(traj.shape[1])
    )
    trajectory_range = traj.max(axis=0) - traj.min(axis=0)
    return {
        "peak_idx": peak_idx.astype(np.int16),
        "peak_step": steps[peak_idx].astype(np.int64),
        "birth_idx": first_alive.astype(np.int16),
        "birth_step": steps[first_alive].astype(np.int64),
        "death_idx": last_alive.astype(np.int16),
        "death_step": steps[last_alive].astype(np.int64),
        "peak_tau": peak_tau.astype(np.float32),
        "support_span_tau": support_span.astype(np.float32),
        "alive_fraction": alive_frac.astype(np.float32),
        "early_level": early_level.astype(np.float32),
        "late_level": late_level.astype(np.float32),
        "mid_peak": mid_peak.astype(np.float32),
        "mid_mean": mid_mean.astype(np.float32),
        "trajectory_range": trajectory_range.astype(np.float32),
    }


def _classify_profiles(traj: np.ndarray, steps: np.ndarray) -> dict[str, np.ndarray]:
    stats = _lifecycle_stats(traj, steps)
    early = stats["early_level"]
    late = stats["late_level"]
    mid_peak = stats["mid_peak"]
    mid_mean = stats["mid_mean"]
    peak_tau = stats["peak_tau"]
    alive_frac = stats["alive_fraction"]
    trajectory_range = stats["trajectory_range"]
    profile = np.full(traj.shape[1], "mixed", dtype=object)
    persistent = ((alive_frac >= 0.70) & (trajectory_range <= 0.35)) | (
        np.minimum(np.minimum(early, mid_mean), late) >= 0.45
    )
    profile[persistent] = "persistent"
    unmatched = profile == "mixed"
    transitional = (
        unmatched
        & (mid_peak >= 0.75)
        & ((mid_peak - np.maximum(early, late)) >= 0.35)
        & (late <= 0.30)
        & (stats["support_span_tau"] <= 0.35)
        & (peak_tau > 0.35)
        & (peak_tau < 0.65)
    )
    profile[transitional] = "transitional"
    unmatched = profile == "mixed"
    early_decay = (
        unmatched & (early >= 0.50) & ((early - late) >= 0.30) & (peak_tau <= 0.45)
    )
    profile[early_decay] = "early_decay"
    unmatched = profile == "mixed"
    late_emerge = (
        unmatched & (late >= 0.50) & ((late - early) >= 0.30) & (peak_tau >= 0.55)
    )
    profile[late_emerge] = "late_emerge"
    return {"profile": profile, **stats}


def compute_lifecycle_profile_composition(
    runs: tuple[Run, Run],
    out_dirs: tuple[Path, ...],
) -> None:
    payload = _build_metric_payload(runs)
    stats_by_run = {
        run.key: _classify_profiles(
            payload[run.key]["norm_rel"], payload[run.key]["steps"]
        )
        for run in runs
    }
    summary_rows: list[list] = []
    feature_rows: list[list] = []
    sidecar: dict[str, dict] = {}
    for run in runs:
        stats = stats_by_run[run.key]
        n = stats["profile"].size
        for profile_name in PROFILE_ORDER:
            count = int((stats["profile"] == profile_name).sum())
            summary_rows.append(
                [run.key, profile_name, PROFILE_LABELS[profile_name], count, count / n]
            )
        for i, feature in enumerate(payload[run.key]["active_feature_idx"]):
            feature_rows.append(
                [
                    run.key,
                    int(feature),
                    str(stats["profile"][i]),
                    int(stats["peak_step"][i]),
                    int(stats["birth_step"][i]),
                    int(stats["death_step"][i]),
                    float(stats["peak_tau"][i]),
                    float(stats["support_span_tau"][i]),
                    float(stats["alive_fraction"][i]),
                    float(stats["early_level"][i]),
                    float(stats["late_level"][i]),
                    float(stats["mid_peak"][i]),
                    float(stats["mid_mean"][i]),
                    float(stats["trajectory_range"][i]),
                ]
            )
        sidecar[run.key] = {
            "steps": torch.as_tensor(payload[run.key]["steps"], dtype=torch.int64),
            "active_feature_idx": torch.as_tensor(
                payload[run.key]["active_feature_idx"], dtype=torch.int64
            ),
            "profile_order": PROFILE_ORDER,
            "profile": list(map(str, stats["profile"])),
            **{
                key: torch.as_tensor(value)
                for key, value in stats.items()
                if key != "profile"
            },
        }

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / "olmo_matched_checkpoint_window_lifecycle_profile_composition"
        _write_csv(
            base.with_name(base.name + "_summary.csv"),
            ["run", "profile", "label", "count", "fraction"],
            summary_rows,
        )
        _write_csv(
            base.with_name(base.name + "_features.csv"),
            [
                "run",
                "feature",
                "profile",
                "peak_step",
                "birth_step",
                "death_step",
                "peak_tau",
                "support_span_tau",
                "alive_fraction",
                "early_level",
                "late_level",
                "mid_peak",
                "mid_mean",
                "trajectory_range",
            ],
            feature_rows,
        )
        torch.save(sidecar, base.with_suffix(".pt"))
        print(f"  -> {base.with_suffix('.pt')}")


# ─── main ──────────────────────────────────────────────────────────────────


def main() -> int:
    fig_dir = REPO / "figures/olmo_matched_checkpoint_window"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs_loaded: list[tuple[Run, np.ndarray, np.ndarray]] = []  # (run, rates, steps)
    for run in (PYTHIA_1B_LS, OLMO_2_7B):
        rates, norms, steps = _load_rates(run)
        traj_rel, _ = _active_traj_relative(rates, norms)
        runs_loaded.append((run, traj_rel, steps))
        print(
            f"{run.key}: rates {rates.shape} → traj_rel {traj_rel.shape} steps[0..2]={steps[:3].tolist()}"
        )

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

    # Metric sidecars
    compute_appendix_metric_grid(
        (PYTHIA_1B_LS, OLMO_2_7B),
        (fig_dir,),
    )
    compute_population_lifecycle_diagnostics(
        (PYTHIA_1B_LS, OLMO_2_7B),
        (fig_dir,),
    )
    compute_decoder_norm_heatmaps(
        (PYTHIA_1B_LS, OLMO_2_7B),
        (fig_dir,),
    )
    compute_wishbone_density_grid(
        (PYTHIA_1B_LS, OLMO_2_7B),
        (fig_dir,),
    )
    compute_lifecycle_profile_composition(
        (PYTHIA_1B_LS, OLMO_2_7B),
        (fig_dir,),
    )

    # PCA wishbone for the new 1B late-start run only (canonical lifecycle plot)
    pca_scores, pca_explained = _pca_scores(runs_loaded[0][1])
    torch.save(
        {
            "runs": {
                run.key: {
                    "label": run.label,
                    "steps": torch.as_tensor(steps, dtype=torch.int64),
                    "traj_rel": torch.from_numpy(traj_rel.astype(np.float32)),
                    "peak_idx": torch.as_tensor(s["peak_idx"], dtype=torch.int64),
                    "early_minus_late": torch.from_numpy(
                        s["early_minus_late"].astype(np.float32)
                    ),
                    "final_rel": torch.from_numpy(s["final_rel"].astype(np.float32)),
                }
                for (run, traj_rel, steps), (_, s) in zip(
                    runs_loaded, summaries, strict=True
                )
            },
            "pythia1b_late_start_pca": {
                "scores": torch.from_numpy(pca_scores.astype(np.float32)),
                "explained_variance_ratio": torch.from_numpy(
                    pca_explained.astype(np.float32)
                ),
            },
            "decay_definition": {
                "decayer_mask": "peak_idx <= last_index_with_step<=1000 AND final_rel < 0.5",
                "early_dominant_mask": "(mean rate over steps<=1000) - (mean rate over steps>=14000) > 0.2",
            },
        },
        fig_dir / "appendix_style_plot_payload.pt",
    )

    # Manifest
    manifest = {
        "runs": [
            {"key": run.key, "label": run.label, "rates_path": str(run.rates_path)}
            for run, _, _ in runs_loaded
        ],
        "summaries": [
            {
                "key": run.key,
                **{k: v for k, v in s.items() if not isinstance(v, np.ndarray)},
            }
            for run, s in summaries
        ],
        "decay_definition": {
            "decayer_mask": "peak_idx <= last_index_with_step<=1000 AND final_rel < 0.5",
            "early_dominant_mask": "(mean rate over steps<=1000) - (mean rate over steps>=14000) > 0.2",
        },
    }
    (fig_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote CSV + .pt sidecars + manifest to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
