"""Build PCA wishbone diagnostics for the four selected paper instruments.

An earlier diagnostic used PCA on each feature's relative-to-peak activation-rate
trajectory to expose "wishbone" structure. This script runs that diagnostic
for the selected paper crosscoders:

  - Pythia-160M W_U d_sae=24576
  - Pythia-1B W_U d_sae=24576
  - Pythia-6.9B W_U d_sae=32768 sparse/high-lambda
  - OLMo-2-7B W_U d_sae=32768

Outputs are written to results/experiments/lifecycle/feature_lifecycle_trajectories/wishbone
as CSV and .pt sidecars.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]

from readout.core.model_specs import DEFAULT_STEPS_32 as _PYTHIA_STEPS_LIST  # noqa: E402
from readout.core.model_specs import OLMO_STEPS_32 as _OLMO_STEPS_LIST  # noqa: E402
from readout.core.paths import ssd_path  # noqa: E402

OUT_DEFAULT = REPO / "results/experiments/lifecycle/feature_lifecycle_trajectories/wishbone"
CACHE = REPO / "figures/feature_lifecycle_trajectories/section52_lifecycle/cache"

PYTHIA_STEPS_32 = np.asarray(_PYTHIA_STEPS_LIST, dtype=np.int64)
OLMO_STEPS_32 = np.asarray(_OLMO_STEPS_LIST, dtype=np.int64)

ACTIVE_NORM_THR = 0.01
ALIVE_FRAC_THR = 0.30
RATE_ACTIVE_THR = 1e-3
EARLY_CUTOFF_STEP = 1000
LATE_CUTOFF_STEP = 3000


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    slug: str
    steps: np.ndarray
    rates_path: Path
    xticks: tuple[int, ...]
    rates_key: str | None = None
    geometry_key: str | None = None


RUNS: tuple[Run, ...] = (
    Run(
        key="pythia160m_d24576",
        label="Pythia-160M W_U d=24576",
        slug="pythia160m_d24576",
        steps=PYTHIA_STEPS_32,
        rates_path=ssd_path("derived", "aggregates", "aggregates_pythia-160m_d24576_seed0.pt"),
        rates_key="wu_cc_dsae24576_seed0",
        xticks=(0, 1000, 14000, 143000),
    ),
    Run(
        key="pythia1b_d24576",
        label="Pythia-1B W_U d=24576",
        slug="pythia1b_d24576",
        steps=PYTHIA_STEPS_32,
        rates_path=ssd_path("derived", "aggregates", "aggregates_pythia-1b_d24576_seed0.pt"),
        xticks=(0, 1000, 14000, 143000),
    ),
    Run(
        key="pythia69b_d32768",
        label="Pythia-6.9B W_U d=32768 sparse",
        slug="pythia69b_sparse_d32768",
        steps=PYTHIA_STEPS_32,
        rates_path=REPO / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/"
        "p69b_sparse_d32768_seed0.pt",
        geometry_key="pythia69b_sparse_d32768",
        xticks=(0, 1000, 14000, 143000),
    ),
    Run(
        key="olmo2_7b_d32768",
        label="OLMo-2-7B W_U d=32768",
        slug="olmo2_7b_d32768",
        steps=OLMO_STEPS_32,
        rates_path=REPO / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/"
        "olmo27b_d32768_seed0.pt",
        xticks=(150, 1000, 14000, 928000),
    ),
)


@dataclass(frozen=True)
class CornerBoundary:
    pc1: float
    pc2: float
    threshold: float


# Manual linear boundaries for splitting the PCA wishbone at its visual corner:
# label 1 means pc1 * PC1 + pc2 * PC2 > threshold. The Pythia panels keep a
# vertical PC1 elbow cut; OLMo uses an angled boundary because its bend is tilted.
MANUAL_CORNER_BOUNDARIES: dict[str, CornerBoundary] = {
    "pythia160m_d24576": CornerBoundary(pc1=1.0, pc2=0.0, threshold=0.15),
    "pythia1b_d24576": CornerBoundary(pc1=1.0, pc2=0.0, threshold=0.55),
    "pythia69b_sparse_d32768": CornerBoundary(pc1=1.0, pc2=0.0, threshold=0.41),
    "olmo2_7b_d32768": CornerBoundary(pc1=1.0, pc2=0.45, threshold=0.02),
}


def nearest_step_idx(steps: np.ndarray, target: int) -> int:
    return int(np.argmin(np.abs(steps - target)))


def load_rates(run: Run) -> np.ndarray:
    if not run.rates_path.exists():
        raise FileNotFoundError(run.rates_path)
    blob = torch.load(run.rates_path, map_location="cpu", weights_only=False)
    if "rates" in blob:
        rates = blob["rates"]
    elif "rates_per_seed" in blob:
        if run.rates_key is not None:
            rates = blob["rates_per_seed"][run.rates_key]
        else:
            rates = next(iter(blob["rates_per_seed"].values()))
    else:
        raise KeyError(f"{run.rates_path} has no rates or rates_per_seed")
    if isinstance(rates, torch.Tensor):
        rates = rates.detach().cpu().numpy()
    rates = rates.astype(np.float64)
    if rates.shape != (run.steps.size, rates.shape[1]):
        raise ValueError(f"{run.slug}: bad rates shape {rates.shape}")
    return rates


def load_arrays(run: Run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norms_path = CACHE / f"{run.key}_decoder_norms.npy"
    rotation_key = run.geometry_key or run.key
    rotation_path = CACHE / f"{rotation_key}_adjacent_rotation_radians.npy"
    if not norms_path.exists():
        raise FileNotFoundError(norms_path)
    if not rotation_path.exists():
        raise FileNotFoundError(rotation_path)
    rates = load_rates(run)
    norms = np.load(norms_path).astype(np.float64)
    rotation = np.load(rotation_path).astype(np.float64)
    if norms.shape != rates.shape:
        raise ValueError(f"{run.slug}: norms {norms.shape}, rates {rates.shape}")
    if rotation.shape != (run.steps.size - 1, rates.shape[1]):
        raise ValueError(f"{run.slug}: rotation {rotation.shape}, rates {rates.shape}")
    return rates, norms, rotation


def run_pca(traj_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = traj_rel - traj_rel.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt.T
    explained = singular_values**2 / np.clip(np.sum(singular_values**2), 1e-12, None)
    return scores, vt, explained


def compute_metrics(
    rates: np.ndarray, norms: np.ndarray, rotation: np.ndarray, steps: np.ndarray
) -> dict[str, np.ndarray]:
    peak_step_rate = rates.argmax(axis=0)
    peak_step_norm = norms.argmax(axis=0)
    above = rates > RATE_ACTIVE_THR
    first_active = above.argmax(axis=0)
    first_active = np.where(above.any(axis=0), first_active, steps.size)
    early_mask = steps < EARLY_CUTOFF_STEP
    late_mask = steps >= LATE_CUTOFF_STEP
    early_late_rate = rates[early_mask].mean(axis=0) - rates[late_mask].mean(axis=0)
    peak_rate = np.clip(rates.max(axis=0, keepdims=True), 1e-12, None)
    lifetime = ((rates / peak_rate) > ALIVE_FRAC_THR).sum(axis=0)
    return {
        "peak_step_rate": peak_step_rate,
        "peak_step_norm": peak_step_norm,
        "prospective_first_active": first_active,
        "early_late_rate": early_late_rate,
        "lifetime": lifetime,
        "drift": rotation.sum(axis=0),
        "norm_step0": norms[0],
        "norm_final": norms[-1],
    }


def active_rate_trajectories(rates: np.ndarray, norms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = norms.max(axis=0) > ACTIVE_NORM_THR
    traj = rates[:, active].T
    peak = np.clip(traj.max(axis=1, keepdims=True), 1e-12, None)
    return traj / peak, active


def kmeans2_pca(scores: np.ndarray, max_iter: int = 100) -> np.ndarray:
    coords = scores[:, :2].astype(np.float64)
    coords = (coords - coords.mean(axis=0, keepdims=True)) / np.clip(coords.std(axis=0, keepdims=True), 1e-12, None)
    pc1 = coords[:, 0]
    centers = np.stack(
        [
            coords[np.argmin(pc1)],
            coords[np.argmax(pc1)],
        ]
    )
    labels = np.zeros(coords.shape[0], dtype=np.int8)
    for _ in range(max_iter):
        dist = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1).astype(np.int8)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in (0, 1):
            mask = labels == cluster
            if np.any(mask):
                centers[cluster] = coords[mask].mean(axis=0)
    if np.median(scores[labels == 0, 0]) > np.median(scores[labels == 1, 0]):
        labels = 1 - labels
    return labels.astype(np.int8)


def manual_corner_labels(run: Run, scores: np.ndarray) -> np.ndarray:
    boundary = MANUAL_CORNER_BOUNDARIES[run.slug]
    side = boundary.pc1 * scores[:, 0] + boundary.pc2 * scores[:, 1]
    return (side > boundary.threshold).astype(np.int8)


def compute_arms(scores: np.ndarray) -> np.ndarray:
    pc1 = scores[:, 0]
    pc2 = scores[:, 1]
    pc1_cut = np.median(pc1)
    pc2_cut = np.median(pc2[pc1 > pc1_cut])
    arm = np.where(
        pc1 < pc1_cut,
        "left",
        np.where(pc2 > pc2_cut, "upper-right", "lower-right"),
    )
    return arm


def save_scores_csv(
    path: Path,
    scores: np.ndarray,
    metrics: dict[str, np.ndarray],
    active: np.ndarray,
    arm: np.ndarray,
) -> None:
    keys = [
        "peak_step_rate",
        "peak_step_norm",
        "prospective_first_active",
        "early_late_rate",
        "lifetime",
        "drift",
        "norm_step0",
        "norm_final",
    ]
    active_ids = np.flatnonzero(active)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_idx", "PC1", "PC2", "PC3", "arm", *keys])
        for row_idx, feature_idx in enumerate(active_ids):
            writer.writerow(
                [
                    int(feature_idx),
                    float(scores[row_idx, 0]),
                    float(scores[row_idx, 1]),
                    float(scores[row_idx, 2]),
                    arm[row_idx],
                    *[float(metrics[key][feature_idx]) for key in keys],
                ]
            )


def run_one(run: Run, out: Path) -> tuple[dict[str, float], np.ndarray]:
    print(f"\n=== {run.label} [{run.slug}] ===")
    rates, norms, rotation = load_arrays(run)
    traj_rel, active = active_rate_trajectories(rates, norms)
    scores, components, explained = run_pca(traj_rel)
    print(
        f"  active {traj_rel.shape[0]} / {rates.shape[1]}  "
        f"PC1 {explained[0] * 100:.1f}%  "
        f"PC2 {explained[1] * 100:.1f}%  "
        f"PC3 {explained[2] * 100:.1f}%"
    )

    metrics = compute_metrics(rates, norms, rotation, run.steps)
    arm = compute_arms(scores)

    save_scores_csv(out / f"{run.slug}_scores_metrics.csv", scores, metrics, active, arm)
    torch.save(
        {
            "scores": torch.from_numpy(scores.astype(np.float32)),
            "components": torch.from_numpy(components.astype(np.float32)),
            "explained_variance_ratio": torch.from_numpy(explained.astype(np.float32)),
            "active_feature_idx": torch.from_numpy(np.flatnonzero(active).astype(np.int64)),
            "arm": arm.tolist(),
            "steps": run.steps.tolist(),
            "label": run.label,
        },
        out / f"{run.slug}_wishbone.pt",
    )
    return (
        {
            "slug": run.slug,
            "label": run.label,
            "n_features": int(rates.shape[1]),
            "n_active": int(traj_rel.shape[0]),
            "pc1_pct": float(explained[0] * 100),
            "pc2_pct": float(explained[1] * 100),
            "pc3_pct": float(explained[2] * 100),
        },
        traj_rel,
    )


def load_grid_payload(out: Path, runs: list[Run]) -> list[tuple[Run, np.ndarray, np.ndarray]]:
    payload = []
    for run in runs:
        sidecar = torch.load(out / f"{run.slug}_wishbone.pt", weights_only=False)
        scores = sidecar["scores"].detach().cpu().numpy()
        explained = sidecar["explained_variance_ratio"].detach().cpu().numpy()
        payload.append((run, scores, explained))
    return payload


def write_summary(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = [
            "slug",
            "label",
            "n_features",
            "n_active",
            "pc1_pct",
            "pc2_pct",
            "pc3_pct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--configs", nargs="+", default=None)
    args = parser.parse_args()

    runs = list(RUNS)
    if args.configs:
        runs = [run for run in runs if any(token in run.slug for token in args.configs)]
        if not runs:
            raise SystemExit(f"no configs matched {args.configs}")

    args.out.mkdir(parents=True, exist_ok=True)
    run_outputs = [run_one(run, args.out) for run in runs]
    summary = [row for row, _traj in run_outputs]
    grid_payload = load_grid_payload(args.out, runs)
    cluster_payload: dict[str, np.ndarray] = {run.slug: kmeans2_pca(scores) for run, scores, _explained in grid_payload}
    split_payload: dict[str, np.ndarray] = {
        run.slug: manual_corner_labels(run, scores) for run, scores, _explained in grid_payload
    }
    torch.save(
        {slug: torch.from_numpy(labels.astype(np.int8)) for slug, labels in cluster_payload.items()},
        args.out / "selected_wishbone_k2_pca_clusters.pt",
    )
    torch.save(
        {slug: torch.from_numpy(labels.astype(np.int8)) for slug, labels in split_payload.items()},
        args.out / "selected_wishbone_manual_corner_split.pt",
    )
    write_summary(args.out / "selected_wishbone_summary.csv", summary)
    print(f"\nDone. Sidecars under {args.out}")


if __name__ == "__main__":
    main()
