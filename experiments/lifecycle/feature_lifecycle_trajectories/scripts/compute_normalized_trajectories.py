"""Ge-style normalized decoder-norm trajectories.

Two-panel figure: 160M (top), 1B (bottom). Each line is a single
crosscoder feature's decoder norm, normalized to its own peak.
Color encodes the peak-snapshot index (purple = early peak, yellow =
late peak). To keep the figure readable we subsample features
stratified by peak-step bucket so every part of the lifecycle is
represented.

Outputs to results/experiments/lifecycle/feature_lifecycle_trajectories/
(normalized_trajectories.csv). Also emits a four-panel selected-instrument
version for the paper narrative (.csv summary + .pt cache; no figure is
rendered).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from readout.core.model_specs import DEFAULT_STEPS_32 as _PYTHIA_STEPS_LIST
from readout.core.model_specs import OLMO_STEPS_32 as _OLMO_STEPS_LIST
from readout.core.paths import repo_root, ssd_path
from readout.core.repro import log_run_provenance

CMAP_NAME = "RdBu_r"
SELECTED_CMAP_NAME = "viridis"
EVENT_SNAP_IDX = 11  # step 1000 — diverging colormap centered here

REPO = repo_root()
SSD = ssd_path()
OUT = REPO / "results/experiments/lifecycle/feature_lifecycle_trajectories"
OUT.mkdir(parents=True, exist_ok=True)
SELECTED_OUT = REPO / "results/experiments/lifecycle/feature_lifecycle_trajectories"
CACHE = REPO / "figures/feature_lifecycle_trajectories/section52_lifecycle/cache"

# Canonical Pythia cross-snap-32 schedule (single source of truth in
# readout.core.model_specs); kept as an int64 ndarray for ndarray-index/.astype use.
GE_STEPS_32 = np.array(_PYTHIA_STEPS_LIST)

ACTIVE_THR = 0.01
LW = 0.3
ALPHA = 0.04
PANEL_BOX_ASPECT = 2 / 3


@dataclass(frozen=True)
class SelectedRun:
    key: str
    label: str
    steps: np.ndarray
    norms_path: Path | None = None
    aggregate_path: Path | None = None
    checkpoint_path: Path | None = None
    checkpoint_wd_key: str = "W_D"


# Canonical OLMo-2-7B cross-snap-32 schedule (single source of truth in
# readout.core.model_specs); kept as an int64 ndarray for ndarray-index/.astype use.
OLMO_STEPS_32 = np.array(_OLMO_STEPS_LIST, dtype=np.int64)

SELECTED_RUNS: tuple[SelectedRun, ...] = (
    SelectedRun(
        key="pythia160m_d24576",
        label=r"Pythia-160M $W_U$, $d_{\rm SAE}=24{,}576$",
        steps=GE_STEPS_32,
        norms_path=Path(
            str(
                ssd_path(
                    "derived",
                    "rates",
                    "wu-d24576-multiseed",
                    "decoder_norms_dsae24576_seed0.npy",
                )
            )
        ),
    ),
    SelectedRun(
        key="pythia1b_d24576",
        label=r"Pythia-1B $W_U$, $d_{\rm SAE}=24{,}576$",
        steps=GE_STEPS_32,
        aggregate_path=Path(str(ssd_path("derived", "aggregates", "aggregates_pythia-1b_d24576_seed0.pt"))),
    ),
    SelectedRun(
        key="pythia69b_d32768",
        label=r"Pythia-6.9B $W_U$, $d_{\rm SAE}=32{,}768$",
        steps=GE_STEPS_32,
        checkpoint_path=Path(
            str(
                ssd_path(
                    "hf_release",
                    "parameter-trajectory-crosscoders",
                    "pythia-6.9b",
                    "W_U",
                    "cross-snapshot-32",
                    "d32768",
                    "seed0-sparse.safetensors",
                )
            )
        ),
    ),
    SelectedRun(
        key="olmo2_7b_d32768",
        label=r"OLMo-2-7B $W_U$, $d_{\rm SAE}=32{,}768$",
        steps=OLMO_STEPS_32,
        checkpoint_path=Path(
            str(
                ssd_path(
                    "hf_release",
                    "parameter-trajectory-crosscoders",
                    "olmo-2-7b",
                    "W_U",
                    "cross-snapshot-32",
                    "d32768",
                    "seed0.safetensors",
                )
            )
        ),
    ),
)


def _load_norms_from_selected_run(run: SelectedRun) -> np.ndarray:
    """Load or derive decoder norms for a selected paper instrument."""
    import gc

    cache_path = CACHE / f"{run.key}_decoder_norms.npy"
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    if run.norms_path is not None:
        norms = np.load(run.norms_path).astype(np.float32)
    elif run.aggregate_path is not None:
        import torch

        blob = torch.load(run.aggregate_path, map_location="cpu", weights_only=False)
        norms = blob["decoder_norms"].float().numpy().astype(np.float32)
    elif run.checkpoint_path is not None:
        from readout.crosscoder.checkpoints import load_checkpoint

        print(f"deriving decoder norms from {run.checkpoint_path}")
        cp = load_checkpoint(run.checkpoint_path)
        W_D = cp.state_dict[run.checkpoint_wd_key].float()
        norms = W_D.norm(dim=-1).numpy().astype(np.float32)
        del W_D, cp
        gc.collect()
    else:
        raise ValueError(f"no source for {run.key}")

    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, norms)
    return norms


def _selected_payload(norms: np.ndarray, steps: np.ndarray) -> dict:
    peak = norms.max(0)
    active = peak > ACTIVE_THR
    sel = np.where(active)[0]
    peak_idx = norms[:, sel].argmax(0)
    peak_step = steps[peak_idx]
    traj = norms[:, sel] / np.clip(peak[sel], 1e-9, None)
    return {
        "active_feature_ids": sel.astype(np.int32),
        "peak_idx": peak_idx.astype(np.int16),
        "peak_step": peak_step.astype(np.int64),
        "steps": steps.astype(np.int64),
        "traj": traj.astype(np.float16),
    }


def make_selected_figure() -> None:
    SELECTED_OUT.mkdir(parents=True, exist_ok=True)
    cache_payload: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for run in SELECTED_RUNS:
        norms = _load_norms_from_selected_run(run)
        payload = _selected_payload(norms, run.steps)
        cache_payload[run.key] = payload

        traj = payload["traj"].astype(np.float32)
        for k, step in enumerate(run.steps):
            vals = traj[k]
            summary_rows.append(
                {
                    "model": run.key,
                    "step": int(step),
                    "n_active_features": int(payload["active_feature_ids"].size),
                    "median": float(np.median(vals)),
                    "q10": float(np.quantile(vals, 0.10)),
                    "q25": float(np.quantile(vals, 0.25)),
                    "q75": float(np.quantile(vals, 0.75)),
                    "q90": float(np.quantile(vals, 0.90)),
                }
            )

    out = SELECTED_OUT / "selected_decoder_norm_trajectories_full"

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    import torch

    torch.save(cache_payload, out.with_suffix(".pt"))
    print(f"wrote {csv_path} and {out.with_suffix('.pt')}")


def make_selected_strip_figure() -> None:
    SELECTED_OUT.mkdir(parents=True, exist_ok=True)
    cache_payload: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for run in SELECTED_RUNS:
        norms = _load_norms_from_selected_run(run)
        payload = _selected_payload(norms, run.steps)
        cache_payload[run.key] = payload

        traj = payload["traj"].astype(np.float32)
        for k, step in enumerate(run.steps):
            vals = traj[k]
            summary_rows.append(
                {
                    "model": run.key,
                    "step": int(step),
                    "n_active_features": int(payload["active_feature_ids"].size),
                    "median": float(np.median(vals)),
                    "q10": float(np.quantile(vals, 0.10)),
                    "q25": float(np.quantile(vals, 0.25)),
                    "q75": float(np.quantile(vals, 0.75)),
                    "q90": float(np.quantile(vals, 0.90)),
                }
            )

    out = SELECTED_OUT / "selected_decoder_norm_trajectories"

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    import torch

    torch.save(cache_payload, out.with_suffix(".pt"))
    print(f"wrote {csv_path} and {out.with_suffix('.pt')}")


def random_sample(norms: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    active = np.where(norms.max(0) > ACTIVE_THR)[0]
    return rng.choice(active, size=min(n, active.size), replace=False)


def stratified_sample(norms: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n features stratified across peak-snapshot deciles."""
    peak_idx = norms.argmax(0)
    peak = norms.max(0)
    active = np.where(peak > ACTIVE_THR)[0]
    pi_active = peak_idx[active]
    deciles = np.quantile(pi_active, np.linspace(0, 1, 11))
    sel: list[int] = []
    per_bucket = max(1, n // 10)
    for d in range(10):
        lo, hi = deciles[d], deciles[d + 1]
        if d == 9:
            mask = (pi_active >= lo) & (pi_active <= hi)
        else:
            mask = (pi_active >= lo) & (pi_active < hi)
        cand = active[mask]
        if cand.size == 0:
            continue
        take = min(per_bucket, cand.size)
        sel.extend(rng.choice(cand, size=take, replace=False).tolist())
    return np.array(sel[:n], dtype=np.int64)


def main() -> None:
    log_run_provenance()
    norms_160 = np.load(SSD / "derived/rates/wu-d8192-multiseed/decoder_norms_all_seeds.npy")[0].astype(np.float32)
    norms_1b = np.load(SSD / "derived/rates/wu-1b-d24576/decoder_norms_dsae24576_seed0.npy").astype(np.float32)

    # CSV cache (mode-independent, long-form). Writes one row per
    # (panel, feature_id, snap_idx) so any of the four x-axis variants
    # can be re-derived.
    rows: list[dict] = []
    for panel_label, norms in (("160M", norms_160), ("1B", norms_1b)):
        peak = norms.max(0)
        peak_idx = norms.argmax(0)
        active = peak > ACTIVE_THR
        sel = np.where(active)[0]
        traj = norms[:, sel] / np.clip(peak[sel], 1e-9, None)
        for j, fid in enumerate(sel):
            for k in range(traj.shape[0]):
                rows.append(
                    {
                        "panel": panel_label,
                        "feature_id": int(fid),
                        "snap_idx": int(k),
                        "step": int(GE_STEPS_32[k]),
                        "norm_normalized": float(traj[k, j]),
                        "peak_snap_idx": int(peak_idx[sel][j]),
                    }
                )
    csv_path = OUT / "normalized_trajectories.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path.name} (n={len(rows)} rows)")
    make_selected_figure()
    make_selected_strip_figure()


if __name__ == "__main__":
    main()
