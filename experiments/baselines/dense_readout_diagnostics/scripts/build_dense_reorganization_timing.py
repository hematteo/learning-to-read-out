"""Build the P0.3 dense early-reorganization diagnostic.

For each model and checkpoint, compute top right-singular directions of
centered W_U with randomized PCA. Then measure adjacent-checkpoint subspace
movement with chordal Grassmann distance and sign-invariant singular-vector
stability.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[4]
from readout.core.model_specs import (
    DEFAULT_STEPS_BY_MODEL,
)
from readout.core.model_specs import (
    MODEL_HF_NAMES as MODEL_NAMES,
)
from readout.core.paths import snapshot_path
from readout.core.repro import git_commit
from readout.crosscoder.snapshots import load_snapshot

K_VALUES = [5, 10, 20, 50]
PYTHIA_MODEL_ORDER = ["pythia-160m", "pythia-1b", "pythia-6.9b"]


@dataclass(frozen=True)
class ModelRun:
    label: str
    model_name: str
    steps: list[int]


def available_steps(model_name: str) -> list[int]:
    import re

    paths = []
    snap0 = snapshot_path(model_name, 0)
    if snap0.parent.exists():
        paths.extend(snap0.parent.glob("*_wu.pt"))
    out = []
    for p in paths:
        if p.name.startswith("._"):
            continue
        m = re.search(r"_step(\d+)_wu\.pt$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def resolve_steps(label: str, model_name: str, mode: str) -> list[int]:
    if mode == "default32":
        steps = DEFAULT_STEPS_BY_MODEL[label]
        missing = [s for s in steps if not snapshot_path(model_name, s).exists()]
        if missing:
            raise FileNotFoundError(f"{model_name} missing default W_U steps: {missing}")
        return list(steps)
    if mode == "available":
        steps = available_steps(model_name)
        if len(steps) < 2:
            raise RuntimeError(f"{model_name} has too few available W_U snapshots")
        return steps
    raise ValueError(f"unknown step mode {mode!r}")


def centered_top_right_singular_vectors(
    model_name: str,
    step: int,
    *,
    rank: int,
    oversample: int,
    niter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Use the same randomized-PCA seed at every checkpoint. Early W_U spectra are
    # close to isotropic, so varying the seed by step would add algorithmic jitter
    # to the adjacent-subspace distances we are trying to measure.
    torch.manual_seed(seed)
    W = load_snapshot(model_name, step, kind="wu", dtype=torch.float32)
    W.sub_(W.mean(dim=0, keepdim=True))
    q = rank + oversample
    _u, s, v = torch.pca_lowrank(W, q=q, center=False, niter=niter)
    order = torch.argsort(s, descending=True)[:rank]
    s = s[order].contiguous().cpu()
    v = v[:, order].contiguous().cpu()
    # pca_lowrank already returns orthonormal columns. QR is cheap insurance
    # against tiny numerical drift before subspace comparisons.
    v, _ = torch.linalg.qr(v, mode="reduced")
    return s, v


def compute_vectors(
    runs: list[ModelRun],
    *,
    rank: int,
    oversample: int,
    niter: int,
    seed: int,
    vector_cache: Path,
    force: bool,
) -> dict:
    if vector_cache.exists() and not force:
        return torch.load(vector_cache, map_location="cpu", weights_only=False)

    payload = {
        "rank": rank,
        "oversample": oversample,
        "niter": niter,
        "seed": seed,
        "models": {},
    }
    for run in runs:
        print(f"[vectors] {run.label}: {len(run.steps)} steps", flush=True)
        model_payload = {"steps": run.steps, "singular_values": {}, "vectors": {}}
        for i, step in enumerate(run.steps, start=1):
            print(f"  [{i:02d}/{len(run.steps):02d}] step {step}", flush=True)
            s, v = centered_top_right_singular_vectors(
                run.model_name,
                step,
                rank=rank,
                oversample=oversample,
                niter=niter,
                seed=seed,
            )
            model_payload["singular_values"][step] = s
            model_payload["vectors"][step] = v
        payload["models"][run.label] = model_payload
        torch.save(payload, vector_cache)
    return payload


def chordal_grassmann(va: torch.Tensor, vb: torch.Tensor, k: int) -> tuple[float, float]:
    overlap = va[:, :k].T @ vb[:, :k]
    sv = torch.linalg.svdvals(overlap).clamp(0.0, 1.0)
    chordal = torch.sqrt(torch.clamp(1.0 - sv.pow(2), min=0.0).sum()).item()
    return chordal, chordal / math.sqrt(k)


def stability_rows(label: str, steps: list[int], vectors: dict, rank: int) -> tuple[list[dict], list[dict]]:
    subspace_rows = []
    sv_rows = []
    for a, b in zip(steps[:-1], steps[1:]):
        va = vectors[a]
        vb = vectors[b]
        for k in K_VALUES:
            if k > rank:
                continue
            dist, dist_norm = chordal_grassmann(va, vb, k)
            overlap = (va[:, :k].T @ vb[:, :k]).abs().numpy()
            diag_mean = float(np.diag(overlap).mean())
            rows, cols = linear_sum_assignment(-overlap)
            matched_mean = float(overlap[rows, cols].mean())
            subspace_rows.append(
                {
                    "model": label,
                    "from_step": a,
                    "to_step": b,
                    "k": k,
                    "grassmann_chordal": dist,
                    "grassmann_chordal_norm": dist_norm,
                    "diag_abs_overlap_mean": diag_mean,
                    "matched_abs_overlap_mean": matched_mean,
                }
            )
        full_overlap = (va[:, :rank].T @ vb[:, :rank]).abs().numpy()
        for sv_idx in range(rank):
            sv_rows.append(
                {
                    "model": label,
                    "from_step": a,
                    "to_step": b,
                    "sv_index": sv_idx + 1,
                    "diag_abs_overlap": float(full_overlap[sv_idx, sv_idx]),
                    "best_abs_overlap": float(full_overlap[sv_idx].max()),
                    "best_match_index": int(full_overlap[sv_idx].argmax() + 1),
                }
            )
    return subspace_rows, sv_rows


def build_metrics(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    subspace_rows = []
    sv_rows = []
    rank = int(payload["rank"])
    for label, model_payload in payload["models"].items():
        steps = list(model_payload["steps"])
        vectors = model_payload["vectors"]
        s_rows, v_rows = stability_rows(label, steps, vectors, rank)
        subspace_rows.extend(s_rows)
        sv_rows.extend(v_rows)
    return pd.DataFrame(subspace_rows), pd.DataFrame(sv_rows)


def persist_metrics(
    subspace_df: pd.DataFrame,
    sv_df: pd.DataFrame,
    *,
    fig_dir: Path,
    paper_dir: Path,
    copy_to_paper: bool,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    if copy_to_paper:
        paper_dir.mkdir(parents=True, exist_ok=True)

    olmo_subspace = subspace_df[subspace_df["model"] == "olmo-2-7b"]

    out_base = fig_dir / "dense_reorganization_timing"
    subspace_df.to_csv(out_base.with_suffix(".csv"), index=False)
    torch.save(
        {
            "subspace_metrics": subspace_df.to_dict(orient="list"),
            "sv_stability": sv_df.to_dict(orient="list"),
        },
        out_base.with_suffix(".pt"),
    )

    if copy_to_paper:
        paper_base = paper_dir / "dense_reorganization_timing"
        subspace_df.to_csv(paper_base.with_suffix(".csv"), index=False)

    if olmo_subspace.empty:
        return

    out_base = fig_dir / "dense_reorganization_timing_olmo"
    olmo_subspace.to_csv(out_base.with_suffix(".csv"), index=False)

    if copy_to_paper:
        paper_base = paper_dir / "dense_reorganization_timing_olmo"
        olmo_subspace.to_csv(paper_base.with_suffix(".csv"), index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=["pythia-160m", "pythia-1b", "pythia-6.9b", "olmo-2-7b"],
        choices=sorted(MODEL_NAMES),
    )
    ap.add_argument("--step-mode", choices=["default32", "available"], default="default32")
    ap.add_argument("--rank", type=int, default=60)
    ap.add_argument("--oversample", type=int, default=16)
    ap.add_argument("--niter", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--copy-to-paper",
        action="store_true",
        help="also mirror outputs into --paper-dir (a LaTeX-tree convenience; "
        "off by default — the paper/ tree is not part of this release)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "results/experiments/dense_readout_diagnostics/dense_reorganization",
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=REPO / "figures/dense_readout_diagnostics",
    )
    ap.add_argument(
        "--paper-dir",
        type=Path,
        default=REPO / "paper/figures/dense_readout",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_to_paper:
        args.paper_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ModelRun(
            label=label,
            model_name=MODEL_NAMES[label],
            steps=resolve_steps(label, MODEL_NAMES[label], args.step_mode),
        )
        for label in args.models
    ]
    vector_cache = args.out_dir / "top_svd_vectors.pt"
    payload = compute_vectors(
        runs,
        rank=args.rank,
        oversample=args.oversample,
        niter=args.niter,
        seed=args.seed,
        vector_cache=vector_cache,
        force=args.force,
    )
    subspace_df, sv_df = build_metrics(payload)
    subspace_path = args.out_dir / "subspace_metrics.csv"
    sv_path = args.out_dir / "sv_stability.csv"
    subspace_df.to_csv(subspace_path, index=False)
    sv_df.to_csv(sv_path, index=False)

    provenance = {
        "experiment": "dense_readout_diagnostics/P0.3",
        "git_commit": git_commit(),
        "models": [
            {
                "label": run.label,
                "model_name": run.model_name,
                "steps": run.steps,
                "snapshot_paths": [str(snapshot_path(run.model_name, s)) for s in run.steps],
            }
            for run in runs
        ],
        "preprocessing": "per-snapshot column mean centering before randomized PCA",
        "rank": args.rank,
        "oversample": args.oversample,
        "niter": args.niter,
        "seed": args.seed,
        "step_mode": args.step_mode,
        "outputs": {
            "vector_cache": str(vector_cache),
            "subspace_metrics": str(subspace_path),
            "sv_stability": str(sv_path),
            "figure_dir": str(args.fig_dir),
            "paper_dir": str(args.paper_dir) if args.copy_to_paper else None,
        },
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    persist_metrics(
        subspace_df,
        sv_df,
        fig_dir=args.fig_dir,
        paper_dir=args.paper_dir,
        copy_to_paper=args.copy_to_paper,
    )
    print(f"[done] wrote {subspace_path}")
    print(f"[done] wrote {sv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
