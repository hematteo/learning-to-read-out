"""Build dense spectral-capacity metrics for W_U snapshots.

This computes the dense spectral-capacity diagnostics in the current
paper framing: Pythia-160M, Pythia-1B, Pythia-6.9B, and OLMo-2-7B over each
model family's canonical 32 cross-snapshot schedule by default.

Outputs are numeric only. Plotting lives in `plot_spectral_capacity.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from src.core.model_specs import (  # noqa: E402
    DEFAULT_STEPS_BY_MODEL,
)
from src.core.model_specs import MODEL_HF_NAMES as MODEL_NAMES  # noqa: E402
from src.core.paths import snapshot_path  # noqa: E402
from src.core.repro import git_commit  # noqa: E402
from src.crosscoder.snapshots import load_snapshot  # noqa: E402

SMOKE_STEPS_BY_MODEL = {
    "pythia-160m": [0, 1000, 143000],
    "pythia-1b": [0, 1000, 143000],
    "pythia-6.9b": [0, 1000, 143000],
    "olmo-2-7b": [150, 1000, 928000],
}

DEFAULT_RANKS = [16, 32, 64, 128, 256, 512, 1024]
OUT_DIR = REPO / "results" / "experiments" / "dense_readout_diagnostics"


def effective_rank_from_singular_values(s: torch.Tensor) -> float:
    eig = s.double().square()
    total = eig.sum()
    if total <= 0:
        return 0.0
    p = eig / total
    p = p[p > 0]
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


def stable_rank_from_singular_values(s: torch.Tensor) -> float:
    eig = s.double().square()
    total = eig.sum()
    top = eig[0] if eig.numel() else torch.tensor(0.0, dtype=torch.float64)
    if top <= 0:
        return 0.0
    return float((total / top).item())


def spectral_metrics(
    model_label: str,
    step: int,
    w: torch.Tensor,
    ranks: list[int],
) -> tuple[dict, list[dict], torch.Tensor]:
    """Return summary row, cumulative rows, and full centered singular values."""
    vocab, d_model = w.shape
    mean = w.mean(dim=0, keepdim=True)
    wc = w - mean
    # torch.linalg.svdvals is not implemented on MPS; CPU is reliable here.
    s = torch.linalg.svdvals(wc).cpu()
    eig = s.double().square()
    total = float(eig.sum().item())
    eff_rank = effective_rank_from_singular_values(s)
    stable_rank = stable_rank_from_singular_values(s)
    spectral_gap = (
        float((s[0] / s[1]).item()) if s.numel() > 1 and s[1] > 0 else math.nan
    )

    summary = {
        "model": model_label,
        "step": step,
        "vocab": vocab,
        "d_model": d_model,
        "mean_row_norm": float(mean.norm().item()),
        "centered_fro_sq": total,
        "top_singular_value": float(s[0].item()),
        "second_singular_value": float(s[1].item()) if s.numel() > 1 else math.nan,
        "spectral_gap": spectral_gap,
        "effective_rank": eff_rank,
        "effective_rank_fraction": eff_rank / d_model,
        "stable_rank": stable_rank,
        "stable_rank_fraction": stable_rank / d_model,
    }

    cumulative = []
    cumsum = torch.cumsum(eig, dim=0)
    for rank in ranks:
        realized = min(rank, s.numel())
        ev = float((cumsum[realized - 1] / total).item()) if total > 0 else 0.0
        cumulative.append(
            {
                "model": model_label,
                "step": step,
                "d_model": d_model,
                "requested_rank": rank,
                "realized_rank": realized,
                "rank_fraction": realized / d_model,
                "cumulative_ev": ev,
            }
        )

    return summary, cumulative, s


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_NAMES),
        default=["pythia-160m", "pythia-1b", "pythia-6.9b", "olmo-2-7b"],
    )
    ap.add_argument("--steps", nargs="+", type=int, default=None)
    ap.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run three representative checkpoints for each selected model.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for numeric outputs.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = args.out_dir / "spectral_metrics.csv"
    cumulative_csv = args.out_dir / "spectral_cumulative.csv"
    spectra_pt = args.out_dir / "spectral_spectra.pt"
    provenance_json = args.out_dir / "spectral_provenance.json"

    if not args.overwrite and any(
        p.exists() for p in (metrics_csv, cumulative_csv, spectra_pt)
    ):
        raise SystemExit(
            "output exists; pass --overwrite to rebuild: "
            f"{metrics_csv}, {cumulative_csv}, {spectra_pt}"
        )

    metric_rows: list[dict] = []
    cumulative_rows: list[dict] = []
    spectra: dict[tuple[str, int], torch.Tensor] = {}
    paths: list[dict] = []
    steps_by_model: dict[str, list[int]] = {}

    t_all = time.time()
    for model_label in args.models:
        model_name = MODEL_NAMES[model_label]
        if args.steps is not None:
            steps = list(args.steps)
        elif args.smoke:
            steps = SMOKE_STEPS_BY_MODEL[model_label]
        else:
            steps = DEFAULT_STEPS_BY_MODEL[model_label]
        steps_by_model[model_label] = list(steps)
        print(f"[{model_label}] {len(steps)} W_U checkpoints")
        for i, step in enumerate(steps, start=1):
            path = snapshot_path(model_name, step, kind="wu")
            paths.append({"model": model_label, "step": step, "path": str(path)})
            print(f"  [{i:02d}/{len(steps):02d}] step {step}: loading {path.name}")
            t0 = time.time()
            w = load_snapshot(model_name, step, kind="wu", dtype=torch.float32)
            summary, cumulative, s = spectral_metrics(model_label, step, w, args.ranks)
            metric_rows.append(summary)
            cumulative_rows.extend(cumulative)
            spectra[(model_label, int(step))] = s
            del w
            print(
                "    "
                f"d={summary['d_model']} eff={summary['effective_rank']:.1f} "
                f"stable={summary['stable_rank']:.1f} "
                f"gap={summary['spectral_gap']:.2f} "
                f"time={time.time() - t0:.1f}s"
            )

    write_csv(metrics_csv, metric_rows)
    write_csv(cumulative_csv, cumulative_rows)
    torch.save(
        {
            "spectra": spectra,
            "metrics": metric_rows,
            "cumulative": cumulative_rows,
            "ranks": args.ranks,
            "steps_by_model": steps_by_model,
            "models": args.models,
        },
        spectra_pt,
    )
    provenance = {
        "experiment": "dense_readout_diagnostics.phase0_2_spectral_capacity",
        "models": args.models,
        "steps_by_model": steps_by_model,
        "ranks": args.ranks,
        "snapshot_paths": paths,
        "preprocessing": "centered W_U = W_U - mean_row(W_U); scalar scaling omitted because spectral EV/ranks are scale-invariant",
        "git_commit": git_commit(),
        "um_ssd_root": os.environ.get("UM_SSD_ROOT"),
        "elapsed_sec": time.time() - t_all,
    }
    provenance_json.write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"wrote {metrics_csv} ({len(metric_rows)} rows)")
    print(f"wrote {cumulative_csv} ({len(cumulative_rows)} rows)")
    print(f"wrote {spectra_pt}")
    print(f"wrote {provenance_json}")


if __name__ == "__main__":
    main()
