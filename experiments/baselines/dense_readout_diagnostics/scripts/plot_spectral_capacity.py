"""Plot dense spectral-capacity diagnostics."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

RESULTS_DIR = REPO / "results" / "experiments" / "dense_readout_diagnostics"
FIG_DIR = REPO / "figures" / "dense_readout_diagnostics"
PAPER_DIR = REPO / "paper" / "figures" / "dense_readout"


def load_csv(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"model"}:
                continue
            try:
                if "." in value or "e" in value.lower():
                    row[key] = float(value)
                else:
                    row[key] = int(value)
            except Exception:
                pass
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no plot rows for {path}")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def x_step(step: int | float) -> float:
    return max(float(step), 1.0)


def plot(args: argparse.Namespace) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.copy_to_paper or args.copy_numeric:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_csv(args.results_dir / "spectral_metrics.csv")
    cumulative = load_csv(args.results_dir / "spectral_cumulative.csv")
    spectra_blob = torch.load(args.results_dir / "spectral_spectra.pt", map_location="cpu", weights_only=False)
    spectra: dict[tuple[str, int], torch.Tensor] = spectra_blob["spectra"]

    model_order = ["pythia-160m", "pythia-1b", "pythia-6.9b", "olmo-2-7b"]
    models = [m for m in model_order if any(r["model"] == m for r in metrics)]
    final_step_by_model = {model: max(int(r["step"]) for r in metrics if r["model"] == model) for model in models}
    plot_rows: list[dict] = []

    for model in models:
        rows = sorted([r for r in metrics if r["model"] == model], key=lambda r: r["step"])
        for r in rows:
            plot_rows.append(
                {
                    "panel": "effective_rank_fraction",
                    "model": model,
                    "step": int(r["step"]),
                    "x": x_step(r["step"]),
                    "y": float(r["effective_rank_fraction"]),
                }
            )
            plot_rows.append(
                {
                    "panel": "stable_rank_fraction",
                    "model": model,
                    "step": int(r["step"]),
                    "x": x_step(r["step"]),
                    "y": float(r["stable_rank_fraction"]),
                }
            )

        final_step = final_step_by_model[model]
        final_rows = sorted(
            [r for r in cumulative if r["model"] == model and int(r["step"]) == final_step],
            key=lambda r: r["realized_rank"],
        )
        for r in final_rows:
            plot_rows.append(
                {
                    "panel": "final_cumulative_ev",
                    "model": model,
                    "step": final_step,
                    "x": int(r["realized_rank"]),
                    "y": float(r["cumulative_ev"]),
                }
            )

        s = spectra[(model, final_step)].double()
        d_model = int(next(r["d_model"] for r in rows if int(r["step"]) == final_step))
        n = min(args.spectrum_components, s.numel())
        rank_fracs = [(i + 1) / d_model for i in range(n)]
        norm_s = (s[:n] / s[0]).tolist()
        for x, y in zip(rank_fracs, norm_s):
            plot_rows.append(
                {
                    "panel": "final_normalized_spectrum",
                    "model": model,
                    "step": final_step,
                    "x": x,
                    "y": float(y),
                }
            )

    stem = "spectral_capacity_multimodel"
    out_dirs = [FIG_DIR] + ([PAPER_DIR] if args.copy_to_paper else [])
    for out_dir in out_dirs:
        csv_path = out_dir / f"{stem}.csv"
        pt_path = out_dir / f"{stem}.pt"
        write_csv(csv_path, plot_rows)
        torch.save(
            {
                "plot_rows": plot_rows,
                "source_results_dir": str(args.results_dir),
                "final_step_by_model": final_step_by_model,
            },
            pt_path,
        )
        print(f"wrote {csv_path}")
        print(f"wrote {pt_path}")

    if args.copy_numeric:
        for name in ("spectral_metrics.csv", "spectral_cumulative.csv"):
            shutil.copy2(args.results_dir / name, PAPER_DIR / name)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument(
        "--spectrum-components",
        type=int,
        default=512,
        help="Number of final singular values to show in the normalized spectrum panel.",
    )
    ap.add_argument(
        "--copy-to-paper",
        action="store_true",
        help="Also mirror the plot-ready CSV/.pt into paper/figures/dense_readout "
        "(a LaTeX-tree convenience; off by default — the paper/ tree is not part "
        "of this release).",
    )
    ap.add_argument(
        "--copy-numeric",
        action="store_true",
        help="Also copy source metric CSVs into paper/figures/dense_readout.",
    )
    return ap.parse_args()


def main() -> None:
    plot(parse_args())


if __name__ == "__main__":
    main()
