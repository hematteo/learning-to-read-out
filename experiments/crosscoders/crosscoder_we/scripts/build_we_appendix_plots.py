"""Build paper-facing W_E companion metrics from canonical cached sidecars.

This script intentionally avoids recomputing crosscoder activations. The
released safetensors contain the trained dictionaries, while the existing SSD
sidecars contain per-feature firing rates and decoder norms for the Pythia-160M
matched-width W_E/W_U 5-seed comparison.

Outputs:
  - figures/crosscoder_we/read_write_asymmetry.{csv,pt}
  - figures/crosscoder_we/we_quality_pareto.{csv,pt}
  - results/experiments/crosscoder_we/{read_write_asymmetry,we_quality_pareto}.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from src.core.model_specs import DEFAULT_STEPS_32 as STEPS_32  # noqa: E402
from src.core.paths import release_path, repo_root, ssd_root

MODEL_SHORT = "pythia-160m"


def _load_we_rates_and_norms(cache_root: Path) -> tuple[np.ndarray, np.ndarray]:
    we_dir = cache_root / "derived" / "rates" / "we-d8192-multiseed"
    rates = []
    norms = []
    for seed in range(5):
        rate_path = we_dir / f"we_rates_dsae8192_seed{seed}.pt"
        norm_path = we_dir / f"we_cc_dsae8192_seed{seed}_norms.npy"
        payload = torch.load(rate_path, map_location="cpu", weights_only=False)
        if payload.get("steps") != STEPS_32:
            raise ValueError(f"unexpected W_E steps in {rate_path}")
        per_seed = payload["rates_per_seed"]
        if len(per_seed) != 1:
            raise ValueError(f"expected one rate tensor in {rate_path}, got {per_seed.keys()}")
        rates.append(next(iter(per_seed.values())).float().numpy())
        norms.append(np.load(norm_path).astype(np.float32))
    return np.stack(rates, axis=0), np.stack(norms, axis=0)


def _load_wu_rates_and_norms(cache_root: Path) -> tuple[np.ndarray, np.ndarray]:
    run3 = cache_root / "derived" / "rates" / "wu-d8192-multiseed"
    rates = np.load(run3 / "firing_rates_all_seeds.npy").astype(np.float32)
    norms = np.load(run3 / "decoder_norms_all_seeds.npy").astype(np.float32)
    if rates.shape != (5, 32, 8192):
        raise ValueError(f"unexpected W_U rates shape {rates.shape}")
    if norms.shape != (5, 32, 8192):
        raise ValueError(f"unexpected W_U norms shape {norms.shape}")
    return rates, norms


def _trajectory_rows(label: str, rates: np.ndarray, norms: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    mean_rate = rates.mean(axis=2)
    l0 = rates.sum(axis=2)
    mean_norm = norms.mean(axis=2)
    for seed in range(rates.shape[0]):
        for i, step in enumerate(STEPS_32):
            rows.append(
                {
                    "matrix": label,
                    "seed": seed,
                    "step": step,
                    "mean_feature_rate": float(mean_rate[seed, i]),
                    "mean_l0": float(l0[seed, i]),
                    "mean_decoder_norm": float(mean_norm[seed, i]),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _quality_rows() -> list[dict]:
    rows: list[dict] = []
    specs = [
        ("W_E", 8192, range(5), "W_E d8192"),
        ("W_E", 24576, range(1), "W_E d24576"),
        ("W_U", 8192, range(5), "W_U d8192"),
        ("W_U", 24576, range(3), "W_U d24576"),
    ]
    for kind, dim, seeds, label in specs:
        for seed in seeds:
            path = release_path(MODEL_SHORT, kind, dim=dim, seed=seed)
            cfg_path = path.with_suffix(".config.json")
            if not cfg_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            q = cfg["quality"]
            rows.append(
                {
                    "matrix": kind,
                    "d_sae": dim,
                    "seed": seed,
                    "label": label,
                    "explained_variance": float(q["explained_variance"]),
                    "mean_l0": float(q["mean_l0"]),
                    "pct_active": float(100.0 * q["mean_l0"] / dim),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssd-root",
        type=Path,
        default=ssd_root(),
        help="canonical SSD root",
    )
    args = parser.parse_args()

    root = repo_root()
    fig_dir = root / "figures" / "crosscoder_we"
    result_dir = root / "results" / "experiments" / "crosscoder_we"

    we_rates, we_norms = _load_we_rates_and_norms(args.ssd_root)
    wu_rates, wu_norms = _load_wu_rates_and_norms(args.ssd_root)
    rows = _trajectory_rows("W_U", wu_rates, wu_norms) + _trajectory_rows("W_E", we_rates, we_norms)

    for base in [fig_dir / "read_write_asymmetry"]:
        _write_csv(base.with_suffix(".csv"), rows)
        torch.save(
            {
                "steps": STEPS_32,
                "wu_rates": torch.from_numpy(wu_rates),
                "wu_norms": torch.from_numpy(wu_norms),
                "we_rates": torch.from_numpy(we_rates),
                "we_norms": torch.from_numpy(we_norms),
            },
            base.with_suffix(".pt"),
        )

    quality_rows = _quality_rows()
    for base in [fig_dir / "we_quality_pareto"]:
        _write_csv(base.with_suffix(".csv"), quality_rows)
        torch.save(quality_rows, base.with_suffix(".pt"))

    result_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(result_dir / "read_write_asymmetry.csv", rows)
    _write_csv(result_dir / "we_quality_pareto.csv", quality_rows)

    print(f"wrote read/write metrics to {fig_dir}")
    print(f"wrote audit CSVs to {result_dir}")


if __name__ == "__main__":
    main()
