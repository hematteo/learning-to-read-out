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
import json

# Same-experiment sibling module; the insert makes this robust to import
# from outside the scripts/ dir (direct file execution already has it first).
import sys as _sys  # noqa: E402
from pathlib import Path
from pathlib import Path as _P  # noqa: E402

import numpy as np
import torch

_sys.path.insert(0, str(_P(__file__).resolve().parent))
from we_common import (  # noqa: E402
    STEPS_32,
    _load_we_rates_and_norms,
    _load_wu_rates_and_norms,
    _quality_rows,
    _write_csv,
)

from readout.core.paths import repo_root, ssd_root
from readout.core.repro import git_commit, log_run_provenance

REPO = repo_root()


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
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "build_we_appendix_plots.provenance.json").write_text(
        json.dumps({**log_run_provenance(), "git_commit": git_commit()}, indent=2)
    )

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
