"""Aggregate the pilot and specificity runs into the tables behind the appendix panels.

Reads the per-run outputs of `run_1b_pilot.py` (one run per token group and
readout step, `1b_<concept>_step<step>_positive/`) and of
`run_specificity_from_pilots.py` (`specificity_step<step>_h<step>/`), and writes:

- `family_expansion_summary.csv`: one row per (concept, snapshot step, h step, k)
  with the top-feature drop, the matched-control drop mean/min/max, and the
  keep-only recoveries. The ablation-curve panel plots
  `top_drop - matched_drop_mean` with the range `top_drop - matched_drop_{max,min}`.
- `specificity_step<step>_h<step>/control_corrected_drop_k<k>.csv`: source x
  measured matrix of the source-ablation drop minus the mean matched-control drop
  (the specificity heatmap).
- `specificity_step<step>_h<step>/diagonal_summary.csv`: per source group, the
  diagonal effect against the largest and mean off-diagonal effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from readout.core.paths import repo_root
from readout.core.repro import git_commit

REPO = repo_root()

CONCEPTS = ["non_latin_scripts", "punctuation", "digits", "function_words"]
STEPS = [1000, 143000]
SPECIFICITY_KS = [16, 32]


def family_rows(run_dir: Path, run_label: str) -> list[dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = pd.read_csv(run_dir / "summary.csv")
    rows = []
    for (h_step, k), g in summary[summary["condition"] != "baseline"].groupby(["h_step", "k"], sort=True):
        top = g[g["condition"] == "top_positive_attr_ablate"].iloc[0]
        top_only = g[g["condition"] == "top_positive_attr_only"].iloc[0]
        matched_drop = g.loc[g["condition"] == "matched_ablate", "drop_from_full"]
        matched_only = g.loc[g["condition"] == "matched_only", "recovery_from_bias"]
        rows.append(
            {
                "concept": manifest["concept"],
                "snapshot_step": manifest["snapshot_step"],
                "h_step": int(h_step),
                "k": int(k),
                "n_in_C": manifest["n_in_C"],
                "n_null_C": manifest["n_null_C"],
                "orig_concept_mass": top["orig_concept_mass"],
                "full_concept_mass": top["full_concept_mass"],
                "bias_concept_mass": top["bias_concept_mass"],
                "recon_gap_orig_minus_full": top["orig_concept_mass"] - top["full_concept_mass"],
                "full_minus_bias": top["full_concept_mass"] - top["bias_concept_mass"],
                "top_drop": top["drop_from_full"],
                "matched_drop_mean": matched_drop.mean(),
                "matched_drop_max": matched_drop.max(),
                "matched_drop_min": matched_drop.min(),
                "top_minus_matched_mean": top["drop_from_full"] - matched_drop.mean(),
                "top_minus_matched_max": top["drop_from_full"] - matched_drop.max(),
                "top_only_recovery": top_only["recovery_from_bias"],
                "matched_only_mean": matched_only.mean(),
                "matched_only_max": matched_only.max(),
                "top_only_minus_matched_mean": top_only["recovery_from_bias"] - matched_only.mean(),
                "run_dir": run_label,
            }
        )
    return rows


def control_corrected_drop(specificity: pd.DataFrame, k: int, concepts: list[str]) -> pd.DataFrame:
    """(source, measured) matrix: source-ablation drop minus mean matched-control drop."""
    sel = specificity[specificity["k"] == k]
    source = sel[sel["condition"] == "source_ablate"].pivot(
        index="source_concept", columns="measured_concept", values="drop_from_full"
    )
    matched = (
        sel[sel["condition"] == "matched_ablate"]
        .groupby(["source_concept", "measured_concept"])["drop_from_full"]
        .mean()
        .unstack()
    )
    return (source - matched).loc[concepts, concepts]


def diagonal_rows(matrix: pd.DataFrame, run: str, k: int) -> list[dict]:
    rows = []
    for concept in matrix.index:
        row = matrix.loc[concept]
        diag = row[concept]
        off = row.drop(concept)
        rows.append(
            {
                "run": run,
                "k": k,
                "source_concept": concept,
                "diag": diag,
                "max_offdiag": off.max(),
                "mean_offdiag": off.mean(),
                "diag_minus_max_offdiag": diag - off.max(),
                "diag_rank": int((row > diag).sum()) + 1,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=REPO / "results/experiments/sparse_feature_causal_tests",
    )
    ap.add_argument("--concepts", nargs="+", default=CONCEPTS)
    ap.add_argument("--steps", type=int, nargs="+", default=STEPS)
    ap.add_argument("--ks", type=int, nargs="+", default=SPECIFICITY_KS)
    args = ap.parse_args()
    root = args.results_root

    family = []
    for concept in args.concepts:
        for step in args.steps:
            name = f"1b_{concept}_step{step}_positive"
            family += family_rows(root / name, f"results/experiments/sparse_feature_causal_tests/{name}")
    pd.DataFrame(family).to_csv(root / "family_expansion_summary.csv", index=False)
    print(f"[aggregate] wrote {root / 'family_expansion_summary.csv'} ({len(family)} rows)", flush=True)

    for step in args.steps:
        run = f"specificity_step{step}_h{step}"
        spec = pd.read_csv(root / run / "specificity.csv")
        diagonal = []
        for k in args.ks:
            matrix = control_corrected_drop(spec, k, args.concepts)
            matrix.rename_axis(index="source_concept", columns=None).to_csv(
                root / run / f"control_corrected_drop_k{k}.csv"
            )
            diagonal += diagonal_rows(matrix, run, k)
        pd.DataFrame(diagonal).to_csv(root / run / "diagonal_summary.csv", index=False)
        print(f"[aggregate] wrote {root / run}/control_corrected_drop_k*.csv, diagonal_summary.csv", flush=True)

    (root / "aggregate_manifest.json").write_text(
        json.dumps(
            {"concepts": args.concepts, "steps": args.steps, "ks": args.ks, "git_commit": git_commit()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
