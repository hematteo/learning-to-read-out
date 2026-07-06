"""Step 6 reporting: held-out best-s selection, paired bootstrap.

Consumes the per-example .pt sidecars produced by run_swap_grid.py
--save-per-example (Step 2a contract). For each (family, alignment, h_step):

  1. Dev/test split by example index (seeded; same split across all s_step
     cells for that (family, alignment, h_step) so the comparison is paired).
  2. Choose best s_step on **dev** mean-margin; report margin + accuracy on
     **test** for that s.
  3. Paired bootstrap 95% CI on (best_s_margin - native_margin) and
     (best_s_acc - native_acc) on the test partition.
  4. Preselected-readout comparison: native, s=1000, s=2000, s=final (the
     final s in the discovered list) — for users who want to fix s in
     advance and not pay a multi-test correction.

Outputs:
  <out-dir>/heldout_best_s_selection.csv
  <out-dir>/paired_bootstrap_ci.csv
  <out-dir>/preselected_readout_comparison.csv
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[4]
PT_RE = re.compile(r"^(?P<fam>.+)__a-(?P<al>[^_]+)__h(?P<h>\d+)__s(?P<s>\d+)\.pt$")


def discover_cells(run_dir: Path) -> list[dict]:
    """Walk shards/*.pt and emit one dict per cell."""
    shards = run_dir / "shards"
    out: list[dict] = []
    for p in sorted(shards.glob("*.pt")):
        m = PT_RE.match(p.name)
        if not m:
            continue
        out.append(
            {
                "path": p,
                "family": m["fam"],
                "alignment": m["al"],
                "h_step": int(m["h"]),
                "s_step": int(m["s"]),
            }
        )
    return out


def _load_per_example(path: Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "margins": d["margins"].float().numpy(),
        "margins_native": d["margins_native"].float().numpy(),
    }


def _split_dev_test(
    n: int, seed: int, dev_frac: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(round(n * dev_frac))
    return idx[:cut], idx[cut:]


def _paired_bootstrap_ci(
    diffs: np.ndarray, *, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap CI on the per-example difference array. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    sample = diffs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (
        float(diffs.mean()),
        float(np.quantile(sample, 0.025)),
        float(np.quantile(sample, 0.975)),
    )


def report_per_group(
    group: list[dict],
    *,
    seed: int,
    preselected_s: list[int],
) -> dict | None:
    """Run Step-6 reductions for one (family, alignment, h_step) group."""
    if not group:
        return None
    first = group[0]
    family, alignment, h = first["family"], first["alignment"], first["h_step"]

    # Load all cells; align by example index. n must match across cells.
    per_cell: dict[int, dict] = {}
    n_ref = None
    margins_native_ref: np.ndarray | None = None
    for cell in group:
        d = _load_per_example(cell["path"])
        if n_ref is None:
            n_ref = d["margins"].shape[0]
            margins_native_ref = d["margins_native"]
        elif d["margins"].shape[0] != n_ref:
            print(
                f"[warn] {family} h={h} s={cell['s_step']}: n mismatch "
                f"({d['margins'].shape[0]} vs {n_ref}); skipping cell",
                flush=True,
            )
            continue
        per_cell[cell["s_step"]] = d

    if n_ref is None or n_ref < 4:
        return None
    dev_idx, test_idx = _split_dev_test(n_ref, seed)

    # Native row uses the (s=h) cell margins_native field; if that cell is
    # missing we fall back to any other cell's margins_native (it's the same
    # tensor across s, by construction in run_swap_grid).
    m_native = per_cell[h]["margins_native"] if h in per_cell else margins_native_ref

    # Best-s on dev mean margin
    s_steps = sorted(per_cell.keys())
    best_s = max(
        s_steps,
        key=lambda s: float(per_cell[s]["margins"][dev_idx].mean()),
    )
    m_best = per_cell[best_s]["margins"]

    # Test-set metrics
    test_native_mg = float(m_native[test_idx].mean())
    test_native_ac = float((m_native[test_idx] > 0).mean())
    test_best_mg = float(m_best[test_idx].mean())
    test_best_ac = float((m_best[test_idx] > 0).mean())

    # Paired bootstrap on (best_s - native) margin and accuracy
    diff_mg = m_best[test_idx] - m_native[test_idx]
    diff_ac = (m_best[test_idx] > 0).astype(float) - (m_native[test_idx] > 0).astype(
        float
    )
    mg_mean, mg_lo, mg_hi = _paired_bootstrap_ci(diff_mg, seed=seed)
    ac_mean, ac_lo, ac_hi = _paired_bootstrap_ci(diff_ac, seed=seed)

    # Preselected readouts
    preselect_rows = []
    for s in preselected_s:
        if s not in per_cell:
            continue
        m_s = per_cell[s]["margins"]
        d_mg = m_s[test_idx] - m_native[test_idx]
        d_ac = (m_s[test_idx] > 0).astype(float) - (m_native[test_idx] > 0).astype(
            float
        )
        mg_m, mg_l, mg_h = _paired_bootstrap_ci(d_mg, seed=seed)
        ac_m, ac_l, ac_h = _paired_bootstrap_ci(d_ac, seed=seed)
        preselect_rows.append(
            {
                "family": family,
                "alignment": alignment,
                "h_step": h,
                "s_step": s,
                "tag": _preselected_tag(s, preselected_s),
                "mean_margin_test": float(m_s[test_idx].mean()),
                "accuracy_test": float((m_s[test_idx] > 0).mean()),
                "delta_margin": mg_m,
                "delta_margin_ci_lo": mg_l,
                "delta_margin_ci_hi": mg_h,
                "delta_accuracy": ac_m,
                "delta_accuracy_ci_lo": ac_l,
                "delta_accuracy_ci_hi": ac_h,
            }
        )

    return {
        "family": family,
        "alignment": alignment,
        "h_step": h,
        "n_dev": len(dev_idx),
        "n_test": len(test_idx),
        "best_s_step_on_dev": best_s,
        "native_mean_margin_test": test_native_mg,
        "best_s_mean_margin_test": test_best_mg,
        "native_accuracy_test": test_native_ac,
        "best_s_accuracy_test": test_best_ac,
        "delta_margin_test": test_best_mg - test_native_mg,
        "delta_margin_ci_lo": mg_lo,
        "delta_margin_ci_hi": mg_hi,
        "delta_accuracy_test": test_best_ac - test_native_ac,
        "delta_accuracy_ci_lo": ac_lo,
        "delta_accuracy_ci_hi": ac_hi,
        "_preselect_rows": preselect_rows,
    }


def _preselected_tag(s: int, preselected: list[int]) -> str:
    if s == max(preselected):
        return "final"
    return f"s={s}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run_swap_grid.py output dir; must contain shards/*.pt sidecars.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--preselected-s", type=int, nargs="+", default=[1000, 2000, 143000]
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = discover_cells(args.run_dir)
    if not cells:
        raise FileNotFoundError(
            f"no per-example .pt sidecars under {args.run_dir / 'shards'}; "
            f"was --save-per-example passed when generating the swap grid?"
        )

    # Group by (family, alignment, h_step)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        groups[(c["family"], c["alignment"], c["h_step"])].append(c)

    main_rows: list[dict] = []
    preselect_rows: list[dict] = []
    for key, group in sorted(groups.items()):
        res = report_per_group(group, seed=args.seed, preselected_s=args.preselected_s)
        if res is None:
            continue
        main_rows.append({k: v for k, v in res.items() if not k.startswith("_")})
        preselect_rows.extend(res["_preselect_rows"])
        print(
            f"  [{res['family']:<14} {res['alignment']:<10} h={res['h_step']:>6}] "
            f"best_s={res['best_s_step_on_dev']:>6} "
            f"Δacc={res['delta_accuracy_test']:+.3f} "
            f"[{res['delta_accuracy_ci_lo']:+.3f}, {res['delta_accuracy_ci_hi']:+.3f}] "
            f"Δmg={res['delta_margin_test']:+.3f}",
            flush=True,
        )

    pd.DataFrame(main_rows).to_csv(
        args.out_dir / "heldout_best_s_selection.csv", index=False
    )
    pd.DataFrame(main_rows)[
        [
            "family",
            "alignment",
            "h_step",
            "best_s_step_on_dev",
            "delta_margin_test",
            "delta_margin_ci_lo",
            "delta_margin_ci_hi",
            "delta_accuracy_test",
            "delta_accuracy_ci_lo",
            "delta_accuracy_ci_hi",
        ]
    ].to_csv(args.out_dir / "paired_bootstrap_ci.csv", index=False)
    pd.DataFrame(preselect_rows).to_csv(
        args.out_dir / "preselected_readout_comparison.csv", index=False
    )
    print(
        f"[done] wrote {len(main_rows)} group summaries to {args.out_dir}", flush=True
    )


if __name__ == "__main__":
    main()
