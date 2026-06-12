"""Endpoint-linear-interpolation baseline for `W_U` trajectories.

For each snapshot t we fit a per-row scalar alpha_v(t) that places
W_U^(t)[v,:] on the line through W_U^(t_1)[v,:] and W_U^(t_K)[v,:]:

    recon_v(t) = W_1[v] + alpha_v(t) (W_K[v] - W_1[v])
    alpha_v(t) = <W_t[v]-W_1[v], W_K[v]-W_1[v]> / ||W_K[v]-W_1[v]||^2

This is the strongest "no-structure-beyond-endpoint-drift" baseline:
each row chooses the best point on its own endpoint line at every
checkpoint. If the per-checkpoint EV under this baseline is high, the
crosscoder isn't gaining much from cross-snapshot feature sharing
beyond endpoint linear interpolation.

Output:
  - figures/run5/a_instrument_validation/endpoint_linear.csv
  - figures/run5/a_instrument_validation/endpoint_linear.pt
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[6]))  # repo root for `src.*`
_sys.path.insert(0, str(_P(__file__).resolve().parent))  # this dir for `baseline_utils`

import argparse
import csv
import time
from pathlib import Path

import torch
from baseline_utils import (
    DEFAULT_SNAP_DIR,
    FIG_DIR,
    SPECS,
    auto_steps_for,
    ensure_dirs,
    iter_snapshots,
    load_snapshot,
    snap_path_for,
)


def evaluate_endpoint_linear(
    spec,
    steps: list[int],
    snap_dir: Path,
) -> tuple[list[dict], dict]:
    """Return (per-snapshot rows, alpha_matrix dict)."""
    t1, tK = steps[0], steps[-1]
    W1 = load_snapshot(snap_path_for(spec.name, t1, snap_dir=snap_dir))  # (V, d)
    WK = load_snapshot(snap_path_for(spec.name, tK, snap_dir=snap_dir))  # (V, d)
    delta = WK - W1  # (V, d)
    denom = (delta * delta).sum(dim=1).clamp_min(1e-12)  # (V,)
    rows = []
    alpha_matrix = torch.zeros(len(steps), W1.shape[0], dtype=torch.float32)

    for i, (step, Wt) in enumerate(
        iter_snapshots(spec, steps=steps, snap_dir=snap_dir)
    ):
        diff_t = Wt - W1
        alpha = ((diff_t * delta).sum(dim=1) / denom).float()  # (V,)
        recon = W1 + alpha.unsqueeze(1) * delta  # (V, d)
        resid = Wt - recon

        # EV in centered (PCA-style) regime, matching the crosscoder convention.
        Wt_centered = Wt - Wt.mean(dim=0, keepdim=True)
        ss_res = (resid * resid).sum().item()
        ss_tot = (Wt_centered * Wt_centered).sum().item()
        ev = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Per-row residual norms for downstream stratification.
        row_resid_sq = (resid * resid).sum(dim=1)
        row_total_sq = (Wt_centered * Wt_centered).sum(dim=1).clamp_min(1e-12)
        ev_row = 1.0 - row_resid_sq / row_total_sq

        rows.append(
            {
                "model": spec.label,
                "step": step,
                "step_index": i,
                "ev_centered": ev,
                "alpha_mean": float(alpha.mean()),
                "alpha_std": float(alpha.std()),
                "alpha_p10": float(alpha.quantile(0.10)),
                "alpha_p50": float(alpha.quantile(0.50)),
                "alpha_p90": float(alpha.quantile(0.90)),
                "ev_row_mean": float(ev_row.mean()),
                "ev_row_p10": float(ev_row.quantile(0.10)),
                "ev_row_p50": float(ev_row.quantile(0.50)),
                "ev_row_p90": float(ev_row.quantile(0.90)),
            }
        )
        alpha_matrix[i] = alpha
    return rows, {"alpha": alpha_matrix, "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["pythia-160m"])
    ap.add_argument("--snap-dir", type=Path, default=DEFAULT_SNAP_DIR)
    ap.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Step list. If omitted, auto-discovers from snap_dir per model.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run only steps 0, 1000, 143000 for fast verification.",
    )
    ap.add_argument("--out-stem", type=str, default="endpoint_linear")
    args = ap.parse_args()

    ensure_dirs()

    all_rows = []
    raw = {}
    for model_label in args.models:
        spec = SPECS[model_label]
        if args.smoke:
            steps = [0, 1000, 143000]
        elif args.steps is not None:
            steps = args.steps
        else:
            steps = auto_steps_for(spec, snap_dir=args.snap_dir)
        print(
            f"[{spec.label}] d_model={spec.d_model} V={spec.vocab} steps={len(steps)}"
        )
        t0 = time.time()
        rows, cache = evaluate_endpoint_linear(spec, steps, args.snap_dir)
        all_rows.extend(rows)
        raw[spec.label] = cache
        for r in rows:
            print(
                f"  step {r['step']:>6d}  ev={r['ev_centered']:.4f}  "
                f"α mean={r['alpha_mean']:+.3f} (p10/p50/p90 = "
                f"{r['alpha_p10']:+.2f}/{r['alpha_p50']:+.2f}/{r['alpha_p90']:+.2f})"
            )
        print(f"[{spec.label}] done in {time.time() - t0:.1f}s")

    csv_path = FIG_DIR / f"{args.out_stem}.csv"
    pt_path = FIG_DIR / f"{args.out_stem}.pt"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    torch.save(raw, pt_path)
    print(f"wrote {csv_path}  ({len(all_rows)} rows)")
    print(f"wrote {pt_path}")


if __name__ == "__main__":
    main()
