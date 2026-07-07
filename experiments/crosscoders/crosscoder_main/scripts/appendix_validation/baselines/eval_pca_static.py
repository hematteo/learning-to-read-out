"""Per-snapshot truncated-SVD baseline for `W_U`.

For each (model, snapshot) we compute the singular spectrum of the
mean-centered weight matrix and report explained variance at a fixed
ladder of requested ranks. Rank is clamped to `d_model`; the realized
value is recorded as `effective_rank`.

The metric matches what the cross-snap crosscoders are evaluated on
(EV in the center-scale-normalized regime — scale is a scalar so PCA
EV is invariant to it). This makes the resulting EV(rank) curves
directly comparable to the dictionary EV in `quality_inventory.csv`.

Output:
  - figures/crosscoder_main/appendix_validation/pca_static.csv
  - figures/crosscoder_main/appendix_validation/pca_static.pt
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parent))  # this dir for `baseline_utils`

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from baseline_utils import (
    DEFAULT_SNAP_DIR,
    FIG_DIR,
    RAW_DIR,
    SPECS,
    auto_steps_for,
    device,
    ensure_dirs,
    iter_snapshots,
)

from readout.core.repro import git_commit, log_run_provenance

DEFAULT_RANKS = [16, 32, 64, 128, 256, 512, 1024]


def per_snapshot_singular_spectrum(W: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """Return singular values of (W - mean(W, dim=0)) on CPU, fp64.

    Note: torch.linalg.svdvals is not implemented on MPS, so we run SVD on
    CPU (or CUDA when available) regardless of `dev`. CPU SVD on the
    50k x d_model matrices we care about takes a few seconds even at 6.9B.
    `dev` is accepted for interface symmetry with the trajectory baselines.
    """
    _ = dev  # SVD path is CPU/CUDA only
    Wc = W - W.mean(dim=0, keepdim=True)
    if torch.cuda.is_available():
        Wc = Wc.cuda()
    S = torch.linalg.svdvals(Wc).double().cpu()
    del Wc
    return S


def ev_at_ranks(S: torch.Tensor, ranks: list[int]) -> list[tuple[int, int, float]]:
    """Return (requested_rank, effective_rank, ev) for each rank in `ranks`."""
    total = (S * S).sum().item()
    out = []
    for r in ranks:
        eff = min(r, S.numel())
        captured = (S[:eff] * S[:eff]).sum().item()
        ev = captured / total if total > 0 else 0.0
        out.append((r, eff, ev))
    return out


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
    ap.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run only steps 0, 1000, 143000 for fast verification.",
    )
    ap.add_argument("--cpu", action="store_true", help="Disable MPS / CUDA.")
    ap.add_argument("--out-stem", type=str, default="pca_static")
    args = ap.parse_args()

    dev = torch.device("cpu") if args.cpu else device()
    ensure_dirs()
    (FIG_DIR / f"{args.out_stem}.provenance.json").write_text(
        json.dumps({**log_run_provenance(), "git_commit": git_commit()}, indent=2)
    )

    rows = []
    spectra: dict[tuple[str, int], torch.Tensor] = {}

    for model_label in args.models:
        spec = SPECS[model_label]
        if args.smoke:
            steps = [0, 1000, 143000]
        elif args.steps is not None:
            steps = args.steps
        else:
            steps = auto_steps_for(spec, snap_dir=args.snap_dir)
        print(f"[{spec.label}] d_model={spec.d_model} V={spec.vocab} steps={len(steps)} dev={dev}")
        t_model = time.time()
        for step, W in iter_snapshots(spec, steps=steps, snap_dir=args.snap_dir):
            t0 = time.time()
            S = per_snapshot_singular_spectrum(W, dev)
            spectra[(spec.label, step)] = S
            for req, eff, ev in ev_at_ranks(S, args.ranks):
                rows.append(
                    {
                        "model": spec.label,
                        "step": step,
                        "d_model": spec.d_model,
                        "vocab": spec.vocab,
                        "requested_rank": req,
                        "effective_rank": eff,
                        "ev": ev,
                    }
                )
            print(f"  step {step:>6d}: SVD in {time.time() - t0:5.1f}s S[:5]={[round(x, 3) for x in S[:5].tolist()]}")
        print(f"[{spec.label}] done in {time.time() - t_model:.1f}s")

    csv_path = FIG_DIR / f"{args.out_stem}.csv"
    pt_path = FIG_DIR / f"{args.out_stem}.pt"
    raw_pt = RAW_DIR / f"{args.out_stem}_spectra.pt"

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    torch.save({"rows": rows, "ranks": args.ranks}, pt_path)
    torch.save({(k[0], int(k[1])): v for k, v in spectra.items()}, raw_pt)

    print(f"wrote {csv_path}  ({len(rows)} rows)")
    print(f"wrote {pt_path}")
    print(f"wrote {raw_pt}  (full singular spectra cache)")


if __name__ == "__main__":
    main()
