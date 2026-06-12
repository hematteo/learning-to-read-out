"""Concatenated-trajectory PCA baseline via streaming randomized SVD.

Treats the trajectory matrix
    A = [W_1 - mu_1 | W_2 - mu_2 | ... | W_K - mu_K]
of shape (V, K*d_model) as a single low-rank target and computes the
top-r singular spectrum without ever materializing A in memory. Uses
Halko randomized SVD with one optional power iteration; one block of A
(one snapshot) is loaded at a time.

EV at rank r is reported against the full Frobenius energy
    F2 = sum_k ||W_k - mu_k||_F^2,
giving a baseline directly comparable to the cross-snap dictionary's
total-trajectory EV.

Output:
  - figures/run5/a_instrument_validation/concat_pca.csv
  - figures/run5/a_instrument_validation/concat_pca.pt
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
)


def streaming_randomized_svd(
    spec,
    steps: list[int],
    snap_dir: Path,
    n_components: int,
    oversample: int,
    n_power_iters: int,
    seed: int,
) -> dict:
    """Halko streaming randomized SVD on the per-snap-centered concat matrix.

    Returns dict with keys:
      - singular_values: torch.Tensor (m,) fp64 (m = n_components + oversample)
      - frob_sq: float (sum of squared centered Frobenius norms)
      - vocab, d_model, K, n_components, oversample
    """
    K = len(steps)
    d = spec.d_model
    m = n_components + oversample

    g = torch.Generator(device="cpu").manual_seed(seed)
    P = torch.randn(K * d, m, generator=g, dtype=torch.float32)  # (Kd, m)

    # Pass 1: V is set on first snapshot; compute Y = A @ P streamingly.
    V = None
    Y = None
    F2 = 0.0
    snap_cache_means: list[torch.Tensor] = []

    for k, (step, Wt) in enumerate(
        iter_snapshots(spec, steps=steps, snap_dir=snap_dir)
    ):
        if V is None:
            V = Wt.shape[0]
            Y = torch.zeros(V, m, dtype=torch.float32)
        mu = Wt.mean(dim=0, keepdim=True)  # (1, d)
        Wc = Wt - mu  # (V, d)
        F2 += float((Wc * Wc).sum().item())
        Y += Wc @ P[k * d : (k + 1) * d, :]  # (V, m)
        snap_cache_means.append(mu.squeeze(0).clone())

    # Optional power iteration (LU re-orth between sweeps for stability).
    for _ in range(n_power_iters):
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        Z = torch.zeros(K * d, m, dtype=torch.float32)
        for k, (step, Wt) in enumerate(
            iter_snapshots(spec, steps=steps, snap_dir=snap_dir)
        ):
            Wc = Wt - snap_cache_means[k]
            Z[k * d : (k + 1) * d, :] = Wc.T @ Q
        Y = torch.zeros(V, m, dtype=torch.float32)
        for k, (step, Wt) in enumerate(
            iter_snapshots(spec, steps=steps, snap_dir=snap_dir)
        ):
            Wc = Wt - snap_cache_means[k]
            Y += Wc @ Z[k * d : (k + 1) * d, :]

    Q, _ = torch.linalg.qr(Y, mode="reduced")  # (V, m)

    # Pass 2: B = Q^T A, streamed.
    B = torch.zeros(m, K * d, dtype=torch.float32)
    for k, (step, Wt) in enumerate(
        iter_snapshots(spec, steps=steps, snap_dir=snap_dir)
    ):
        Wc = Wt - snap_cache_means[k]
        B[:, k * d : (k + 1) * d] = Q.T @ Wc

    S = torch.linalg.svdvals(B).double().cpu()  # (m,)

    return {
        "singular_values": S,
        "frob_sq": F2,
        "vocab": V,
        "d_model": d,
        "K": K,
        "n_components": n_components,
        "oversample": oversample,
    }


def ev_at_ranks(out: dict, ranks: list[int]) -> list[tuple[int, int, float]]:
    S = out["singular_values"]
    F2 = out["frob_sq"]
    rows = []
    for r in ranks:
        eff = min(r, S.numel())
        captured = float((S[:eff] * S[:eff]).sum().item())
        ev = captured / F2 if F2 > 0 else 0.0
        rows.append((r, eff, ev))
    return rows


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
    ap.add_argument("--ranks", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    ap.add_argument(
        "--n-components",
        type=int,
        default=256,
        help="Top-r ladder; sketch dim is n_components + oversample.",
    )
    ap.add_argument("--oversample", type=int, default=64)
    ap.add_argument("--power-iters", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Use only steps {0, 1000, 143000} (K=3) for fast verification.",
    )
    ap.add_argument("--out-stem", type=str, default="concat_pca")
    args = ap.parse_args()

    ranks = [r for r in args.ranks if r <= args.n_components]
    ensure_dirs()

    rows = []
    raw = {}
    for model_label in args.models:
        spec = SPECS[model_label]
        if args.smoke:
            steps = [0, 1000, 143000]
        elif args.steps is not None:
            steps = args.steps
        else:
            steps = auto_steps_for(spec, snap_dir=args.snap_dir)
        K_d = len(steps) * spec.d_model
        sketch_mb = (K_d * (args.n_components + args.oversample) * 4) / 1024 / 1024
        print(
            f"[{spec.label}] V={spec.vocab} d={spec.d_model} K={len(steps)} "
            f"n_comp={args.n_components} +os={args.oversample} pi={args.power_iters} "
            f"sketch={sketch_mb:.1f}MB"
        )
        t0 = time.time()
        out = streaming_randomized_svd(
            spec,
            steps,
            args.snap_dir,
            n_components=args.n_components,
            oversample=args.oversample,
            n_power_iters=args.power_iters,
            seed=args.seed,
        )
        for req, eff, ev in ev_at_ranks(out, ranks):
            rows.append(
                {
                    "model": spec.label,
                    "K": len(steps),
                    "d_model": spec.d_model,
                    "vocab": spec.vocab,
                    "n_components": args.n_components,
                    "oversample": args.oversample,
                    "power_iters": args.power_iters,
                    "requested_rank": req,
                    "effective_rank": eff,
                    "ev": ev,
                    "frob_sq": out["frob_sq"],
                }
            )
        raw[spec.label] = {
            "singular_values": out["singular_values"],
            "frob_sq": out["frob_sq"],
            "steps": steps,
        }
        print(
            f"  S[:5]={[round(x, 3) for x in out['singular_values'][:5].tolist()]} "
            f"F2={out['frob_sq']:.3e}"
        )
        for req, eff, ev in ev_at_ranks(out, ranks):
            print(f"  rank {req:>4d} (eff {eff:>4d}): EV={ev:.4f}")
        print(f"[{spec.label}] done in {time.time() - t0:.1f}s")

    csv_path = FIG_DIR / f"{args.out_stem}.csv"
    pt_path = FIG_DIR / f"{args.out_stem}.pt"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    torch.save(raw, pt_path)
    print(f"wrote {csv_path}  ({len(rows)} rows)")
    print(f"wrote {pt_path}")


if __name__ == "__main__":
    main()
