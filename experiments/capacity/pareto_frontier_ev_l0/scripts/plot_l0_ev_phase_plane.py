"""Plot #2: L0 vs EV phase-plane scatter across crosscoder families.

For each checkpoint we compute (L0_per_snap, EV_per_snap) at each of K
snapshots using the canonical joint forward pass. EV is per-snapshot mean-centered:
    ev[i] = 1 - resid.var() / x_dev.var()
L0 is
    l0[i] = (joint_pre > thr / dec_norm[i]).sum(-1).mean()

We connect Run-3 seed-0's 32 snapshots chronologically with a thin gray line
to show the consolidation as a "phase transition" near step 1000.

Mac CPU only. Drops checkpoint weights with `del` between checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402

from readout.core.paths import ssd_path
from readout.core.repro import log_run_provenance
from readout.crosscoder.extract_rates import compute_rates_canonical  # noqa: E402,F401

DEVICE = "cpu"
MODEL_SLUG = "EleutherAI_pythia-160m"


def per_snap_l0_ev(ckpt_path: Path, snap_dir: Path, label: str) -> dict:
    """Compute per-snapshot (L0, EV) using the canonical joint forward pass.

    Memory-safe: streams W_U snapshots (one at a time) into the joint pre-act
    accumulator. Only the (V, D) joint_pre tensor + the current snapshot is
    held in RAM, never the full (K, V, d) staged tensor. For d=24576 the
    inner per-snap loop also reuses preallocated buffers.
    """
    print(f"  loading {ckpt_path.name} ...", flush=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    steps = list(ck["steps"])
    W_E = sd["W_E"].float()  # (K, d, D)
    W_D = sd["W_D"].float()  # (K, D, d)
    b_E = sd["b_E"].float()  # (K, D)
    b_D = sd.get("b_D")
    if b_D is not None:
        b_D = b_D.float()
    thr = sd["activation_function.log_jumprelu_threshold"].float().exp()  # (D,)
    stats = ck["preprocess_stats"]
    K, d, D = W_E.shape

    print(f"    {label}: K={K}, d={d}, d_sae={D}", flush=True)

    # Joint pre-activation, accumulated by streaming snapshots one-by-one.
    # Memory: (V, D) float32 — same regardless of K.
    pre = None  # type: torch.Tensor | None
    for h in range(K):
        x_h = torch.load(snap_dir / f"{MODEL_SLUG}_step{steps[h]}_wu.pt", map_location="cpu").float()
        if stats is not None:
            x_h = (x_h - stats["mean"][h].squeeze(0)) / stats["scale"][h].squeeze()
        head_pre = x_h @ W_E[h] + b_E[h]
        if pre is None:
            pre = head_pre.contiguous()
        else:
            pre.add_(head_pre)
        del x_h, head_pre

    dec_norm = W_D.norm(dim=-1).clamp_min(1e-12)  # (K, D)

    # Per-snap (L0, EV). For each k:
    #   gate_k[v, j] = (pre[v, j] * dec_norm[k, j]) > thr[j]
    #   acts_k       = gate_k * pre  (scaled by dec_norm * 1/dec_norm cancels)
    #   recon_k      = acts_k @ W_D[k] + b_D[k]
    # Need the per-snap z-scored input for EV, so reload snapshots a second
    # time (keeps memory low — only one snap's input held at any moment).
    evs = np.zeros(K, dtype=np.float32)
    l0s = np.zeros(K, dtype=np.float32)
    for k in range(K):
        scaled = pre * dec_norm[k]  # (V, D)
        gate = scaled > thr  # (V, D) bool
        acts = torch.where(gate, scaled, torch.zeros_like(scaled)) / dec_norm[k]
        recon = acts @ W_D[k]
        if b_D is not None:
            recon = recon + b_D[k]

        x_k = torch.load(snap_dir / f"{MODEL_SLUG}_step{steps[k]}_wu.pt", map_location="cpu").float()
        if stats is not None:
            x_k = (x_k - stats["mean"][k].squeeze(0)) / stats["scale"][k].squeeze()

        resid = x_k - recon
        evs[k] = float(1.0 - resid.var() / x_k.var())
        l0s[k] = float(gate.float().sum(dim=-1).mean())
        del scaled, gate, acts, recon, resid, x_k

    del ck, sd, W_E, W_D, b_E, thr, stats, pre, dec_norm
    import gc

    gc.collect()

    return {"steps": steps, "ev": evs, "l0": l0s, "label": label, "K": K, "D": D}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--release-root",
        type=Path,
        default=ssd_path(),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/pareto_frontier_ev_l0"),
    )
    args = ap.parse_args()
    log_run_provenance()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = args.release_root / "snapshots"
    R3 = args.release_root / "derived" / "rates" / "wu-d8192-multiseed"
    R4 = args.release_root / "run4_dsae16384"
    T46 = args.release_root / "derived" / "rates" / "wu-d24576-multiseed"
    T44 = args.release_root / "derived" / "rates" / "wu-d8192-cs16"

    families: list[tuple[str, Path, str, str]] = []
    for s in range(5):
        families.append((f"Run3 seed{s} (d=8192)", R3 / f"wu_cc_ge_exact_seed{s}.pt", "run3", "o"))
    families.append(("Run4 seed0 (d=16384)", R4 / "wu_cc_dsae16384_seed0.pt", "run4", "^"))
    for s in range(3):
        families.append(
            (
                f"T4.6 seed{s} (d=24576)",
                T46 / f"wu_cc_dsae24576_seed{s}.pt",
                "t46",
                "D",
            )
        )
    families.append(("T4.4 16-snap (d=8192)", T44 / "wu_cc_dsae8192_seed0.pt", "t44", "*"))

    cache_path = args.out_dir / "l0_ev_phase_plane_cache.pt"
    csv_path = args.out_dir / "l0_ev_phase_plane.csv"
    # Resume-friendly: load any prior cache, then compute only missing labels.
    if cache_path.exists():
        print(f"loading existing cache from {cache_path}", flush=True)
        results = torch.load(cache_path, weights_only=False)
        print(f"  {len(results)} checkpoints already cached", flush=True)
    else:
        results = []
    cached_labels = {r["label"] for r in results}
    missing = [(lbl, p, f, m) for (lbl, p, f, m) in families if lbl not in cached_labels]
    if missing:
        print(f"computing {len(missing)} missing checkpoint(s):", flush=True)
        for label, _, _, _ in missing:
            print(f"  - {label}", flush=True)
        for label, path, family, marker in missing:
            if not path.exists():
                print(f"WARNING missing {path}; skipping", flush=True)
                continue
            res = per_snap_l0_ev(path, snap_dir, label)
            res["family"] = family
            res["marker"] = marker
            results.append(res)
            # Persist incrementally so a mid-run crash doesn't lose earlier work.
            torch.save(results, cache_path)
        print(f"cached {len(results)} (L0, EV) arrays to {cache_path}", flush=True)
    else:
        print(
            f"all {len(results)} checkpoints already cached; skipping compute",
            flush=True,
        )

    # Always write/refresh the CSV from the in-memory results — simple wide
    # format with one row per (label, snapshot index).
    import csv as _csv

    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["label", "family", "d_sae", "snap_idx", "step", "L0", "EV"])
        for r in results:
            for i, step in enumerate(r["steps"]):
                w.writerow(
                    [
                        r["label"],
                        r["family"],
                        r["D"],
                        i,
                        int(step),
                        f"{float(r['l0'][i]):.6f}",
                        f"{float(r['ev'][i]):.6f}",
                    ]
                )
    print(f"wrote per-snapshot (L0, EV) table to {csv_path}", flush=True)

    if not results:
        raise RuntimeError("No checkpoints loaded.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
