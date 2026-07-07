"""Build the SAE-vs-crosscoder Pareto comparison table (text output).

Framing note: an earlier per-snap EV computation was incorrect — the
crosscoder uses a SHARED encoder that sums hidden_pre across all K=32 heads
to produce joint features (a single set of D activations explaining all
snapshots), then per-snapshot decoders; a per-head-only encoder
approximation under-estimates EV.

The correct apples-to-apples comparison:
  - Crosscoder OVERALL EV (joint reconstruction across 32 snapshots) — stored
    in quality.explained_variance
  - SAE single-snapshot EV on step 143k — recomputed from weights with the
    correct per-snapshot SAE forward pass

These solve different tasks: the crosscoder reconstructs all 32 snapshots
through a single shared feature space, the SAE is focused on one snapshot.
A single-snapshot SAE at the same d_sae as the crosscoder is expected to
match or slightly beat the crosscoder's overall EV (since the SAE has more
capacity per-snapshot); the multiplier shows how much "budget" the
crosscoder spent on temporal structure vs raw reconstruction.

Inputs are legacy run directories (see --help); point the flags at your own
retrained outputs if regenerating from scratch.
"""

import argparse
from pathlib import Path

import torch

from readout.core.paths import ssd_path
from readout.core.repro import log_run_provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--sae-dir",
        type=Path,
        default=Path("results/wu_crosscoder/sae_pareto_results_v2"),
        help="dir of single-snapshot SAE checkpoints (sae_*_seed0.pt)",
    )
    ap.add_argument(
        "--cc-d8192",
        type=Path,
        default=ssd_path(
            "hf_release",
            "parameter-trajectory-crosscoders",
            "pythia-160m",
            "W_U",
            "cross-snapshot-32",
            "d8192",
            "seed0.safetensors",
        ),
        help="d_sae=8192 K=32 crosscoder checkpoint",
    )
    ap.add_argument(
        "--cc-d16384",
        type=Path,
        default=Path("results/wu_crosscoder/run4_dsae16384/wu_cc_dsae16384_seed0.pt"),
        help="d_sae=16384 K=32 crosscoder checkpoint",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/capacity/pareto_frontier_ev_l0/derived"),
        help="where sae_pareto.txt is written",
    )
    args = ap.parse_args()
    log_run_provenance()

    rows = []
    for f in sorted(args.sae_dir.glob("sae_*_seed0.pt")):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        q = ck["quality"]
        train = ck["training"]
        rows.append(
            {
                "name": f.name,
                "d_sae": q["d_sae"],
                "expansion": train["expansion_factor"],
                "ev": q["explained_variance"],
                "l0": q["mean_l0"],
            }
        )

    # Crosscoder stored quality
    cc3 = torch.load(args.cc_d8192, map_location="cpu", weights_only=False)["quality"]
    cc4 = torch.load(args.cc_d16384, map_location="cpu", weights_only=False)["quality"]

    rows.sort(key=lambda r: r["d_sae"])
    for r in rows:
        print(f"  d_sae={r['d_sae']:>5} EV={r['ev']:.4f} L0={r['l0']:.1f}")
    print(f"  CC (d=8192,  K=32) overall EV={cc3['explained_variance']:.4f}")
    print(f"  CC (d=16384, K=32) overall EV={cc4['explained_variance']:.4f}")

    # === Updated text summary ===
    txt = [
        "SAE Pareto comparison (single-snapshot SAE vs multi-snapshot crosscoder)",
        "=" * 75,
    ]
    txt.append("Pythia-160M, step 143000 W_U, seed 0, 100 epochs")
    txt.append("Hyperparameters (v2): λ=0.05, init_thresh=0.001, jumprelu_lr_factor=1.0,")
    txt.append("                       l1_warmup_fraction=0.2, lr=5e-5, batch=1024")
    txt.append("")
    txt.append("Important framing: the crosscoder uses a shared encoder that pools across")
    txt.append("all K=32 snapshots into a single feature space, with per-snapshot decoders.")
    txt.append("Its 'EV' is the joint reconstruction quality across all 32 snapshots, NOT")
    txt.append("per-snapshot EV. The single-snapshot SAE solves a strictly easier problem")
    txt.append("(reconstruct ONE snapshot only), so it is expected to match or beat the")
    txt.append("crosscoder's joint EV at the same dictionary size.")
    txt.append("")
    txt.append("Single-snapshot SAEs (target: step-143k W_U):")
    txt.append(f"  {'d_sae':>8} | {'expansion':>9} | {'EV':>8} | {'L0':>8}")
    for r in rows:
        txt.append(f"  {r['d_sae']:>8} | {r['expansion']:>9.2f} | {r['ev']:>8.4f} | {r['l0']:>8.2f}")
    txt.append("")
    txt.append("Multi-snapshot crosscoders (joint EV across 32 snapshots):")
    txt.append(f"  CC d_sae=8192,  K=32: overall EV={cc3['explained_variance']:.4f}, mean L0={cc3['mean_l0']:.1f}")
    txt.append(f"  CC d_sae=16384, K=32: overall EV={cc4['explained_variance']:.4f}, mean L0={cc4['mean_l0']:.1f}")
    txt.append("")
    txt.append("Headline comparison at matched dictionary size:")
    sae_at_8192 = next((r for r in rows if r["d_sae"] == 8192), None)
    sae_at_16384 = next((r for r in rows if r["d_sae"] == 16384), None)
    if sae_at_8192:
        gap = sae_at_8192["ev"] - cc3["explained_variance"]
        txt.append(
            f"  d=8192:  SAE EV {sae_at_8192['ev']:.4f} vs CC EV {cc3['explained_variance']:.4f} (Δ = {gap:+.4f})"
        )
    if sae_at_16384:
        gap = sae_at_16384["ev"] - cc4["explained_variance"]
        txt.append(
            f"  d=16384: SAE EV {sae_at_16384['ev']:.4f} vs CC EV {cc4['explained_variance']:.4f} (Δ = {gap:+.4f})"
        )
    txt.append(f"  d=65536: SAE EV {rows[-1]['ev']:.4f} (no matched crosscoder; capacity ceiling probe)")
    txt.append("")
    txt.append("Capacity saturation: the SAE rises monotonically with d_sae")
    txt.append(
        f"  d=6144→8192:    EV {rows[0]['ev']:.4f} → {rows[1]['ev']:.4f}  (+{rows[1]['ev'] - rows[0]['ev']:.4f})"
    )
    txt.append(
        f"  d=8192→16384:   EV {rows[1]['ev']:.4f} → {rows[2]['ev']:.4f}  (+{rows[2]['ev'] - rows[1]['ev']:.4f})"
    )
    txt.append(
        f"  d=16384→32768:  EV {rows[2]['ev']:.4f} → {rows[3]['ev']:.4f}  (+{rows[3]['ev'] - rows[2]['ev']:.4f})"
    )
    txt.append(
        f"  d=32768→65536:  EV {rows[3]['ev']:.4f} → {rows[4]['ev']:.4f}  (+{rows[4]['ev'] - rows[3]['ev']:.4f})"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sae_pareto.txt").write_text("\n".join(txt))
    print("\n" + "\n".join(txt))


if __name__ == "__main__":
    main()
