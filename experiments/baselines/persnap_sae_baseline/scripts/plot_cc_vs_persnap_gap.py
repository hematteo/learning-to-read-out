"""Plot #11: Crosscoder vs per-snap-SAE delta-EV.

For each snapshot index i (training step `steps[i]`):
  crosscoder_EV[i] = (1 - resid.var() / x_dev.var())  using joint encode + decode
                     head i of the Run-3 seed-0 crosscoder
  per_snap_EV[i]   = same formula on snapshot i, using the SAE trained ONLY on
                     snapshot i (single-head, K=1).

We plot:
  Top:    per-snap SAE EV trajectory + crosscoder EV trajectory vs step.
  Bottom: gap = (crosscoder_EV - per_snap_EV), marking step 1000.

Per-snap SAEs are loaded sequentially and dropped immediately. Mac CPU only.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[4]))
import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.core.paths import ssd_path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from src.crosscoder.extract_rates import compute_rates_canonical  # noqa: E402,F401

DEVICE = "cpu"
MODEL_SLUG = "EleutherAI_pythia-160m"


def _load_zscored(snap_dir: Path, slug: str, step: int, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x = torch.load(snap_dir / f"{slug}_step{step}_wu.pt", map_location="cpu").float()
    return (x - mean) / scale


def crosscoder_per_snap_ev(ckpt_path: Path, snap_dir: Path) -> dict:
    """Per-snapshot EV under the joint canonical forward pass."""
    print(f"  loading crosscoder {ckpt_path.name} ...", flush=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    steps = list(ck["steps"])
    W_E = sd["W_E"]
    W_D = sd["W_D"]
    b_E = sd["b_E"]
    b_D = sd.get("b_D", None)
    thr = sd["activation_function.log_jumprelu_threshold"].exp()
    stats = ck["preprocess_stats"]
    K, d, D = W_E.shape
    print(f"    crosscoder: K={K}, d={d}, d_sae={D}", flush=True)

    # Stage all K z-scored snapshots once.
    first = torch.load(snap_dir / f"{MODEL_SLUG}_step{steps[0]}_wu.pt", map_location="cpu").float()
    x_all = torch.empty(K, *first.shape, dtype=torch.float32)
    x_all[0] = (first - stats["mean"][0].squeeze(0)) / stats["scale"][0].squeeze()
    del first
    for i, step in enumerate(steps[1:], start=1):
        x_i = torch.load(snap_dir / f"{MODEL_SLUG}_step{step}_wu.pt", map_location="cpu").float()
        x_all[i] = (x_i - stats["mean"][i].squeeze(0)) / stats["scale"][i].squeeze()
        del x_i
    V = x_all.shape[1]

    # Joint pre-activation.
    pre = torch.zeros(V, D, dtype=torch.float32)
    for h in range(K):
        pre += x_all[h] @ W_E[h]
    pre += b_E.sum(dim=0)

    dec_norm = W_D.norm(dim=-1).clamp_min(1e-12)  # (K, D)
    evs = np.zeros(K, dtype=np.float32)
    for k in range(K):
        scaled = pre * dec_norm[k]
        gate = scaled > thr
        acts = torch.where(gate, scaled, torch.zeros_like(scaled)) / dec_norm[k]
        recon = acts @ W_D[k]
        if b_D is not None:
            recon = recon + b_D[k]
        x_dev = x_all[k]
        resid = x_dev - recon
        evs[k] = float(1.0 - resid.var() / x_dev.var())
        del scaled, gate, acts, recon, resid

    del ck, sd, W_E, W_D, b_E, b_D, thr, stats, pre, dec_norm, x_all
    return {"steps": steps, "ev": evs}


def per_snap_sae_ev(per_snap_dir: Path, snap_dir: Path, steps: list[int]) -> np.ndarray:
    """Per-step EV for the K=1 single-snapshot SAEs.

    Each SAE is loaded sequentially and dropped to keep RAM low.
    """
    out = np.full(len(steps), np.nan, dtype=np.float32)
    for i, step in enumerate(steps):
        path = per_snap_dir / f"wu_sae_dsae8192_step{step}.pt"
        if not path.exists():
            print(f"    WARNING: missing {path}; skipping step {step}", flush=True)
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        pp = ck["preprocess_stats"]
        W_E = sd["W_E"]  # (1, d, D)
        W_D = sd["W_D"]  # (1, D, d)
        b_E = sd["b_E"]  # (1, D)
        b_D = sd.get("b_D", None)
        thr = sd["activation_function.log_jumprelu_threshold"].exp()  # (D,)

        mean = pp["mean"].squeeze()  # (d,)
        scale = pp["scale"].squeeze()  # scalar
        x_raw = torch.load(snap_dir / f"{MODEL_SLUG}_step{step}_wu.pt", map_location="cpu").float()
        x = (x_raw - mean) / scale  # (V, d)
        del x_raw

        pre = x @ W_E[0] + b_E[0]  # (V, D)
        dec_norm = W_D[0].norm(dim=-1).clamp_min(1e-12)  # (D,)
        scaled = pre * dec_norm
        gate = scaled > thr
        acts = torch.where(gate, scaled, torch.zeros_like(scaled)) / dec_norm
        recon = acts @ W_D[0]
        if b_D is not None:
            recon = recon + b_D[0]
        resid = x - recon
        ev = float(1.0 - resid.var() / x.var())
        out[i] = ev
        l0 = float(gate.float().sum(dim=-1).mean())
        print(
            f"    step {step:>7}: per-snap SAE EV={ev:.4f} L0={l0:.1f}",
            flush=True,
        )
        del ck, sd, pp, W_E, W_D, b_E, b_D, thr, x, pre, dec_norm, scaled
        del gate, acts, recon, resid
    return out


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
        default=Path("figures/persnap_sae_baseline"),
    )
    ap.add_argument(
        "--per-snap-job-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing wu_sae_dsae8192_step*.pt. Default = "
            "<release-root>/cluster_results/t1_3_per_snap_sae/20260428T230005Z_job28571218/"
        ),
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snap_dir = args.release_root / "snapshots"
    cc_path = (
        args.release_root / "hf_release" / "parameter-trajectory-crosscoders"
        / "pythia-160m" / "W_U" / "cross-snapshot-32" / "d8192" / "seed0.safetensors"
    )
    per_snap_dir = (
        args.per_snap_job_dir
        if args.per_snap_job_dir is not None
        else (
            args.release_root / "hf_release" / "parameter-trajectory-crosscoders"
            / "pythia-160m" / "W_U" / "per-snapshot-saes" / "d8192"
        )
    )

    print(f"Crosscoder: {cc_path}")
    print(f"Per-snap dir: {per_snap_dir}")

    cc = crosscoder_per_snap_ev(cc_path, snap_dir)
    steps = cc["steps"]
    cc_ev = cc["ev"]

    print("Computing per-snap SAE EV (sequential load) ...", flush=True)
    sae_ev = per_snap_sae_ev(per_snap_dir, snap_dir, steps)

    valid = ~np.isnan(sae_ev)
    gap = np.full_like(cc_ev, np.nan)
    gap[valid] = cc_ev[valid] - sae_ev[valid]

    # Summary text.
    txt = args.out_dir / "crosscoder_vs_persnap_gap.txt"
    with open(txt, "w") as f:
        f.write("Crosscoder vs per-snap SAE per-snapshot EV\n")
        f.write("step\tcc_EV\tpersnap_EV\tgap\n")
        for i, s in enumerate(steps):
            sae_str = f"{sae_ev[i]:.6f}" if not np.isnan(sae_ev[i]) else "NaN"
            gap_str = f"{gap[i]:.6f}" if not np.isnan(gap[i]) else "NaN"
            f.write(f"{s}\t{cc_ev[i]:.6f}\t{sae_str}\t{gap_str}\n")
        f.write("\n")
        f.write(f"mean cc EV   = {np.nanmean(cc_ev):.6f}\n")
        f.write(f"mean sae EV  = {np.nanmean(sae_ev):.6f}\n")
        f.write(f"mean gap     = {np.nanmean(gap):.6f}\n")
        f.write(f"max gap      = {np.nanmax(gap):.6f}\n")
        f.write(f"min gap      = {np.nanmin(gap):.6f}\n")
    print(f"Wrote {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
