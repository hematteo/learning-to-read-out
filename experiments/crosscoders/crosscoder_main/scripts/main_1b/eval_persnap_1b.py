"""Per-snapshot (FVU, L0, dead) for the 1B W_U cross-snap-32 crosscoders.

Memory-efficient: reads W_E[k]/W_D[k] one snapshot at a time from safetensors
instead of loading the whole (32, 2048, D) tensor into memory.

Inputs:
  ${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/
      W_U/cross-snapshot-32/d{D}/seed0.{safetensors,config.json}
  ${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step{S}_wu.pt

Outputs (one file per (d_sae, seed)):
  experiments/crosscoders/crosscoder_main/derived/main_1b/persnap_1b_d{D}_seed0.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from readout.core.paths import ssd_root

REPO = Path(__file__).resolve().parents[5]
SSD = ssd_root()
HF = SSD / "hf_release/parameter-trajectory-crosscoders"
SNAPS = SSD / "snapshots/pythia-1b"
OUT = REPO / "experiments/crosscoders/crosscoder_main/derived/main_1b"
OUT.mkdir(parents=True, exist_ok=True)


def load_snap(step: int) -> torch.Tensor:
    p = SNAPS / f"EleutherAI_pythia-1b_step{step}_wu.pt"
    return torch.load(p, map_location="cpu", weights_only=True).float()


def per_snap_preprocess_stats(steps):
    means, scales = [], []
    for s in steps:
        W = load_snap(s)
        mu = W.mean(dim=0, keepdim=True)
        sc = (W - mu).pow(2).mean().sqrt().view(1, 1)
        means.append(mu)
        scales.append(sc)
    return torch.stack(means), torch.stack(scales)


def eval_one(d_sae: int, seed: int = 0, device: str = "cpu", verbose: bool = True):
    sf = HF / f"pythia-1b/W_U/cross-snapshot-32/d{d_sae}/seed{seed}.safetensors"
    cfg = json.loads(sf.with_name(f"seed{seed}.config.json").read_text())
    steps = list(cfg["steps"])
    K = len(steps)

    with safe_open(sf, framework="pt") as f:
        # Load small tensors fully.
        log_thr = f.get_tensor("activation_function.log_jumprelu_threshold").to(device).float()
        thr = log_thr.exp()
        b_E = f.get_tensor("b_E").to(device).float()  # (K, D)
        b_D = f.get_tensor("b_D").to(device).float()  # (K, d)
        # Decoder norms: read W_D one snap at a time to compute per-(K,D) norms.
        W_D_slice = f.get_slice("W_D")
        dec_norm = torch.empty(K, d_sae, dtype=torch.float32, device=device)
        for k in range(K):
            wd_k = W_D_slice[k].to(device).float()
            dec_norm[k] = wd_k.pow(2).sum(dim=1).sqrt().clamp_min(1e-8)
            del wd_k

        # Preprocess stats (preprocess_stats not in safetensors; reconstruct).
        means, scales = per_snap_preprocess_stats(steps)
        means = means.to(device).squeeze(1).float()  # (K, d_model)
        scales = scales.to(device).squeeze(-1).squeeze(-1).float()  # (K,)

        # Phase 1: accumulate joint pre-activation.
        # joint_pre[v, j] = sum_k (x_k[v] @ W_E[k][:, j]) + sum_k b_E[k, j]
        W_E_slice = f.get_slice("W_E")
        V = None
        joint_pre = None
        for k, s in enumerate(steps):
            t0 = time.time()
            we_k = W_E_slice[k].to(device).float()  # (d_model, D)
            W_raw = load_snap(s).to(device)
            if V is None:
                V = W_raw.shape[0]
                joint_pre = torch.zeros(V, d_sae, device=device)
            x_k = (W_raw - means[k]) / scales[k]
            chunk = 4096
            for v0 in range(0, V, chunk):
                v1 = min(V, v0 + chunk)
                joint_pre[v0:v1] += x_k[v0:v1] @ we_k
            del we_k, W_raw, x_k
            if verbose:
                print(f"  phase1 k={k:2d} step={s} ({time.time() - t0:.1f}s)", flush=True)
        joint_pre = joint_pre + b_E.sum(dim=0)

        # Phase 2: per-snap recon quality, streaming W_D[k].
        rows = []
        for k, s in enumerate(steps):
            t0 = time.time()
            wd_k = W_D_slice[k].to(device).float()  # (D, d_model)
            W_raw = load_snap(s).to(device)
            x_k = (W_raw - means[k]) / scales[k]
            ss_tot = (x_k - x_k.mean(dim=0, keepdim=True)).pow(2).sum().item()

            scaled = joint_pre * dec_norm[k]
            acts_scaled = torch.where(scaled > thr, scaled, torch.zeros_like(scaled))
            acts_k = acts_scaled / dec_norm[k]  # (V, D)

            # recon = acts_k @ W_D[k] in V-chunks to bound memory.
            recon = torch.empty_like(x_k)
            chunk = 4096
            for v0 in range(0, V, chunk):
                v1 = min(V, v0 + chunk)
                recon[v0:v1] = acts_k[v0:v1] @ wd_k + b_D[k]
            ss_res = (x_k - recon).pow(2).sum().item()
            ev = 1.0 - ss_res / max(ss_tot, 1e-12)
            fvu = 1.0 - ev
            l0 = (acts_k > 0).float().sum(dim=1).mean().item()
            rate = (acts_k > 0).float().mean(dim=0)
            dead = (rate < 1e-4).float().mean().item()
            rows.append(
                {
                    "step": int(s),
                    "ev": ev,
                    "fvu": fvu,
                    "l0": l0,
                    "dead_rate": dead,
                }
            )
            del wd_k, W_raw, x_k, scaled, acts_scaled, acts_k, recon
            if verbose:
                print(
                    f"  phase2 k={k:2d} step={s} fvu={fvu:.4f} l0={l0:.1f} ({time.time() - t0:.1f}s)",
                    flush=True,
                )

    out_csv = OUT / f"persnap_1b_d{d_sae}_seed{seed}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "ev", "fvu", "l0", "dead_rate"])
        for r in rows:
            w.writerow([r["step"], r["ev"], r["fvu"], r["l0"], r["dead_rate"]])
    print(f"  → {out_csv}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--d-saes",
        type=int,
        nargs="+",
        default=[8192, 16384, 24576],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    for d in args.d_saes:
        out = OUT / f"persnap_1b_d{d}_seed{args.seed}.csv"
        if out.exists():
            print(f"[skip] d={d}: {out} already exists")
            continue
        print(f"[eval] 1B W_U cross-snap-32 d={d} seed={args.seed}")
        t0 = time.time()
        eval_one(d_sae=d, seed=args.seed, device=args.device)
        print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
