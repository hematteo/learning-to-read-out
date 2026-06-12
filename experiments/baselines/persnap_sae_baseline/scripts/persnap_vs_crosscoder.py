"""Per-snapshot SAE recovery vs Run 3 crosscoder recovery.

Loads:
  - 32 persnap JumpReLU SAEs (one per Ge step) trained by
    scripts/train_persnap_saes.py
  - Run 3 seed 0 crosscoder (the 32-snap d_sae=8192 result)

Computes per-snapshot reconstruction recovery R = 1 - MSE / Var(x) on the
matched center_scale-preprocessed W_U rows for both the persnap SAEs and
the joint crosscoder, and reports the metrics to stdout.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[4]))
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.core.paths import ssd_path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from src.crosscoder.wu_adapter import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_STEPS,
    extract_wu,
)

DEV = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

R3_CKPT = Path(
    str(ssd_path("wu_crosscoder", "run3_ge_exact_results", "wu_cc_ge_exact_seed0.pt"))
)
PERSNAP_DIR = ssd_path("wu_crosscoder", "per_snap_sae")


def load_snapshots_for_steps(steps: list[int]) -> torch.Tensor:
    arrs = [
        extract_wu(DEFAULT_MODEL, s, DEFAULT_CACHE, dtype=torch.float32) for s in steps
    ]
    return torch.stack(arrs, dim=0)  # (K, V, d)


def crosscoder_persnap_recovery(ckpt_path: Path) -> dict[int, dict]:
    """Per-snapshot R = 1 - MSE/Var(x) and mean L0 for the joint crosscoder.

    Computes only the quantities we need for the figure.
    """
    print(f"Loading crosscoder {ckpt_path.name}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    pp = ck["preprocess_stats"]
    steps = list(ck["steps"])
    K = len(steps)

    W_E = sd["W_E"].to(DEV).float()  # (K, d, D)
    b_E = sd["b_E"].to(DEV).float()  # (K, D)
    W_D = sd["W_D"].to(DEV).float()  # (K, D, d)
    b_D = sd["b_D"].to(DEV).float()  # (K, d)
    log_thr = sd["activation_function.log_jumprelu_threshold"].to(DEV).float()  # (D,)
    dec_norm = W_D.pow(2).sum(dim=2).sqrt().clamp_min(1e-8)  # (K, D)

    snaps_raw = load_snapshots_for_steps(steps)  # (K, V, d) cpu
    means = pp["mean"].squeeze(1)  # (K, d)
    scales = pp["scale"].squeeze(-1).squeeze(-1)  # (K,)
    snaps_x = (snaps_raw - means.unsqueeze(1)) / scales.unsqueeze(-1).unsqueeze(-1)
    V = snaps_x.shape[1]

    D = W_E.shape[2]
    chunk = 2048 if D <= 8192 else 1024
    ss_res = torch.zeros(K, device=DEV)
    ss_tot = torch.zeros(K, device=DEV)
    l0_sum = torch.zeros(K, device=DEV)
    n_seen = torch.zeros(K, device=DEV)

    snaps_x_dev = snaps_x.to(DEV)
    snap_mean = snaps_x_dev.mean(dim=1, keepdim=True)
    thr = torch.exp(log_thr)

    for v0 in range(0, V, chunk):
        v1 = min(V, v0 + chunk)
        x_chunk = snaps_x_dev[:, v0:v1, :]
        pre_joint = torch.bmm(x_chunk, W_E).sum(dim=0) + b_E.sum(dim=0)  # (chunk, D)

        ss_tot += (x_chunk - snap_mean).pow(2).sum(dim=(1, 2))

        for k in range(K):
            scaled = pre_joint * dec_norm[k]
            acts_scaled = torch.where(scaled > thr, scaled, torch.zeros_like(scaled))
            acts = acts_scaled / dec_norm[k]
            recon = acts @ W_D[k] + b_D[k]
            ss_res[k] += (x_chunk[k] - recon).pow(2).sum()
            l0_sum[k] += (acts > 0).float().sum()

        n_seen += v1 - v0

    recovery = (1.0 - ss_res / ss_tot).cpu().numpy()
    l0 = (l0_sum / n_seen).cpu().numpy()
    return {
        steps[k]: {"recovery": float(recovery[k]), "l0": float(l0[k])} for k in range(K)
    }


def persnap_sae_recovery(persnap_dir: Path, steps: list[int]) -> dict[int, dict]:
    """Per-step recovery & L0 for the 32 single-snapshot SAEs.

    Each SAE is a K=1 crosscoder. We re-extract its snapshot, apply the
    saved center_scale stats, and compute R = 1 - MSE/Var(x) on the
    same center_scale-space the SAE was trained on.
    """
    out: dict[int, dict] = {}
    for s in steps:
        path = persnap_dir / f"wu_sae_dsae8192_step{s}.pt"
        if not path.exists():
            print(f"WARNING: missing {path}; skipping step {s}")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        pp = ck["preprocess_stats"]

        # K=1 single-snap crosscoder.
        W_E = sd["W_E"].to(DEV).float()  # (1, d, D)
        b_E = sd["b_E"].to(DEV).float()  # (1, D)
        W_D = sd["W_D"].to(DEV).float()  # (1, D, d)
        b_D = sd["b_D"].to(DEV).float()  # (1, d)
        log_thr = sd["activation_function.log_jumprelu_threshold"].to(DEV).float()
        dec_norm = W_D.pow(2).sum(dim=2).sqrt().clamp_min(1e-8)  # (1, D)
        thr = torch.exp(log_thr)

        x_raw = extract_wu(
            DEFAULT_MODEL, s, DEFAULT_CACHE, dtype=torch.float32
        )  # (V, d)
        mean = pp["mean"].squeeze()  # (d,)
        scale = pp["scale"].squeeze()  # scalar
        x = ((x_raw - mean) / scale).to(DEV)  # (V, d) in center_scale space

        V = x.shape[0]
        chunk = 2048
        ss_res = torch.zeros((), device=DEV)
        ss_tot = torch.zeros((), device=DEV)
        l0_sum = torch.zeros((), device=DEV)
        x_mean = x.mean(dim=0, keepdim=True)

        for v0 in range(0, V, chunk):
            v1 = min(V, v0 + chunk)
            x_c = x[v0:v1]  # (chunk, d)
            pre = x_c @ W_E[0] + b_E[0]  # (chunk, D)
            scaled = pre * dec_norm[0]
            acts_scaled = torch.where(scaled > thr, scaled, torch.zeros_like(scaled))
            acts = acts_scaled / dec_norm[0]
            recon = acts @ W_D[0] + b_D[0]
            ss_res = ss_res + (x_c - recon).pow(2).sum()
            ss_tot = ss_tot + (x_c - x_mean).pow(2).sum()
            l0_sum = l0_sum + (acts > 0).float().sum()

        recovery = (1.0 - ss_res / ss_tot).item()
        l0 = (l0_sum / V).item()
        out[s] = {"recovery": recovery, "l0": l0}
        print(f"  step {s:>7}: SAE R={recovery:.4f} L0={l0:.1f}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persnap-dir", type=Path, default=PERSNAP_DIR)
    parser.add_argument("--crosscoder", type=Path, default=R3_CKPT)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if not args.crosscoder.exists():
        raise FileNotFoundError(
            f"Run 3 seed 0 crosscoder not found at {args.crosscoder}"
        )

    print(f"Device: {DEV}")
    cc = crosscoder_persnap_recovery(args.crosscoder)
    print(
        f"Crosscoder persnap recovery: mean R={np.mean([v['recovery'] for v in cc.values()]):.4f}"
    )

    steps = list(args.steps) if args.steps else list(DEFAULT_STEPS)
    sae = persnap_sae_recovery(args.persnap_dir, steps)
    if not sae:
        raise RuntimeError(
            f"No persnap SAEs found in {args.persnap_dir}. "
            f"Run scripts/train_persnap_saes.py first."
        )

    print("\nstep        | cc R    | sae R   | cc L0  | sae L0")
    for s in sorted(set(cc) | set(sae)):
        cc_r = f"{cc[s]['recovery']:.4f}" if s in cc else "  --  "
        sae_r = f"{sae[s]['recovery']:.4f}" if s in sae else "  --  "
        cc_l = f"{cc[s]['l0']:.1f}" if s in cc else "  --  "
        sae_l = f"{sae[s]['l0']:.1f}" if s in sae else "  --  "
        print(f"{s:>10}  | {cc_r}  | {sae_r}  | {cc_l}  | {sae_l}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
