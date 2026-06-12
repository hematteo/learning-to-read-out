"""Snapshot-wise fidelity (Ge Fig. 3 analogue).

For each Pythia step k ∈ steps:
  - cross-snap-32 d=8192 seed0 (W_U): per-k EV, L0
  - cross-snap-32 d=24576 seed0 (W_U): per-k EV, L0
  - cross-snap-16 d=8192 seed0 (W_U): per-k EV, L0 (only its 16 training steps)
  - per-snap-SAE d=8192 step{k}: EV, L0 at its own step
  - final-snap-SAE d=8192 (trained at step 143000) evaluated on every step.

Outputs:
  experiments/crosscoders/crosscoder_main/derived/appendix_validation/persnap_fidelity.csv  (model, step, EV, FVU, L0)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[5]))
import csv
import gc
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.core.paths import ssd_root

REPO = Path(__file__).resolve().parents[5]
SSD = ssd_root()
HF = SSD / "hf_release/parameter-trajectory-crosscoders"
SNAP = SSD / "snapshots/pythia-160m"
SLUG = "EleutherAI_pythia-160m"
OUT = REPO / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/persnap_fidelity.csv"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

PYTHIA_STEPS = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    14000,
    21000,
    27000,
    34000,
    47000,
    61000,
    75000,
    89000,
    102000,
    116000,
    130000,
    143000,
]


def load_snap(step):
    return torch.load(SNAP / f"{SLUG}_step{step}_wu.pt", map_location="cpu", weights_only=False).float()


def preprocess_stats(steps):
    means, scales = [], []
    for s in steps:
        x = load_snap(s)
        m = x.mean(dim=0, keepdim=True)
        c = x - m
        d = x.shape[-1]
        ms = c.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        sc = (ms / d).clamp_min(1e-8).sqrt().squeeze(-1)
        means.append(m.unsqueeze(0))
        scales.append(sc.reshape(1, 1, 1))
    return (
        torch.cat(means, dim=0).squeeze(1),
        torch.cat(scales, dim=0).squeeze(-1).squeeze(-1),
    )


def eval_crosssnap(ckpt_rel, steps_eval):
    """Per-snap EV/L0 for a cross-snap jumprelu crosscoder.

    `steps_eval` is the list of steps the crosscoder was trained on; it must
    match the safetensors K dimension. Returns list of (step, ev, l0) per k.
    """
    sd = load_file(str(HF / ckpt_rel))
    W_E = sd["W_E"].float()
    b_E = sd["b_E"].float()
    W_D = sd["W_D"].float()
    b_D = sd["b_D"].float()
    log_thr = sd["activation_function.log_jumprelu_threshold"].float()
    thr = log_thr.exp().to(DEV)
    dec_norm = W_D.pow(2).sum(dim=2).sqrt().clamp_min(1e-8).float()
    K, d_model, D = W_E.shape
    assert K == len(steps_eval), f"K={K} vs steps={len(steps_eval)}"

    means, scales = preprocess_stats(steps_eval)
    V = None
    pre_joint = None
    # Phase 1: pre_joint
    for k, s in enumerate(steps_eval):
        snap = load_snap(s)
        if V is None:
            V = snap.shape[0]
            pre_joint = torch.zeros(V, D, dtype=torch.float32)
        x_k = (snap - means[k]) / scales[k]
        del snap
        W_E_k = W_E[k].to(DEV)
        for v0 in range(0, V, 4096):
            v1 = min(V, v0 + 4096)
            xc = x_k[v0:v1].to(DEV)
            pre_joint[v0:v1] += (xc @ W_E_k).cpu()
            del xc
        del W_E_k, x_k
        gc.collect()
    pre_joint += b_E.sum(dim=0)

    # Phase 2: per-k SS_res, SS_tot, L0
    ss_res = [0.0] * K
    ss_tot = [0.0] * K
    l0_sum = [0.0] * K
    for k, s in enumerate(steps_eval):
        snap = load_snap(s)
        x_k = (snap - means[k]) / scales[k]
        del snap
        dn_k = dec_norm[k].to(DEV)
        W_D_k = W_D[k].to(DEV)
        b_D_k = b_D[k].to(DEV)
        for v0 in range(0, V, 4096):
            v1 = min(V, v0 + 4096)
            x_chunk = x_k[v0:v1].to(DEV)
            pj = pre_joint[v0:v1].to(DEV)
            scaled = pj * dn_k
            fires = scaled > thr
            acts = pj * fires.float()
            recon = acts @ W_D_k + b_D_k
            ss_res[k] += (x_chunk - recon).pow(2).sum().float().cpu().item()
            ss_tot[k] += x_chunk.pow(2).sum().float().cpu().item()
            l0_sum[k] += fires.sum().float().cpu().item()
            del x_chunk, pj, scaled, fires, acts, recon
        del dn_k, W_D_k, b_D_k, x_k
        gc.collect()
    if DEV.type == "mps":
        torch.mps.empty_cache()

    out = []
    for k, s in enumerate(steps_eval):
        ev = 1.0 - ss_res[k] / ss_tot[k] if ss_tot[k] > 0 else float("nan")
        l0 = l0_sum[k] / V
        out.append({"step": s, "ev": ev, "l0": l0})
    return out


def eval_singlesnap_on_all(ckpt_rel, train_step, eval_steps):
    """K=1 jumprelu SAE trained at `train_step`, evaluated on each of `eval_steps`.

    Uses the SAE's preprocess (computed from the training step) and applies
    the same scaling to each evaluation step's W_U. This matches how a
    practitioner would deploy the SAE on a different snapshot.
    """
    sd = load_file(str(HF / ckpt_rel))
    W_E = sd["W_E"].float()  # (1, d, D)
    b_E = sd["b_E"].float()
    W_D = sd["W_D"].float()
    b_D = sd["b_D"].float()
    log_thr = sd["activation_function.log_jumprelu_threshold"].float()
    thr = log_thr.exp().to(DEV)
    dec_norm = W_D.pow(2).sum(dim=2).sqrt().clamp_min(1e-8).float()

    # Preprocess stats: only at train_step (the SAE saw)
    train_x = load_snap(train_step)
    m = train_x.mean(dim=0, keepdim=True)
    centered = train_x - m
    d = train_x.shape[-1]
    ms = centered.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    sc = (ms / d).clamp_min(1e-8).sqrt().squeeze(-1)
    pp_scale = sc.squeeze().item()
    del train_x, centered

    W_E0 = W_E[0].to(DEV)
    b_E0 = b_E[0].to(DEV)
    W_D0 = W_D[0].to(DEV)
    b_D0 = b_D[0].to(DEV)
    dn0 = dec_norm[0].to(DEV)

    out = []
    for s in eval_steps:
        x = load_snap(s)
        # apply same preprocess scaling, but center using THIS snapshot's mean
        # (more honest; matches how you'd use the SAE — center the new W_U
        # row-wise then divide by the train scale).
        x_centered = x - x.mean(dim=0, keepdim=True)
        x_norm = x_centered / pp_scale
        del x, x_centered
        ss_res = ss_tot = l0_sum = 0.0
        V = x_norm.shape[0]
        for v0 in range(0, V, 4096):
            v1 = min(V, v0 + 4096)
            xc = x_norm[v0:v1].to(DEV)
            pre = xc @ W_E0 + b_E0
            scaled = pre * dn0
            fires = scaled > thr
            acts = pre * fires.float()
            recon = acts @ W_D0 + b_D0
            ss_res += (xc - recon).pow(2).sum().float().cpu().item()
            ss_tot += xc.pow(2).sum().float().cpu().item()
            l0_sum += fires.sum().float().cpu().item()
            del xc, pre, scaled, fires, acts, recon
        if DEV.type == "mps":
            torch.mps.empty_cache()
        ev = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        l0 = l0_sum / V
        out.append({"step": s, "ev": ev, "l0": l0})
    return out


def main():
    rows = []

    targets = [
        (
            "cross-snap-32 d=8192 (W_U)",
            "pythia-160m/W_U/cross-snapshot-32/d8192/seed0.safetensors",
            PYTHIA_STEPS,
        ),
        (
            "cross-snap-32 d=24576 (W_U)",
            "pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors",
            PYTHIA_STEPS,
        ),
        # cross-snap-16 was trained on 16 steps; read steps from its config.json
    ]
    cs16_cfg = json.loads((HF / "pythia-160m/W_U/cross-snapshot-16/d8192/seed0.config.json").read_text())
    cs16_steps = cs16_cfg["steps"]
    targets.append(
        (
            "cross-snap-16 d=8192 (W_U)",
            "pythia-160m/W_U/cross-snapshot-16/d8192/seed0.safetensors",
            cs16_steps,
        )
    )

    for label, rel, steps in targets:
        print(f"\n=== {label} ({len(steps)} steps) ===", flush=True)
        t0 = time.time()
        recs = eval_crosssnap(rel, steps)
        print(f"  {time.time() - t0:.1f}s", flush=True)
        for r in recs:
            rows.append(
                {
                    "family": label,
                    "step": r["step"],
                    "ev": r["ev"],
                    "fvu": 1 - r["ev"],
                    "l0": r["l0"],
                }
            )

    # final-snap d=8192 (trained at step 143000) evaluated on all 32 steps
    print("\n=== final-snap d=8192 (W_U) cross-eval ===", flush=True)
    t0 = time.time()
    recs = eval_singlesnap_on_all("pythia-160m/W_U/final-snapshot-saes/d8192.safetensors", 143000, PYTHIA_STEPS)
    print(f"  {time.time() - t0:.1f}s", flush=True)
    for r in recs:
        rows.append(
            {
                "family": "final-snap d=8192 (cross-eval)",
                "step": r["step"],
                "ev": r["ev"],
                "fvu": 1 - r["ev"],
                "l0": r["l0"],
            }
        )

    # Per-snap SAEs: read from existing inventory CSV (one ckpt per step)
    inv = list(
        csv.DictReader(
            (REPO / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/full_inventory.csv").open()
        )
    )
    for r in inv:
        if r["family"] == "per-snap-SAE":
            try:
                step = int(r["seed"].replace("step", ""))
            except (ValueError, AttributeError):
                continue
            ev = float(r["EV"]) if r["EV"] not in ("", "—") else None
            l0 = float(r["mean_L0"]) if r["mean_L0"] not in ("", "—") else None
            rows.append(
                {
                    "family": "per-snap d=8192",
                    "step": step,
                    "ev": ev,
                    "fvu": (1 - ev) if ev else None,
                    "l0": l0,
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "step", "ev", "fvu", "l0"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
