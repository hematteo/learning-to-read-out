"""Per-snapshot Explained Variance (EV) and L0 sparsity for the 8 arms in
the Phase 2 grid. Mirrors Ge et al. 2026 (arXiv:2509.17196) Fig 2 / Fig 3 —
the canonical crosscoder QC plot is EV vs L0 (Pareto), with per-snapshot
EV and L0 trajectories alongside.

Reconstruction definition (cross-snapshot crosscoder, single shared feature
basis, K snapshot heads):
    joint_pre[v, j]      = sum_h (x_h[v] @ W_E[h, :, j] + b_E[h, j])
    feature_acts[v, j]   = arch-specific activation of joint_pre
                              (JumpReLU: pre * (pre > thr); BatchTopK:
                               top-(V*k); Gated: gate * relu(r_mag·pre + b_mag))
    recon_h[v, :]        = feature_acts[v, :] @ W_D[h] + b_D[h]
    EV_h                 = 1 - var(x_h - recon_h) / var(x_h)   (per snapshot h)
    L0_h                 = mean over v of (feature_acts[v] != 0).sum()

We compute EV per snapshot (so we can plot per-snapshot trajectories and
mean EV across snapshots, the latter being Ge's "EV" headline value).

Output: results/experiments/capacity/pareto_frontier_ev_l0/ev_l0_phase2.json
with per-arm per-snapshot EV and L0 arrays, plus arm-level summaries.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[4]))
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.core.paths import ssd_path
from src.crosscoder.checkpoints import unwrap_ckpt

SSD = ssd_path()
WU_CACHE = SSD / "snapshots"
WE_CACHE = SSD / "we_snapshots"

OUT = Path("results/experiments/capacity/pareto_frontier_ev_l0/ev_l0_phase2.json")


@dataclass
class Arm:
    name: str
    label: str
    ckpt: Path
    snap_dir: Path
    family: str  # "WU8192" / "WU16384" / "WU24576" / "WE8192" / "WE24576"
    is_new: bool


ARMS: list[Arm] = [
    # ---- baselines (already had rates pre-Phase-2) ----
    Arm(
        "run3_s0",
        "Run 3 seed 0 (160M W_U d=8192, JumpReLU)",
        SSD / "hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d8192/seed0.safetensors",
        WU_CACHE,
        "WU8192",
        is_new=False,
    ),
    Arm(
        "t4_6_s0",
        "T4.6 seed 0 (160M W_U d=24576, JumpReLU)",
        SSD / "hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors",
        WU_CACHE,
        "WU24576",
        is_new=False,
    ),
    Arm(
        "we_pilot",
        "we_pilot (160M W_E d=8192, JumpReLU)",
        SSD / "hf_release/parameter-trajectory-crosscoders/pythia-160m/W_E/cross-snapshot-32/d8192/seed0.safetensors",
        WE_CACHE,
        "WE8192",
        is_new=False,
    ),
    # ---- 5 newly-extracted arms ----
    Arm(
        "t4_4_16snap",
        "T4.4 16-snap (160M W_U d=8192, JumpReLU, K=16)",
        SSD / "hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-16/d8192/seed0.safetensors",
        WU_CACHE,
        "WU8192",
        is_new=True,
    ),
    Arm(
        "t1_5_bt",
        "T1.5 BatchTopK (160M W_U d=8192, BatchTopK k=203)",
        SSD / "hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/architecture-comparison/d8192/batchtopk/seed0.safetensors",
        WU_CACHE,
        "WU8192",
        is_new=True,
    ),
    Arm(
        "t1_5_gt",
        "T1.5 Gated retune (160M W_U d=8192, Gated lambda=0.05)",
        SSD / "hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/architecture-comparison/d8192/gated-retuned/seed0.safetensors",
        WU_CACHE,
        "WU8192",
        is_new=True,
    ),
    Arm(
        "run4_d16384",
        "run4 (160M W_U d=16384, JumpReLU)",
        SSD / "hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d16384/seed0.safetensors",
        WU_CACHE,
        "WU16384",
        is_new=True,
    ),
    Arm(
        "t3_2_we_d24576",
        "T3.2 W_E (160M W_E d=24576, JumpReLU)",
        SSD / "hf_release/parameter-trajectory-crosscoders/pythia-160m/W_E/cross-snapshot-32/d24576/seed0.safetensors",
        WE_CACHE,
        "WE24576",
        is_new=True,
    ),
]


def _detect_arch(ck: dict, sd: dict) -> str:
    cfg = ck.get("config") if isinstance(ck.get("config"), dict) else {}
    inner = cfg.get("config", cfg) if isinstance(cfg.get("config"), dict) else cfg
    arch = cfg.get("arch") or inner.get("arch") or cfg.get("act_fn") or inner.get("act_fn")
    if arch:
        return arch
    if "activation_function.log_jumprelu_threshold" in sd:
        return "jumprelu"
    if "activation_function.b_gate" in sd:
        return "gated"
    if any(k in cfg for k in ("batchtopk_k",)):
        return "batchtopk"
    raise RuntimeError("could not detect arch from ckpt")


@torch.no_grad()
def compute_arm(arm: Arm, device: torch.device) -> dict:
    """Compute per-snapshot EV and L0 for one arm."""
    print(f"\n=== {arm.name} ({arm.label}) ===", flush=True)
    t0 = time.time()
    ck = unwrap_ckpt(torch.load(arm.ckpt, map_location="cpu", weights_only=False))
    sd = ck["state_dict"]
    steps = list(ck["steps"])
    model_name = ck["model_name"]
    slug = model_name.replace("/", "_")
    arch = _detect_arch(ck, sd)
    W_E = sd["W_E"]  # (K, d, D) CPU
    b_E = sd["b_E"]  # (K, D)    CPU
    W_D = sd["W_D"]  # (K, D, d) CPU
    b_D = sd.get("b_D")  # (K, d) CPU, may be None
    K, d, D = W_E.shape
    pstats = ck.get("preprocess_stats")

    def _load_x(i: int) -> torch.Tensor:
        x_cpu = torch.load(arm.snap_dir / f"{slug}_step{steps[i]}_wu.pt", map_location="cpu").float()
        x_dev = x_cpu.to(device, non_blocking=True)
        if pstats is not None:
            mean_i = pstats["mean"][i].squeeze(0).to(device)
            scale_i = pstats["scale"][i].squeeze().to(device)
            x_dev = (x_dev - mean_i) / scale_i
        return x_dev

    # ---- Pass 1: compute joint_pre (single forward across all snapshots) ----
    joint_pre: torch.Tensor | None = None
    print("  pass 1: joint_pre", flush=True)
    for i, step in enumerate(steps):
        x_use = _load_x(i)
        W_Ei = W_E[i].to(device, non_blocking=True)
        b_Ei = b_E[i].to(device, non_blocking=True)
        head_pre = x_use @ W_Ei + b_Ei
        joint_pre = head_pre if joint_pre is None else joint_pre + head_pre
        del W_Ei, b_Ei, head_pre, x_use
        if device.type == "mps":
            torch.mps.empty_cache()
    assert joint_pre is not None

    # ---- Compute feature_acts via arch-specific rule ----
    print(f"  arch={arch}: computing feature_acts", flush=True)
    if arch == "jumprelu":
        thr = sd["activation_function.log_jumprelu_threshold"].exp().to(device)  # (D,)
        # feature_acts = joint_pre * (joint_pre > thr)  (no per-head correction
        # in the activation step — that's the rate convention. For
        # reconstruction we use the JumpReLU function applied to joint_pre.)
        feature_acts = torch.where(joint_pre > thr, joint_pre, torch.zeros_like(joint_pre))
    elif arch == "batchtopk":
        cfg = ck.get("config", {})
        k_topk = (
            cfg.get("batchtopk_k")
            or cfg.get("k")
            or cfg.get("config", {}).get("batchtopk_k")
            or cfg.get("config", {}).get("k")
        )
        if k_topk is None:
            raise KeyError("batchtopk_k missing from config")
        k_topk = int(k_topk)
        acts = torch.relu(joint_pre)
        flat = acts.reshape(-1)
        k_total = min(joint_pre.shape[0] * k_topk, flat.numel())
        _, idx = flat.topk(k_total, sorted=False)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[idx] = True
        feature_acts = (flat * mask).reshape_as(acts)
    elif arch == "gated":
        if "activation_function.b_gate" in sd:
            b_gate = sd["activation_function.b_gate"].to(device)
            r_mag = sd["activation_function.r_mag"].to(device)
            b_mag = sd["activation_function.b_mag"].to(device)
        else:
            raise RuntimeError("gated ckpt missing activation params")
        gate_pre = joint_pre + b_gate
        gate_hard = (gate_pre > 0).float()
        magnitude = torch.relu(r_mag.exp() * joint_pre + b_mag)
        feature_acts = gate_hard * magnitude
    else:
        raise ValueError(f"unknown arch {arch}")

    nonzero_mask = feature_acts != 0  # (V, D)
    l0_global = float(nonzero_mask.float().sum(dim=-1).mean().item())  # mean L0 over tokens
    active_features = int(nonzero_mask.any(dim=0).sum().item())
    dead_features = D - active_features

    # ---- Pass 2: per-snapshot reconstruction + EV (re-load x_h streaming) ----
    print("  pass 2: per-snapshot reconstruction + EV", flush=True)
    ev_per_step: list[float] = []
    nrmse_per_step: list[float] = []
    for i, step in enumerate(steps):
        x_use = _load_x(i)
        W_Di = W_D[i].to(device, non_blocking=True)  # (D, d)
        recon = feature_acts @ W_Di
        if b_D is not None:
            recon = recon + b_D[i].to(device)
        residual = x_use - recon
        var_total = x_use.var()
        var_resid = residual.var()
        ev = float((1.0 - var_resid / var_total.clamp_min(1e-12)).item())
        # Normalised RMSE (residual L2 / activation L2; Ge's NRMSE definition)
        nrmse = float((residual.norm() / x_use.norm().clamp_min(1e-12)).item())
        ev_per_step.append(ev)
        nrmse_per_step.append(nrmse)
        del W_Di, recon, residual, x_use
        if device.type == "mps":
            torch.mps.empty_cache()

    elapsed = time.time() - t0
    rec = {
        "name": arm.name,
        "label": arm.label,
        "family": arm.family,
        "is_new": arm.is_new,
        "arch": arch,
        "K": K,
        "d_model": d,
        "d_sae": D,
        "steps": steps,
        "ev_per_step": ev_per_step,
        "nrmse_per_step": nrmse_per_step,
        "ev_mean": sum(ev_per_step) / len(ev_per_step),
        "ev_min": min(ev_per_step),
        "ev_max": max(ev_per_step),
        "l0_mean_over_v": l0_global,
        "active_features": active_features,
        "dead_features": dead_features,
        "elapsed_s": elapsed,
    }
    print(
        f"  EV mean {rec['ev_mean']:.4f}  L0 {rec['l0_mean_over_v']:.2f}  "
        f"dead {rec['dead_features']}/{D}  ({elapsed:.1f}s)",
        flush=True,
    )

    # cleanup
    del joint_pre, feature_acts, nonzero_mask
    if device.type == "mps":
        torch.mps.empty_cache()
    return rec


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}", flush=True)
    out: dict = {"device": str(device), "arms": {}}
    for arm in ARMS:
        out["arms"][arm.name] = compute_arm(arm, device)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
