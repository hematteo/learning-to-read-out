"""Compute per-snapshot (EV, L0, dead_rate) for any SAE/crosscoder ckpt.

Three modes (auto-detected from family):
  cross-snap   joint encoder forward across K snapshots; for each snap k,
               reconstruct the snap's W_U using W_D[k] and the *joint* sparse
               codes.
  final-snap   single SAE (W_E, W_D, b_E, b_D, threshold); evaluated on each
               snapshot via a fresh encode each time. Captures how a
               terminal-trained SAE degrades on earlier snapshots.
  per-snap     for d8192 per-snapshot SAEs: evaluate each step's SAE on its
               own snapshot only (one (step, ev, l0) row per ckpt).

Outputs (per ckpt → one CSV row per snapshot):
  experiments/crosscoders/crosscoder_main/derived/appendix_validation/per_snap_eval/<ckpt-id>.csv

Re-runs are no-ops if the CSV exists; pass --force to recompute.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import safe_open

from readout.core.model_specs import DEFAULT_STEPS_32 as PYTHIA_STEPS_32
from readout.core.paths import repo_root, ssd_root
from readout.crosscoder import inference  # noqa: E402

REPO = repo_root()
SSD = ssd_root()
HF = SSD / "hf_release/parameter-trajectory-crosscoders"
SNAPSHOTS = SSD / "snapshots"
DEFAULT_OUT = (
    REPO
    / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/per_snap_eval"
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_ckpt_state(ckpt_path: Path) -> tuple[dict, dict | None]:
    """Returns (state_dict, companion_cfg). Handles .safetensors + .pt."""
    if ckpt_path.suffix == ".safetensors":
        sd = {}
        with safe_open(ckpt_path, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        cfg_path = ckpt_path.with_name(f"{ckpt_path.stem}.config.json")
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    else:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
        sd = ck["state_dict"]
        cfg = {
            k: ck[k]
            for k in ("steps", "preprocess_stats", "model_name", "config")
            if k in ck
        }
    return sd, cfg


def reconstruct_preprocess_stats(snap_dir: Path, prefix: str, steps: list[int]) -> dict:
    """center_scale per-snap (mean, scale), rebuilt when the ckpt ships none."""
    return inference.reconstruct_preprocess_stats(load_snap(snap_dir, prefix, s) for s in steps)


def load_snap(snap_dir: Path, prefix: str, step: int) -> torch.Tensor:
    return torch.load(
        snap_dir / f"{prefix}_step{step}_wu.pt", map_location="cpu", weights_only=True
    ).float()


# ─── eval implementations ──────────────────────────────────────────────────────


def eval_cross_snapshot(
    ckpt_path: Path, snap_dir: Path, snap_prefix: str, device: str = DEVICE
) -> list[dict]:
    """Joint-encoder per-snap (EV, L0, dead_rate)."""
    sd, cfg = load_ckpt_state(ckpt_path)
    steps = list(cfg["steps"]) if cfg and "steps" in cfg else PYTHIA_STEPS_32
    K = len(steps)
    W_E = sd["W_E"].to(device).float()  # (K, d, D)
    b_E = sd["b_E"].to(device).float()  # (K, D)
    W_D = sd["W_D"].to(device).float()  # (K, D, d)
    b_D = sd["b_D"].to(device).float()  # (K, d)
    thr = inference.jumprelu_threshold(sd, device=device)  # (D,)
    dec_norm = inference.decoder_norms(sd, eps=1e-8, device=device)  # (K, D)

    K_, d, D = W_E.shape
    print(f"  cross-snapshot eval: K={K} D={D} d={d} on {device}", flush=True)

    if cfg and cfg.get("preprocess_stats"):
        ps = cfg["preprocess_stats"]
    else:
        print("  preprocess_stats missing → reconstruct from snapshots", flush=True)
        ps = reconstruct_preprocess_stats(snap_dir, snap_prefix, steps)

    means = ps["mean"].to(device).squeeze(1).float()  # (K, d)
    scales = ps["scale"].to(device).squeeze(-1).squeeze(-1).float()  # (K,)

    # Stream snapshots one-by-one to compute joint pre-activations.
    # joint_pre = sum_k (x_k @ W_E[k]) + sum_k(b_E[k]) — additive, so we accumulate.
    V = None
    joint_pre = None
    for i, s in enumerate(steps):
        W_raw = load_snap(snap_dir, snap_prefix, s).to(device)
        if V is None:
            V = W_raw.shape[0]
            joint_pre = torch.zeros(V, D, device=device)
        x_i = (W_raw - means[i]) / scales[i]
        # Chunk across V to avoid (V, D) tensor blow-ups for big D
        chunk = 4096
        for v0 in range(0, V, chunk):
            v1 = min(V, v0 + chunk)
            joint_pre[v0:v1] += x_i[v0:v1] @ W_E[i]
        del W_raw, x_i
    joint_pre = joint_pre + b_E.sum(dim=0)  # (V, D)

    # Per-snap reconstruction quality
    rows = []
    for i, s in enumerate(steps):
        W_raw = load_snap(snap_dir, snap_prefix, s).to(device)
        x_i = (W_raw - means[i]) / scales[i]
        x_centered = x_i - x_i.mean(dim=0, keepdim=True)
        ss_tot = x_centered.pow(2).sum().item()
        # Activations: jumprelu w.r.t. dec_norm[i]
        acts_i, _ = inference.jumprelu_feature_acts(joint_pre, thr, dec_norm[i])  # (V, D)
        recon = acts_i @ W_D[i] + b_D[i]  # (V, d)
        ss_res = (x_i - recon).pow(2).sum().item()
        ev = 1.0 - ss_res / max(ss_tot, 1e-12)
        l0 = (acts_i > 0).float().sum(dim=1).mean().item()
        # Per-feature firing rate over V; dead = rate < 1e-4
        rate = (acts_i > 0).float().mean(dim=0)
        dead = (rate < 1e-4).float().mean().item()
        rows.append(
            {"step": s, "ev": ev, "l0": l0, "dead_rate": dead, "kind": "cross-snap"}
        )
        del W_raw, x_i, acts_i, recon, x_centered
    return rows


def eval_final_snapshot(
    ckpt_path: Path,
    snap_dir: Path,
    snap_prefix: str,
    device: str = DEVICE,
    eval_steps: list[int] | None = None,
) -> list[dict]:
    """Single SAE (final-snapshot trained); evaluate on each snapshot.

    The final-snapshot SAEs in hf_release have shapes:
        W_E: (d, D), b_E: (D,), W_D: (D, d), b_D: (d,), threshold: (D,)
    (no leading K). They were trained on terminal-step W_U with center_scale
    preprocessing. To evaluate on snap k we apply the *snap-k* preprocessing
    (mean / scale) before encode; this matches the standard "back-projection"
    convention.
    """
    sd, cfg = load_ckpt_state(ckpt_path)
    W_E = sd["W_E"].to(device).float()
    b_E = sd["b_E"].to(device).float()
    W_D = sd["W_D"].to(device).float()
    b_D = sd["b_D"].to(device).float()
    log_thr = sd["activation_function.log_jumprelu_threshold"].to(device).float()
    thr = log_thr.exp()

    # Final-snap SAE shape may be (d, D) for W_E and (D, d) for W_D, or include
    # a leading singleton K=1 dim. Normalize to 2D.
    if W_E.ndim == 3 and W_E.shape[0] == 1:
        W_E, b_E, W_D, b_D = W_E[0], b_E[0], W_D[0], b_D[0]
    dec_norm = W_D.pow(2).sum(dim=1).sqrt().clamp_min(1e-8)  # (D,)
    D, d = W_D.shape if W_D.ndim == 2 else (W_D.shape[1], W_D.shape[2])

    steps = eval_steps or PYTHIA_STEPS_32
    print(f"  final-snap eval: D={D} d={d} over {len(steps)} snaps", flush=True)

    rows = []
    for s in steps:
        W_raw = load_snap(snap_dir, snap_prefix, s).to(device)
        # Use snap's own center_scale preprocessing
        mu = W_raw.mean(dim=0, keepdim=True)
        sc = (W_raw - mu).pow(2).mean().sqrt().clamp_min(1e-8)
        x = (W_raw - mu) / sc
        x_centered = x - x.mean(dim=0, keepdim=True)
        ss_tot = x_centered.pow(2).sum().item()

        pre = x @ W_E + b_E  # (V, D)
        acts, _ = inference.jumprelu_feature_acts(pre, thr, dec_norm)
        recon = acts @ W_D + b_D  # (V, d)
        ss_res = (x - recon).pow(2).sum().item()
        ev = 1.0 - ss_res / max(ss_tot, 1e-12)
        l0 = (acts > 0).float().sum(dim=1).mean().item()
        rate = (acts > 0).float().mean(dim=0)
        dead = (rate < 1e-4).float().mean().item()
        rows.append(
            {"step": s, "ev": ev, "l0": l0, "dead_rate": dead, "kind": "final-snap"}
        )
        del W_raw, x, pre, acts, recon
    return rows


def eval_one_per_snap_sae(
    ckpt_path: Path, snap_dir: Path, snap_prefix: str, step: int, device: str = DEVICE
) -> dict:
    """Per-snapshot SAE evaluated on its own snapshot only."""
    sd, _ = load_ckpt_state(ckpt_path)
    W_E = sd["W_E"].to(device).float()
    b_E = sd["b_E"].to(device).float()
    W_D = sd["W_D"].to(device).float()
    b_D = sd["b_D"].to(device).float()
    log_thr = sd["activation_function.log_jumprelu_threshold"].to(device).float()
    thr = log_thr.exp()
    if W_E.ndim == 3 and W_E.shape[0] == 1:
        W_E, b_E, W_D, b_D = W_E[0], b_E[0], W_D[0], b_D[0]
    dec_norm = W_D.pow(2).sum(dim=1).sqrt().clamp_min(1e-8)

    W_raw = load_snap(snap_dir, snap_prefix, step).to(device)
    mu = W_raw.mean(dim=0, keepdim=True)
    sc = (W_raw - mu).pow(2).mean().sqrt().clamp_min(1e-8)
    x = (W_raw - mu) / sc
    x_centered = x - x.mean(dim=0, keepdim=True)
    ss_tot = x_centered.pow(2).sum().item()

    pre = x @ W_E + b_E
    scaled = pre * dec_norm
    acts_scaled = torch.where(scaled > thr, scaled, torch.zeros_like(scaled))
    acts = acts_scaled / dec_norm
    recon = acts @ W_D + b_D
    ss_res = (x - recon).pow(2).sum().item()
    ev = 1.0 - ss_res / max(ss_tot, 1e-12)
    l0 = (acts > 0).float().sum(dim=1).mean().item()
    rate = (acts > 0).float().mean(dim=0)
    dead = (rate < 1e-4).float().mean().item()
    return {"step": step, "ev": ev, "l0": l0, "dead_rate": dead, "kind": "per-snap"}


# ─── dispatcher / CLI ──────────────────────────────────────────────────────────


@dataclass
class Job:
    ckpt_id: str  # canonical id used as CSV filename
    ckpt_path: Path
    snap_dir: Path
    snap_prefix: str
    family: str  # "cross-snap" | "final-snap" | "per-snap"


def write_csv(rows: list[dict], out: Path, ckpt_id: str):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt_id", "step", "kind", "ev", "l0", "dead_rate"])
        for r in rows:
            w.writerow(
                [ckpt_id, r["step"], r["kind"], r["ev"], r["l0"], r["dead_rate"]]
            )
    print(f"  → {out}")


def run_job(job: Job, force: bool = False):
    out = DEFAULT_OUT / f"{job.ckpt_id}.csv"
    if out.exists() and not force:
        print(f"[skip] {job.ckpt_id} (CSV exists; pass --force to recompute)")
        return
    t0 = time.time()
    print(f"[eval] {job.ckpt_id} ({job.family}) ckpt={job.ckpt_path.name}")
    if job.family == "cross-snap":
        rows = eval_cross_snapshot(job.ckpt_path, job.snap_dir, job.snap_prefix)
    elif job.family == "final-snap":
        rows = eval_final_snapshot(job.ckpt_path, job.snap_dir, job.snap_prefix)
    elif job.family == "per-snap":
        # per-snap files are named step<N>.safetensors; one row per ckpt
        step = int(job.ckpt_path.stem.replace("step", ""))
        rows = [
            eval_one_per_snap_sae(job.ckpt_path, job.snap_dir, job.snap_prefix, step)
        ]
    else:
        raise ValueError(f"unknown family: {job.family}")
    write_csv(rows, out, job.ckpt_id)
    print(f"  done in {time.time() - t0:.1f}s")


def jobs_for_160m_d8192_family() -> list[Job]:
    """Everything needed for the snapshot-wise reconstruction plot."""
    snap_dir = SNAPSHOTS / "pythia-160m"
    prefix = "EleutherAI_pythia-160m"
    jobs = []
    # cross-snap-32 d8192 (the headline jumprelu)
    jobs.append(
        Job(
            "pythia-160m_d8192_cross-snap-32_jumprelu_seed0",
            HF / "pythia-160m/W_U/cross-snapshot-32/d8192/seed0.safetensors",
            snap_dir,
            prefix,
            "cross-snap",
        )
    )
    # cross-snap-16 d8192 (ablation)
    jobs.append(
        Job(
            "pythia-160m_d8192_cross-snap-16_jumprelu_seed0",
            HF / "pythia-160m/W_U/cross-snapshot-16/d8192/seed0.safetensors",
            snap_dir,
            prefix,
            "cross-snap",
        )
    )
    # NOTE: arch variants (batchtopk, gated, gated-retuned) skipped here.
    # They use different activation functions (top-K, gated) — would need
    # separate per-snap eval implementations. Their aggregate (EV, L0) is
    # already on the Pareto-front plot via the quality field; the snapshot-
    # wise reconstruction plot below uses the jumprelu cross-snap-32 as the
    # canonical "cross-snap" curve.
    # final-snap SAE d=8192
    jobs.append(
        Job(
            "pythia-160m_d8192_final-snap-sae",
            HF / "pythia-160m/W_U/final-snapshot-saes/d8192.safetensors",
            snap_dir,
            prefix,
            "final-snap",
        )
    )
    # per-snap-saes d=8192 (32 step files)
    for p in sorted(
        (HF / "pythia-160m/W_U/per-snapshot-saes/d8192").glob("step*.safetensors")
    ):
        if p.name.startswith("._"):
            continue
        step_id = p.stem
        jobs.append(
            Job(
                f"pythia-160m_d8192_per-snap-sae_{step_id}",
                p,
                snap_dir,
                prefix,
                "per-snap",
            )
        )
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preset",
        choices=["160m_d8192_family", "all-headline"],
        default="160m_d8192_family",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.preset == "160m_d8192_family":
        jobs = jobs_for_160m_d8192_family()
    else:
        raise SystemExit("only 160m_d8192_family wired so far")
    print(f"running {len(jobs)} eval jobs on {DEVICE}")
    for j in jobs:
        run_job(j, force=args.force)


if __name__ == "__main__":
    main()
