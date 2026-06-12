"""Recompute EV / mean_l0 / dead_rate for any W_U crosscoder checkpoint.

Handles both .pt and .safetensors formats. Resolves the W_U snapshot cache
based on the checkpoint's stored model_name and steps list, applies the
preprocessing mode the checkpoint was trained with (center_scale typical),
runs encode/decode per-head to keep memory bounded, and writes a sidecar
metrics_recomputed.json next to the checkpoint plus an aggregated CSV.

Usage:
    python scripts/eval/recompute_metrics.py \\
        --root ${UM_SSD_ROOT}/hf_release \\
        --out  ${UM_SSD_ROOT}/hf_release/recompute_summary.csv \\
        --device cuda   # auto-detects cuda/cpu by default; mps also works
"""

from __future__ import annotations

import argparse
import csv
import json
import sys as _sys
from pathlib import Path
from time import time

import torch

_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.crosscoder.checkpoints import load_checkpoint
from src.crosscoder.snapshots import load_snapshots
from src.crosscoder.wu_adapter import preprocess_snapshots


def compute_metrics(sd, X_norm: torch.Tensor, batch_rows: int, device: str) -> dict:
    """Crosscoder forward (matches Crosscoder.encode/decode in llamascopium):
      pre[h] = x[:, h, :] @ W_E[h] + b_E[h]            # per head
      pre    = sum_h pre[h]                             # shared (B, d_sae)
      pre_replicated = repeat(pre, '... d -> ... K d')  # (B, K, d_sae)
      scaled = pre_replicated * decoder_norm            # (B, K, d_sae); decoder_norm: (K, d_sae)
      fired  = scaled > threshold                       # JumpReLU mask, per-head
      acts   = scaled * fired / decoder_norm            # (B, K, d_sae); per-head feature_acts
      rec[h] = acts[:, h, :] @ W_D[h] + b_D[h]          # per-head reconstruction
    l0 reported = (fired.sum(-1)).mean over (B, K)  — same as quick_quality.
    """
    W_E = sd["W_E"]  # (K, d, d_sae)
    W_D = sd["W_D"]  # (K, d_sae, d)
    b_E = sd["b_E"]  # (K, d_sae)
    b_D = sd["b_D"]  # (K, d)
    threshold = sd["activation_function.log_jumprelu_threshold"].exp()

    K, V, d = X_norm.shape
    d_sae = W_E.shape[-1]

    W_E_d = W_E.to(device)
    b_E_d = b_E.to(device)
    W_D_d = W_D.to(device)
    b_D_d = b_D.to(device)
    thr_d = threshold.to(device)
    decoder_norm = torch.norm(W_D_d, dim=-1)  # (K, d_sae)

    total_rec = 0.0
    total_var = 0.0
    total_l0 = 0.0
    n_batches = 0
    feature_fires = torch.zeros(d_sae, dtype=torch.float64)
    n_seen = 0

    X_perm = X_norm.permute(1, 0, 2).contiguous()  # (V, K, d)

    for i in range(0, V, batch_rows):
        x = X_perm[i : i + batch_rows].to(device)  # (B, K, d)
        B = x.shape[0]
        hidden_pre_per_head = torch.einsum("bkd,kde->bke", x, W_E_d) + b_E_d  # (B, K, d_sae)
        accumulated = hidden_pre_per_head.sum(dim=1)  # (B, d_sae)
        pre_rep = accumulated.unsqueeze(1).expand(B, K, d_sae)  # (B, K, d_sae)
        scaled = pre_rep * decoder_norm  # (B, K, d_sae)
        fired_mask = scaled > thr_d
        acts_scaled = scaled * fired_mask
        feature_acts = acts_scaled / decoder_norm  # (B, K, d_sae)

        recon = torch.einsum("bke,kef->bkf", feature_acts, W_D_d) + b_D_d  # (B, K, d)

        total_rec += (recon - x).pow(2).mean().item()
        total_var += x.var().item()
        total_l0 += fired_mask.float().sum(dim=-1).mean().item()
        n_batches += 1

        fa_cpu = fired_mask.to(torch.bool).cpu().reshape(-1, d_sae)
        feature_fires += fa_cpu.sum(dim=0).to(torch.float64)
        n_seen += fa_cpu.shape[0]

        del x, hidden_pre_per_head, accumulated, pre_rep, scaled
        del fired_mask, acts_scaled, feature_acts, recon, fa_cpu
        if device == "mps":
            torch.mps.empty_cache()

    ev = 1.0 - total_rec / total_var if total_var > 0 else 0.0
    dead_rate = float((feature_fires / n_seen <= 1e-6).float().mean()) if n_seen > 0 else float("nan")
    return {
        "explained_variance": float(ev),
        "reconstruction_mse": total_rec / n_batches,
        "mean_l0": total_l0 / n_batches,
        "dead_rate": dead_rate,
        "d_sae": int(d_sae),
        "n_snapshots": int(K),
        "n_rows": int(V),
        "n_batches": n_batches,
    }


def process_one(path: Path, device: str, batch_rows: int, snapshot_dir: Path | None = None) -> dict:
    t0 = time()
    cp = load_checkpoint(path)
    if cp.model_name is None or cp.steps is None:
        raise RuntimeError(f"missing model_name or steps in {path}")

    mode = cp.training.get("input_preprocess") or "none"
    snapshots = load_snapshots(cp.model_name, cp.steps, dir_override=snapshot_dir)
    X_norm, _ = preprocess_snapshots(snapshots, mode=mode)

    metrics = compute_metrics(cp.state_dict, X_norm, batch_rows=batch_rows, device=device)
    metrics["preprocess_mode"] = mode
    metrics["model_name"] = cp.model_name
    metrics["runtime_s"] = round(time() - t0, 1)

    sidecar = path.parent / f"{path.stem}.metrics_recomputed.json"
    payload = {"recomputed": metrics, "stored": cp.quality}
    sidecar.write_text(json.dumps(payload, indent=2))
    return payload


def find_crosscoder_ckpts(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.safetensors"):
        if p.name.startswith("._"):
            continue
        cfg = p.with_suffix(".config.json")
        if not cfg.exists():
            continue
        try:
            kind = json.loads(cfg.read_text()).get("kind", "")
        except json.JSONDecodeError:
            continue
        # cross-snapshot and wu-crosscoder are both crosscoder architectures.
        # final-snapshot and per-snapshot are vanilla SAEs (different forward).
        if kind in ("wu-crosscoder", "cross-snapshot"):
            out.append(p)
    for p in root.rglob("*.pt"):
        if p.name.startswith("._"):
            continue
        if "wu_cc" in p.name or "cc_olmo" in p.name or "cc_pythia" in p.name:
            out.append(p)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-rows", type=int, default=4096)
    ap.add_argument("--filter", default=None, help="substring filter on ckpt path")
    ap.add_argument("--single", type=Path, default=None, help="run one ckpt only")
    ap.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="override snapshot location: load the canonical "
        "{slug}_step{N}_{kind}.pt files from this dir instead of "
        "${UM_SSD_ROOT}/snapshots/<model>/",
    )
    args = ap.parse_args()

    if args.single is not None:
        ckpts = [args.single]
    else:
        ckpts = find_crosscoder_ckpts(args.root)
        if args.filter:
            ckpts = [p for p in ckpts if args.filter in str(p)]
    print(f"Found {len(ckpts)} checkpoints")

    rows = []
    for i, p in enumerate(ckpts, 1):
        rel = p.relative_to(args.root) if p.is_relative_to(args.root) else p
        print(f"[{i}/{len(ckpts)}] {rel}", flush=True)
        try:
            payload = process_one(
                p,
                device=args.device,
                batch_rows=args.batch_rows,
                snapshot_dir=args.snapshot_dir,
            )
            r = payload["recomputed"]
            s = payload.get("stored") or {}
            ev_s = s.get("explained_variance")
            l0_s = s.get("mean_l0")
            ev_s_str = f"{ev_s:.4f}" if isinstance(ev_s, float) else "n/a"
            l0_s_str = f"{l0_s:.1f}" if isinstance(l0_s, (int, float)) else "n/a"
            print(
                f"  EV new={r['explained_variance']:.4f} stored={ev_s_str}  "
                f"l0 new={r['mean_l0']:.1f} stored={l0_s_str}  runtime={r['runtime_s']}s",
                flush=True,
            )
            rows.append(
                {
                    "path": str(rel),
                    "model": r["model_name"],
                    "preprocess": r["preprocess_mode"],
                    "d_sae": r["d_sae"],
                    "n_snapshots": r["n_snapshots"],
                    "n_rows": r["n_rows"],
                    "ev_new": r["explained_variance"],
                    "ev_stored": s.get("explained_variance"),
                    "l0_new": r["mean_l0"],
                    "l0_stored": s.get("mean_l0"),
                    "dead_new": r["dead_rate"],
                    "dead_stored": s.get("dead_rate"),
                    "mse_new": r["reconstruction_mse"],
                    "mse_stored": s.get("reconstruction_mse"),
                    "runtime_s": r["runtime_s"],
                }
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            rows.append({"path": str(rel), "error": f"{type(e).__name__}: {e}"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
