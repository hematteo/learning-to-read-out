"""Recompute EV/MSE/L0/dead_rate for hf_release ckpts using the canonical formula.

Streams snapshots and weights from CPU/disk so the same script runs on a
24GB Mac (MPS) and a CUDA host.

Architectures handled:
  - jumprelu cross-snap-K  (K∈{1,16,32}; covers final-snap and per-snap as K=1)
  - batchtopk K=32         (W_U/architecture-comparison/d8192/batchtopk)
  - gated K=32             (gated, gated-retuned)

For each ckpt:
  1. Load weights from safetensors (CPU).
  2. Reconstruct preprocess_stats deterministically from raw W_U snapshots.
  3. Phase 1: stream snapshots, accumulate pre_joint = Σ_k(x_k @ W_E[k]) + Σ_k(b_E[k]).
     Per-k transfer of W_E[k] to device; matmul in chunks.
  4. Phase 2: stream snapshots again, compute per-k fires/recon/SS in chunks,
     accumulating canonical SS_res, SS_tot, L0, fires-per-feature, AND the
     training-formula MSE/var/L0 over batches of size training_batch_size.
  5. Report:
     - canonical (single-pass, batch-independent): EV, MSE, L0, dead_rate
     - training-formula at training batch_size: EV (= 1 - Σmse_b / Σvar_b), MSE, L0
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from readout.crosscoder import inference


def reconstruct_preprocess_stats_streaming(snap_paths, slug):
    """Compute per-snap (mean, scale) without loading all K at once.

    Matches preprocess_snapshots(mode='center_scale') from wu_adapter.py.
    """
    means, scales = [], []
    for p in snap_paths:
        x = load_snapshot(p)
        m = x.mean(dim=0, keepdim=True)  # (1, d)
        centered = x - m
        d = x.shape[-1]
        mean_sq = centered.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        s = (mean_sq / d).clamp_min(1e-8).sqrt().squeeze(-1)
        means.append(m.unsqueeze(0))  # (1, 1, d)
        scales.append(s.reshape(1, 1, 1))
    return torch.cat(means, dim=0).squeeze(1), torch.cat(scales, dim=0).squeeze(
        -1
    ).squeeze(-1)


def snap_path_for(snap_dir: Path, model_name: str, matrix: str, step: int) -> Path:
    base = model_name.split("/")[-1]
    slug = model_name.replace("/", "_")
    suffix = "wu" if matrix == "W_U" else "we"
    p = snap_dir / base / f"{slug}_step{step}_{suffix}.pt"
    if p.exists():
        return p
    p2 = snap_dir / f"{slug}_step{step}_{suffix}.pt"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"No snapshot at {p}")


def load_snapshot(p: Path) -> torch.Tensor:
    from readout.crosscoder.snapshots import load_snapshot_at

    return load_snapshot_at(p)


def evaluate_jumprelu(sd, snap_paths, training_batch_size, dev, vchunk=4096):
    """Streaming canonical eval for jumprelu cross-snap-K (works for K=1 too)."""
    K = len(snap_paths)
    W_E = sd["W_E"].float()  # (K, d, D) cpu
    b_E = sd["b_E"].float()  # (K, D) cpu
    W_D = sd["W_D"].float()  # (K, D, d) cpu
    b_D = sd["b_D"].float()  # (K, d) cpu
    log_thr = sd["activation_function.log_jumprelu_threshold"].float()
    thr = log_thr.exp().to(dev)
    dec_norm = W_D.pow(2).sum(dim=2).sqrt().clamp_min(1e-8).float()  # (K, D) cpu
    K_, d_model, D = W_E.shape
    assert K_ == K

    means, scales = reconstruct_preprocess_stats_streaming(snap_paths, "")

    # Phase 1: pre_joint on CPU (large for big D, V)
    V = None
    pre_joint = None
    for k, p in enumerate(snap_paths):
        snap = load_snapshot(p)  # (V, d_model) cpu
        if V is None:
            V = snap.shape[0]
            pre_joint = torch.zeros(V, D, dtype=torch.float32)
        x_k = (snap - means[k]) / scales[k]
        del snap
        W_E_k = W_E[k].to(dev)  # (d_model, D)
        for v0 in range(0, V, vchunk):
            v1 = min(V, v0 + vchunk)
            xc = x_k[v0:v1].to(dev)
            pj = xc @ W_E_k
            pre_joint[v0:v1] += pj.cpu()
            del xc, pj
        del W_E_k, x_k
        gc.collect()
    pre_joint += b_E.sum(dim=0)  # CPU broadcast (D,)

    # Phase 2: per-snap fires/recon/SS, both canonical & training formula.
    ss_res = torch.zeros(K, dtype=torch.float64)
    ss_tot = torch.zeros(K, dtype=torch.float64)
    L0_total = torch.tensor(0.0, dtype=torch.float64)
    fires_per_feat = torch.zeros(D, dtype=torch.float64)
    n_seen = 0
    qq_total_mse = qq_total_var = qq_total_l0 = 0.0
    qq_n_batches = 0

    bs = training_batch_size
    # qq formula iterates over batches of vocab; for each batch, encode runs
    # all K simultaneously. We mirror that: iterate vocab batch first, then K.
    # Each batch needs (B, K, d) of normalized x.
    # Stream snapshots once for ss_tot first (cheap)
    # Actually we can compute ss_tot during phase 2 too.

    # For phase 2 batched eval, precompute normalized snaps in chunks by reloading.
    # Strategy: for each vocab batch, load each snapshot's row-slice into device
    # form a (B, K, d) tensor on device, run encode/decode.

    for v0 in range(0, V, bs):
        v1 = min(V, v0 + bs)
        c = v1 - v0
        # Build x_b shape (c, K, d) on device by streaming snapshots
        x_b = torch.empty(c, K, d_model, device=dev, dtype=torch.float32)
        for k, p in enumerate(snap_paths):
            snap = load_snapshot(p)
            x_b[:, k, :] = ((snap[v0:v1] - means[k]) / scales[k]).to(dev)
            del snap

        pj_b = pre_joint[v0:v1].to(dev)  # (c, D)
        recon = torch.empty(c, K, d_model, device=dev, dtype=torch.float32)
        feat_qq = torch.empty(c, K, D, device=dev, dtype=torch.float32)
        for k in range(K):
            dn_k = dec_norm[k].to(dev)
            acts, fires = inference.jumprelu_feature_acts(pj_b, thr, dn_k)
            feat_qq[:, k, :] = acts
            W_D_k = W_D[k].to(dev)
            b_D_k = b_D[k].to(dev)
            recon[:, k, :] = acts @ W_D_k + b_D_k
            del W_D_k, b_D_k, dn_k
            # Canonical SS
            diff_k = x_b[:, k, :] - recon[:, k, :]
            ss_res[k] += diff_k.pow(2).sum().float().cpu().double()
            ss_tot[k] += x_b[:, k, :].pow(2).sum().float().cpu().double()
            L0_total += fires.sum().float().cpu().double()
            fires_per_feat += fires.float().sum(dim=0).cpu().double()
        n_seen += c * K

        # Training formula
        diff = recon - x_b
        qq_total_mse += diff.pow(2).mean().item()
        qq_total_var += x_b.var().item()
        qq_total_l0 += (feat_qq > 0).float().sum(dim=-1).mean().item()
        qq_n_batches += 1
        del x_b, pj_b, recon, feat_qq, diff
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()

    ev = (1.0 - (ss_res.sum() / ss_tot.sum())).item()
    mse = (ss_res.sum() / (V * K * d_model)).item()
    mean_l0 = (L0_total / (V * K)).item()
    rate = (fires_per_feat / n_seen).numpy()
    dead_rate = float((rate <= 1e-6).mean())
    qq_ev = 1.0 - qq_total_mse / qq_total_var
    return {
        "canon_ev": ev,
        "canon_mse": mse,
        "canon_l0": mean_l0,
        "dead_rate": dead_rate,
        "qq_ev": qq_ev,
        "qq_mse": qq_total_mse / qq_n_batches,
        "qq_l0": qq_total_l0 / qq_n_batches,
        "V": V,
        "K": K,
        "D": D,
        "d_model": d_model,
    }


def evaluate_batchtopk(
    sd, snap_paths, batchtopk_k, training_batch_size, dev, vchunk=4096
):
    K = len(snap_paths)
    W_E = sd["W_E"].float()
    b_E = sd["b_E"].float()
    W_D = sd["W_D"].float()
    b_D = sd["b_D"].float()
    K_, d_model, D = W_E.shape
    assert K_ == K
    means, scales = reconstruct_preprocess_stats_streaming(snap_paths, "")

    V = None
    pre_joint = None
    for k, p in enumerate(snap_paths):
        snap = load_snapshot(p)
        if V is None:
            V = snap.shape[0]
            pre_joint = torch.zeros(V, D, dtype=torch.float32)
        x_k = (snap - means[k]) / scales[k]
        del snap
        W_E_k = W_E[k].to(dev)
        for v0 in range(0, V, vchunk):
            v1 = min(V, v0 + vchunk)
            xc = x_k[v0:v1].to(dev)
            pj = xc @ W_E_k
            pre_joint[v0:v1] += pj.cpu()
            del xc, pj
        del W_E_k, x_k
        gc.collect()
    pre_joint += b_E.sum(dim=0)

    ss_res = torch.zeros(K, dtype=torch.float64)
    ss_tot = torch.zeros(K, dtype=torch.float64)
    L0_total = torch.tensor(0.0, dtype=torch.float64)
    fires_per_feat = torch.zeros(D, dtype=torch.float64)
    n_seen = 0
    qq_total_mse = qq_total_var = qq_total_l0 = 0.0
    qq_n_batches = 0

    bs = training_batch_size
    for v0 in range(0, V, bs):
        v1 = min(V, v0 + bs)
        c = v1 - v0
        x_b = torch.empty(c, K, d_model, device=dev, dtype=torch.float32)
        for k, p in enumerate(snap_paths):
            snap = load_snapshot(p)
            x_b[:, k, :] = ((snap[v0:v1] - means[k]) / scales[k]).to(dev)
            del snap
        pj_b = pre_joint[v0:v1].to(dev)
        relud = torch.relu(pj_b)
        flat = relud.reshape(-1)
        k_total = min(c * batchtopk_k, flat.numel())
        _, idx = flat.topk(k_total, sorted=False)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[idx] = True
        feat = (flat * mask).reshape_as(relud)  # (c, D)
        recon = torch.empty(c, K, d_model, device=dev, dtype=torch.float32)
        for k in range(K):
            W_D_k = W_D[k].to(dev)
            b_D_k = b_D[k].to(dev)
            recon[:, k, :] = feat @ W_D_k + b_D_k
            diff_k = x_b[:, k, :] - recon[:, k, :]
            ss_res[k] += diff_k.pow(2).sum().float().cpu().double()
            ss_tot[k] += x_b[:, k, :].pow(2).sum().float().cpu().double()
            del W_D_k, b_D_k
        fires = feat > 0  # (c, D) — shared across heads
        L0_total += fires.sum().float().cpu().double() * K
        fires_per_feat += fires.float().sum(dim=0).cpu().double() * K
        n_seen += c * K
        diff = recon - x_b
        qq_total_mse += diff.pow(2).mean().item()
        qq_total_var += x_b.var().item()
        qq_total_l0 += fires.float().sum(dim=-1).mean().item()
        qq_n_batches += 1
        del x_b, pj_b, recon, feat, fires, relud, flat, mask, idx
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()

    ev = (1.0 - (ss_res.sum() / ss_tot.sum())).item()
    mse = (ss_res.sum() / (V * K * d_model)).item()
    mean_l0 = (L0_total / (V * K)).item()
    rate = (fires_per_feat / n_seen).numpy()
    dead_rate = float((rate <= 1e-6).mean())
    qq_ev = 1.0 - qq_total_mse / qq_total_var
    return {
        "canon_ev": ev,
        "canon_mse": mse,
        "canon_l0": mean_l0,
        "dead_rate": dead_rate,
        "qq_ev": qq_ev,
        "qq_mse": qq_total_mse / qq_n_batches,
        "qq_l0": qq_total_l0 / qq_n_batches,
        "V": V,
        "K": K,
        "D": D,
        "d_model": d_model,
    }


def evaluate_gated(sd, snap_paths, training_batch_size, dev, vchunk=4096):
    K = len(snap_paths)
    W_E = sd["W_E"].float()
    b_E = sd["b_E"].float()
    W_D = sd["W_D"].float()
    b_D = sd["b_D"].float()
    b_gate = sd["activation_function.b_gate"].to(dev).float()
    b_mag = sd["activation_function.b_mag"].to(dev).float()
    r_mag = sd["activation_function.r_mag"].to(dev).float()
    K_, d_model, D = W_E.shape
    assert K_ == K
    means, scales = reconstruct_preprocess_stats_streaming(snap_paths, "")

    V = None
    pre_joint = None
    for k, p in enumerate(snap_paths):
        snap = load_snapshot(p)
        if V is None:
            V = snap.shape[0]
            pre_joint = torch.zeros(V, D, dtype=torch.float32)
        x_k = (snap - means[k]) / scales[k]
        del snap
        W_E_k = W_E[k].to(dev)
        for v0 in range(0, V, vchunk):
            v1 = min(V, v0 + vchunk)
            xc = x_k[v0:v1].to(dev)
            pj = xc @ W_E_k
            pre_joint[v0:v1] += pj.cpu()
            del xc, pj
        del W_E_k, x_k
        gc.collect()
    pre_joint += b_E.sum(dim=0)

    ss_res = torch.zeros(K, dtype=torch.float64)
    ss_tot = torch.zeros(K, dtype=torch.float64)
    L0_total = torch.tensor(0.0, dtype=torch.float64)
    fires_per_feat = torch.zeros(D, dtype=torch.float64)
    n_seen = 0
    qq_total_mse = qq_total_var = qq_total_l0 = 0.0
    qq_n_batches = 0

    bs = training_batch_size
    for v0 in range(0, V, bs):
        v1 = min(V, v0 + bs)
        c = v1 - v0
        x_b = torch.empty(c, K, d_model, device=dev, dtype=torch.float32)
        for k, p in enumerate(snap_paths):
            snap = load_snapshot(p)
            x_b[:, k, :] = ((snap[v0:v1] - means[k]) / scales[k]).to(dev)
            del snap
        pj_b = pre_joint[v0:v1].to(dev)
        gate = (pj_b + b_gate > 0).float()
        magnitude = torch.relu(r_mag.exp() * pj_b + b_mag)
        feat = gate * magnitude  # (c, D)
        recon = torch.empty(c, K, d_model, device=dev, dtype=torch.float32)
        for k in range(K):
            W_D_k = W_D[k].to(dev)
            b_D_k = b_D[k].to(dev)
            recon[:, k, :] = feat @ W_D_k + b_D_k
            diff_k = x_b[:, k, :] - recon[:, k, :]
            ss_res[k] += diff_k.pow(2).sum().float().cpu().double()
            ss_tot[k] += x_b[:, k, :].pow(2).sum().float().cpu().double()
            del W_D_k, b_D_k
        fires = feat > 0
        L0_total += fires.sum().float().cpu().double() * K
        fires_per_feat += fires.float().sum(dim=0).cpu().double() * K
        n_seen += c * K
        diff = recon - x_b
        qq_total_mse += diff.pow(2).mean().item()
        qq_total_var += x_b.var().item()
        qq_total_l0 += fires.float().sum(dim=-1).mean().item()
        qq_n_batches += 1
        del x_b, pj_b, recon, feat, fires, gate, magnitude
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()

    ev = (1.0 - (ss_res.sum() / ss_tot.sum())).item()
    mse = (ss_res.sum() / (V * K * d_model)).item()
    mean_l0 = (L0_total / (V * K)).item()
    rate = (fires_per_feat / n_seen).numpy()
    dead_rate = float((rate <= 1e-6).mean())
    qq_ev = 1.0 - qq_total_mse / qq_total_var
    return {
        "canon_ev": ev,
        "canon_mse": mse,
        "canon_l0": mean_l0,
        "dead_rate": dead_rate,
        "qq_ev": qq_ev,
        "qq_mse": qq_total_mse / qq_n_batches,
        "qq_l0": qq_total_l0 / qq_n_batches,
        "V": V,
        "K": K,
        "D": D,
        "d_model": d_model,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--snap-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument(
        "--device",
        default="mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    ap.add_argument("--filter", default=None)
    ap.add_argument(
        "--max-d-sae",
        type=int,
        default=None,
        help="skip ckpts with d_sae > this (memory cap)",
    )
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(f"Device: {dev}")
    root = Path(args.ckpt_root)
    snap_dir = Path(args.snap_dir)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done = set()
    if out.exists():
        rows = list(csv.DictReader(out.open()))
        done = {r["ckpt"] for r in rows}

    fieldnames = [
        "ckpt",
        "matrix",
        "model",
        "K",
        "d_sae",
        "d_model",
        "V",
        "act_fn",
        "training_batch_size",
        "canon_ev",
        "canon_mse",
        "canon_l0",
        "dead_rate",
        "qq_ev",
        "qq_mse",
        "qq_l0",
        "stored_ev",
        "stored_mse",
        "stored_l0",
        "stored_dead_rate",
        "diff_qq_ev",
        "diff_canon_ev",
        "elapsed_s",
    ]

    safetensors_files = sorted(root.rglob("*.safetensors"))
    safetensors_files = [p for p in safetensors_files if not p.name.startswith("._")]
    if args.filter:
        safetensors_files = [p for p in safetensors_files if args.filter in str(p)]
    print(f"found {len(safetensors_files)} ckpts; {len(done)} already done")

    def write_csv():
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)

    for i, st_path in enumerate(safetensors_files):
        rel = str(st_path.relative_to(root))
        if rel in done:
            print(f"[{i + 1}/{len(safetensors_files)}] {rel}: skip (done)", flush=True)
            continue
        cfg_path = st_path.parent / (st_path.stem + ".config.json")
        if not cfg_path.exists():
            print(f"  no config for {rel}, skip")
            continue
        cfg = json.loads(cfg_path.read_text())
        model_name = cfg["model_name"]
        steps = cfg["steps"]
        K = len(steps)
        config = cfg.get("config", {})
        training = cfg.get("training", {})
        q_stored = cfg.get("quality", {})
        train_bs = training.get("batch_size", 1024)
        matrix = "W_E" if "/W_E/" in str(st_path) else "W_U"
        act = config.get("act_fn") or config.get("arch")
        d_sae = q_stored.get("d_sae") or config.get("d_sae")
        if act not in ("jumprelu", "batchtopk", "gated"):
            print(f"[{i + 1}] {rel}: unknown arch '{act}', skipping")
            continue
        if args.max_d_sae and d_sae and d_sae > args.max_d_sae:
            print(f"[{i + 1}] {rel}: d_sae {d_sae} > max {args.max_d_sae}, skipping")
            continue

        try:
            snap_paths = [snap_path_for(snap_dir, model_name, matrix, s) for s in steps]
        except FileNotFoundError as e:
            print(f"[{i + 1}] {rel}: SNAPSHOTS MISSING ({e}), skipping")
            continue

        print(
            f"[{i + 1}/{len(safetensors_files)}] {rel} arch={act} K={K} d_sae={d_sae} train_bs={train_bs}",
            flush=True,
        )
        sd = load_file(str(st_path))

        t0 = time.time()
        try:
            if act == "jumprelu":
                r = evaluate_jumprelu(sd, snap_paths, train_bs, dev)
            elif act == "batchtopk":
                r = evaluate_batchtopk(
                    sd, snap_paths, config.get("batchtopk_k", 64), train_bs, dev
                )
            elif act == "gated":
                r = evaluate_gated(sd, snap_paths, train_bs, dev)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            continue
        elapsed = time.time() - t0

        sev = q_stored.get("explained_variance")
        smse = q_stored.get("reconstruction_mse")
        sl0 = q_stored.get("mean_l0")
        sdr = q_stored.get("dead_rate")
        row = {
            "ckpt": rel,
            "matrix": matrix,
            "model": model_name,
            "K": K,
            "d_sae": r["D"],
            "d_model": r["d_model"],
            "V": r["V"],
            "act_fn": act,
            "training_batch_size": train_bs,
            "canon_ev": f"{r['canon_ev']:.6f}",
            "canon_mse": f"{r['canon_mse']:.6f}",
            "canon_l0": f"{r['canon_l0']:.4f}",
            "dead_rate": f"{r['dead_rate']:.6f}",
            "qq_ev": f"{r['qq_ev']:.6f}",
            "qq_mse": f"{r['qq_mse']:.6f}",
            "qq_l0": f"{r['qq_l0']:.4f}",
            "stored_ev": "" if sev is None else f"{sev}",
            "stored_mse": "" if smse is None else f"{smse}",
            "stored_l0": "" if sl0 is None else f"{sl0}",
            "stored_dead_rate": "" if sdr is None else f"{sdr}",
            "diff_qq_ev": "" if sev is None else f"{r['qq_ev'] - sev:+.6f}",
            "diff_canon_ev": "" if sev is None else f"{r['canon_ev'] - sev:+.6f}",
            "elapsed_s": f"{elapsed:.1f}",
        }
        rows.append(row)
        write_csv()
        diff_str = f" (diff_qq_ev={(r['qq_ev'] - sev):+.4f})" if sev is not None else ""
        print(
            f"  canon EV={r['canon_ev']:.4f} L0={r['canon_l0']:.1f} dead={r['dead_rate']:.4f}; "
            f"qq EV={r['qq_ev']:.4f}{diff_str}; ({elapsed:.1f}s)",
            flush=True,
        )
        del sd
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()

    print("done")


if __name__ == "__main__":
    main()
