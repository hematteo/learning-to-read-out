"""Extract per-snapshot per-feature firing-rate matrices from saved crosscoders.

Direct-tensor path: uses W_E, b_E, JumpReLU threshold, and preprocess_stats
from the checkpoint. Bypasses the Crosscoder class to avoid the
prepare_input/preprocess mismatch in the legacy permutation test.

The CLI defaults to ``compute_rates_canonical`` (joint pre-activation across
all K heads, with per-head decoder-norm-effective threshold ``thr / dec_norm[k]``),
matching ``llamascopium.Crosscoder.encode``. Use ``--legacy-per-head`` to
opt back into the old per-head pre-activation against the global threshold
for diagnostic comparisons.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .checkpoints import unwrap_ckpt


def _maybe_compute_preprocess_stats(stats, mode, slug, snap_dir, steps, device="cpu"):
    """Return preprocess stats compatible with extract_rates.

    If ``stats`` is non-None, return it unchanged. Otherwise compute the
    per-snapshot mean/scale on the fly using the same formulas as
    ``preprocess_snapshots`` in ``wu_adapter.py`` (mode='center_scale' only).
    Returns None if mode is 'none' or unrecognized.
    """
    if stats is not None:
        return stats
    if mode in (None, "none"):
        return None
    if mode not in ("center", "center_scale"):
        raise ValueError(f"Unknown preprocess mode for stats reconstruction: {mode}")
    means: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for i, step in enumerate(steps):
        x = torch.load(Path(snap_dir) / f"{slug}_step{step}_wu.pt", map_location="cpu").float()
        m = x.mean(dim=0, keepdim=True)  # (1, d)
        centered = x - m
        if mode == "center":
            s = torch.ones(1, 1, dtype=x.dtype)
        else:  # center_scale
            d = x.shape[-1]
            mean_sq = centered.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
            s = (mean_sq / d).clamp_min(1e-8).sqrt().squeeze(-1)
        means.append(m.unsqueeze(0))  # (1, 1, d)
        scales.append(s.reshape(1, 1, 1))
    return {
        "mean": torch.cat(means, dim=0),  # (K, 1, d)
        "scale": torch.cat(scales, dim=0),  # (K, 1, 1)
        "mode": mode,
    }


def compute_rates(ckpt_path, snap_dir, device="cpu"):
    ck = unwrap_ckpt(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    sd = ck["state_dict"]
    steps = ck["steps"]
    model_name = ck["model_name"]
    slug = model_name.replace("/", "_")
    W_E = sd["W_E"]  # (K, d, D)
    b_E = sd["b_E"]  # (K, D)
    thr = sd["activation_function.log_jumprelu_threshold"].exp()  # (D,)
    stats = _maybe_compute_preprocess_stats(
        ck.get("preprocess_stats"),
        ck.get("preprocess_mode"),
        slug,
        snap_dir,
        steps,
        device=device,
    )
    K, d, D = W_E.shape
    rates = torch.zeros(K, D)
    for i, step in enumerate(steps):
        x = torch.load(snap_dir / f"{slug}_step{step}_wu.pt", map_location="cpu").float()
        if stats is not None:
            x = (x - stats["mean"][i].squeeze(0)) / stats["scale"][i].squeeze()
        pre = (x.to(device) @ W_E[i].to(device) + b_E[i].to(device)).cpu()
        rates[i] = (pre > thr).float().mean(dim=0)
    return rates, list(steps)


def compute_rates_canonical(ckpt_path, snap_dir, device="cpu"):
    """Per-snapshot per-feature firing rate under the model's canonical encode.

    Matches llamascopium.Crosscoder.encode (sparsity_include_decoder_norm=True):
        joint_pre[v, j]   = sum_k (x[k, v] @ W_E[k, :, j] + b_E[k, j])
        dec_norm[k, j]    = ||W_D[k, j, :]||
        fires[v, k, j]    = (joint_pre[v, j] * dec_norm[k, j]) > thr[j]
        rate[k, j]        = fires[:, k, j].mean()

    joint_pre is shared across heads; per-head variation comes through
    decoder norm. Inputs are preprocessed identically to training.
    """
    ck = unwrap_ckpt(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    sd = ck["state_dict"]
    steps = ck["steps"]
    model_name = ck["model_name"]
    slug = model_name.replace("/", "_")
    W_E = sd["W_E"]  # (K, d, D)
    b_E = sd["b_E"]  # (K, D)
    W_D = sd["W_D"]  # (K, D, d)
    thr = sd["activation_function.log_jumprelu_threshold"].exp()  # (D,)
    stats = _maybe_compute_preprocess_stats(
        ck.get("preprocess_stats"),
        ck.get("preprocess_mode"),
        slug,
        snap_dir,
        steps,
        device=device,
    )
    K, d, D = W_E.shape

    dec_norm = torch.linalg.norm(W_D, dim=-1).to(device)  # (K, D)

    joint_pre = None
    for i, step in enumerate(steps):
        x = torch.load(snap_dir / f"{slug}_step{step}_wu.pt", map_location="cpu").float()
        if stats is not None:
            x = (x - stats["mean"][i].squeeze(0)) / stats["scale"][i].squeeze()
        head_pre = x.to(device) @ W_E[i].to(device) + b_E[i].to(device)  # (V, D)
        joint_pre = head_pre if joint_pre is None else joint_pre + head_pre

    thr_dev = thr.to(device)  # (D,)
    rates = torch.zeros(K, D, device=device)
    for k in range(K):
        thr_eff = thr_dev / dec_norm[k].clamp_min(1e-12)  # (D,)
        rates[k] = (joint_pre > thr_eff).float().mean(dim=0)
    return rates.cpu(), list(steps)


def rate_rotation_r(ckpt_path, rates):
    """Pearson r between |Δrate| and (1 - cos(W_D[i], W_D[i+1])) over all features
    pooled across consecutive snapshot pairs (matches run5_1b_sweep_analysis.py)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    W_D = ck["state_dict"]["W_D"].cpu().numpy()  # (K, D, d)
    rates_np = rates.cpu().numpy() if isinstance(rates, torch.Tensor) else np.asarray(rates)
    K = rates_np.shape[0]
    rate_changes, rotations = [], []
    for i in range(K - 1):
        d_rate = np.abs(rates_np[i + 1] - rates_np[i])
        v1, v2 = W_D[i], W_D[i + 1]
        cos = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-12)
        rate_changes.append(d_rate)
        rotations.append(1.0 - cos)
    rate_all = np.concatenate(rate_changes)
    rot_all = np.concatenate(rotations)
    if rate_all.std() == 0 or rot_all.std() == 0:
        return float("nan")
    return float(np.corrcoef(rate_all, rot_all)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--append-manifest",
        action="store_true",
        default=False,
        help="Update results/crosscoder_sweep_manifest.csv with rate_rotation_r per ckpt.",
    )
    ap.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("results/crosscoder_sweep_manifest.csv"),
        help="Manifest CSV path (only used when --append-manifest is set).",
    )
    ap.add_argument(
        "--legacy-per-head",
        action="store_true",
        default=False,
        help="Diagnostic-only: use the legacy per-head pre-activation against "
        "the global threshold (compute_rates) instead of the canonical joint "
        "pre-activation with per-head decoder-norm-effective threshold "
        "(compute_rates_canonical).",
    )
    args = ap.parse_args()

    rate_fn = compute_rates if args.legacy_per_head else compute_rates_canonical
    print(f"Using rate function: {rate_fn.__name__}", flush=True)

    rates_per_seed = {}
    rotation_rs: dict[Path, float] = {}
    steps: list[int] = []
    for i, ckpt in enumerate(args.ckpts):
        print(f"[{i + 1}/{len(args.ckpts)}] {ckpt.name}...", flush=True)
        rates, ckpt_steps = rate_fn(ckpt, args.cache_dir, device=args.device)
        # rates_per_seed shares a single steps axis in the saved file, so every
        # checkpoint passed in one invocation must use the same schedule.
        if steps and list(ckpt_steps) != list(steps):
            raise ValueError(
                f"{ckpt.name} has step schedule {list(ckpt_steps)} but a prior "
                f"checkpoint had {list(steps)}; all --ckpts must share one schedule."
            )
        steps = ckpt_steps
        rates_per_seed[ckpt.stem] = rates
        r = rate_rotation_r(ckpt, rates)
        rotation_rs[ckpt] = r
        # Reference snapshot for the diagnostic print: prefer step 1000 when the
        # schedule includes it, else fall back to the first snapshot.
        ref_idx = steps.index(1000) if 1000 in steps else 0
        ref_label = "step1000" if 1000 in steps else f"step{steps[0]}"
        print(
            f"  -> {rates.shape}, mean={rates.mean().item():.4f}, "
            f"{ref_label}_idx_rate={rates[ref_idx].mean().item():.4f}, "
            f"rate_rotation_r={r:.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rates_per_seed": rates_per_seed, "steps": steps}, args.output)
    print(f"Saved to {args.output}")

    if args.append_manifest:
        from readout.crosscoder.manifest import (
            MANIFEST_HEADER,
            _atomic_write,
            _ensure_manifest_dir,
            _fmt,
            _read_existing,
            update_or_insert,
        )

        manifest_csv = Path(args.manifest_path)
        _ensure_manifest_dir(manifest_csv, strict=True)
        header, rows = _read_existing(manifest_csv)
        if header != MANIFEST_HEADER:
            merged = list(MANIFEST_HEADER)
            for c in header:
                if c not in merged:
                    merged.append(c)
            header = merged
        else:
            header = list(MANIFEST_HEADER)
        for ckpt, r in rotation_rs.items():
            new_row = {
                "path": str(Path(ckpt).resolve()),
                "rate_rotation_r": _fmt(r),
            }
            rows, action = update_or_insert(rows, new_row, key="path")
            print(f"Manifest {action}: {ckpt.name} (rate_rotation_r={r:.4f})")
        _atomic_write(manifest_csv, header, rows)
