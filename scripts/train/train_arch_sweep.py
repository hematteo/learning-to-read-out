"""SAE architecture sweep at d_sae=8192 on Pythia-160M W_U.

Trains one of {jumprelu, batchtopk, gated} on the Ge-exact 32-step W_U
snapshot cache and saves a checkpoint compatible with extract_rates.py
(JumpReLU) or the arch-sweep rate plotter (BatchTopK / Gated).

Usage:
  uv run python scripts/train/train_arch_sweep.py --arch jumprelu --output ...
  uv run python scripts/train/train_arch_sweep.py --arch batchtopk --output ...
  uv run python scripts/train/train_arch_sweep.py --arch gated --output ...

Use --dry-run for the validation pass: 4 snapshots, 2 epochs, batch_size=256.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import argparse
import subprocess
from pathlib import Path

import torch

from src.core.paths import ssd_path, ssd_root

# Repo root is already on sys.path (inserted above); ROOT is the git rev-parse cwd.
ROOT = Path(__file__).resolve().parents[2]

from src.crosscoder.crosscoder_arch_sweep import (  # noqa: E402
    quick_quality,
    train_arch_sweep,
)
from src.crosscoder.wu_adapter import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_STEPS,
    load_snapshots,
    preprocess_snapshots,
)

DEFAULT_OUTPUT_DIR = ssd_path("wu_crosscoder", "arch_sweep")


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["jumprelu", "batchtopk", "gated"], required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Tiny run: 4 snapshots, 2 epochs, bs=256")
    ap.add_argument("--d-sae", type=int, default=8192)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--input-preprocess",
        choices=["none", "center", "center_scale"],
        default="center_scale",
        help="Match the production 160M W_U run's default ('center_scale'); see experiments.yaml: crosscoder_main.",
    )
    # Schedule (mirror the production 160M W_U run: l1_warmup=0.1, no LR warmup/decay).
    ap.add_argument("--l1-warmup-fraction", type=float, default=0.1)
    ap.add_argument("--lr-warmup-fraction", type=float, default=0.0)
    ap.add_argument("--lr-decay-fraction", type=float, default=0.0)
    # JumpReLU
    ap.add_argument("--init-threshold", type=float, default=0.1)
    ap.add_argument("--bandwidth", type=float, default=4.0)
    ap.add_argument("--l1-coefficient", type=float, default=0.3)
    ap.add_argument("--frequency-scale", type=float, default=0.01)
    ap.add_argument("--tanh-stretch", type=float, default=1.0)
    ap.add_argument("--jumprelu-lr-factor", type=float, default=0.1)
    # BatchTopK: 203 matches the production 160M W_U run's seed-0 mean_l0
    # (≈203), so the comparison controls for actual sparsity rather than
    # nominal K. CLI override available.
    ap.add_argument("--batchtopk-k", type=int, default=203)
    # Gated
    ap.add_argument("--gated-l1-coefficient", type=float, default=0.3)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    # SSD check (large outputs go to ${UM_SSD_ROOT}).
    output = args.output or (DEFAULT_OUTPUT_DIR / f"{args.arch}_dsae{args.d_sae}_seed{args.seed}.pt")
    if str(output).startswith(str(ssd_root())) and not ssd_root().is_dir():
        raise SystemExit("ERROR: external SSD ${UM_SSD_ROOT} is not mounted. Attach and retry.")

    # Determinism (fp32 only; no bf16; hard stop rule).
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(False)  # einsum + topk on CUDA aren't fully deterministic; seed RNG only.

    if args.dry_run:
        steps = DEFAULT_STEPS[:4]
        n_epochs = 2
        batch_size = 256
        d_sae = min(args.d_sae, 256)  # tiny d_sae for dry-run speed
        print(
            f"DRY-RUN: arch={args.arch} steps={steps} epochs={n_epochs} bs={batch_size} d_sae={d_sae}",
            flush=True,
        )
    else:
        steps = list(DEFAULT_STEPS)
        n_epochs = args.n_epochs
        batch_size = args.batch_size
        d_sae = args.d_sae

    snapshots, resolved_steps = load_snapshots(
        model_name=args.model,
        steps=steps,
        cache_dir=args.cache_dir,
        dtype=torch.float32,
    )
    print(f"Snapshots loaded: {tuple(snapshots.shape)} (K, V, d)", flush=True)

    snapshots, preprocess_stats = preprocess_snapshots(snapshots, mode=args.input_preprocess)

    # For BatchTopK on the dry-run, clamp k below d_sae.
    batchtopk_k = min(args.batchtopk_k, d_sae // 2)

    model = train_arch_sweep(
        snapshots,
        arch=args.arch,
        d_sae=d_sae,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        init_threshold=args.init_threshold,
        bandwidth=args.bandwidth,
        l1_coefficient=args.l1_coefficient,
        frequency_scale=args.frequency_scale,
        tanh_stretch=args.tanh_stretch,
        jumprelu_lr_factor=args.jumprelu_lr_factor,
        batchtopk_k=batchtopk_k,
        gated_l1_coefficient=args.gated_l1_coefficient,
        lr_warmup_fraction=args.lr_warmup_fraction,
        lr_decay_fraction=args.lr_decay_fraction,
        l1_warmup_fraction=args.l1_warmup_fraction,
    )

    metrics = quick_quality(model, snapshots, batch_size=batch_size, device=args.device)
    print(f"Quality: {metrics}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "arch": args.arch,
            "d_sae": d_sae,
            "n_snapshots": snapshots.shape[0],
            "d_model": snapshots.shape[-1],
            "batchtopk_k": batchtopk_k if args.arch == "batchtopk" else None,
        },
        "steps": resolved_steps,
        "model_name": args.model,
        "seed": args.seed,
        "quality": metrics,
        "training": {
            "lr": args.lr,
            "l1_coefficient": args.l1_coefficient,
            "gated_l1_coefficient": args.gated_l1_coefficient,
            "init_threshold": args.init_threshold,
            "bandwidth": args.bandwidth,
            "frequency_scale": args.frequency_scale,
            "tanh_stretch": args.tanh_stretch,
            "jumprelu_lr_factor": args.jumprelu_lr_factor,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "input_preprocess": args.input_preprocess,
            "l1_warmup_fraction": args.l1_warmup_fraction,
            "lr_warmup_fraction": args.lr_warmup_fraction,
            "lr_decay_fraction": args.lr_decay_fraction,
            "git_hash": _git_hash(),
            "amp_dtype": "fp32",
        },
        "preprocess_stats": preprocess_stats,
    }
    torch.save(payload, output)
    print(f"Saved to {output}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
