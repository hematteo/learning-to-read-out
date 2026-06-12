"""Per-snapshot JumpReLU SAEs on Pythia-160M W_U (per-snapshot SAE baseline).

Trains 32 single-snapshot SAEs at d_sae=8192, one per Ge-exact training step,
mirroring the Run 3 crosscoder hyperparameters as closely as possible
(JumpReLU + tanh-quad sparsity per Ge eq. 9, decoder-norm-aware threshold,
fp32, decoder-transpose encoder init, center_scale persnap preprocessing).

A "single-snapshot SAE" here is a llamascopium Crosscoder with K=1 hook
points: this gives bit-perfect parity with Run 3's loss/init/optimizer
plumbing, so the persnap-vs-crosscoder comparison is apples-to-apples.

Outputs:
  ${UM_SSD_ROOT}/wu_crosscoder/per_snap_sae/wu_sae_dsae8192_step{step}.pt
"""

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[4]))
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from src.core.paths import ssd_path, ssd_root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.crosscoder.wu_adapter import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_STEPS,
    extract_wu,
    preprocess_snapshots,
    quick_quality,
    train,
)

DEFAULT_OUTPUT_DIR = ssd_path("wu_crosscoder", "per_snap_sae")


def git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def require_external_ssd(path: Path) -> None:
    mount = ssd_root()
    if str(path).startswith(str(mount)) and not mount.exists():
        raise RuntimeError(
            f"External SSD {mount} is not mounted. Attach the drive (T1.3 requires"
            f" persistent storage at {path})."
        )


def train_one_snapshot(
    step: int,
    *,
    model_name: str,
    cache_dir: Path,
    output_dir: Path,
    expansion_factor: float,
    lr: float,
    l1_coefficient: float,
    frequency_scale: float,
    tanh_stretch: float,
    jumprelu_lr_factor: float,
    init_threshold: float,
    n_epochs: int,
    batch_size: int,
    lr_warmup_fraction: float,
    lr_decay_fraction: float,
    decoder_transpose_init: float,
    seed: int,
    device: str,
    log_every: int,
    overwrite: bool,
) -> dict:
    output_path = output_dir / f"wu_sae_dsae8192_step{step}.pt"
    if output_path.exists() and not overwrite:
        print(
            f"[step {step}] exists, skipping ({output_path.name}); use --overwrite to retrain"
        )
        return {"step": step, "skipped": True, "path": str(output_path)}

    print(f"[step {step}] loading W_U from {cache_dir}")
    W_U = extract_wu(model_name, step, cache_dir, dtype=torch.float32)  # (V, d)
    snap = W_U.unsqueeze(0)  # (1, V, d)  — K=1 single-snapshot adapter
    snap_proc, preprocess_stats = preprocess_snapshots(snap, mode="center_scale")

    if preprocess_stats is not None:
        mean_norm = preprocess_stats["mean"].norm(dim=-1).item()
        scale = preprocess_stats["scale"].item()
        print(
            f"[step {step}] preprocess center_scale: mean-shift={mean_norm:.3f} scale={scale:.4f}"
        )

    t0 = time.perf_counter()
    sae = train(
        snap_proc,
        expansion_factor=expansion_factor,
        lr=lr,
        l1_coefficient=l1_coefficient,
        frequency_scale=frequency_scale,
        tanh_stretch_coefficient=tanh_stretch,
        jumprelu_lr_factor=jumprelu_lr_factor,
        init_threshold=init_threshold,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
        seed=seed,
        log_every=log_every,
        amp_dtype=None,  # fp32 only — bf16+tanh-STE diverges (HANDOFF hazard #3)
        use_sparse_adam=False,
        init_encoder_with_decoder_transpose_factor=decoder_transpose_init,
        l1_warmup_fraction=0.1,
        lr_warmup_fraction=lr_warmup_fraction,
        lr_decay_fraction=lr_decay_fraction,
    )
    elapsed = time.perf_counter() - t0

    metrics = quick_quality(sae, snap_proc, batch_size=batch_size, device=device)
    print(
        f"[step {step}] EV={metrics['explained_variance']:.4f} L0={metrics['mean_l0']:.1f} ({elapsed:.1f}s)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "config": sae.cfg.model_dump(),
            "step": step,
            "steps": [step],
            "model_name": model_name,
            "seed": seed,
            "quality": metrics,
            "training": {
                "lr": lr,
                "l1_coefficient": l1_coefficient,
                "frequency_scale": frequency_scale,
                "tanh_stretch_coefficient": tanh_stretch,
                "jumprelu_lr_factor": jumprelu_lr_factor,
                "init_threshold": init_threshold,
                "decoder_transpose_init": decoder_transpose_init,
                "amp_dtype": "fp32",
                "optimizer": "adam",
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "expansion_factor": expansion_factor,
                "d_sae": sae.cfg.d_sae,
                "input_preprocess": "center_scale",
                "l1_warmup_fraction": 0.1,
                "lr_warmup_fraction": lr_warmup_fraction,
                "lr_decay_fraction": lr_decay_fraction,
                "elapsed_seconds": elapsed,
            },
            "preprocess_stats": preprocess_stats,
            "git_hash": git_hash(),
        },
        output_path,
    )
    print(f"[step {step}] saved {output_path}")
    return {
        "step": step,
        "path": str(output_path),
        "ev": metrics["explained_variance"],
        "l0": metrics["mean_l0"],
        "elapsed": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Subset of Ge steps to train; default = all 32.",
    )
    parser.add_argument(
        "--expansion-factor",
        type=float,
        default=10.6667,
        help="d_sae / d_model. 10.6667 -> d_sae=8192 at d_model=768.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--l1-coefficient", type=float, default=0.3)
    parser.add_argument("--frequency-scale", type=float, default=0.01)
    parser.add_argument(
        "--tanh-stretch",
        type=float,
        default=1.0,
        help="Matches Run 3 tanh_stretch_coefficient=1.0 exactly.",
    )
    parser.add_argument("--jumprelu-lr-factor", type=float, default=0.1)
    parser.add_argument("--init-threshold", type=float, default=0.1)
    parser.add_argument("--decoder-transpose-init", type=float, default=1.0)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr-warmup-fraction", type=float, default=0.1)
    parser.add_argument("--lr-decay-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Smoke run: 2 snapshots (step 0 and step 143000), 2 epochs, batch 256.",
    )
    args = parser.parse_args()

    require_external_ssd(args.output_dir)

    print(f"git_hash={git_hash()} torch={torch.__version__} device={args.device}")
    print(f"output_dir={args.output_dir}")
    print(f"cache_dir={args.cache_dir}")

    if args.dry_run:
        steps = [0, 143000]
        n_epochs = 2
        batch_size = 256
        log_every = 5
        print(f"DRY-RUN: steps={steps} n_epochs={n_epochs} batch={batch_size}")
    else:
        steps = list(args.steps) if args.steps else list(DEFAULT_STEPS)
        n_epochs = args.n_epochs
        batch_size = args.batch_size
        log_every = args.log_every

    torch.manual_seed(args.seed)

    t_total = time.perf_counter()
    summary = []
    for step in steps:
        info = train_one_snapshot(
            step,
            model_name=args.model,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            expansion_factor=args.expansion_factor,
            lr=args.lr,
            l1_coefficient=args.l1_coefficient,
            frequency_scale=args.frequency_scale,
            tanh_stretch=args.tanh_stretch,
            jumprelu_lr_factor=args.jumprelu_lr_factor,
            init_threshold=args.init_threshold,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr_warmup_fraction=args.lr_warmup_fraction,
            lr_decay_fraction=args.lr_decay_fraction,
            decoder_transpose_init=args.decoder_transpose_init,
            seed=args.seed,
            device=args.device,
            log_every=log_every,
            overwrite=args.overwrite,
        )
        summary.append(info)

    total = time.perf_counter() - t_total
    print(f"\n=== T1.3 done: {len(summary)} snapshots in {total / 60:.1f} min ===")
    for info in summary:
        if info.get("skipped"):
            print(f"  step {info['step']:>7}: skipped ({info['path']})")
        else:
            print(
                f"  step {info['step']:>7}: EV={info['ev']:.4f} L0={info['l0']:.1f} "
                f"({info['elapsed']:.1f}s) -> {os.path.basename(info['path'])}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
