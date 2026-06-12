"""Train a Pythia W_U crosscoder on an OLMo-like late-start schedule.

The paper-facing run (appendix checkpoint-window control) is Pythia-1B; the
script also works for other Pythia sizes (d_model is read from the snapshots).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[4]))
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.paths import ssd_path
from src.crosscoder import wu_adapter

LATE_START_STEPS_32 = [
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
    10000,
    13000,
    14000,
    21000,
    23000,
    27000,
    33000,
    34000,
    43000,
    47000,
    53000,
    61000,
    73000,
    83000,
    89000,
    93000,
    102000,
    113000,
    123000,
    130000,
    143000,
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Defaults to ${UM_SSD_ROOT}/snapshots/<model short name>.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to results/experiments/olmo_matched_checkpoint_window/...",
    )
    ap.add_argument("--d-sae", type=int, default=24576)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--l1-coefficient", type=float, default=0.3)
    ap.add_argument("--tanh-stretch", type=float, default=1.0)
    ap.add_argument("--jumprelu-lr-factor", type=float, default=0.1)
    ap.add_argument("--init-threshold", type=float, default=0.1)
    ap.add_argument("--l1-warmup-fraction", type=float, default=0.1)
    ap.add_argument("--lr-warmup-fraction", type=float, default=0.10)
    ap.add_argument("--lr-decay-fraction", type=float, default=0.20)
    ap.add_argument("--optimizer", choices=["adam", "sparse_adam"], default="adam")
    ap.add_argument(
        "--amp-dtype",
        choices=["fp32", "bf16"],
        default="bf16",  # run of record (pythia-1b late-start) trained with bf16 amp
    )
    ap.add_argument("--n-epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
    )
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--append-manifest", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    model_short = args.model.split("/")[-1]  # e.g. "pythia-1b"
    model_tag = model_short.replace("-", "")  # e.g. "pythia1b"
    if args.cache_dir is None:
        args.cache_dir = ssd_path("snapshots", model_short)
    if args.output is None:
        args.output = (
            Path("results/experiments/olmo_matched_checkpoint_window")
            / f"{model_tag}_wu_late_start32_dsae{args.d_sae}_seed{args.seed}.pt"
        )
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16}[args.amp_dtype]

    use_sparse_adam = args.optimizer == "sparse_adam"

    print(f"late-start schedule ({args.model}):", LATE_START_STEPS_32, flush=True)
    snapshots, resolved_steps = wu_adapter.load_snapshots(
        model_name=args.model,
        steps=LATE_START_STEPS_32,
        cache_dir=args.cache_dir,
        dtype=torch.float32,
    )
    print(f"Snapshots: {tuple(snapshots.shape)}", flush=True)
    # snapshots: (K, V, d_model). Compute expansion_factor against the actual
    # snapshot d_model so this script works for 160m (768) and 1b (2048).
    d_model = int(snapshots.shape[-1])
    expansion_factor = args.d_sae / d_model
    print(f"d_model={d_model} expansion_factor={expansion_factor}", flush=True)
    snapshots, preprocess_stats = wu_adapter.preprocess_snapshots(snapshots, mode="center_scale")

    crosscoder = wu_adapter.train(
        snapshots,
        expansion_factor=expansion_factor,
        lr=args.lr,
        l1_coefficient=args.l1_coefficient,
        frequency_scale=0.01,
        tanh_stretch_coefficient=args.tanh_stretch,
        jumprelu_lr_factor=args.jumprelu_lr_factor,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        init_threshold=args.init_threshold,
        amp_dtype=amp_dtype,
        use_sparse_adam=use_sparse_adam,
        init_encoder_with_decoder_transpose_factor=1.0,
        l1_warmup_fraction=args.l1_warmup_fraction,
        lr_warmup_fraction=args.lr_warmup_fraction,
        lr_decay_fraction=args.lr_decay_fraction,
        log_every=args.log_every,
    )
    metrics = wu_adapter.quick_quality(crosscoder, snapshots, batch_size=args.batch_size, device=args.device)
    print(f"Quality: {metrics}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": crosscoder.state_dict(),
            "config": crosscoder.cfg.model_dump(),
            "steps": resolved_steps,
            "model_name": args.model,
            "seed": args.seed,
            "quality": metrics,
            "training": {
                "lr": args.lr,
                "l1_coefficient": args.l1_coefficient,
                "frequency_scale": 0.01,
                "tanh_stretch_coefficient": args.tanh_stretch,
                "jumprelu_lr_factor": args.jumprelu_lr_factor,
                "init_threshold": args.init_threshold,
                "decoder_transpose_init": 1.0,
                "amp_dtype": args.amp_dtype,
                "optimizer": args.optimizer,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "expansion_factor": expansion_factor,
                "d_model": d_model,
                "d_sae": crosscoder.cfg.d_sae,
                "input_preprocess": "center_scale",
                "l1_warmup_fraction": args.l1_warmup_fraction,
                "lr_warmup_fraction": args.lr_warmup_fraction,
                "lr_decay_fraction": args.lr_decay_fraction,
                "schedule_note": f"32 {model_short} W_U snapshots starting at step 256",
            },
            "preprocess_stats": preprocess_stats,
        },
        args.output,
    )
    print(f"Saved to {args.output}", flush=True)

    if args.append_manifest:
        from src.crosscoder.manifest import append_manifest_row

        row = append_manifest_row(
            args.output,
            notes=f"olmo_matched_checkpoint_window late-start {model_short}",
            strict=True,
        )
        print(f"Manifest row updated: {row['checkpoint_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
