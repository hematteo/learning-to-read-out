"""Head-parallel crosscoder training entry point for large W_U production runs.

Wraps wu_adapter.train() with the torch.distributed init + DeviceMesh setup
that llamascopium's Crosscoder requires for head-parallel mode (Ge §A.3).
With NPROC processes and N_SNAPSHOTS source snapshots, each process handles
N_SNAPSHOTS/NPROC encoder/decoder pairs; pre-activations are summed via
All-Reduce.

Launch with:
    torchrun --nproc-per-node=4 experiments/crosscoders/crosscoder_olmo/scripts/train_distributed.py [...]

This script is for production-scale fits such as OLMo-2-7B and Pythia-6.9B
at d_SAE=32768. Single-GPU pilots should use wu_adapter.py directly via the
relevant launch wrapper. The DeviceMesh path is wired through wu_adapter, but
a 2-GPU smoke run should precede any 4-GPU production launch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(
    0, str(REPO_ROOT)
)  # so `from src.crosscoder.wu_adapter import ...` works


def setup_distributed() -> tuple[int, int, "torch.distributed.device_mesh.DeviceMesh"]:
    """Initialise the process group and a single-axis device mesh on which
    head parallelism distributes snapshot heads.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)

    # Single-axis mesh over all GPUs; named "head" so the Crosscoder DTensor
    # placements know which axis to shard the snapshot dimension along.
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("head",))

    if rank == 0:
        print(f"[dist] world_size={world_size} rank={rank} mesh={mesh}", flush=True)
    return rank, world_size, mesh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument(
        "--d-sae",
        type=int,
        required=True,
        help="Direct d_sae spec (production typically uses 32768).",
    )
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--jumprelu-lr-factor", type=float, default=0.3)
    ap.add_argument("--l1-coefficient", type=float, default=0.3)
    ap.add_argument(
        "--tanh-stretch",
        type=float,
        default=1.0,
        help="Tanh-quad sparsity stretch. Ge-aligned W_U runs use 1.0.",
    )
    ap.add_argument("--init-threshold", type=float, default=0.1)
    ap.add_argument("--decoder-transpose-init", type=float, default=1.0)
    ap.add_argument(
        "--input-preprocess",
        choices=["none", "center", "center_scale"],
        default="center_scale",
    )
    ap.add_argument("--l1-warmup-fraction", type=float, default=0.1)
    ap.add_argument(
        "--lr-warmup-fraction",
        type=float,
        default=0.10,
        help="Linear LR warmup fraction. Ge §A.4 specifies 0.10.",
    )
    ap.add_argument(
        "--lr-decay-fraction",
        type=float,
        default=0.20,
        help="Linear LR decay fraction at end of training. Ge §A.4 specifies 0.20.",
    )
    ap.add_argument(
        "--optimizer",
        choices=["adam", "sparse_adam"],
        default=None,
        help="Explicit optimizer selection. Overrides --use-sparse-adam when set.",
    )
    ap.add_argument("--use-sparse-adam", action="store_true", default=False)
    ap.add_argument("--amp-dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--n-epochs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument(
        "--auxk-coefficient",
        type=float,
        default=0.0,
        help="Gao et al. 2024 auxk dead-feature revival weight. 0=disabled. "
        "Set 0.03125 (=1/32) for high-capacity runs at risk of dead-feature collapse.",
    )
    ap.add_argument(
        "--auxk-k",
        type=int,
        default=512,
        help="Number of dead features to revive per step under auxk loss.",
    )
    ap.add_argument(
        "--dead-window-steps",
        type=int,
        default=2000,
        help="Steps after which features that haven't fired are considered dead "
        "and eligible for auxk revival.",
    )
    ap.add_argument(
        "--ckpt-every-epochs",
        type=int,
        default=0,
        help="Rolling-checkpoint frequency in epochs (0 = disabled). The "
        "checkpoint is written to <output>.latest.ckpt.pt with atomic rename.",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Explicit resume path. If unset and --ckpt-every-epochs>0, the "
        "trainer auto-detects <output>.latest.ckpt.pt and resumes if present.",
    )
    ap.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Override rolling-checkpoint location. Default is "
        "<output>.latest.ckpt.pt. Use this to put rolling ckpts on tmpfs "
        "(e.g. /dev/shm/...) when the working filesystem is small.",
    )
    args = ap.parse_args()

    if args.optimizer is not None:
        args.use_sparse_adam = args.optimizer == "sparse_adam"
    optimizer_name = "sparse_adam" if args.use_sparse_adam else "adam"

    rank, world_size, mesh = setup_distributed()

    from src.core.repro import log_run_provenance, seed_everything

    seed_everything(args.seed)
    if rank == 0:
        log_run_provenance(seed=args.seed)

    # Each rank loads the same W_U snapshot tensors (data-replicated). The
    # crosscoder's per-snapshot encoder/decoder *params* are sharded across
    # the mesh; the snapshot inputs themselves are not.
    from src.crosscoder.wu_adapter import load_snapshots, preprocess_snapshots, train

    if rank == 0:
        print(
            f"Loading {len(args.steps)} W_U snapshots from {args.cache_dir}", flush=True
        )
    snapshots, resolved_steps = load_snapshots(
        model_name=args.model,
        steps=args.steps,
        cache_dir=args.cache_dir,
    )
    if rank == 0:
        print(f"Snapshots: {tuple(snapshots.shape)}", flush=True)

    snapshots, preprocess_stats = preprocess_snapshots(
        snapshots, mode=args.input_preprocess
    )
    if rank == 0 and preprocess_stats is not None:
        scales = preprocess_stats["scale"].squeeze().tolist()
        if not isinstance(scales, list):
            scales = [scales]
        print(
            f"Preprocess '{args.input_preprocess}': scales[:3]={scales[:3]}", flush=True
        )

    # Sanity: head-parallelism requires world_size divides n_snapshots.
    K = snapshots.shape[0]
    if K % world_size != 0:
        if rank == 0:
            print(
                f"ERROR: world_size={world_size} must divide n_snapshots={K}",
                flush=True,
            )
        return 2

    # Compute expansion_factor from explicit d_sae for compatibility with the
    # existing build_crosscoder API.
    d_model = snapshots.shape[-1]
    expansion = args.d_sae / d_model

    amp_dtype_map = {"fp32": None, "bf16": torch.bfloat16}

    # Head-parallel implementation note, scoped 2026-04-27 after reading
    # lib/Language-Model-SAEs/src/llamascopium/{models/crosscoder.py, optim.py}.
    # The library already supports head parallelism end-to-end — the only gap
    # is that wu_adapter.build_crosscoder does not accept a device_mesh kwarg.
    #
    # What's already supported (verified):
    #   - Crosscoder.__init__ accepts `device_mesh: Optional[DeviceMesh]`
    #     (crosscoder.py:101) and routes weight construction to a DTensor
    #     branch (lines 120-158) sharded along the 'head' axis.
    #   - Crosscoder.init_parameters handles the distributed branch via
    #     dim_maps().local_slices() + DTensor.from_local (lines 178-217).
    #   - SparseAdam handles DTensor gradients explicitly (optim.py:202-204).
    #   - prepare_input only asserts non-DTensor in the padding path
    #     (crosscoder.py:559); for OLMo-2-7B with d_model=4096 == row dim,
    #     no padding fires.
    #
    # Remaining launch risk:
    #   1. init_encoder_with_decoder_transpose at crosscoder.py:543 does
    #      `einops.rearrange(self.W_D, ...).clone().contiguous() * factor`
    #      followed by `self.W_E.copy_(transposed_decoder)`. einops handles
    #      DTensor shape-semantics, but the .copy_ requires placement-compatible
    #      source and destination. W_E and W_D have different placements
    #      (Shard(0) on n_heads vs same Shard(0) on n_heads — should be
    #      compatible but worth confirming on a 2-GPU smoke run before
    #      launching the 4-GPU production fit).
    #
    # First production launch plan:
    #   - 2-GPU smoke: 4 snapshots, d_SAE=4096, 5 epochs, verify loss matches
    #     a 1-GPU control on the same data within fp32 tolerance.
    #   - If smoke passes: 4-GPU production at d_SAE=32768.
    #   - If init_encoder_with_decoder_transpose fails on DTensor copy_, fall
    #     back to constructing W_E without the transpose-init (factor=0) and
    #     accept the EV cost (Run 2 baseline was EV ~0.39 without transpose-init
    #     vs ~0.78 with; degraded but trainable).
    if rank == 0:
        print(
            "[head-parallel] device_mesh wired through Crosscoder; init_encoder_with_decoder_transpose "
            "needs a 2-GPU smoke run to confirm DTensor copy_ compatibility before 4-GPU launch.",
            flush=True,
        )

    if args.ckpt_every_epochs > 0:
        if args.ckpt_path is not None:
            rolling_ckpt_path = args.ckpt_path
        else:
            rolling_ckpt_path = args.output.parent / (
                args.output.name + ".latest.ckpt.pt"
            )
    else:
        rolling_ckpt_path = None
    crosscoder = train(
        snapshots,
        expansion_factor=expansion,
        lr=args.lr,
        l1_coefficient=args.l1_coefficient,
        tanh_stretch_coefficient=args.tanh_stretch,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device="cuda",
        seed=args.seed,
        log_every=args.log_every,
        jumprelu_lr_factor=args.jumprelu_lr_factor,
        init_threshold=args.init_threshold,
        use_sparse_adam=args.use_sparse_adam,
        amp_dtype=amp_dtype_map[args.amp_dtype],
        init_encoder_with_decoder_transpose_factor=args.decoder_transpose_init,
        l1_warmup_fraction=args.l1_warmup_fraction,
        lr_warmup_fraction=args.lr_warmup_fraction,
        lr_decay_fraction=args.lr_decay_fraction,
        auxk_coefficient=args.auxk_coefficient,
        auxk_k=args.auxk_k,
        dead_window_steps=args.dead_window_steps,
        device_mesh=mesh,
        ckpt_every_epochs=args.ckpt_every_epochs,
        ckpt_path=rolling_ckpt_path,
        resume_from=args.resume_from,
        is_rank_0=(rank == 0),
    )

    # SAVE FIRST, EVAL SECOND: quick_quality can OOM on the post-training all-gather
    # when Adam state is still resident, and if eval crashes before the save the
    # run's compute is lost.
    # Gather the (small) state-dict shards now so the model is durable, THEN run
    # quick_quality, THEN re-save with metrics attached if eval succeeded.

    # Free Adam state and any cached blocks before the all-gather. Each
    # full_tensor() needs ~(K × in_dim × d_sae × 4) bytes of fresh GPU memory
    # for the output buffer; on A40 (44 GB) at K=32, d_sae=32768 that's 17 GB
    # per gather, and Adam state alone takes another 17 GB.
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # Gather DTensor shards into full tensors before serializing, then move
    # each to CPU immediately so they don't pile up on GPU. nn.Module's
    # state_dict() returns DTensors as-is; torch.save of a DTensor only
    # serializes the local shard. .full_tensor() is an all-gather collective —
    # every rank must participate, so this block runs OUTSIDE the `if rank == 0`
    # guard.
    from torch.distributed.tensor import DTensor as _DTensor  # type: ignore

    sd_local = crosscoder.state_dict()
    sd_gathered = {}
    for k, v in sd_local.items():
        if isinstance(v, _DTensor):
            full = v.full_tensor().detach().cpu()
            sd_gathered[k] = full
            del full
            torch.cuda.empty_cache()
        else:
            sd_gathered[k] = v.detach().cpu() if hasattr(v, "detach") else v
    del sd_local
    gc.collect()
    torch.cuda.empty_cache()

    training_block = {
        "lr": args.lr,
        "l1_coefficient": args.l1_coefficient,
        "tanh_stretch_coefficient": args.tanh_stretch,
        "jumprelu_lr_factor": args.jumprelu_lr_factor,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "expansion_factor": expansion,
        "d_sae": args.d_sae,
        "input_preprocess": args.input_preprocess,
        "l1_warmup_fraction": args.l1_warmup_fraction,
        "lr_warmup_fraction": args.lr_warmup_fraction,
        "lr_decay_fraction": args.lr_decay_fraction,
        "init_threshold": args.init_threshold,
        "decoder_transpose_init": args.decoder_transpose_init,
        "auxk_coefficient": args.auxk_coefficient,
        "auxk_k": args.auxk_k,
        "dead_window_steps": args.dead_window_steps,
        "amp_dtype": args.amp_dtype,
        "optimizer": optimizer_name,
    }

    def _build_payload(quality):
        return {
            "state_dict": sd_gathered,
            "config": crosscoder.cfg.model_dump(),
            "steps": resolved_steps,
            "model_name": args.model,
            "seed": args.seed,
            "world_size": world_size,
            "quality": quality,
            "training": training_block,
            "preprocess_stats": preprocess_stats,
        }

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_build_payload(quality=None), args.output)
        print(f"Saved (pre-eval) to {args.output}", flush=True)

    # Eval is best-effort. If it OOMs or otherwise crashes, the model is already
    # on disk; we just won't have inline metrics. Re-run quick_quality offline
    # against the saved checkpoint with a smaller batch_size to recover them.
    from src.crosscoder.wu_adapter import quick_quality

    metrics = None
    try:
        metrics = quick_quality(
            crosscoder, snapshots, batch_size=args.batch_size, device="cuda"
        )
    except Exception as exc:
        if rank == 0:
            print(
                f"WARN: quick_quality failed ({type(exc).__name__}: {exc}); "
                f"checkpoint is saved, run eval offline.",
                flush=True,
            )

    if rank == 0 and metrics is not None:
        print(f"Final quality: {metrics}", flush=True)
        torch.save(_build_payload(quality=metrics), args.output)
        print(f"Re-saved with quality metrics to {args.output}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
