#!/usr/bin/env python3
"""Fit one W_U trajectory crosscoder per recipe-control condition.

For each arm, the trajectory crosscoder is trained on that arm's stacked W_U
snapshots ``(K, V, d)`` (``embed_out.weight`` at every checkpoint on disk; K is
discovered by globbing, ~16 per released arm) with the production trainer
``readout.crosscoder.wu_adapter`` (``preprocess_snapshots -> train -> quick_quality``).
Only the data loader differs from the published-model fits: the control's
``model_fp16.pt`` checkpoints replace Pythia HF ``step{N}`` revisions.

Hyperparameters follow the paper's Pythia-160M dictionary column
(``tab:repro-dictionary-hparams``), mapped to the 31M control (d_model=256):
  input_preprocess  center_scale
  expansion_factor  32.0   -> d_sae = 32 * 256 = 8192
  lambda1 (l1)      0.3
  lr                5e-5
  jumprelu_lr_factor 0.1
  init_threshold    0.1
  tanh stretch c    1.0
  frequency_scale a 0.01
  batch size        1024 rows
  epochs            300    (-> ceil(V/1024)=50 steps/epoch * 300 = 15000 steps)
  sparsity warmup   0.05   (l1_warmup_fraction)
  LR warmup/decay   0 / 0
  AuxK              off
  optimizer         Adam
  AMP               fp32

Output per condition: ``<out-dir>/cc_<cond>_d<d_sae>_seed<seed>.pt`` holding
``state_dict``, ``config``, ``steps``, ``quality`` (EV / mean L0 / dead rate),
``training``, the invertible ``center_scale`` ``preprocess_stats``, and provenance.
Resumable: an existing output is skipped; within a condition a rolling checkpoint
(``.rolling.pt``) is written every ``--ckpt-every-epochs`` and auto-resumed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rc_common import add_common_args, default_results_dir, pick_device, write_provenance

from readout.core.repro import git_commit, log_run_provenance, seed_everything
from readout.crosscoder.wu_adapter import preprocess_snapshots, quick_quality, train
from readout.probes.recipe_control_models import load_condition_wu

# Pythia-160M dictionary recipe, mapped to the 31M control (d_model=256).
RECIPE = dict(
    expansion_factor=32.0,  # d_sae = 32 * 256 = 8192
    lr=5e-5,
    l1_coefficient=0.3,
    frequency_scale=0.01,  # alpha
    tanh_stretch_coefficient=1.0,  # c
    n_epochs=300,
    batch_size=1024,
    jumprelu_lr_factor=0.1,
    init_threshold=0.1,
    use_sparse_adam=False,  # Adam, as in the Pythia column
    amp_dtype=None,  # fp32
    init_encoder_with_decoder_transpose_factor=1.0,
    l1_warmup_fraction=0.05,
    lr_warmup_fraction=0.0,
    lr_decay_fraction=0.0,
    auxk_coefficient=0.0,  # AuxK off
)


def output_path(out_dir: Path, cond: str, d_sae: int, seed: int) -> Path:
    return out_dir / f"cc_{cond}_d{d_sae}_seed{seed}.pt"


def fit_condition(args: argparse.Namespace, cond: str, device: str) -> Path | None:
    snaps, steps = load_condition_wu(args.ckpt_root, cond)
    K, V, d = snaps.shape
    recipe = dict(RECIPE)
    if args.expansion_factor is not None:
        recipe["expansion_factor"] = args.expansion_factor
    if args.n_epochs is not None:
        recipe["n_epochs"] = args.n_epochs
    if args.batch_size is not None:
        recipe["batch_size"] = args.batch_size
    d_sae = int(recipe["expansion_factor"] * d)
    out = output_path(args.out_dir, cond, d_sae, args.seed)
    if out.exists() and not args.force:
        print(f"[skip] {cond}: {out} already exists", flush=True)
        return None
    print(f"[{cond}] snapshots {(K, V, d)} (K,V,d) steps={steps} -> d_sae={d_sae}", flush=True)

    snaps, pstats = preprocess_snapshots(snaps, mode="center_scale")
    scales = pstats["scale"].reshape(-1).tolist()
    print(f"[{cond}] center_scale per-snapshot scale[:3]={scales[:3]}", flush=True)

    rolling = out.with_suffix(".rolling.pt")
    crosscoder = train(
        snaps,
        device=device,
        seed=args.seed,
        log_every=args.log_every,
        ckpt_every_epochs=args.ckpt_every_epochs,
        ckpt_path=str(rolling),
        **recipe,
    )
    metrics = quick_quality(crosscoder, snaps, batch_size=recipe["batch_size"], device=device)
    print(f"[{cond}] quality {metrics}", flush=True)

    payload = dict(
        state_dict={k: v.detach().cpu() for k, v in crosscoder.state_dict().items()},
        config=crosscoder.cfg.model_dump(),
        steps=steps,
        condition=cond,
        seed=args.seed,
        quality=metrics,
        training={
            **recipe,
            "amp_dtype": "fp32",
            "optimizer": "adam",
            "input_preprocess": "center_scale",
            "d_sae": crosscoder.cfg.d_sae,
            "n_snapshots": len(steps),
        },
        preprocess_stats=pstats,
        provenance={**log_run_provenance(args.seed), "git_commit_full": git_commit()},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    print(f"[{cond}] saved -> {out}", flush=True)
    if rolling.exists():  # the final dictionary is authoritative
        rolling.unlink()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=default_results_dir() / "crosscoders",
        help="where cc_<cond>_d<d_sae>_seed<seed>.pt land",
    )
    ap.add_argument("--expansion-factor", type=float, default=None, help="override RECIPE (smoke tests)")
    ap.add_argument("--n-epochs", type=int, default=None, help="override RECIPE (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=None, help="override RECIPE (smoke tests)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every-epochs", type=int, default=25, help="rolling checkpoint cadence (0 = off)")
    ap.add_argument("--force", action="store_true", help="refit even if the output exists")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    write_provenance(args.out_dir, "train_control_crosscoder", args, args.seed)
    for cond in args.conditions:
        fit_condition(args, cond, device)


if __name__ == "__main__":
    main()
