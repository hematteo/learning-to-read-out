#!/usr/bin/env python3
"""Within-trajectory readout swap grid for each recipe-control condition.

The published-model protocol (``temporal_localization_patching/run_aligned_swap_grid.py``)
applied per arm: fix the post-final-LN hidden states from checkpoint h_t and decode
them through every training-time readout W_U^(s) of the same arm, recording
next-token NLL, KL-to-native, top-1 agreement, and centered-logit R^2 on the held-out
slice. The question is whether the recipe shifts the (h_t, s) compatibility basin:
if a later readout (s > h_t) decodes h_t better than its own, the hidden state is
ahead of its readout (the expression lag), and the argmin-NLL s per h_t is the
basin plotted in fig:app-recipe-control-basin.

Reuses ``readout.probes.readout_swap.{swap_cell_metrics, align_readout}`` (the
5-mode gauge ladder none / mean / scale / row_norm / procrustes) and
``readout.core.resume`` (one JSON shard per cell, atomic writes, exists-skip,
aggregation), so a killed job resumes where it stopped.

Eval corpus: the arms' own held-out Pile slice (``fetch_heldout_slice.py``), so the
control is tokenizer-matched and unseen by every arm.

Output: ``<out-dir>/trajectory_swap_<cond>.csv`` per condition and
``trajectory_swap_all.csv``; rows = (condition, alignment, h_step, s_step, metrics).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rc_common import (
    add_common_args,
    cache_hidden,
    default_results_dir,
    load_eval_blocks,
    pick_device,
    write_provenance,
)

from readout.core.repro import seed_everything
from readout.core.resume import aggregate_json_shards, atomic_write_json, iter_undone
from readout.probes.readout_swap import ALIGN_MODES, align_readout, swap_cell_metrics
from readout.probes.recipe_control_models import condition_config, condition_steps, load_body, load_readout

DEFAULT_H_STEPS = [256, 512, 1024, 2048, 4769]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, eval_bin=True)
    ap.add_argument(
        "--h-steps",
        type=int,
        nargs="+",
        default=DEFAULT_H_STEPS,
        help="hidden state checkpoints (rows); mapped to the nearest available step per condition",
    )
    ap.add_argument("--s-steps", type=int, nargs="+", default=None, help="readout checkpoints (cols); default all")
    ap.add_argument("--alignments", nargs="+", default=list(ALIGN_MODES))
    ap.add_argument("--eval-tokens", type=int, default=400_000)
    ap.add_argument("--batch-seqs", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=default_results_dir() / "trajectory")
    args = ap.parse_args()

    bad = sorted(set(args.alignments) - set(ALIGN_MODES))
    if bad:
        raise SystemExit(f"unknown alignments {bad}; valid={list(ALIGN_MODES)}")

    seed_everything(args.seed)
    dev = pick_device(args.device)
    ids_seqs = load_eval_blocks(args.eval_bin, args.eval_tokens, args.seq_len)
    print(f"[eval] {args.eval_bin}: {ids_seqs.shape[0]} blocks x {args.seq_len}", flush=True)

    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(args.out_dir, "run_trajectory_swap", args, args.seed)

    for cond in args.conditions:
        cfg = condition_config(args.ckpt_root, cond)
        msize = cfg["model_size"]
        avail = condition_steps(args.ckpt_root, cond)
        h_steps = sorted({min(avail, key=lambda a: abs(a - h)) for h in args.h_steps})
        s_steps = sorted(args.s_steps) if args.s_steps else avail
        print(
            f"[cond] {cond} size={msize} mult={cfg.get('readout_lr_mult')} warmup={cfg.get('warmup_steps')} "
            f"h={h_steps} |s|={len(s_steps)}",
            flush=True,
        )
        readouts = {s: load_readout(args.ckpt_root, cond, s)["wu"] for s in sorted(set(s_steps) | set(h_steps))}
        items = [(cond, al, h, s) for al in args.alignments for h in h_steps for s in s_steps]

        def shard_path(it):
            c, al, h, s = it
            return shard_dir / f"{c}__a-{al}__h{h}__s{s}.json"

        h_cache: dict[int, torch.Tensor] = {}  # each (cond, h_t) body forwards once
        for it in iter_undone(items, shard_path, label="cell"):
            c, al, h_step, s_step = it
            if h_step not in h_cache:
                body = load_body(args.ckpt_root, c, h_step, msize, args.seq_len, dev)
                h_cache[h_step] = cache_hidden(body, ids_seqs, dev, args.batch_seqs)
                del body
                if dev == "cuda":
                    torch.cuda.empty_cache()
            W_native = readouts[h_step]  # same-checkpoint readout
            W_aligned = align_readout(readouts[s_step], W_native, al)
            m = swap_cell_metrics(
                h_cache[h_step], W_aligned, W_native, ids_seqs, device=dev, batch_seqs=args.batch_seqs
            )
            row = {
                "condition": c,
                "readout_lr_mult": cfg.get("readout_lr_mult"),
                "warmup_steps": cfg.get("warmup_steps"),
                "alignment": al,
                "h_step": h_step,
                "s_step": s_step,
                **m,
            }
            atomic_write_json(shard_path(it), row)
            print(
                f"  [{c:>12} {al:>10}] h={h_step:>5} s={s_step:>5} nll={m['nll']:.4f} "
                f"(delta={m['delta_nll']:+.4f}) KL={m['kl_to_native']:.3f} agree={m['top1_agreement']:.3f}",
                flush=True,
            )

        n = aggregate_json_shards(
            shard_dir, args.out_dir / f"trajectory_swap_{cond}.csv", pattern=f"{cond}__*.json", key="s_step"
        )
        print(f"[done] {cond}: {n} cells -> trajectory_swap_{cond}.csv", flush=True)

    total = aggregate_json_shards(shard_dir, args.out_dir / "trajectory_swap_all.csv", key="s_step")
    print(f"[done] combined {total} cells -> trajectory_swap_all.csv", flush=True)


if __name__ == "__main__":
    main()
