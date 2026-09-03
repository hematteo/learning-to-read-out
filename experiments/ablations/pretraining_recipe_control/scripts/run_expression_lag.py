#!/usr/bin/env python3
"""Availability versus expression (the readout expression lag) per recipe-control condition.

Probe-space companion to the weight-space swap grid (``run_trajectory_swap.py``).
For each (condition, hidden state checkpoint h_t) it records the availability triple:

  * probe_acc          k-fold logistic probe of the target from the post-final-LN
                       hidden states h_t (availability: is it linearly there?),
                       with label-shuffle and random-label nulls and a bootstrap CI
                       (``readout.probes.availability_probe``)
  * native_readout_acc the same-checkpoint readout W_U^(h_t)'s own accuracy on the
                       target (expression: does the readout show it?)
  * best_readout_acc   the best accuracy over all of this condition's readouts
                       W_U^(s), and the argmax s

  availability_gap = probe_acc - native_readout_acc   (available but not yet expressed)
  readout_rescue   = best_readout_acc - native_readout_acc   (a later readout expresses it)

Target: the next token's frequency tier (``--n-bins`` tiers of equal occurrence
mass on the eval slice, so the base rate is ~1/n_bins). At 31M scale the paper's
contrastive tasks sit near floor; the frequency tier is the lexical axis the
vocabulary-family probes use. Readout accuracy is the argmax token's tier versus
the true next token's tier; a fixed seeded position subsample keeps the logistic
regression tractable.

Output: ``<out-dir>/expression_lag.csv`` (one row per condition x h_t), built from
per-cell JSON shards (resumable).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from rc_common import (
    add_common_args,
    default_results_dir,
    load_eval_blocks,
    pick_device,
    write_provenance,
)

from readout.core.repro import seed_everything
from readout.core.resume import aggregate_json_shards, atomic_write_json, iter_undone
from readout.probes.availability_probe import availability_probe
from readout.probes.recipe_control_models import condition_config, condition_steps, load_body, load_readout

DEFAULT_H_STEPS = [256, 512, 1024, 2048, 4769]


def freq_class_map(ids_flat: np.ndarray, n_bins: int, vocab: int) -> np.ndarray:
    """Map each vocab id to a frequency tier by equal occurrence mass, (vocab,) int.

    Tokens are ordered commonest-first and split so each tier holds ~1/n_bins of the
    slice's token positions (not 1/n_bins of the vocabulary; under Zipf the latter
    puts ~all positions in one tier and collapses the probe to its base rate).
    Tier 0 = commonest; unseen tokens fall in the rarest tier.
    """
    counts = np.bincount(ids_flat, minlength=vocab).astype(np.int64)
    order = np.argsort(-counts, kind="stable")
    cum = np.cumsum(counts[order]).astype(np.float64)
    total = float(cum[-1])
    frac = (cum - counts[order]) / max(total, 1.0)  # cumulative mass before each token
    tier_in_order = np.clip((frac * n_bins).astype(np.int64), 0, n_bins - 1)
    cls = np.empty(vocab, dtype=np.int64)
    cls[order] = tier_in_order
    return cls


@torch.no_grad()
def cache_hidden_and_targets(model, ids_seqs, device, batch_seqs, vocab_class, max_positions, seed):
    """One forward: (h, target tier, next-token id) over a seeded position subsample.

    h is the post-final-LN hidden at position p; the target is the tier of the token at p+1.
    """
    H, NT = [], []
    for s in range(0, ids_seqs.shape[0], batch_seqs):
        x = ids_seqs[s : s + batch_seqs].to(device)
        h = model.gpt_neox(input_ids=x).last_hidden_state.float().cpu()  # (b, T, d)
        hh = h[:, :-1, :]  # hidden at p, target at p+1
        H.append(hh.reshape(-1, hh.shape[-1]))
        NT.append(x[:, 1:].cpu().reshape(-1))
    Hc = torch.cat(H).numpy()
    NTc = torch.cat(NT).numpy()
    rng = np.random.default_rng(seed)
    if Hc.shape[0] > max_positions:
        idx = rng.choice(Hc.shape[0], size=max_positions, replace=False)
        Hc, NTc = Hc[idx], NTc[idx]
    return Hc, vocab_class[NTc], NTc


@torch.no_grad()
def readout_class_acc(h, next_ids, wu, vocab_class, device, batch=4096) -> float:
    """Accuracy of readout W_U at the next token's tier: tier[argmax_v h @ W_U^T] vs tier[next]."""
    W = wu.to(device).float()
    cls = torch.from_numpy(vocab_class).to(device)
    y_true = cls[torch.from_numpy(next_ids).to(device)]
    ht = torch.from_numpy(h).to(device).float()
    correct = 0
    for i in range(0, ht.shape[0], batch):
        pred_tok = (ht[i : i + batch] @ W.T).argmax(dim=-1)  # (b,)
        correct += int((cls[pred_tok] == y_true[i : i + batch]).sum())
    return correct / ht.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, eval_bin=True)
    ap.add_argument("--h-steps", type=int, nargs="+", default=DEFAULT_H_STEPS)
    ap.add_argument("--n-bins", type=int, default=4, help="frequency tiers (probe target)")
    ap.add_argument("--eval-tokens", type=int, default=400_000)
    ap.add_argument("--batch-seqs", type=int, default=8)
    ap.add_argument("--max-positions", type=int, default=20000, help="probe matrix row cap")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=default_results_dir() / "expression_lag")
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = pick_device(args.device)
    ids_seqs = load_eval_blocks(args.eval_bin, args.eval_tokens, args.seq_len)
    first_cfg = condition_config(args.ckpt_root, args.conditions[0])
    vocab = int(
        load_readout(args.ckpt_root, args.conditions[0], condition_steps(args.ckpt_root, args.conditions[0])[0])[
            "wu"
        ].shape[0]
    )
    vocab_class = freq_class_map(ids_seqs.numpy().reshape(-1), args.n_bins, vocab)
    print(
        f"[eval] {ids_seqs.shape[0]} blocks; freq tiers={args.n_bins} (vocab ids per tier "
        f"{np.bincount(vocab_class).tolist()}); model_size={first_cfg.get('model_size')}",
        flush=True,
    )

    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(args.out_dir, "run_expression_lag", args, args.seed)

    items = []
    for cond in args.conditions:
        avail = condition_steps(args.ckpt_root, cond)
        for h in sorted({min(avail, key=lambda a: abs(a - h)) for h in args.h_steps}):
            items.append((cond, h))

    def shard_path(it):
        c, h = it
        return shard_dir / f"{c}__h{h}.json"

    for it in iter_undone(items, shard_path, label="probe-cell"):
        cond, h_step = it
        cfg = condition_config(args.ckpt_root, cond)
        body = load_body(args.ckpt_root, cond, h_step, cfg["model_size"], args.seq_len, dev)
        h, y, next_ids = cache_hidden_and_targets(
            body, ids_seqs, dev, args.batch_seqs, vocab_class, args.max_positions, args.seed
        )
        del body
        if dev == "cuda":
            torch.cuda.empty_cache()

        probe = availability_probe(h, y, n_splits=args.n_splits, seed=args.seed)

        native_acc = best_acc = float("nan")
        best_s = -1
        for s in condition_steps(args.ckpt_root, cond):
            acc = readout_class_acc(h, next_ids, load_readout(args.ckpt_root, cond, s)["wu"], vocab_class, dev)
            if s == h_step:
                native_acc = acc
            if np.isnan(best_acc) or acc > best_acc:
                best_acc, best_s = acc, s

        row = {
            "condition": cond,
            "readout_lr_mult": cfg.get("readout_lr_mult"),
            "warmup_steps": cfg.get("warmup_steps"),
            "h_step": h_step,
            "n_bins": args.n_bins,
            "probe_acc": probe["probe_acc"],
            "probe_acc_label_shuffle": probe["probe_acc_label_shuffle"],
            "probe_acc_random_label": probe["probe_acc_random_label"],
            "probe_acc_ci_lo": probe.get("probe_acc_ci_lo"),
            "probe_acc_ci_hi": probe.get("probe_acc_ci_hi"),
            "native_readout_acc": native_acc,
            "best_readout_acc": best_acc,
            "best_readout_s": best_s,
            "availability_gap": probe["probe_acc"] - native_acc,
            "readout_rescue": best_acc - native_acc,
            "n": probe["n"],
            "skip_reason": probe["skip_reason"],
        }
        atomic_write_json(shard_path(it), row)
        print(
            f"  [{cond:>12} h={h_step:>5}] probe={row['probe_acc']:.3f} (shuf {row['probe_acc_label_shuffle']:.3f}) "
            f"native={native_acc:.3f} best={best_acc:.3f}@s{best_s} gap={row['availability_gap']:+.3f}",
            flush=True,
        )

    n = aggregate_json_shards(shard_dir, args.out_dir / "expression_lag.csv", key="h_step")
    print(f"[done] {n} cells -> expression_lag.csv", flush=True)


if __name__ == "__main__":
    main()
