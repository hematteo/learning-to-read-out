#!/usr/bin/env python3
"""Why matched-loss conditions differ in readout geometry: the loss is (near) gauge-invariant.

The post-final-LN readout has an exact rescaling symmetry. With
    logits = LN_{gamma,beta}(g) @ W_U^T,   LN_{gamma,beta}(x) = gamma * (x - mu) / sigma + beta,
the joint transform  gamma -> a*gamma, beta -> a*beta, W_U -> W_U / a  leaves every
logit unchanged (scaling gamma and W_U without beta does not: the beta @ W_U^T term
breaks it). The readout LR knob moves the trained models mainly along this
loss-flat direction, which is why W_U row norms and the LN gain move by tens of
percent while the loss barely moves.

Two artifacts:

  PART 1 (fig:app-recipe-control-temperature, ``temperature_conservation.csv``)
    For each condition's final checkpoint on the held-out slice: the raw W_U row norm
    mean, the final-LN gain norm, their product, the effective logit scale (std of
    the centered logits), and the NLL. The factors swing; the product and the logit
    scale are conserved.

  PART 2 (fig:app-recipe-control-gauge-landscape, ``gauge_landscape.csv``)
    For ``--landscape-cond``, a 2D grid of held-out NLL:
      alpha = along the gauge direction (gamma, beta scaled by alpha, W_U / alpha)
      beta  = off gauge (W_U scaled by beta alone)
    The loss is flat along alpha (a valley) and steep along beta (walls).

Eval is deterministic (fixed slice, fp32, no sampling).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rc_common import (
    add_common_args,
    cache_hidden,
    default_results_dir,
    load_eval_blocks,
    pick_device,
    write_csv,
    write_provenance,
)

from readout.core.repro import seed_everything
from readout.probes.recipe_control_models import condition_config, load_body, load_readout, resolve_step


@torch.no_grad()
def nll_and_scale(g, ids, gamma, beta, W, eps, device, micro) -> tuple[float, float]:
    """Mean next-token NLL (nats/token) and effective logit scale (std of centered logits)."""
    d = g.shape[-1]
    vocab = W.shape[0]
    ce_sum = logit_sq = 0.0
    logit_n = ntok = 0
    for s in range(0, g.shape[0], micro):
        gb = g[s : s + micro].to(device)
        h = F.layer_norm(gb, (d,), gamma, beta, eps)
        logits = (h @ W.T)[:, :-1, :].reshape(-1, vocab)
        tgt = ids[s : s + micro, 1:].reshape(-1).to(device)
        ce_sum += float(F.cross_entropy(logits, tgt, reduction="sum"))
        lc = logits - logits.mean(dim=-1, keepdim=True)
        logit_sq += float((lc * lc).sum())
        logit_n += lc.numel()
        ntok += int(tgt.numel())
    return ce_sum / ntok, (logit_sq / logit_n) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, eval_bin=True)
    ap.add_argument("--landscape-cond", default="baseline", help="condition whose checkpoint defines the landscape")
    ap.add_argument("--eval-tokens", type=int, default=200_000)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--grid", type=int, default=21, help="points per landscape axis")
    ap.add_argument("--log2-range", type=float, default=2.0, help="axes span 2^[-r, +r]")
    ap.add_argument("--out-dir", type=Path, default=default_results_dir() / "gauge")
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = pick_device(args.device)
    ids_seqs = load_eval_blocks(args.eval_bin, args.eval_tokens, args.seq_len)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(args.out_dir, "run_gauge_landscape", args, args.seed)
    print(f"[eval] {ids_seqs.shape[0]} blocks x {args.seq_len}", flush=True)

    def final_checkpoint(cond):
        cfg = condition_config(args.ckpt_root, cond)
        st = resolve_step(args.ckpt_root, cond, "final")
        model = load_body(args.ckpt_root, cond, st, cfg["model_size"], args.seq_len, dev)
        rd = load_readout(args.ckpt_root, cond, st, dev)
        _, g = cache_hidden(model, ids_seqs, dev, args.micro, return_pre_ln=True)
        ln = model.gpt_neox.final_layer_norm
        return cfg, st, ln.weight.detach().float(), ln.bias.detach().float(), ln.eps, rd["wu"], g

    # PART 1: temperature conservation across conditions
    rows1 = []
    for cond in args.conditions:
        cfg, st, gamma, beta, eps, W, g = final_checkpoint(cond)
        nll, scale = nll_and_scale(g, ids_seqs, gamma, beta, W, eps, dev, args.micro)
        rn = W.norm(dim=1).mean().item()
        gain = gamma.norm().item()
        rows1.append(
            dict(
                condition=cond,
                step=st,
                readout_lr_mult=cfg.get("readout_lr_mult"),
                warmup_steps=cfg.get("warmup_steps"),
                wu_row_norm=round(rn, 4),
                lnf_gain=round(gain, 4),
                norm_x_gain=round(rn * gain, 3),
                logit_scale=round(scale, 4),
                nll=round(nll, 5),
            )
        )
        print(f"[temp] {cond}: nll={nll:.4f} logit_scale={scale:.3f} rn={rn:.4f} gain={gain:.4f}", flush=True)
        del g
        if dev == "cuda":
            torch.cuda.empty_cache()
    write_csv(args.out_dir / "temperature_conservation.csv", rows1)

    # PART 2: 2D loss landscape (gauge axis vs off-gauge axis)
    _, st, gamma0, beta0, eps, W0, g = final_checkpoint(args.landscape_cond)
    axis = np.logspace(-args.log2_range, args.log2_range, args.grid, base=2.0)
    rows2 = []
    for a in axis:  # alpha: along the gauge (gamma, beta * a; W / a) -> loss should be flat
        for b in axis:  # beta: off the gauge (W * b only) -> loss should curve
            nll, scale = nll_and_scale(g, ids_seqs, gamma0 * a, beta0 * a, (W0 / a) * b, eps, dev, args.micro)
            rows2.append(
                dict(
                    condition=args.landscape_cond,
                    step=st,
                    alpha=round(float(a), 6),
                    beta=round(float(b), 6),
                    log2_alpha=round(float(np.log2(a)), 4),
                    log2_beta=round(float(np.log2(b)), 4),
                    nll=round(nll, 5),
                    logit_scale=round(scale, 4),
                )
            )
        print(f"[land] alpha={a:.3f} done", flush=True)
    write_csv(args.out_dir / "gauge_landscape.csv", rows2)

    gauge = [r["nll"] for r in rows2 if abs(r["log2_beta"]) < 1e-6]
    offg = [r["nll"] for r in rows2 if abs(r["log2_alpha"]) < 1e-6]
    print(f"\n[done] along gauge: {min(gauge):.4f}..{max(gauge):.4f} (spread {max(gauge) - min(gauge):.4f})")
    print(f"       off  gauge: {min(offg):.4f}..{max(offg):.4f} (spread {max(offg) - min(offg):.4f})")


if __name__ == "__main__":
    main()
