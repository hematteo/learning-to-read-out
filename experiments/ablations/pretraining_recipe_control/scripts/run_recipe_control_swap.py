#!/usr/bin/env python3
"""Cross-condition readout swap grid at the final checkpoint.

Body from condition A + readout W_U from condition B, same hidden states, on the
held-out slice. All arms share initialization and data order, so the bodies are
matched and any off-diagonal cost isolates whether a readout side recipe change
moved readout expression / hidden state-readout alignment rather than a global
output scale. This is the cross-condition supplement to the within-trajectory grid
(``run_trajectory_swap.py``).

Reuses the tested swap machinery:
  * ``readout.probes.readout_swap.align_readout``     gauge alignment ladder
    (none / row_norm / procrustes): a cost that survives procrustes is real, not
    gauge; one that row_norm removes was the per-token output-scale gauge.
  * ``readout.probes.readout_swap.swap_cell_metrics`` per-cell kernel (NLL,
    KL-to-native, top-1 agreement, centered-logit R^2).
  * ``readout.probes.weight_swap``                    correctness oracle
    (``--verify``): a physical embed_out swap + ``eval_bpt`` must match the fast path.

The fast path forwards each body once (``rc_common.cache_hidden`` returns the post-LN
hidden h and the pre-LN residual g); every readout graft is a re-LN + matmul.

Variants per cell:
  none / row_norm / procrustes : graft B's W_U (aligned to A's gauge); A keeps its LN.
  wu_plus_ln                   : graft B's W_U and B's final-LN gain+bias together (the
                                 co-adapted readout module), to separate the
                                 W_U-norm / LN-gain trade from a directional change.
                                 The native reference stays A's own (h_A, W_U^A).

CSV columns: body,readout,variant,step,nll,nll_native,delta_nll,kl_to_native,
top1,top1_native,top1_agreement,centered_logit_r2,n_tokens
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
from readout.probes.readout_swap import align_readout, swap_cell_metrics
from readout.probes.recipe_control_models import condition_config, load_body, load_readout, resolve_step
from readout.probes.weight_swap import eval_bpt, get_weight, swap_weight

VARIANTS = ["none", "row_norm", "procrustes", "wu_plus_ln"]


def load_condition(ckpt_root: Path, cond: str, step: str, seq_len: int, device: str) -> dict:
    """Body model (eval, fp32) + readout tensors for one condition."""
    cfg = condition_config(ckpt_root, cond)
    st = resolve_step(ckpt_root, cond, step)
    model = load_body(ckpt_root, cond, st, cfg["model_size"], seq_len, device)
    rd = load_readout(ckpt_root, cond, st, device)
    return dict(
        model=model,
        wu=rd["wu"],
        ln_w=rd["ln_w"],
        ln_b=rd["ln_b"],
        step=st,
        d_model=int(rd["wu"].shape[1]),
        ln_eps=model.gpt_neox.final_layer_norm.eps,
        wu_mult=cfg.get("readout_lr_mult"),
        warmup=cfg.get("warmup_steps"),
        model_size=cfg["model_size"],
    )


def run_grid(conds: dict, labels: list[str], ids_seqs, variants: list[str], micro: int, device: str) -> list[dict]:
    rows = []
    for body in labels:
        bd = conds[body]
        h_body, g = cache_hidden(bd["model"], ids_seqs, device, micro, return_pre_ln=True)
        for ro in labels:
            rd = conds[ro]
            for var in variants:
                if var == "wu_plus_ln":
                    # swapped path: B's LN on A's residual; native reference: A's true (h_A, W_U^A)
                    h_swap = F.layer_norm(g, (bd["d_model"],), rd["ln_w"].cpu(), rd["ln_b"].cpu(), bd["ln_eps"])
                    W, h_native = rd["wu"], h_body
                else:  # gauge ladder: A keeps its LN; B's W_U aligned to A's gauge
                    h_swap, W, h_native = h_body, align_readout(rd["wu"], bd["wu"], var), None
                m = swap_cell_metrics(h_swap, W, bd["wu"], ids_seqs, device=device, batch_seqs=micro, h_native=h_native)
                rows.append(
                    dict(
                        body=body,
                        readout=ro,
                        variant=var,
                        step=bd["step"],
                        **{k: round(m[k], 6) for k in ("nll", "nll_native", "delta_nll", "kl_to_native")},
                        **{k: round(m[k], 6) for k in ("top1", "top1_native", "top1_agreement", "centered_logit_r2")},
                        n_tokens=m["n_tokens"],
                    )
                )
        print(f"[swap] body {body} done ({ids_seqs.shape[0]} blocks x {len(labels)} readouts)", flush=True)
    return rows


def verify_cell(conds: dict, body: str, readout: str, ids_seqs, micro: int, fast_nll: float) -> float:
    """Oracle: a physical embed_out swap + eval_bpt must match the fast 'none' cell (nats)."""
    bd, rd = conds[body], conds[readout]
    model = bd["model"]
    dev = str(next(model.parameters()).device)
    orig = get_weight(model, "wu").data.clone()
    swap_weight(model, rd["wu"], target="wu")
    bpt = eval_bpt(model, ids_seqs, batch_size=micro, device=dev)
    swap_weight(model, orig, target="wu")  # restore
    nats = bpt * np.log(2)
    print(
        f"[verify] cell ({body}<-{readout}, none): eval_bpt={nats:.5f} nats  fast={fast_nll:.5f} nats  "
        f"|diff|={abs(nats - fast_nll):.5f}",
        flush=True,
    )
    return abs(nats - fast_nll)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, eval_bin=True)
    ap.add_argument("--step", default="final", help="checkpoint step ('final' or an int; nearest on a miss)")
    ap.add_argument("--eval-tokens", type=int, default=2_000_000)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--micro", type=int, default=4, help="sequences per forward")
    ap.add_argument("--out-csv", type=Path, default=default_results_dir() / "swap" / "swap_grid_recipe_control.csv")
    ap.add_argument("--verify", action="store_true", help="cross-check one cell with weight_swap + eval_bpt")
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = pick_device(args.device)
    ids_seqs = load_eval_blocks(args.eval_bin, args.eval_tokens, args.seq_len)
    print(f"[eval] {args.eval_bin}: {ids_seqs.shape[0]} blocks x {args.seq_len}", flush=True)

    conds = {lab: load_condition(args.ckpt_root, lab, args.step, args.seq_len, dev) for lab in args.conditions}
    for lab, c in conds.items():
        print(f"[cond] {lab}: step{c['step']} size={c['model_size']} wu_mult={c['wu_mult']} warmup={c['warmup']}")

    rows = run_grid(conds, args.conditions, ids_seqs, args.variants, args.micro, dev)

    verify = None
    if args.verify and len(args.conditions) >= 2:
        table = {(r["body"], r["readout"], r["variant"]): r["nll"] for r in rows}
        b, r = args.conditions[2 % len(args.conditions)], args.conditions[0]
        verify = verify_cell(conds, b, r, ids_seqs, args.micro, table[(b, r, "none")])
        if verify > 0.02:
            print(f"[verify] WARNING: fast path disagrees with the oracle by {verify:.4f} nats", flush=True)

    write_csv(args.out_csv, rows)
    write_provenance(args.out_csv.parent, "run_recipe_control_swap", args, args.seed, {"verify_abs_diff_nats": verify})
    print(f"\nwrote {len(rows)} rows -> {args.out_csv}\n", flush=True)

    dnll = {(r["body"], r["readout"], r["variant"]): r["delta_nll"] for r in rows}
    for var in args.variants:
        print(f"=== delta_nll grid (nats/token) variant={var}  [row=body, col=readout] ===")
        print("body\\readout".ljust(14) + "".join(c[:11].rjust(12) for c in args.conditions))
        for body in args.conditions:
            print(body[:13].ljust(14) + "".join(f"{dnll[(body, ro, var)]:+12.4f}" for ro in args.conditions))
        print()


if __name__ == "__main__":
    main()
