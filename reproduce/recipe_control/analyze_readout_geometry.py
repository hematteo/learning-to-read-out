#!/usr/bin/env python3
"""Readout-geometry analysis across conditions x checkpoints (CPU, no GPU).

For each checkpoint it loads the fp16 state_dict and characterises how the
output readout W_U (embed_out.weight, shape (V, d)) develops, vs the input
embedding W_E and the final-LN gain. Emits a tidy CSV + a console summary.

Metrics per checkpoint:
  W_U row-norm: mean/std/median/max + tail fraction (rows > 1.5x median)
  W_U spectrum: stable_rank = ||W||_F^2 / s1^2 ; eff_rank = exp(entropy of s_i/sum)
                top1_frac = s1^2 / sum(s_i^2) ; s1, s2
  W_E row-norm mean (untied input embedding, for contrast)
  lnf_gain: ||final_layer_norm.weight|| , mean, max
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def wu_geometry(W: torch.Tensor) -> dict:
    W = W.float()  # (V, d)
    rn = W.norm(dim=1)
    med = rn.median()
    s = torch.linalg.svdvals(W)  # (d,) descending
    s2 = s * s
    p = s / s.sum()
    entropy = -(p * (p + 1e-12).log()).sum()
    return dict(
        rn_mean=rn.mean().item(),
        rn_std=rn.std().item(),
        rn_med=med.item(),
        rn_max=rn.max().item(),
        rn_tail_frac=(rn > 1.5 * med).float().mean().item(),
        stable_rank=(s2.sum() / s2[0]).item(),
        eff_rank=torch.exp(entropy).item(),
        top1_frac=(s2[0] / s2.sum()).item(),
        s1=s[0].item(),
        s2v=s[1].item(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True, type=Path)
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline", "warmup_short", "wu_lr_0p25", "wu_lr_4x"],
    )
    ap.add_argument("--out-csv", type=Path, default=Path("readout_geometry.csv"))
    args = ap.parse_args()

    rows = []
    for cond in args.conditions:
        cdir = args.ckpt_root / cond
        cfg = json.loads((cdir / "config.json").read_text())
        steps = sorted(
            int(p.name[4:])
            for p in (cdir / "ckpts").glob("step*")
            if p.name[4:].isdigit()
        )
        for st in steps:
            f = cdir / "ckpts" / f"step{st}" / "model_fp16.pt"
            if not f.is_file():
                continue
            sd = torch.load(f, map_location="cpu")
            wu = sd["embed_out.weight"]
            we_key = next((k for k in sd if k.endswith("embed_in.weight")), None)
            lnf_key = next((k for k in sd if "final_layer_norm.weight" in k), None)
            g = wu_geometry(wu)
            g["we_rn_mean"] = (
                sd[we_key].float().norm(dim=1).mean().item() if we_key else float("nan")
            )
            lnf = sd[lnf_key].float() if lnf_key else None
            g["lnf_norm"] = lnf.norm().item() if lnf is not None else float("nan")
            g["lnf_max"] = lnf.max().item() if lnf is not None else float("nan")
            tokens = st * cfg["global_batch"] * 2048
            rows.append(
                dict(
                    cond=cond,
                    wu_mult=cfg["readout_lr_mult"],
                    warmup=cfg["warmup_steps"],
                    step=st,
                    tokens=tokens,
                    **g,
                )
            )

    fields = list(rows[0].keys())
    with args.out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out_csv}\n")

    # console summary: final logged step per condition
    print(
        f"{'cond':<12}{'tok':>8}{'rn_mean':>9}{'rn_max':>8}{'stable_rk':>10}"
        f"{'eff_rk':>8}{'top1%':>7}{'we_rn':>7}{'lnf':>8}"
    )
    for cond in args.conditions:
        cr = [r for r in rows if r["cond"] == cond]
        if not cr:
            continue
        r = max(cr, key=lambda x: x["step"])
        print(
            f"{cond:<12}{r['tokens'] / 1e6:>7.0f}M{r['rn_mean']:>9.3f}{r['rn_max']:>8.3f}"
            f"{r['stable_rank']:>10.2f}{r['eff_rank']:>8.2f}{r['top1_frac'] * 100:>6.1f}%"
            f"{r['we_rn_mean']:>7.3f}{r['lnf_norm']:>8.2f}"
        )


if __name__ == "__main__":
    main()
