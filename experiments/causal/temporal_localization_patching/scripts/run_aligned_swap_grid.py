"""Aligned readout-swap grid: gauge/scale-controlled version of temporal patching.

The bare swap-grid (`temporal_patch_grid.py`) replaces W_U_t with W_U_s and
asks whether step-1000 is the swap-grid optimum. A reviewer's first move is:
"the basin is just gauge alignment — match means/scales/orthogonal frame
and the optimum disappears."

This script answers that. For each (h_t, W_U_s) cell we evaluate the swap
under N alignment ladders applied to W_U_s before the dot product:

    none        : raw W_U_s (the existing baseline)
    mean        : shift W_U_s so its column mean equals W_U_t's
    scale       : center, then rescale to match Frobenius norm of W_U_t
    row_norm    : per-row L2 calibration (token-wise rescale to W_U_t's row L2)
    procrustes  : centered orthogonal Procrustes (W_U_s @ R, R unitary)

Resume-safe: per-cell JSON shards under shards/, atomic writes, exists-check.
Outputs CSVs aggregated on completion. One row per (alignment, h_t, W_U_s).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
TPM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TPM_DIR))
import temporal_patch_metrics as TPM  # noqa: E402

from readout.core.repro import git_commit  # noqa: E402
from readout.core.resume import (  # noqa: E402
    aggregate_json_shards,
    atomic_write_json,
    iter_undone,
)
from readout.probes.readout_swap import ALIGN_MODES  # noqa: E402
from readout.probes.readout_swap import align_readout as align

# Gauge-alignment ladder (none/mean/scale/row_norm/procrustes). Canonical
# implementation lives in readout.probes.readout_swap.align_readout; imported here
# so there is a single source of truth (regression-tested in tests/test_readout_swap.py).
ALIGNMENTS = list(ALIGN_MODES)


def cell_metrics(
    h_LN: torch.Tensor,
    W: torch.Tensor,
    W_native: torch.Tensor,
    ids_seqs: torch.Tensor,
    *,
    device: str,
    batch_seqs: int,
) -> dict:
    """Per-cell global metrics: NLL, KL to native, top-1 agreement, logit R^2."""
    targets = ids_seqs[:, 1:]
    Wd = W.to(device).float()
    Wnd = W_native.to(device).float()

    nll_sum = 0.0
    nll_native_sum = 0.0
    kl_sum = 0.0
    top1 = 0
    top1_native = 0
    top1_agree = 0
    n_tok = 0
    res_sse = 0.0
    cent_native_var = 0.0

    for s in range(0, h_LN.shape[0], batch_seqs):
        h = h_LN[s : s + batch_seqs].to(device).float()
        tgt = targets[s : s + batch_seqs].to(device)
        logits = (h @ Wd.T)[:, :-1, :]
        logits_native = (h @ Wnd.T)[:, :-1, :]
        logp = F.log_softmax(logits, dim=-1)
        logp_native = F.log_softmax(logits_native, dim=-1)

        nll_sum += float(-logp.gather(-1, tgt.unsqueeze(-1)).sum().item())
        nll_native_sum += float(-logp_native.gather(-1, tgt.unsqueeze(-1)).sum().item())
        p = logp.exp()
        kl_sum += float((p * (logp - logp_native)).sum(-1).sum().item())

        am = logp.argmax(dim=-1)
        am_native = logp_native.argmax(dim=-1)
        top1 += int((am == tgt).sum().item())
        top1_native += int((am_native == tgt).sum().item())
        top1_agree += int((am == am_native).sum().item())

        diff = logits - logits_native
        res_sse += float(diff.pow(2).sum().item())
        cn = logits_native - logits_native.mean(dim=-1, keepdim=True)
        cent_native_var += float(cn.pow(2).sum().item())

        n_tok += int(tgt.numel())
        del logits, logits_native, logp, logp_native, p, am, am_native
        if device == "cuda":
            torch.cuda.empty_cache()

    nll = nll_sum / n_tok
    nll_native = nll_native_sum / n_tok
    return {
        "n_tokens": n_tok,
        "nll": nll,
        "nll_native": nll_native,
        "delta_nll": nll - nll_native,
        "ppl": math.exp(nll),
        "kl_to_native": kl_sum / n_tok,
        "top1": top1 / n_tok,
        "top1_native": top1_native / n_tok,
        "top1_agreement": top1_agree / n_tok,
        "centered_logit_r2": (
            1.0 - res_sse / cent_native_var if cent_native_var > 0 else float("nan")
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=["pythia-160m", "pythia-1b", "pythia-6.9b"],
        default="pythia-1b",
    )
    ap.add_argument("--d-sae", type=int, default=24576)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--h-eval-steps", type=int, nargs="+", default=[256, 512, 1000, 2000, 143000]
    )
    ap.add_argument(
        "--snap-steps",
        type=int,
        nargs="+",
        default=None,
        help="Subset of canonical 32 W_U snapshot steps; default = all 32.",
    )
    ap.add_argument(
        "--alignments",
        nargs="+",
        default=ALIGNMENTS,
        help=f"Subset of {ALIGNMENTS}",
    )
    ap.add_argument("--batch-seqs", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/experiments/causal/temporal_localization_patching/aligned_readout_swaps/run0"
        ),
    )
    args = ap.parse_args()

    if args.model == "pythia-160m":
        TPM.ACTIVE_CFG = TPM.CFG_PYTHIA_160M
        TPM.ACTIVE_D_SAE = 8192
    elif args.model == "pythia-1b":
        TPM.ACTIVE_CFG = TPM.CFG_PYTHIA_1B
        TPM.ACTIVE_D_SAE = args.d_sae
    elif args.model == "pythia-6.9b":
        TPM.ACTIVE_CFG = TPM.CFG_PYTHIA_6_9B
        TPM.ACTIVE_D_SAE = args.d_sae
    TPM.CORPUS_TOKENS = TPM.CORPUS_TOKENS_PYTHIA
    TPM._refresh_aliases()
    cfg = TPM.ACTIVE_CFG

    snap_steps = args.snap_steps or list(cfg.steps_canonical)
    bad = sorted(set(args.alignments) - set(ALIGNMENTS))
    if bad:
        raise ValueError(f"unknown alignments: {bad}; valid={ALIGNMENTS}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    eval_data = torch.load(TPM.CORPUS_TOKENS, weights_only=False)
    ids = eval_data["ids"]
    n_full = (len(ids) // TPM.SEQ_LEN) * TPM.SEQ_LEN
    ids_seqs = ids[:n_full].view(-1, TPM.SEQ_LEN)
    print(
        f"[setup] model={cfg.model_name}  align={args.alignments}  "
        f"cells = {len(args.alignments)} alignments x "
        f"{len(args.h_eval_steps)} h_t x {len(snap_steps)} W_s = "
        f"{len(args.alignments) * len(args.h_eval_steps) * len(snap_steps)}",
        flush=True,
    )

    manifest = {
        "model": cfg.model_name,
        "seed": args.seed,
        "alignments": args.alignments,
        "h_eval_steps": args.h_eval_steps,
        "snapshot_steps": snap_steps,
        "n_eval_seqs": int(ids_seqs.shape[0]),
        "seq_len": TPM.SEQ_LEN,
        "git_commit": git_commit(),
    }
    atomic_write_json(args.out_dir / "manifest.json", manifest)

    items = [
        (al, h, s)
        for al in args.alignments
        for h in args.h_eval_steps
        for s in snap_steps
    ]

    def shard_path(item) -> Path:
        al, h, s = item
        return shard_dir / f"a-{al}_h{h}_s{s}.json"

    h_LN_cache: dict[int, dict] = {}
    W_cache: dict[int, torch.Tensor] = {}

    t0 = time.time()
    for item in iter_undone(items, shard_path, label="cell"):
        al, h_step, snap_step = item
        if h_step not in h_LN_cache:
            cached = TPM.cache_or_build_hLN(h_step, ids_seqs)
            h_LN_cache[h_step] = cached
        h_LN = h_LN_cache[h_step]["h_LN"]
        # Older caches from build_hln_cache.py omit W_U_orig; fall back to the
        # snapshot W_U at h_step (which IS the native readout for Pythia models).
        W_native = h_LN_cache[h_step].get("W_U_orig")
        if W_native is None:
            W_native = TPM.load_snapshot(h_step).float()

        if snap_step not in W_cache:
            W_cache[snap_step] = TPM.load_snapshot(snap_step).float()
        W_s = W_cache[snap_step]

        W_aligned = align(W_s, W_native, al)
        m = cell_metrics(
            h_LN,
            W_aligned,
            W_native,
            ids_seqs,
            device=args.device,
            batch_seqs=args.batch_seqs,
        )
        row = {
            "model": cfg.model_name,
            "seed": args.seed,
            "alignment": al,
            "h_eval_step": h_step,
            "snapshot_step": snap_step,
            **m,
        }
        atomic_write_json(shard_path(item), row)
        print(
            f"  [{al:>10}] h_t={h_step:>6} W_s={snap_step:>6} "
            f"nll={m['nll']:.4f} (Δ={m['delta_nll']:+.4f}) "
            f"KL={m['kl_to_native']:.3f} top1agree={m['top1_agreement']:.3f}",
            flush=True,
        )
        del W_aligned
        if args.device == "cuda":
            torch.cuda.empty_cache()

        # Bound the W cache so we don't hold all 32 snapshots forever.
        if len(W_cache) > 8:
            old_step = next(iter(W_cache))
            del W_cache[old_step]
        gc.collect()

    n_done = sum(1 for it in items if shard_path(it).exists())
    if n_done == len(items):
        n = aggregate_json_shards(
            shard_dir, args.out_dir / "summary.csv", key="snapshot_step"
        )
        print(f"[done] aggregated {n} rows -> {args.out_dir}/summary.csv", flush=True)
    else:
        print(f"[partial] {n_done}/{len(items)} cells; rerun to complete", flush=True)

    print(f"[elapsed] {time.time() - t0:.1f}s", flush=True)
    Path(args.out_dir / "manifest.json").write_text(
        json.dumps({**manifest, "elapsed_s": round(time.time() - t0, 2)}, indent=2)
    )


if __name__ == "__main__":
    main()
