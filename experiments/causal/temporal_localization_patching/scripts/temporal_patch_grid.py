"""Full h_t × W_U_s causal grid.

Caches h_LN from several checkpoints (default: 256, 512, 1000, 2000, 143000)
and evaluates every snapshot's W_U_s against each h_t. Produces a
co-adaptation map answering:

  Q1.  Does W_U_1000 improve h_512 more than nearby readouts?
  Q2.  Does h_1000 uniquely prefer W_U_1000?

Per (h_t, W_U_s) cell, computes:

  global_nll       per-token cross-entropy on the eval corpus (nats / token)
  top1, top5       fraction of tokens where target is argmax / in top-5
  entropy          mean entropy of the predicted distribution
  kl_to_native     KL( softmax(h_t W_U_s.T)  ||  softmax(h_t W_U_t.T) )

For each proto-token concept C (the seven families from temporal_patch_metrics):

  target_nll[C]    mean per-token NLL on tokens whose target falls in C
  concept_mass[C]  logsumexp_C - logsumexp_{matched_null_C} (per-context mean)

Outputs:

  results/temporal_patch_grid/<tag>/
      manifest.json
      summary_global.csv   one row per (h_t, W_U_s)
      summary_concept.csv  one row per (h_t, W_U_s, concept)
      raw.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import readout.dynamics.temporal_patch as TPM  # noqa: E402

DEVICE = TPM.DEVICE


def per_cell_metrics(
    h_LN: torch.Tensor,
    W_U: torch.Tensor,
    W_U_native: torch.Tensor,
    ids_seqs: torch.Tensor,
    target_concept_ix: np.ndarray,
    n_concepts: int,
    batch_seqs: int = 4,
) -> dict:
    """Compute global + per-concept target-NLL for one (h_t, W_U_s) cell.

    target_concept_ix: (N*(T-1),) int array; -1 = not in any concept,
    else index into [0, n_concepts).
    """
    N, T, d = h_LN.shape
    targets = ids_seqs[:, 1:]  # (N, T-1)
    W_dev = W_U.to(DEVICE)
    W_native_dev = W_U_native.to(DEVICE)

    nll_sum = 0.0
    top1 = 0
    top5 = 0
    ent_sum = 0.0
    kl_sum = 0.0
    n_tok = 0
    nll_per_concept = np.zeros(n_concepts, dtype=np.float64)
    n_per_concept = np.zeros(n_concepts, dtype=np.int64)

    for s in range(0, N, batch_seqs):
        h = h_LN[s : s + batch_seqs].to(DEVICE)
        tgt = targets[s : s + batch_seqs].to(DEVICE)
        logits = (h @ W_dev.T)[:, :-1, :].float()
        logits_native = (h @ W_native_dev.T)[:, :-1, :].float()
        logp = F.log_softmax(logits, dim=-1)
        logp_native = F.log_softmax(logits_native, dim=-1)

        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll_sum += float(nll.sum().item())

        top5_idx = logp.topk(5, dim=-1).indices
        eq = top5_idx == tgt.unsqueeze(-1)
        top1 += int(eq[..., 0].sum().item())
        top5 += int(eq.any(dim=-1).sum().item())

        p = logp.exp()
        ent_sum += float(-(p * logp).sum(-1).sum().item())
        kl = (p * (logp - logp_native)).sum(-1)
        kl_sum += float(kl.sum().item())
        n_tok += int(nll.numel())

        # Per-concept stratification of the target NLL.
        nll_flat = nll.reshape(-1).cpu().numpy()  # (b * (T-1),)
        b = tgt.shape[0]
        # The chunk of target_concept_ix corresponding to this batch:
        start_flat = s * (T - 1)
        end_flat = start_flat + b * (T - 1)
        ix = target_concept_ix[start_flat:end_flat]
        for c in range(n_concepts):
            mask = ix == c
            if mask.sum():
                nll_per_concept[c] += float(nll_flat[mask].sum())
                n_per_concept[c] += int(mask.sum())

        del logits, logits_native, logp, logp_native, p
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    nll_mean = nll_sum / n_tok
    return {
        "n_tokens": n_tok,
        "nll": nll_mean,
        "perplexity": math.exp(nll_mean),
        "top1": top1 / n_tok,
        "top5": top5 / n_tok,
        "entropy": ent_sum / n_tok,
        "kl_to_native": kl_sum / n_tok,
        "nll_per_concept": (nll_per_concept / np.maximum(1, n_per_concept)).tolist(),
        "n_per_concept": n_per_concept.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=["pythia-160m", "pythia-1b", "pythia-6.9b", "olmo-2-7b"],
        default="pythia-1b",
    )
    ap.add_argument("--d-sae", type=int, default=24576)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--h-eval-steps", type=int, nargs="+", default=[256, 512, 1000, 2000, 143000]
    )
    ap.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Subset of GE_STEPS_32 for W_U_s; default = all 32.",
    )
    ap.add_argument("--batch-seqs", type=int, default=4)
    ap.add_argument(
        "--max-cm-contexts",
        type=int,
        default=256,
        help="Per-concept context cap for the concept-mass metric.",
    )
    ap.add_argument("--n-match", type=int, default=99)
    ap.add_argument("--n-concept-tokens", type=int, default=256)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("results/temporal_patch_grid/run0")
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
    elif args.model == "olmo-2-7b":
        TPM.ACTIVE_CFG = TPM.CFG_OLMO_2_7B
        TPM.ACTIVE_D_SAE = args.d_sae
    else:
        raise ValueError(f"Unknown --model {args.model}")
    # OLMo uses its own tokenized eval corpus; Pythia models share one.
    TPM.CORPUS_TOKENS = (
        TPM.CORPUS_TOKENS_OLMO
        if args.model == "olmo-2-7b"
        else TPM.CORPUS_TOKENS_PYTHIA
    )
    TPM._refresh_aliases()
    cfg = TPM.ACTIVE_CFG

    snap_steps = args.steps if args.steps is not None else list(cfg.steps_canonical)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[seed {args.seed}] model={cfg.model_name}  "
        f"h_t × W_U_s = {len(args.h_eval_steps)} × {len(snap_steps)} = "
        f"{len(args.h_eval_steps) * len(snap_steps)} cells",
        flush=True,
    )

    # Eval corpus + tokenizer + concepts.
    eval_data = torch.load(TPM.CORPUS_TOKENS, weights_only=False)
    ids = eval_data["ids"]
    n_full = (len(ids) // TPM.SEQ_LEN) * TPM.SEQ_LEN
    ids_seqs = ids[:n_full].view(-1, TPM.SEQ_LEN)
    targets_flat = ids_seqs[:, 1:].reshape(-1).numpy()

    terminal_step = cfg.steps_canonical[-1]
    print(
        f"  loading terminal W_U (step={terminal_step}) + token meta + concept registry...",
        flush=True,
    )
    W_U_terminal = TPM.load_snapshot(terminal_step)
    V = W_U_terminal.shape[0]
    tok = TPM.load_tokenizer()
    meta = TPM.build_token_meta(tok, ids[:n_full], V_explicit=V)
    concepts = TPM.build_concept_registry(meta)
    pool = TPM.build_pools(meta, W_U_terminal)
    concept_names = list(concepts.keys())
    n_concepts = len(concept_names)

    # Per-target-token concept index: -1 if not in any, else index in concept_names.
    # If a token belongs to multiple concepts (shouldn't happen with the current
    # disjoint predicates but be safe), use first match.
    print("  assigning per-target-token concept index...", flush=True)
    concept_membership = np.stack(
        [concepts[n].member_mask for n in concept_names]
    )  # (C, V)
    target_concept_ix = np.full(len(targets_flat), -1, dtype=np.int32)
    for c, name in enumerate(concept_names):
        m = concept_membership[c, targets_flat]
        # Only assign if not already (first concept wins).
        target_concept_ix = np.where(
            (target_concept_ix == -1) & m, c, target_concept_ix
        )
    counts = {
        n: int((target_concept_ix == c).sum()) for c, n in enumerate(concept_names)
    }
    print("  per-target concept counts:", counts, flush=True)
    print(f"  unassigned targets: {int((target_concept_ix == -1).sum()):,}", flush=True)

    # Pre-build per-concept context-mass evaluation sets, fixed across all cells.
    rng = np.random.default_rng(TPM.RNG_SEED + args.seed)
    print("  building per-concept candidate sets (fixed across cells)...", flush=True)
    concept_sets: dict[str, dict] = {}
    for cn, c in concepts.items():
        det_rng = np.random.default_rng(hash(("concept_sample", cn)) % 2**32)
        in_C_all = np.where(c.member_mask)[0]
        if len(in_C_all) > args.n_concept_tokens:
            w = meta.log_freq[in_C_all] + 1e-3
            w = w / w.sum()
            in_C = np.sort(
                det_rng.choice(in_C_all, size=args.n_concept_tokens, replace=False, p=w)
            )
        else:
            in_C = in_C_all
        null_C = TPM.build_concept_null(c, meta, pool, in_C, rng)
        concept_sets[cn] = {"in_C": in_C, "null_C": null_C, "concept": c}

    manifest = {
        "model": cfg.model_name,
        "seed": args.seed,
        "h_eval_steps": args.h_eval_steps,
        "snapshot_steps": snap_steps,
        "batch_seqs": args.batch_seqs,
        "n_concept_tokens": args.n_concept_tokens,
        "n_match": args.n_match,
        "max_cm_contexts": args.max_cm_contexts,
        "concept_names": concept_names,
        "concept_target_counts": counts,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summary_global = args.out_dir / "summary_global.csv"
    summary_concept = args.out_dir / "summary_concept.csv"
    for p in (summary_global, summary_concept):
        if p.exists():
            p.unlink()

    raw_records: list[dict] = []

    # Outer loop: h_t. Load h_LN once per h_t.
    for h_step in args.h_eval_steps:
        print(f"\n=== h_t = {h_step} ===", flush=True)
        cached = TPM.cache_or_build_hLN(h_step, ids_seqs)
        h_LN = cached["h_LN"]
        W_U_native = cached["W_U_orig"]

        # Pre-compute concept-mass contexts for this h_t (per-target distractors
        # are independent of W_U_s and only depend on h_LN @ targets).
        # The h is gathered once at the predictive position per concept.
        ctx_by_concept: dict[str, dict] = {}
        for cn, cs in concept_sets.items():
            h_C, target_ids = TPM.collect_concept_contexts(
                h_LN,
                ids_seqs,
                cs["concept"],
                max_per_concept=args.max_cm_contexts,
                rng=rng,
            )
            ctx_by_concept[cn] = {"h": h_C, "target_ids": target_ids}

        for snap_step in snap_steps:
            t0 = time.time()
            W = TPM.load_snapshot(snap_step)
            m = per_cell_metrics(
                h_LN,
                W,
                W_U_native,
                ids_seqs,
                target_concept_ix,
                n_concepts,
                batch_seqs=args.batch_seqs,
            )

            # Concept-mass per concept under this (h_t, W_U_s).
            cm_per_concept: dict[str, float] = {}
            for cn, cs in concept_sets.items():
                ctx = ctx_by_concept[cn]
                if ctx["h"].numel() == 0:
                    cm_per_concept[cn] = float("nan")
                    continue
                cm = TPM.concept_mass_score(ctx["h"], W, cs["in_C"], cs["null_C"])
                cm_per_concept[cn] = float(cm.mean())

            # Global row.
            global_row = {
                "model": cfg.model_name,
                "seed": args.seed,
                "h_eval_step": h_step,
                "snapshot_step": snap_step,
                "n_tokens": m["n_tokens"],
                "nll": m["nll"],
                "perplexity": m["perplexity"],
                "top1": m["top1"],
                "top5": m["top5"],
                "entropy": m["entropy"],
                "kl_to_native": m["kl_to_native"],
                "wallclock_s": round(time.time() - t0, 2),
            }
            new_file = not summary_global.exists()
            with summary_global.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(global_row.keys()))
                if new_file:
                    w.writeheader()
                w.writerow(global_row)

            # Per-concept rows.
            for c, cn in enumerate(concept_names):
                row = {
                    "model": cfg.model_name,
                    "seed": args.seed,
                    "h_eval_step": h_step,
                    "snapshot_step": snap_step,
                    "concept": cn,
                    "n_tokens_in_concept": m["n_per_concept"][c],
                    "target_nll": m["nll_per_concept"][c],
                    "concept_mass": cm_per_concept[cn],
                }
                new_file = not summary_concept.exists()
                with summary_concept.open("a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if new_file:
                        w.writeheader()
                    w.writerow(row)
                raw_records.append(row)

            print(
                f"  W_U_s={snap_step:>6}  "
                f"nll={m['nll']:.3f}  top1={m['top1']:.3f}  "
                f"H={m['entropy']:.3f}  KL={m['kl_to_native']:.3f}  "
                f"[{global_row['wallclock_s']:.1f}s]",
                flush=True,
            )
            del W
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        del h_LN, W_U_native, cached
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    torch.save({"records": raw_records, "manifest": manifest}, args.out_dir / "raw.pt")
    print(f"\nDone. Wrote {summary_global} and {summary_concept}.", flush=True)


if __name__ == "__main__":
    main()
