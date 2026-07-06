"""Feature-level rescue / knockout for the step-1000 compatibility basin.

The full readout-swap grid shows that swapping in W_U at step 1000 reduces
NLL when paired with hidden states from later checkpoints. This script
asks the next causal question: which sparse W_U-crosscoder features carry
the basin? Two paired experiments per (h_eval_step, subset):

  knockout : start from the reconstructed W_U at step 1000 and SUBTRACT
             a feature subset's rank-1 contribution. Compare NLL on h_eval
             before and after the knockout. A trajectory subset that drives
             the basin should knockout-out most of the recon_1000 advantage.

  addback  : start from the native W_U at h_eval_step (the "control" gauge)
             and ADD a feature subset's rank-1 contribution from step 1000.
             A trajectory subset that drives the basin should addback most
             of the swap advantage.

Subsets evaluated for each h_eval_step:
  trajectory_top_K  : top K features by (rate@1000 - rate@256), positive
  random_K          : K uniform-random features (seeded; same K)
  late_top_K        : top K features by rate@143000 - rate@1000 (control)
  norm_matched_K    : K features whose decoder norm at 1000 matches the
                      trajectory subset distribution (matched gauge control)

All firing rates / decoder norms are computed from the crosscoder forward
pass over the 32 W_U snapshots — no precomputed aggregates sidecar needed.

Resume-safe per (h_eval_step, subset_name) shard.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
TPM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TPM_DIR))
import temporal_patch_metrics as TPM  # noqa: E402

from readout.core.paths import release_path  # noqa: E402
from readout.core.repro import git_commit  # noqa: E402
from readout.core.resume import (  # noqa: E402
    aggregate_json_shards,
    atomic_write_json,
    iter_undone,
)
from readout.crosscoder.checkpoints import load_checkpoint  # noqa: E402
from readout.crosscoder.snapshots import load_snapshots  # noqa: E402
from readout.crosscoder.wu_adapter import preprocess_snapshots  # noqa: E402


def encode_at_step(
    sd: dict[str, torch.Tensor],
    accumulated: torch.Tensor,  # (V, D), already summed over K snapshots
    step_idx: int,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (feat_acts, decoder_norm_t, fired) at one snapshot index."""
    W_D = sd["W_D"][step_idx].to(device)  # (D, d)
    threshold = sd["activation_function.log_jumprelu_threshold"].exp().to(device)
    decoder_norm = torch.linalg.norm(W_D, dim=-1)  # (D,)
    scaled = accumulated * decoder_norm[None, :]
    fired = scaled > threshold[None, :]
    feature_acts = (scaled * fired) / decoder_norm.clamp_min(1e-12)[None, :]
    return feature_acts, decoder_norm, fired


def reconstruct(
    feature_acts: torch.Tensor,
    sd: dict[str, torch.Tensor],
    step_idx: int,
    stats_step: dict[str, torch.Tensor] | None,
    *,
    device: str,
) -> torch.Tensor:
    """W_recon (V, d) in raw model-space units."""
    W_D = sd["W_D"][step_idx].to(device)
    b_D = sd["b_D"][step_idx].to(device)
    recon_norm = feature_acts @ W_D + b_D
    if stats_step is not None:
        scale = stats_step["scale"].to(device).float()
        mean = stats_step["mean"].to(device).float()
        return recon_norm * scale + mean
    return recon_norm


def subset_contribution(
    feature_acts: torch.Tensor,
    sd: dict[str, torch.Tensor],
    step_idx: int,
    stats_step: dict[str, torch.Tensor] | None,
    subset: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    """Rank-|S| contribution sum_{k in S} a_k(t) * D_k(t) * scale_t -> (V, d)."""
    if subset.numel() == 0:
        V, _ = feature_acts.shape
        d = sd["W_D"][step_idx].shape[-1]
        return torch.zeros(V, d, device=device, dtype=torch.float32)
    W_D = sd["W_D"][step_idx].to(device)  # (D, d)
    a = feature_acts[:, subset]  # (V, |S|)
    d = W_D[subset]  # (|S|, d)
    contrib = a @ d  # (V, d) in normalized space
    if stats_step is not None:
        scale = stats_step["scale"].to(device).float()
        contrib = contrib * scale
    return contrib


def eval_logits(
    h_LN: torch.Tensor,
    ids_seqs: torch.Tensor,
    W: torch.Tensor,
    W_native: torch.Tensor,
    *,
    device: str,
    batch_seqs: int,
) -> dict[str, float]:
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
    for s in range(0, h_LN.shape[0], batch_seqs):
        h = h_LN[s : s + batch_seqs, :-1, :].to(device).float()
        tgt = targets[s : s + batch_seqs].to(device)
        logits = h @ Wd.T
        logits_native = h @ Wnd.T
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
        n_tok += int(tgt.numel())
        del logits, logits_native, logp, logp_native, p
        if device == "cuda":
            torch.cuda.empty_cache()
    nll = nll_sum / n_tok
    nll_native = nll_native_sum / n_tok
    return {
        "n_tokens": n_tok,
        "nll": nll,
        "nll_native_at_h": nll_native,
        "delta_nll": nll - nll_native,
        "kl_to_native_at_h": kl_sum / n_tok,
        "top1": top1 / n_tok,
        "top1_native_at_h": top1_native / n_tok,
        "top1_agreement": top1_agree / n_tok,
    }


def select_subsets(
    rates: dict[int, torch.Tensor],
    decoder_norms: dict[int, torch.Tensor],
    *,
    K: int,
    early_step: int,
    target_step: int,
    late_step: int,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """Return {name -> LongTensor of feature indices, length K each}."""
    D = rates[target_step].shape[0]
    delta_traj = (rates[target_step] - rates[early_step]).cpu().numpy()
    delta_late = (rates[late_step] - rates[target_step]).cpu().numpy()
    norms = decoder_norms[target_step].cpu().numpy()

    traj_idx = np.argsort(-delta_traj)[:K]
    late_idx = np.argsort(-delta_late)[:K]
    rand_idx = rng.choice(D, size=K, replace=False)

    # Norm-matched: pool of features NOT in traj_idx, sample K with closest
    # decoder norm @ target_step to traj_idx's norms (1-NN per traj feature).
    traj_set = set(int(x) for x in traj_idx)
    available = np.array([i for i in range(D) if i not in traj_set])
    if len(available) >= K:
        avail_norms = norms[available]
        targets_norms = norms[traj_idx]
        order = np.argsort(targets_norms)
        chosen = []
        used = set()
        for j in order:
            t = targets_norms[j]
            d2 = (avail_norms - t) ** 2
            d2[list(used)] = np.inf
            pick = int(np.argmin(d2))
            chosen.append(int(available[pick]))
            used.add(pick)
        norm_idx = np.array(chosen)
    else:
        norm_idx = rand_idx.copy()

    return {
        "trajectory_top_K": torch.from_numpy(traj_idx).long(),
        "random_K": torch.from_numpy(rand_idx).long(),
        "late_top_K": torch.from_numpy(late_idx).long(),
        "norm_matched_K": torch.from_numpy(norm_idx).long(),
        "empty": torch.tensor([], dtype=torch.long),  # for baseline rows
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=["pythia-1b", "pythia-160m", "pythia-6.9b"],
        default="pythia-1b",
    )
    ap.add_argument("--d-sae", type=int, default=24576)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--h-eval-steps", type=int, nargs="+", default=[256, 512, 1000, 2000])
    ap.add_argument("--target-step", type=int, default=1000)
    ap.add_argument("--early-step", type=int, default=256)
    ap.add_argument("--late-step", type=int, default=143000)
    ap.add_argument("--K", type=int, default=256, help="subset size")
    ap.add_argument("--batch-rows", type=int, default=2048)
    ap.add_argument("--batch-seqs", type=int, default=2)
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--snapshot-dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/experiments/causal/readout_feature_rescue/run0_pythia1b_d24576_seed0"),
    )
    ap.add_argument(
        "--eval-tokens",
        type=Path,
        default=REPO / "_archive/legacy_crosscoder_160m/intervention/eval_tokens.pt",
    )
    args = ap.parse_args()

    if args.model == "pythia-1b":
        TPM.ACTIVE_CFG = TPM.CFG_PYTHIA_1B
    elif args.model == "pythia-160m":
        TPM.ACTIVE_CFG = TPM.CFG_PYTHIA_160M
    elif args.model == "pythia-6.9b":
        TPM.ACTIVE_CFG = TPM.CFG_PYTHIA_6_9B
    TPM.ACTIVE_D_SAE = args.d_sae
    TPM.CORPUS_TOKENS = TPM.CORPUS_TOKENS_PYTHIA
    TPM._refresh_aliases()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.ckpt or release_path(args.model, "W_U", dim=args.d_sae, seed=args.seed)
    cp = load_checkpoint(ckpt)
    sd = cp.state_dict
    steps_canonical = list(cp.steps)
    step_to_idx = {s: i for i, s in enumerate(steps_canonical)}

    needed_steps = sorted({args.target_step, args.early_step, args.late_step, *args.h_eval_steps})
    for s in needed_steps:
        if s not in step_to_idx:
            raise ValueError(f"step {s} not in checkpoint schedule {steps_canonical}")

    print(f"[setup] ckpt={ckpt}", flush=True)
    print(
        f"[setup] target={args.target_step} early={args.early_step} "
        f"late={args.late_step} h_eval={args.h_eval_steps} K={args.K}",
        flush=True,
    )

    mode = cp.training.get("input_preprocess") or "none"
    snapshots_full = load_snapshots(cp.model_name, steps_canonical)
    if args.snapshot_dtype == "bf16":
        snapshots_full = snapshots_full.to(torch.bfloat16)
    elif args.snapshot_dtype == "fp16":
        snapshots_full = snapshots_full.to(torch.float16)
    x_norm, stats = preprocess_snapshots(snapshots_full.float(), mode=mode)
    x_perm = x_norm.permute(1, 0, 2).contiguous()  # (V, K, d)
    del x_norm
    gc.collect()
    V, K, d = x_perm.shape

    # ---- Pre-compute the joint-encoder activation `accumulated` (V, D) once.
    # accumulated[v, k_feat] = sum_t x_perm[v, t, :] @ W_E[t][:, k_feat] + b_E[t, k_feat]
    print(f"[encode] accumulating joint encoder over {K} snapshots, V={V}", flush=True)
    W_E = sd["W_E"].to(args.device).float()  # (K, d, D)
    b_E = sd["b_E"].to(args.device).float()  # (K, D)
    D_sae = W_E.shape[-1]
    accumulated = torch.zeros((V, D_sae), dtype=torch.float32, device=args.device)
    for start in range(0, V, args.batch_rows):
        stop = min(start + args.batch_rows, V)
        x = x_perm[start:stop].to(args.device).float()
        local = torch.zeros((stop - start, D_sae), dtype=torch.float32, device=args.device)
        for k in range(K):
            local += x[:, k, :] @ W_E[k] + b_E[k]
        accumulated[start:stop] = local
        del x, local
        if args.device == "cuda":
            torch.cuda.empty_cache()
    del W_E, b_E

    # Per-step rates and decoder norms (only for needed_steps to bound memory).
    print("[encode] per-step rates + decoder norms", flush=True)
    rates: dict[int, torch.Tensor] = {}
    decoder_norms: dict[int, torch.Tensor] = {}
    feat_acts_at: dict[int, torch.Tensor] = {}  # only kept for target_step
    for s in needed_steps:
        idx = step_to_idx[s]
        feat_acts, dn, _ = encode_at_step(sd, accumulated, idx, device=args.device)
        rates[s] = (feat_acts > 0).float().mean(dim=0).cpu()
        decoder_norms[s] = dn.cpu()
        if s == args.target_step:
            feat_acts_at[s] = feat_acts  # keep on device
        else:
            del feat_acts
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ---- Subsets (computed once)
    rng = np.random.default_rng(args.rng_seed)
    subsets = select_subsets(
        rates,
        decoder_norms,
        K=args.K,
        early_step=args.early_step,
        target_step=args.target_step,
        late_step=args.late_step,
        rng=rng,
    )
    print(f"[subsets] {[(n, len(v)) for n, v in subsets.items()]}", flush=True)

    # ---- Reconstruct W at target_step (full and per-subset deltas).
    target_idx = step_to_idx[args.target_step]
    stats_target = None if stats is None else {"scale": stats["scale"][target_idx], "mean": stats["mean"][target_idx]}
    feat_acts_target = feat_acts_at[args.target_step]
    W_recon_target = reconstruct(feat_acts_target, sd, target_idx, stats_target, device=args.device).cpu()
    print(
        f"[recon] target_step={args.target_step} ||W_recon||={float(W_recon_target.norm()):.3f}",
        flush=True,
    )

    snapshots_full = None  # free
    gc.collect()

    # Manifest (small, for downstream plotting).
    manifest = {
        "model": cp.model_name,
        "ckpt": str(ckpt),
        "d_sae": args.d_sae,
        "seed": args.seed,
        "target_step": args.target_step,
        "early_step": args.early_step,
        "late_step": args.late_step,
        "h_eval_steps": args.h_eval_steps,
        "K": args.K,
        "rng_seed": args.rng_seed,
        "subset_indices": {n: v.tolist() for n, v in subsets.items()},
        "rates_at_target": rates[args.target_step].tolist(),
        "decoder_norms_at_target": decoder_norms[args.target_step].tolist(),
        "git_commit": git_commit(),
    }
    atomic_write_json(args.out_dir / "manifest.json", manifest)

    # ---- Eval grid: (h_eval_step, subset_name, mode in {knockout, addback}).
    items: list[tuple[int, str, str]] = []
    for h_step in args.h_eval_steps:
        items.append((h_step, "baseline_native_h", "baseline_native"))
        items.append((h_step, "baseline_recon_target", "baseline_recon"))
        for sname in subsets:
            if sname == "empty":
                continue
            items.append((h_step, sname, "knockout"))
            items.append((h_step, sname, "addback"))

    def shard_path(item: tuple[int, str, str]) -> Path:
        h, sn, mo = item
        return shard_dir / f"h{h}_{sn}_{mo}.json"

    # Pre-load eval ids.
    data = torch.load(args.eval_tokens, map_location="cpu", weights_only=False)
    ids = data["ids"].long()
    n_full = (len(ids) // TPM.SEQ_LEN) * TPM.SEQ_LEN
    ids_seqs = ids[:n_full].view(-1, TPM.SEQ_LEN)
    if args.max_seqs is not None:
        ids_seqs = ids_seqs[: args.max_seqs]
    print(
        f"[eval] using {ids_seqs.shape[0]} sequences x {TPM.SEQ_LEN} tokens",
        flush=True,
    )

    h_LN_cache: dict[int, dict] = {}
    for item in iter_undone(items, shard_path, label="cell"):
        h_step, sname, mode_ev = item
        if h_step not in h_LN_cache:
            h_LN_cache[h_step] = TPM.cache_or_build_hLN(h_step, ids_seqs)
        h_LN = h_LN_cache[h_step]["h_LN"]
        W_native_at_h = h_LN_cache[h_step].get("W_U_orig")
        if W_native_at_h is None:
            W_native_at_h = TPM.load_snapshot(h_step).float()

        if mode_ev == "baseline_native":
            W = W_native_at_h
            subset_used = []
        elif mode_ev == "baseline_recon":
            W = W_recon_target
            subset_used = []
        else:
            S = subsets[sname]
            contrib = subset_contribution(
                feat_acts_target,
                sd,
                target_idx,
                stats_target,
                S.to(args.device),
                device=args.device,
            ).cpu()
            if mode_ev == "knockout":
                W = W_recon_target - contrib
            else:  # addback
                W = W_native_at_h.float() + contrib
            subset_used = S.tolist()

        m = eval_logits(
            h_LN,
            ids_seqs,
            W,
            W_native_at_h,
            device=args.device,
            batch_seqs=args.batch_seqs,
        )
        row = {
            "model": cp.model_name,
            "h_eval_step": h_step,
            "subset": sname,
            "mode": mode_ev,
            "K": args.K if subset_used else 0,
            **m,
        }
        atomic_write_json(shard_path(item), row)
        print(
            f"  h={h_step} {sname:>22} {mode_ev:<14} "
            f"NLL={m['nll']:.4f} (Δ={m['delta_nll']:+.4f}) "
            f"KL={m['kl_to_native_at_h']:.3f} agree={m['top1_agreement']:.3f}",
            flush=True,
        )
        del W
        if args.device == "cuda":
            torch.cuda.empty_cache()

    n_done = sum(1 for it in items if shard_path(it).exists())
    if n_done == len(items):
        n = aggregate_json_shards(shard_dir, args.out_dir / "rescue_summary.csv", key="h_eval_step")
        print(
            f"[done] aggregated {n} rows -> {args.out_dir}/rescue_summary.csv",
            flush=True,
        )
    else:
        print(f"[partial] {n_done}/{len(items)} cells", flush=True)


if __name__ == "__main__":
    main()
