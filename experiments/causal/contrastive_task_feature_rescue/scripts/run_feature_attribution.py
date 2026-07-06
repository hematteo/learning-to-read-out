"""Sparse readout-feature attribution and intervention for contrastive tasks.

Pipeline (per task family, per (h_step, snapshot_step) cell):

  1. Encode all V vocab rows through the W_U crosscoder at ``snapshot_step``
     to get per-vocab feature activations a_{v,f} and decoders D_f.

  2. Reconstruct W_recon = sum_f a_{v,f} D_f * scale_t (matches
     ``run_step1000_feature_rescue.reconstruct``). Verify it
     approximately matches the native W_U at snapshot_step.

  3. For each example with answers (y+, y-), compute per-feature attribution
     to the contrastive margin:

         attr_f = (a_{y+, f} - a_{y-, f}) * (h · D_f * scale_t)

     summed over examples gives a per-feature score for each task family.

  4. Top-k interventions (paired):
       - **ablate**: zero out top-k features in W_recon, re-evaluate margin.
       - **preserve**: keep only top-k features (zero everything else), re-eval.
     Compare against decoder-norm + firing-rate matched random-feature controls.

  5. Specificity matrix: rows = task families (their top-k feature sets),
     columns = task families (eval). Cell = margin drop under that task's
     top-k ablation evaluated on this task's examples. Diagonal-heavy means
     the rescue is task-specific.

Outputs per cell (resume-safe shards under ``shards/``):
  - feature_scores: (D_sae,) attribution per feature
  - top_k_indices: list[int] of length K
  - margins:       baseline / ablate / preserve / control_ablate / control_preserve
  - specificity:   (n_families, n_families) margin-drop matrix

Usage:
    uv run python experiments/causal/contrastive_task_feature_rescue/scripts/run_feature_attribution.py \\
        --model pythia-1b \\
        --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \\
        --hidden-dir   results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden \\
        --snapshot-step 1000 --h-step 1000
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from readout.core.paths import (
    release_path,  # noqa: E402
    repo_root,  # noqa: E402
)
from readout.core.repro import git_commit  # noqa: E402
from readout.core.resume import atomic_write_json  # noqa: E402
from readout.crosscoder import inference  # noqa: E402
from readout.crosscoder.checkpoints import load_checkpoint  # noqa: E402
from readout.crosscoder.snapshots import load_snapshots  # noqa: E402
from readout.crosscoder.wu_adapter import preprocess_snapshots  # noqa: E402
from readout.probes import contrastive_tasks as CT  # noqa: E402
from readout.probes import readout_swap as RS  # noqa: E402

REPO = repo_root()


# ---------------------------------------------------------------------------
# Crosscoder encode/decode — math lives in readout.crosscoder.inference
# (equivalence-tested against the llamascopium module); these adapters carry
# this script's per-step `stats_step` payload convention.
# ---------------------------------------------------------------------------
def encode_at_step(
    sd: dict[str, torch.Tensor],
    encoder_preact: torch.Tensor,  # (V, D_sae) — joint encoder *preactivation*
    step_idx: int,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-step decoder-norm rescale + JumpReLU to the joint preactivation.

    `encoder_preact` is the sum over snapshots of `x[:, k, :] @ W_E[k] + b_E[k]`;
    *no* nonlinearity has been applied yet. JumpReLU happens here, with a
    per-step decoder-norm scale (so the same preactivation produces different
    final feature activations at different snapshot steps).

    Returns (feat_acts (V, D_sae), decoder_norm (D_sae,)) at one snapshot.
    """
    feature_acts, decoder_norm, _fired = inference.encode_at_step(sd, encoder_preact, step_idx, device=device)
    return feature_acts, decoder_norm


def reconstruct_W(
    feature_acts: torch.Tensor,
    sd: dict[str, torch.Tensor],
    step_idx: int,
    stats_step: dict[str, torch.Tensor] | None,
    *,
    device: str,
) -> torch.Tensor:
    if stats_step is None:
        return inference.decode_step(sd, feature_acts, step_idx, device=device)
    return inference.reconstruct_step(
        sd,
        feature_acts,
        step_idx,
        mean_k=stats_step["mean"].to(device).float(),
        scale_k=stats_step["scale"].to(device).float(),
        device=device,
    )


def feature_contribution(
    feature_acts: torch.Tensor,  # (V, D)
    sd: dict[str, torch.Tensor],
    step_idx: int,
    stats_step: dict[str, torch.Tensor] | None,
    subset: torch.Tensor,  # (K,)
    *,
    device: str,
) -> torch.Tensor:
    """sum_{f in subset} a_{:,f} * D_f * scale_t  ->  (V, d). No bias term."""
    contrib = inference.subset_decode(sd, feature_acts, step_idx, subset, device=device)
    if stats_step is not None:
        contrib = contrib * stats_step["scale"].to(device).float()
    return contrib


# ---------------------------------------------------------------------------
# Per-feature margin attribution
# ---------------------------------------------------------------------------
def per_feature_attribution(
    h: torch.Tensor,  # (N, d)
    feature_acts: torch.Tensor,  # (V, D_sae)
    W_D: torch.Tensor,  # (D_sae, d)
    scale_step: torch.Tensor | None,  # currently always scalar in center_scale mode
    y_plus: torch.Tensor,  # (N,)
    y_minus: torch.Tensor,  # (N,)
) -> torch.Tensor:
    """Mean-over-examples per-feature attribution to the contrastive margin.

    attr_f = mean_n  (a_{y+_n, f} - a_{y-_n, f}) * (h_n · D_f) * scale_t

    Note on `scale_step`: in `center_scale` preprocessing (the only mode
    used in the released crosscoders so far) the scale is a single scalar
    per snapshot, so the multiplication commutes through the dot product.
    The per-dim branch below is unvalidated — if a future preprocessing
    mode introduces a per-d scale it must be tested before using here.
    """
    device = h.device
    a_plus = feature_acts[y_plus.to(device)]  # (N, D)
    a_minus = feature_acts[y_minus.to(device)]  # (N, D)
    delta_a = (a_plus - a_minus).float()  # (N, D)
    Wd = W_D.float().to(device)  # (D, d)
    h_dot = h.float() @ Wd.T  # (N, D), h · D_f
    if scale_step is not None:
        s = scale_step.to(device).float()
        if s.numel() == 1:
            h_dot = h_dot * s.reshape(())  # scalar (center_scale)
        else:
            # Untested branch — guarded so it is obvious if it ever fires.
            raise NotImplementedError(
                "per-dim scale path is unvalidated; verify the algebra before enabling for a new preprocessing mode."
            )
    return (delta_a * h_dot).mean(dim=0).cpu()


def margins_with_W(
    h: torch.Tensor,
    W: torch.Tensor,
    y_plus: torch.Tensor,
    y_minus: torch.Tensor,
) -> torch.Tensor:
    return RS.compute_margin(h, W, y_plus, y_minus)


# ---------------------------------------------------------------------------
# Matched-random control
# ---------------------------------------------------------------------------
def matched_random_features(
    target_idx: np.ndarray,  # (K,)
    decoder_norms: np.ndarray,  # (D,)
    rates: np.ndarray,  # (D,)
    rng: np.random.Generator,
    *,
    sign_filter: np.ndarray | None = None,  # bool mask, True = eligible (e.g., positive attr)
) -> np.ndarray:
    """Pick K features outside ``target_idx`` whose (norm, rate) closely match.

    1-NN in the (log-norm, log-rate) plane; sample without replacement.
    If ``sign_filter`` is given, the candidate pool is intersected with
    ``sign_filter == True`` (e.g., the set of features whose attribution
    has the same sign as the top-K) — this catches sign-confounding in
    the matched-control comparison.
    """
    D = decoder_norms.shape[0]
    target_set = set(int(x) for x in target_idx)
    eligible = np.ones(D, dtype=bool)
    eligible[list(target_set)] = False
    if sign_filter is not None:
        eligible &= sign_filter
    available = np.flatnonzero(eligible)
    if len(available) < len(target_idx):
        # Not enough sign-matched candidates; fall back to unfiltered pool
        # (caller is responsible for noticing the dropped constraint).
        eligible = np.ones(D, dtype=bool)
        eligible[list(target_set)] = False
        available = np.flatnonzero(eligible)
    log_n = np.log(np.maximum(decoder_norms, 1e-12))
    log_r = np.log(np.maximum(rates, 1e-12))
    n_avail, r_avail = log_n[available], log_r[available]
    chosen: list[int] = []
    used: set[int] = set()
    order = rng.permutation(len(target_idx))
    for j in order:
        i = int(target_idx[j])
        d2 = (n_avail - log_n[i]) ** 2 + (r_avail - log_r[i]) ** 2
        for k in used:
            d2[k] = np.inf
        pick = int(np.argmin(d2))
        chosen.append(int(available[pick]))
        used.add(pick)
    return np.asarray(chosen, dtype=np.int64)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
MODEL_HF = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "olmo-2-7b": "allenai/OLMo-2-1124-7B",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=list(MODEL_HF.keys()),
        default="pythia-1b",
    )
    ap.add_argument("--d-sae", type=int, default=24576)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument(
        "--datasets-dir",
        type=Path,
        required=True,
        help="<family>.jsonl files (built by Exp 1's build_task_datasets.py).",
    )
    ap.add_argument(
        "--hidden-dir",
        type=Path,
        required=True,
        help="Cached hidden states from Exp 1 (run_swap_grid.py): expects files of the form <family>_h<step>.pt.",
    )
    ap.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Task families to evaluate (default = every <family>.jsonl found).",
    )
    ap.add_argument(
        "--snapshot-step",
        type=int,
        required=True,
        help="Crosscoder snapshot step S to use as the readout/feature gauge.",
    )
    ap.add_argument(
        "--h-step",
        type=int,
        required=True,
        help="Hidden-state step T (must have a cached <family>_h<T>.pt).",
    )
    ap.add_argument("--K", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument(
        "--rank-by",
        choices=["pos", "abs"],
        default="pos",
        help="Top-K ranking criterion: 'pos' = features with the largest "
        "positive attribution to the y+/y- margin (default; tests "
        "'features that PROMOTE y+'); 'abs' = features with the largest "
        "|attribution| (tests 'features that move the margin in either "
        "direction'). 'abs' is the more honest 'do these features carry "
        "the task' control.",
    )
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument(
        "--split-frac",
        type=float,
        default=0.5,
        help="Fraction of examples used for attribution. The remaining "
        "(1 - frac) is held out for ablate/preserve evaluation. "
        "Set to 1.0 to disable the split (legacy / leakage-prone).",
    )
    ap.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="RNG seed for the attribution / evaluation split.",
    )
    ap.add_argument(
        "--min-split-n",
        type=int,
        default=12,
        help="Minimum per-side n for the held-out split. Families with fewer "
        "examples fall back to the legacy same-set protocol; the manifest "
        "records which families were split vs. fallback.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-rows", type=int, default=2048)
    ap.add_argument(
        "--snapshot-dtype",
        choices=["fp32", "bf16", "fp16"],
        default="bf16",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/experiments/causal/contrastive_task_feature_rescue/run0"),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    model_hf = MODEL_HF[args.model]

    # Discover families.
    if args.families is None:
        family_paths = [p for p in sorted(args.datasets_dir.glob("*.jsonl")) if not p.name.endswith("_corrupt.jsonl")]
        families = [p.stem for p in family_paths]
    else:
        families = args.families
    if not families:
        raise FileNotFoundError(f"no <family>.jsonl files found under {args.datasets_dir}")
    print(f"[setup] families={families}", flush=True)

    # Load crosscoder + snapshots.
    ckpt = args.ckpt or release_path(args.model, "W_U", dim=args.d_sae, seed=args.seed)
    cp = load_checkpoint(ckpt)
    sd = cp.state_dict
    steps_canonical = list(cp.steps)
    step_to_idx = {s: i for i, s in enumerate(steps_canonical)}
    if args.snapshot_step not in step_to_idx:
        raise ValueError(f"snapshot_step {args.snapshot_step} not in {steps_canonical}")
    target_idx = step_to_idx[args.snapshot_step]
    print(f"[setup] ckpt={ckpt}", flush=True)
    print(f"[setup] snapshot_step={args.snapshot_step} -> idx {target_idx}", flush=True)

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

    # Joint encoder forward over snapshots (matches Exp 1 of run_step1000_feature_rescue).
    print(f"[encode] joint encoder accumulation V={V} K={K}", flush=True)
    W_E = sd["W_E"].to(args.device).float()
    b_E = sd["b_E"].to(args.device).float()
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
    feature_acts, decoder_norm = encode_at_step(
        sd,
        accumulated,
        target_idx,
        device=args.device,
    )
    rates = (feature_acts > 0).float().mean(dim=0).cpu().numpy()  # (D_sae,)
    decoder_norms_np = decoder_norm.cpu().numpy()

    # Reconstructed and native W_U at snapshot_step (CPU).
    stats_target = (
        None
        if stats is None
        else {
            "scale": stats["scale"][target_idx],
            "mean": stats["mean"][target_idx],
        }
    )
    W_recon = reconstruct_W(
        feature_acts,
        sd,
        target_idx,
        stats_target,
        device=args.device,
    ).cpu()
    W_native_s = snapshots_full[target_idx].float()  # native W_U at snapshot_step
    # Also slice out the native W_U at h_step (the readout the model actually
    # uses when forwarding hidden states at h_step). Required for the true
    # rescue baseline margin(t, t).
    if args.h_step in step_to_idx:
        W_native_h = snapshots_full[step_to_idx[args.h_step]].float()
        h_native_step_used = args.h_step
    else:
        # h_step is outside the canonical snapshot schedule (e.g., a finer
        # grid). Fall back to the snapshot-step native; flag this in the
        # output so callers know the rescue delta is degenerate.
        W_native_h = W_native_s
        h_native_step_used = args.snapshot_step
        print(
            f"[warn] h_step={args.h_step} not in canonical schedule; "
            f"using snapshot_step={args.snapshot_step} as h-native fallback",
            flush=True,
        )
    # ---- Per-snapshot reconstruction quality (EV / R^2 of W_recon^s vs W_native^s).
    # Lets the report show whether the attribution-time reconstruction is
    # roughly at-par with other steps, and whether reconstruction quality
    # itself is varying across the trajectory in a way that could be confused
    # with "readout reorganization".
    print("[recon_ev] computing per-snapshot reconstruction EV", flush=True)
    recon_ev: list[dict] = []
    snapshots_full_keep = snapshots_full  # used for per-step loop below
    for s_idx, s_step in enumerate(steps_canonical):
        feats_s, _ = encode_at_step(sd, accumulated, s_idx, device=args.device)
        stats_s = None if stats is None else {"scale": stats["scale"][s_idx], "mean": stats["mean"][s_idx]}
        W_recon_s = reconstruct_W(feats_s, sd, s_idx, stats_s, device=args.device).cpu()
        W_native_s_ = snapshots_full_keep[s_idx].float()
        diff = W_native_s_ - W_recon_s
        var_native = float(W_native_s_.var().item())
        var_resid = float(diff.var().item())
        ev = 1.0 - (var_resid / var_native) if var_native > 0 else float("nan")
        recon_ev.append({"step": int(s_step), "ev": ev, "var_native": var_native})
        del feats_s, W_recon_s, W_native_s_, diff
        if args.device == "cuda":
            torch.cuda.empty_cache()
    atomic_write_json(
        args.out_dir / "reconstruction_ev.json",
        {
            "model": model_hf,
            "ckpt": str(ckpt),
            "d_sae": args.d_sae,
            "seed": args.seed,
            "snapshot_step_used_for_attribution": args.snapshot_step,
            "ev_at_attribution_step": next(r["ev"] for r in recon_ev if r["step"] == args.snapshot_step),
            "recon_ev": recon_ev,
        },
    )
    snapshots_full = None
    snapshots_full_keep = None
    gc.collect()

    # Decoder rows + scale (for attribution).
    W_D = sd["W_D"][target_idx].cpu().float()
    scale_step = None if stats is None else stats["scale"][target_idx].cpu().float()

    # ---- Per-family: load examples, hidden states, attribution + interventions.
    rng = np.random.default_rng(args.rng_seed)
    split_rng = np.random.default_rng(args.split_seed)
    per_family_top: dict[str, np.ndarray] = {}
    per_family_results: dict[str, dict] = {}
    per_family_split_info: dict[str, dict] = {}

    for fam in families:
        ex = CT.load_examples(args.datasets_dir / f"{fam}.jsonl")
        if not ex:
            continue
        h_path = args.hidden_dir / f"{fam}_h{args.h_step}.pt"
        if not h_path.exists():
            print(f"[skip] {fam}: missing {h_path.name}", flush=True)
            continue
        h = torch.load(h_path, map_location="cpu", weights_only=False).float()
        if h.shape[0] != len(ex):
            print(
                f"[warn] {fam}: hidden ({h.shape[0]}) != examples ({len(ex)}); truncating to min",
                flush=True,
            )
            n = min(h.shape[0], len(ex))
            h = h[:n]
            ex = ex[:n]
        y_plus_full = torch.tensor([e.y_plus for e in ex], dtype=torch.long)
        y_minus_full = torch.tensor([e.y_minus for e in ex], dtype=torch.long)

        # --- Held-out split for attribution vs evaluation.
        # split_a: used to compute the per-feature attribution and select top-K.
        # split_b: held-out, used to evaluate ablate/preserve interventions.
        # If --split-frac == 1.0 OR per-side n < min-split-n, fall back to the
        # legacy same-set protocol (record this in the manifest).
        n_total = h.shape[0]
        n_a = int(round(n_total * args.split_frac))
        n_b = n_total - n_a
        use_split = args.split_frac < 1.0 and min(n_a, n_b) >= args.min_split_n
        if use_split:
            perm = split_rng.permutation(n_total)
            idx_a = np.sort(perm[:n_a])
            idx_b = np.sort(perm[n_a:])
        else:
            # Legacy / fallback: attribute and evaluate on the same examples.
            idx_a = np.arange(n_total)
            idx_b = np.arange(n_total)
        per_family_split_info[fam] = {
            "n_total": int(n_total),
            "n_attr": int(len(idx_a)),
            "n_eval": int(len(idx_b)),
            "use_split": bool(use_split),
            "split_seed": int(args.split_seed),
        }

        h_a = h[idx_a]
        h_b = h[idx_b]
        y_plus_a = y_plus_full[idx_a]
        y_minus_a = y_minus_full[idx_a]
        y_plus_b = y_plus_full[idx_b]
        y_minus_b = y_minus_full[idx_b]

        # --- Per-feature attribution computed ON SPLIT A only.
        attrs = per_feature_attribution(
            h_a,
            feature_acts.cpu(),
            W_D,
            scale_step,
            y_plus_a,
            y_minus_a,
        )
        attrs_np = attrs.cpu().numpy()
        if args.rank_by == "pos":
            order = np.argsort(-attrs_np)
        else:  # "abs"
            order = np.argsort(-np.abs(attrs_np))
        pos_mask = attrs_np > 0

        # --- Margin baselines on the EVAL side (split B; same as A under fallback).
        #   m_h_native:  native W_U at h_step  (margin(t, t); rescue denominator)
        #   m_native_s:  native W_U at snapshot_step
        #   m_recon:     reconstructed W_U at snapshot_step
        m_h_native = margins_with_W(h_b, W_native_h, y_plus_b, y_minus_b)
        m_native_s = margins_with_W(h_b, W_native_s, y_plus_b, y_minus_b)
        m_recon = margins_with_W(h_b, W_recon, y_plus_b, y_minus_b)
        # native_rescue_delta = swap rescue measured purely on native readouts:
        #   margin(t, s; W_native^s)  -  margin(t, t; W_native^t).
        # This is the quantity Experiment 1 reports per cell. recon_rescue_delta
        # uses the reconstructed W_U^s instead and is a degraded version.
        native_rescue_delta = float(m_native_s.float().mean().item()) - float(m_h_native.float().mean().item())
        recon_rescue_delta = float(m_recon.float().mean().item()) - float(m_h_native.float().mean().item())

        # --- In-set leakage baseline: same metric reported ON SPLIT A so
        # the leakage gap is visible. Only meaningful when use_split is True.
        m_h_native_a = margins_with_W(h_a, W_native_h, y_plus_a, y_minus_a)
        m_recon_a = margins_with_W(h_a, W_recon, y_plus_a, y_minus_a)

        per_K: dict[int, dict] = {}
        for K_ in args.K:
            top = order[:K_].astype(np.int64)
            # Two controls:
            #   ctrl     — matched on (log decoder_norm, log firing_rate) only.
            #   ctrl_sgn — additionally matched on attribution sign (positive
            #              when --rank-by pos; |attr|-bucket when --rank-by abs).
            #              This catches the sign-confounding interpretation
            #              of preserve-K beating recon.
            ctrl = matched_random_features(top, decoder_norms_np, rates, rng)
            ctrl_sgn = matched_random_features(
                top,
                decoder_norms_np,
                rates,
                rng,
                sign_filter=pos_mask if args.rank_by == "pos" else None,
            )

            top_t = torch.from_numpy(top).long().to(args.device)
            ctrl_t = torch.from_numpy(ctrl).long().to(args.device)
            ctrl_sgn_t = torch.from_numpy(ctrl_sgn).long().to(args.device)

            contrib_top = feature_contribution(
                feature_acts,
                sd,
                target_idx,
                stats_target,
                top_t,
                device=args.device,
            ).cpu()
            contrib_ctrl = feature_contribution(
                feature_acts,
                sd,
                target_idx,
                stats_target,
                ctrl_t,
                device=args.device,
            ).cpu()
            contrib_ctrl_sgn = feature_contribution(
                feature_acts,
                sd,
                target_idx,
                stats_target,
                ctrl_sgn_t,
                device=args.device,
            ).cpu()

            def complement_contrib(idx_arr: np.ndarray) -> torch.Tensor:
                comp = torch.from_numpy(np.setdiff1d(np.arange(D_sae), idx_arr)).long().to(args.device)
                return feature_contribution(
                    feature_acts,
                    sd,
                    target_idx,
                    stats_target,
                    comp,
                    device=args.device,
                ).cpu()

            # Ablate: keep everything except the chosen subset's rank-1 piece.
            # Preserve: keep only the chosen subset's rank-1 piece (plus the
            # bias/mean baseline, which cancels in the contrastive margin).
            W_ablate = W_recon - contrib_top
            W_preserve = W_recon - complement_contrib(top)
            W_ablate_ctrl = W_recon - contrib_ctrl
            W_preserve_ctrl = W_recon - complement_contrib(ctrl)
            W_ablate_ctrl_sgn = W_recon - contrib_ctrl_sgn
            W_preserve_ctrl_sgn = W_recon - complement_contrib(ctrl_sgn)

            # Held-out (split B) evaluation — the headline numbers.
            m_ab = margins_with_W(h_b, W_ablate, y_plus_b, y_minus_b)
            m_pres = margins_with_W(h_b, W_preserve, y_plus_b, y_minus_b)
            m_ab_ctrl = margins_with_W(h_b, W_ablate_ctrl, y_plus_b, y_minus_b)
            m_pres_ctrl = margins_with_W(h_b, W_preserve_ctrl, y_plus_b, y_minus_b)
            m_ab_ctrl_sgn = margins_with_W(h_b, W_ablate_ctrl_sgn, y_plus_b, y_minus_b)
            m_pres_ctrl_sgn = margins_with_W(h_b, W_preserve_ctrl_sgn, y_plus_b, y_minus_b)
            # In-set (split A) evaluation — for the leakage-vs-held-out gap.
            m_ab_a = margins_with_W(h_a, W_ablate, y_plus_a, y_minus_a)
            m_pres_a = margins_with_W(h_a, W_preserve, y_plus_a, y_minus_a)

            # ----------------------------------------------------------------
            # Logit decomposition (eval split): for each readout, report
            # mean h · W[y+] and mean h · W[y-] separately. Lets the analyst
            # see whether ablate / preserve is moving the y+ logit, the y-
            # logit, or both.
            # ----------------------------------------------------------------
            def _logit_pair(h_, W_, yp_, ym_):
                Wp = W_[yp_.long()].to(h_.device).float()
                Wm = W_[ym_.long()].to(h_.device).float()
                return (
                    (h_.float() * Wp).sum(-1).mean().item(),
                    (h_.float() * Wm).sum(-1).mean().item(),
                )

            l_recon = _logit_pair(h_b, W_recon, y_plus_b, y_minus_b)
            l_h_native = _logit_pair(h_b, W_native_h, y_plus_b, y_minus_b)
            l_native_s = _logit_pair(h_b, W_native_s, y_plus_b, y_minus_b)
            l_ablate = _logit_pair(h_b, W_ablate, y_plus_b, y_minus_b)
            l_preserve = _logit_pair(h_b, W_preserve, y_plus_b, y_minus_b)
            l_ablate_ctl = _logit_pair(h_b, W_ablate_ctrl, y_plus_b, y_minus_b)
            l_pres_ctl = _logit_pair(h_b, W_preserve_ctrl, y_plus_b, y_minus_b)
            l_ablate_sgn = _logit_pair(h_b, W_ablate_ctrl_sgn, y_plus_b, y_minus_b)
            l_pres_sgn = _logit_pair(h_b, W_preserve_ctrl_sgn, y_plus_b, y_minus_b)

            logit_decomp = {
                "recon": {"yp": l_recon[0], "ym": l_recon[1]},
                "h_native": {"yp": l_h_native[0], "ym": l_h_native[1]},
                "native_s": {"yp": l_native_s[0], "ym": l_native_s[1]},
                "ablate_top": {"yp": l_ablate[0], "ym": l_ablate[1]},
                "preserve_top": {"yp": l_preserve[0], "ym": l_preserve[1]},
                "ablate_ctrl": {"yp": l_ablate_ctl[0], "ym": l_ablate_ctl[1]},
                "preserve_ctrl": {"yp": l_pres_ctl[0], "ym": l_pres_ctl[1]},
                "ablate_ctrl_sgn": {"yp": l_ablate_sgn[0], "ym": l_ablate_sgn[1]},
                "preserve_ctrl_sgn": {"yp": l_pres_sgn[0], "ym": l_pres_sgn[1]},
            }
            # Convenience: explicit deltas relative to recon.
            ry_p = l_recon[0]
            ry_m = l_recon[1]
            for k, lp in list(logit_decomp.items()):
                lp["dy_plus"] = lp["yp"] - ry_p
                lp["dy_minus"] = lp["ym"] - ry_m
                lp["dmargin"] = lp["dy_plus"] - lp["dy_minus"]

            per_K[K_] = {
                "K": K_,
                "rank_by": args.rank_by,
                "top_indices": top.tolist(),
                "ctrl_indices": ctrl.tolist(),
                "ctrl_sign_matched_indices": ctrl_sgn.tolist(),
                # Eval-side baselines (split B; the rescue denominator)
                "summary_h_native": RS.margin_summary(m_h_native),
                "summary_native_s": RS.margin_summary(m_native_s),
                "summary_recon": RS.margin_summary(m_recon),
                # Eval-side interventions (split B)
                "summary_ablate_top": RS.margin_summary(m_ab),
                "summary_preserve_top": RS.margin_summary(m_pres),
                "summary_ablate_ctrl": RS.margin_summary(m_ab_ctrl),
                "summary_preserve_ctrl": RS.margin_summary(m_pres_ctrl),
                "summary_ablate_ctrl_sign_matched": RS.margin_summary(m_ab_ctrl_sgn),
                "summary_preserve_ctrl_sign_matched": RS.margin_summary(m_pres_ctrl_sgn),
                # In-set diagnostic (split A) — exposes leakage gap
                "summary_recon_inset": RS.margin_summary(m_recon_a),
                "summary_h_native_inset": RS.margin_summary(m_h_native_a),
                "summary_ablate_top_inset": RS.margin_summary(m_ab_a),
                "summary_preserve_top_inset": RS.margin_summary(m_pres_a),
                # Rescue deltas (margin units, mean over eval split):
                #   native:  margin(t, s; W_native^s) - margin(t, t; W_native^t)
                #   recon:   margin(t, s; W_recon^s)  - margin(t, t; W_native^t)
                "rescue_delta_native": native_rescue_delta,
                "rescue_delta_recon": recon_rescue_delta,
                "h_native_step_used": h_native_step_used,
                "split": per_family_split_info[fam],
                # Per-readout logit decomposition (eval split):
                #   yp / ym are mean h · W[y+] / h · W[y-]
                #   dy_plus / dy_minus / dmargin are deltas vs `recon`
                # Tells us whether ablate/preserve moves the y+ logit, the
                # y- logit, or both.
                "logit_decomp": logit_decomp,
            }
            split_tag = "split" if use_split else "same"
            print(
                f"  [{fam} K={K_:>4} rank={args.rank_by} {split_tag}"
                f" n_a={len(idx_a)} n_b={len(idx_b)}] "
                f"recon={per_K[K_]['summary_recon']['accuracy']:.3f} "
                f"h_nat={per_K[K_]['summary_h_native']['accuracy']:.3f} "
                f"r_native={per_K[K_]['rescue_delta_native']:+.3f} "
                f"r_recon={per_K[K_]['rescue_delta_recon']:+.3f} "
                f"abl={per_K[K_]['summary_ablate_top']['accuracy']:.3f} "
                f"pres={per_K[K_]['summary_preserve_top']['accuracy']:.3f} "
                f"abl_ctrl={per_K[K_]['summary_ablate_ctrl']['accuracy']:.3f} "
                f"pres_ctrl={per_K[K_]['summary_preserve_ctrl']['accuracy']:.3f} "
                f"abl_sgn={per_K[K_]['summary_ablate_ctrl_sign_matched']['accuracy']:.3f} "
                f"pres_sgn={per_K[K_]['summary_preserve_ctrl_sign_matched']['accuracy']:.3f}",
                flush=True,
            )

        # Persist per-family attribution + interventions.
        out_p = shard_dir / f"{fam}__h{args.h_step}__s{args.snapshot_step}.json"
        atomic_write_json(
            out_p,
            {
                "model": model_hf,
                "family": fam,
                "h_step": args.h_step,
                "snapshot_step": args.snapshot_step,
                "n": len(ex),
                "rank_by": args.rank_by,
                "h_native_step_used": h_native_step_used,
                "n_features_positive_attr": int(pos_mask.sum()),
                "split": per_family_split_info[fam],
                "feature_attributions_top1024": [
                    {"feature": int(i), "attribution": float(attrs_np[i])} for i in order[:1024]
                ],
                "per_K": per_K,
            },
        )
        # Companion .pt with full attribution vector + the K decoder rows
        # we used. The decoder rows let downstream scripts run the converse
        # intervention (project K directions out of h) without needing to
        # reload the full crosscoder.
        # W_D shape (D_sae, d); take the union of all top-K we evaluated.
        K_max = max(args.K)
        top_union = order[:K_max].astype(np.int64)
        D_top = sd["W_D"][target_idx][torch.from_numpy(top_union).long()].cpu().float()
        torch.save(
            {
                "attrs": attrs,
                "feature_order": order,
                "decoder_rows_top_Kmax": D_top,  # (K_max, d)
                "decoder_rows_top_Kmax_indices": top_union,  # (K_max,)
                "scale_step": (None if scale_step is None else scale_step.float()),
            },
            out_p.with_suffix(".pt"),
        )

        per_family_top[fam] = order[: max(args.K)].astype(np.int64)
        per_family_results[fam] = per_K

    # --- Specificity matrix: row = ablation source, col = eval family.
    # Source feature set is selected from src's split-A (already done above).
    # Evaluation is done on the eval family's split-B for held-out honesty.
    if len(per_family_top) >= 2:
        K_spec = max(args.K)
        fam_list = list(per_family_top.keys())
        n = len(fam_list)
        spec = np.full((n, n), np.nan, dtype=np.float64)
        spec_split_rng = np.random.default_rng(args.split_seed)  # re-derive splits
        spec_split_cache: dict[str, dict] = {}

        def _eval_split(eval_fam: str):
            if eval_fam in spec_split_cache:
                return spec_split_cache[eval_fam]
            ex_eval = CT.load_examples(args.datasets_dir / f"{eval_fam}.jsonl")
            h_path = args.hidden_dir / f"{eval_fam}_h{args.h_step}.pt"
            if not ex_eval or not h_path.exists():
                spec_split_cache[eval_fam] = None
                return None
            h_eval = torch.load(h_path, map_location="cpu", weights_only=False).float()
            n_eval = min(h_eval.shape[0], len(ex_eval))
            h_eval = h_eval[:n_eval]
            ex_eval = ex_eval[:n_eval]
            n_a = int(round(n_eval * args.split_frac))
            n_b = n_eval - n_a
            use_split_local = args.split_frac < 1.0 and min(n_a, n_b) >= args.min_split_n
            if use_split_local:
                p = spec_split_rng.permutation(n_eval)
                idx_b_local = np.sort(p[n_a:])
            else:
                idx_b_local = np.arange(n_eval)
            h_b_local = h_eval[idx_b_local]
            yp = torch.tensor([ex_eval[i].y_plus for i in idx_b_local.tolist()], dtype=torch.long)
            ym = torch.tensor([ex_eval[i].y_minus for i in idx_b_local.tolist()], dtype=torch.long)
            spec_split_cache[eval_fam] = (h_b_local, yp, ym)
            return spec_split_cache[eval_fam]

        for i, src in enumerate(fam_list):
            top_t = torch.from_numpy(per_family_top[src][:K_spec]).long().to(args.device)
            contrib_top = feature_contribution(
                feature_acts,
                sd,
                target_idx,
                stats_target,
                top_t,
                device=args.device,
            ).cpu()
            W_ablate = W_recon - contrib_top
            for j, eval_fam in enumerate(fam_list):
                pkg = _eval_split(eval_fam)
                if pkg is None:
                    continue
                h_eval, yp, ym = pkg
                m0 = margins_with_W(h_eval, W_recon, yp, ym).mean().item()
                m1 = margins_with_W(h_eval, W_ablate, yp, ym).mean().item()
                spec[i, j] = m0 - m1  # margin drop under src-family ablation
        atomic_write_json(
            args.out_dir / "specificity_matrix.json",
            {
                "K": K_spec,
                "families": fam_list,
                "matrix": spec.tolist(),
                "snapshot_step": args.snapshot_step,
                "h_step": args.h_step,
            },
        )
        torch.save(
            {"matrix": torch.from_numpy(spec), "families": fam_list},
            args.out_dir / "specificity_matrix.pt",
        )
        print(f"[specificity] wrote {n}x{n} matrix", flush=True)

    atomic_write_json(
        args.out_dir / "manifest.json",
        {
            "model": model_hf,
            "ckpt": str(ckpt),
            "d_sae": args.d_sae,
            "seed": args.seed,
            "snapshot_step": args.snapshot_step,
            "h_step": args.h_step,
            "h_native_step_used": h_native_step_used,
            "K": args.K,
            "rank_by": args.rank_by,
            "split_frac": args.split_frac,
            "split_seed": args.split_seed,
            "min_split_n": args.min_split_n,
            "families": list(per_family_top.keys()),
            "per_family_split": per_family_split_info,
            "git_commit": git_commit(),
        },
    )
    print(f"[done] -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
