"""Evaluate the SCHEDULE.md decision rules against a trained OLMo crosscoder.

Computes the four pilot-stage / seven production-stage gates and emits a
JSON verdict. Designed to be run immediately after wu_adapter.py finishes
training so a positive pilot can auto-trigger the production launch.

Usage:
    python experiments/crosscoders/crosscoder_olmo/scripts/eval_decision_rules.py \
        --crosscoder /workspace/results/olmo_cross_family/pilot/cc_olmo27b_dsae16384_seed0.pt \
        --wu-cache /workspace/wu_cache_olmo \
        --stage pilot \
        --out /workspace/results/olmo_cross_family/pilot/verdict_seed0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src" / "crosscoder"))

# Pre-registered thresholds from SCHEDULE.md (committed 2026-04-27).
# EV bumped from 0.40 to 0.55 per code review: Run 5 1B/d_SAE=8192 hit 0.477,
# so 0.40 admits an undertrained crosscoder (Run 2 failure mode).
PILOT_THRESHOLDS = {
    "rate_rotation_r_min": 0.5,  # PRIMARY: baseline-window-free
    "perm_test_p_value_max": 0.01,  # SECONDARY: reduced-baseline CUSUM
    "ev_min": 0.55,  # bumped from 0.40 — Run 5 floor + headroom
    "dead_feature_rate_max": 0.30,
}

PRODUCTION_THRESHOLDS = {
    "perm_test_p_value_max": 0.001,
    "rate_rotation_r_min": 0.7,
    "decoder_cosine_step1000_vs_step150_max": 0.7,  # significant rotation by step 1000
    "ev_min": 0.65,
    "dead_feature_rate_max": 0.10,
    "monosemantic_features_in_top50_min": 5,
    "causal_ablation_target_script_drop_min_nats": 1.0,
}


def compute_firing_rates(crosscoder, snapshots, batch_size=2048, device="cuda"):
    """Per-feature, per-snapshot firing rates. Returns (K, d_sae) tensor."""
    from wu_adapter import batch_iter  # type: ignore

    crosscoder.eval()
    hook_points = crosscoder.cfg.hook_points
    K, V, _ = snapshots.shape
    d_sae = crosscoder.cfg.d_sae

    counts = torch.zeros(K, d_sae, device=device)
    n_seen = 0

    with torch.no_grad():
        for batch in batch_iter(
            snapshots, hook_points, batch_size, shuffle=False, device=device
        ):
            x, enc_kw, _ = crosscoder.prepare_input(batch)
            feat, _ = crosscoder.encode(x, return_hidden_pre=True, **enc_kw)
            # feat shape: (batch, K, d_sae) — count per (snapshot, feature)
            counts += (feat > 0).float().sum(dim=0)
            n_seen += feat.shape[0]
    return (counts / n_seen).cpu()  # (K, d_sae)


def compute_decoder_cosines(crosscoder, baseline_idx=0):
    """Per-feature decoder cosine vs reference snapshot. Returns (K, d_sae)."""
    W_D = crosscoder.W_D  # (K, d_sae, d_model) typically
    if W_D.dim() != 3:
        raise RuntimeError(f"unexpected W_D shape {tuple(W_D.shape)}")
    W_D = W_D.detach().cpu().float()
    K = W_D.shape[0]
    ref = W_D[baseline_idx]  # (d_sae, d_model)
    ref_norm = ref / ref.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosines = torch.zeros(K, W_D.shape[1])
    for i in range(K):
        cur = W_D[i]
        cur_norm = cur / cur.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cosines[i] = (ref_norm * cur_norm).sum(dim=-1)
    return cosines


def _preprocess_mode_from_blob(blob):
    training = blob.get("training", {})
    if "input_preprocess" in training:
        return training["input_preprocess"]
    stats = blob.get("preprocess_stats")
    if isinstance(stats, dict):
        return stats.get("mode", "none")
    return blob.get("preprocess_mode", "none")


def _cusum_max(rates, baseline_n: int) -> float:
    """CUSUM-max statistic with MAD-floor robustness (Run 3 fix, pre-registered).

    For each feature, estimates baseline mean and MAD from the first
    ``baseline_n`` snapshots, z-scores the trajectory, and returns the max
    absolute cumulative sum over (snapshot, feature). Robust σ floor =
    25th percentile of non-zero per-feature MADs.
    """
    base = rates[:baseline_n]
    mu = base.mean(dim=0, keepdim=True)
    median = base.median(dim=0, keepdim=True).values
    mad = (base - median).abs().median(dim=0).values
    sigma = 1.4826 * mad
    nonzero = sigma[sigma > 0]
    if nonzero.numel() > 0:
        sigma = sigma.clamp_min(nonzero.quantile(0.25))
    sigma = sigma.clamp_min(1e-8)
    z = (rates - mu) / sigma
    return z.cumsum(dim=0).abs().max().item()


def reduced_baseline_cusum_p(rates, baseline_n=6, n_perms=1000, seed=0):
    """CUSUM-max permutation test with snapshot-label permutation.

    Re-derives mu and sigma from the *permuted* baseline window each
    permutation, not from the real one. This makes the null sensitive only
    to temporal ordering, not to position-specific magnitudes — fixes the
    conservative-bias bug that the prior implementation introduced by
    z-scoring permuted snapshots against the unpermuted baseline.

    Returns (observed, p_value, null_mean).
    """
    K, _ = rates.shape
    obs = _cusum_max(rates, baseline_n)

    g = torch.Generator(device="cpu").manual_seed(seed)
    null = torch.zeros(n_perms)
    for i in range(n_perms):
        perm = torch.randperm(K, generator=g)
        null[i] = _cusum_max(rates[perm], baseline_n)

    # Empirical p-value (one-sided, upper)
    p = ((null >= obs).sum().item() + 1) / (n_perms + 1)
    return obs, p, null.mean().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crosscoder", type=Path, required=True)
    ap.add_argument("--wu-cache", type=Path, required=True)
    ap.add_argument("--stage", choices=["pilot", "production"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--baseline-n",
        type=int,
        default=6,
        help="Number of pre-event snapshots for CUSUM baseline.",
    )
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.core.repro import log_run_provenance, seed_everything

    seed_everything(args.seed)
    log_run_provenance(seed=args.seed)

    from wu_adapter import build_crosscoder, load_snapshots, preprocess_snapshots  # type: ignore

    print(f"Loading crosscoder from {args.crosscoder}", flush=True)
    blob = torch.load(args.crosscoder, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    steps = blob["steps"]
    model_name = blob["model_name"]
    quality_at_train = blob.get("quality", {})
    training_meta = blob.get("training", {})

    print(f"  model: {model_name}", flush=True)
    print(f"  steps: {steps}", flush=True)
    print(
        f"  d_sae: {cfg.get('d_sae')}, n_heads: {len(cfg.get('hook_points', []))}",
        flush=True,
    )
    print(
        f"  EV at train end: {quality_at_train.get('explained_variance')}", flush=True
    )

    snapshots, _ = load_snapshots(
        model_name=model_name, steps=steps, cache_dir=args.wu_cache
    )
    snapshots, _ = preprocess_snapshots(
        snapshots, mode=_preprocess_mode_from_blob(blob)
    )

    K, V, d = snapshots.shape
    print(f"Snapshots: {(K, V, d)}", flush=True)

    n_snapshots = len(cfg.get("hook_points", []))
    crosscoder = build_crosscoder(
        n_snapshots=n_snapshots,
        d_model=cfg["d_model"],
        expansion_factor=cfg["expansion_factor"],
        device=args.device,
        init_threshold=training_meta.get("init_threshold", 0.1),
        init_encoder_with_decoder_transpose_factor=0.0,  # no init effect; we load weights
    )
    crosscoder.load_state_dict(blob["state_dict"])
    crosscoder.to(args.device)
    crosscoder.eval()

    print("Computing per-snapshot firing rates...", flush=True)
    rates = compute_firing_rates(
        crosscoder, snapshots, batch_size=args.batch_size, device=args.device
    )
    print(
        f"  rates shape: {tuple(rates.shape)}, mean={rates.mean().item():.4f}",
        flush=True,
    )

    print(
        f"Running reduced-baseline CUSUM permutation test (baseline_n={args.baseline_n})...",
        flush=True,
    )
    obs, p_value, null_mean = reduced_baseline_cusum_p(
        rates, baseline_n=args.baseline_n, n_perms=args.n_perms
    )
    print(f"  observed CUSUM = {obs:.3f}", flush=True)
    print(f"  null mean      = {null_mean:.3f}", flush=True)
    print(f"  p-value        = {p_value:.6f}", flush=True)

    print("Computing rate-rotation correlation...", flush=True)
    cosines = compute_decoder_cosines(crosscoder, baseline_idx=0)
    # Step-1000 is index 5 in our schedule (after 150, 600, 700, 850, 900)
    step1000_idx = steps.index(1000) if 1000 in steps else 5
    delta_rate = rates[step1000_idx] - rates[0]  # (d_sae,)
    rotation = 1.0 - cosines[step1000_idx]  # (d_sae,)
    rate_rotation_r = torch.corrcoef(torch.stack([delta_rate, rotation]))[0, 1].item()
    print(f"  rate-rotation r = {rate_rotation_r:.3f}", flush=True)

    print("Computing dead-feature rate (final snapshot)...", flush=True)
    final_rates = rates[-1]
    dead_rate = (final_rates < 1e-6).float().mean().item()
    print(f"  dead rate at step {steps[-1]} = {dead_rate:.3f}", flush=True)

    print("Computing decoder cosine drop step150 -> step1000...", flush=True)
    mean_cos_step1000 = cosines[step1000_idx].mean().item()
    print(
        f"  mean decoder cos vs snap[0] at step 1000: {mean_cos_step1000:.3f}",
        flush=True,
    )

    ev = quality_at_train.get("explained_variance")
    if ev is None:
        # Recompute defensively
        from wu_adapter import quick_quality  # type: ignore

        m = quick_quality(
            crosscoder, snapshots, batch_size=args.batch_size, device=args.device
        )
        ev = m["explained_variance"]

    # Apply pre-registered thresholds
    if args.stage == "pilot":
        thr = PILOT_THRESHOLDS
        gates = {
            "perm_test_p_value": (p_value, "<=", thr["perm_test_p_value_max"]),
            "rate_rotation_r": (rate_rotation_r, ">=", thr["rate_rotation_r_min"]),
            "ev": (ev, ">=", thr["ev_min"]),
            "dead_feature_rate": (dead_rate, "<=", thr["dead_feature_rate_max"]),
        }
    else:
        thr = PRODUCTION_THRESHOLDS
        gates = {
            "perm_test_p_value": (p_value, "<=", thr["perm_test_p_value_max"]),
            "rate_rotation_r": (rate_rotation_r, ">=", thr["rate_rotation_r_min"]),
            "decoder_cosine_step1000_vs_step150": (
                mean_cos_step1000,
                "<=",
                thr["decoder_cosine_step1000_vs_step150_max"],
            ),
            "ev": (ev, ">=", thr["ev_min"]),
            "dead_feature_rate": (dead_rate, "<=", thr["dead_feature_rate_max"]),
            # Two production-only checks not computed here (need auto-interp + causal ablation):
            # "monosemantic_features_in_top50": ...
            # "causal_ablation_target_script_drop_nats": ...
        }

    def passes(observed, op, threshold):
        return observed <= threshold if op == "<=" else observed >= threshold

    gate_results = {}
    all_pass = True
    for name, (obs_val, op, threshold) in gates.items():
        ok = passes(obs_val, op, threshold)
        all_pass = all_pass and ok
        gate_results[name] = {
            "observed": float(obs_val),
            "op": op,
            "threshold": float(threshold),
            "pass": bool(ok),
        }

    # Pre-registered partial-credit decision rule (SCHEDULE.md §"Stage 1" / §"Stage 2"):
    #   Pilot:      proceed iff A passes AND ≥2 of {B, C, D} pass.
    #   Production: headline holds iff A passes AND ≥4 of 6 secondary checks pass;
    #               this script computes 4 of 6 secondaries (perm, decoder-cos, EV, dead),
    #               the remaining 2 (auto-interp monosemanticity, causal script ablation)
    #               are external. We surface `proceed` requiring A AND ≥2 of these 4
    #               (so that the headline rule of ≥4-of-6 remains achievable when the
    #               external checks land), and emit `external_checks_pending=True` to
    #               make the gap explicit.
    primary_name = "rate_rotation_r"
    primary_pass = gate_results[primary_name]["pass"]
    secondary_pass_count = sum(
        v["pass"] for k, v in gate_results.items() if k != primary_name
    )
    secondary_total = len(gate_results) - 1
    if args.stage == "pilot":
        secondary_required = 2
        external_pending = False
    else:
        secondary_required = 2  # 4-of-6 headline minus 2 external = 2 of 4 here
        external_pending = True

    proceed = primary_pass and secondary_pass_count >= secondary_required

    verdict = {
        "stage": args.stage,
        "model": model_name,
        "d_sae": cfg.get("d_sae"),
        "steps": steps,
        "thresholds": {k: float(v) for k, v in thr.items()},
        "gates": gate_results,
        "all_gates_pass": all_pass,
        "decision": {
            "primary_gate": primary_name,
            "primary_pass": bool(primary_pass),
            "secondary_pass_count": int(secondary_pass_count),
            "secondary_total": int(secondary_total),
            "secondary_required": int(secondary_required),
            "external_checks_pending": bool(external_pending),
            "proceed": bool(proceed),
        },
        "raw": {
            "perm_test_observed_cusum": obs,
            "perm_test_null_mean": null_mean,
            "perm_test_n_perms": args.n_perms,
            "perm_test_baseline_n": args.baseline_n,
            "rate_rotation_r": rate_rotation_r,
            "ev": ev,
            "dead_feature_rate": dead_rate,
            "decoder_cosine_step1000_vs_step150": mean_cos_step1000,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nVerdict written to {args.out}", flush=True)
    print(
        f"  Primary ({primary_name}): {'PASS' if primary_pass else 'FAIL'}; "
        f"secondaries: {secondary_pass_count}/{secondary_total} pass "
        f"(need ≥{secondary_required})",
        flush=True,
    )
    print(f"  PROCEED: {proceed}  ALL GATES PASS: {all_pass}", flush=True)
    return 0 if proceed else 1


if __name__ == "__main__":
    sys.exit(main())
