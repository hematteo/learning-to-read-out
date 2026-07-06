"""Vectorized 100k-permutation test on saved (K, D) rate matrices.

Reproduces the robust-floor max-feature CUSUM statistic from
permutation_test.py but processes many permutations per batch on the
device.

Usage:
    python permutation_test_fast.py \
        --rates run3_rates.pt \
        --n-perms 100000 \
        --batch 1000 \
        --output run3_perm_100k.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2


def cusum_max_feature_batched(rates_perm, variance_estimator_n=10):
    """Robust-floor CUSUM-max stat, vectorized over a leading batch dim.

    rates_perm: (B, K, D) batch of (possibly permuted) rate matrices.
    Returns: (B,) tensor of max-feature SIGNED CUSUM statistics.

    Signed (not abs): the readout-consolidation claim is a one-sided directional
    increase in firing rates near step ~1000. Collapsing the sign would also
    score downward excursions and inflate the null distribution.
    """
    B, K, D = rates_perm.shape
    n_base = min(variance_estimator_n, max(K // 3, 2))
    if n_base < 3:
        # With a 2-sample baseline the MAD is identically zero (torch's
        # even-count median takes the lower value), every sigma collapses to
        # the NaN mask, and the statistic silently returns NaN. No shipped
        # schedule is this short (smallest is K=9).
        raise ValueError(
            f"K={K} snapshots gives an n_base={n_base} MAD baseline that is "
            f"identically zero; the robust-floor CUSUM statistic needs K >= 9."
        )
    base = rates_perm[:, :n_base]  # (B, n_base, D)
    baseline_mean = base.mean(dim=1, keepdim=True)  # (B, 1, D)
    median = base.median(dim=1, keepdim=True).values  # (B, 1, D)
    mad = (base - median).abs().median(dim=1).values  # (B, D)
    sigma_raw = 1.4826 * mad  # (B, D)
    sigma_masked = torch.where(
        sigma_raw > 0, sigma_raw, torch.full_like(sigma_raw, float("nan"))
    )
    floors = torch.nanquantile(sigma_masked, 0.25, dim=-1)  # (B,)
    floors = torch.clamp(floors, min=1e-8)
    sigma = torch.maximum(sigma_raw, floors.unsqueeze(-1))  # (B, D)
    z = (rates_perm - baseline_mean) / sigma.unsqueeze(1)  # (B, K, D)
    cusum = z.cumsum(dim=1)  # (B, K, D)
    return cusum.reshape(B, -1).max(dim=1).values  # (B,)


def cusum_max_feature_single(rates):
    return cusum_max_feature_batched(rates.unsqueeze(0))[0].item()


def permutation_test_vectorized(
    rates, n_permutations, batch=1000, seed=0, device="cpu"
):
    """Per-feature independent shuffle: each feature's K-step trajectory is
    shuffled independently. Sharing one snapshot permutation across all D
    features (the previous behavior) preserves cross-feature correlations and
    underestimates the null, producing anti-conservative p-values.

    Memory: the (B, K, D) permutation tensor dominates. To stay safe on a
    16 GB Mac with K=32, D=8192 we cap the effective batch so the permuted
    tensor stays under ~256 MB.
    """
    K, D = rates.shape
    rates = rates.to(device)
    obs = cusum_max_feature_single(rates)

    # Cap batch so the (B, K, D) permuted tensor + perms tensor stay under ~256 MB.
    # Float32 K*D = 32*8192*4 = 1 MB per batch item; perms (int64) = 32*8192*8 = 2 MB.
    bytes_per_item = K * D * (4 + 8)
    max_batch = max(1, int(2**28 / bytes_per_item))
    effective_batch = min(batch, max_batch)
    if effective_batch < batch:
        print(
            f"    [perm] capping batch {batch}→{effective_batch} for memory safety",
            flush=True,
        )

    g = torch.Generator(device="cpu").manual_seed(seed)
    null = torch.empty(n_permutations)

    n_done = 0
    last_log = time.time()
    while n_done < n_permutations:
        b = min(effective_batch, n_permutations - n_done)
        # Per-feature independent shuffle: argsort of random scores generates
        # an independent permutation of K for each (batch, feature) slot.
        perms = torch.argsort(torch.rand(b, K, D, generator=g, device="cpu"), dim=1).to(
            device
        )  # (B, K, D)
        permuted = rates.unsqueeze(0).expand(b, K, D).gather(1, perms)
        stats = cusum_max_feature_batched(permuted)
        null[n_done : n_done + b] = stats.cpu()
        del perms, permuted, stats
        n_done += b
        if time.time() - last_log > 5.0:
            print(f"    {n_done}/{n_permutations}", flush=True)
            last_log = time.time()

    null_np = null.numpy()
    p = (1 + int((null_np >= obs).sum())) / (n_permutations + 1)
    obs_zsigma = float((obs - null_np.mean()) / null_np.std())
    return obs, p, null_np, obs_zsigma, effective_batch


def permutation_test_seed_averaged(
    rates_per_seed, n_permutations, batch=1000, seed=0, device="cpu"
):
    """Single permutation test on the seed-averaged rate matrix.

    Training seeds share W_U snapshots (only SAE init differs), so per-seed
    observed CUSUM stats are highly correlated. Fisher-combining their p-values
    overstates evidence. Instead: average rates across seeds first, then
    permute the averaged (K, D) matrix once. One p-value, no independence
    assumption.
    """
    stacked = torch.stack(
        [
            r if isinstance(r, torch.Tensor) else torch.as_tensor(r)
            for r in rates_per_seed
        ],
        dim=0,
    )  # (S, K, D)
    rates_mean = stacked.mean(dim=0)  # (K, D)
    return permutation_test_vectorized(
        rates_mean,
        n_permutations=n_permutations,
        batch=batch,
        seed=seed,
        device=device,
    )


def fisher_combine(p_values):
    p = np.asarray(p_values, dtype=float).clip(min=1e-300)
    stat = -2.0 * np.log(p).sum()
    return float(1.0 - chi2.cdf(stat, 2 * len(p)))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Vectorized 100k-perm CUSUM test, Fisher-combined across seeds. "
            "NOTE: the output is a SEED-AGGREGATE Fisher-combined p, not a "
            "per-checkpoint statistic, so no manifest row is updated here. "
            "The aggregate p is mirrored to "
            "results/crosscoder_seed_aggregate_p.json for the paper."
        )
    )
    ap.add_argument(
        "--rates",
        type=Path,
        required=True,
        help="dict {seed_label -> (K, D) rate tensor}",
    )
    ap.add_argument("--n-perms", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--aggregate-json",
        type=Path,
        default=Path("results/crosscoder_seed_aggregate_p.json"),
        help="Where to mirror the seed-aggregate Fisher-combined p (per-row manifest update is not applicable).",
    )
    ap.add_argument(
        "--seed-average",
        action="store_true",
        help=(
            "Average rates across training seeds first, then run a single "
            "permutation test on the averaged (K, D) matrix. Statistically "
            "correct when training seeds share W_U snapshots; replaces the "
            "(invalid) Fisher combination across seeds."
        ),
    )
    args = ap.parse_args()

    if not args.seed_average:
        print(
            "WARNING: Fisher combination across training seeds is "
            "statistically invalid (seeds share W_U). Use --seed-average "
            "for the correct test.",
            flush=True,
        )

    blob = torch.load(args.rates, weights_only=False)
    rates_per_seed = (
        blob["rates_per_seed"]
        if isinstance(blob, dict) and "rates_per_seed" in blob
        else blob
    )
    results = {}
    p_values = []
    effective_batches = {}

    if args.seed_average:
        rates_list = list(rates_per_seed.values())
        labels = list(rates_per_seed.keys())
        print(
            f"[seed-averaged] averaging across {len(rates_list)} seeds "
            f"({labels}); starting {args.n_perms} perms (batch {args.batch}, "
            f"shuffle_seed={args.shuffle_seed})...",
            flush=True,
        )
        t0 = time.time()
        obs, p, null, obs_z, eff_b = permutation_test_seed_averaged(
            rates_list,
            n_permutations=args.n_perms,
            batch=args.batch,
            seed=args.shuffle_seed,
            device=args.device,
        )
        dt = time.time() - t0
        sort_idx = np.argsort(null)[::-1]
        n_tail = max(5000, int(0.05 * len(null)))
        null_tail = null[sort_idx[:n_tail]]
        results["seed_averaged"] = {
            "observed": float(obs),
            "p_value": float(p),
            "n_permutations": args.n_perms,
            "requested_batch": int(args.batch),
            "effective_batch": int(eff_b),
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_max": float(null.max()),
            "null_p95": float(np.quantile(null, 0.95)),
            "null_p99": float(np.quantile(null, 0.99)),
            "null_p999": float(np.quantile(null, 0.999)),
            "null_tail_top5pct": null_tail.astype(np.float32),
            "obs_in_null_sigma_units": obs_z,
            "n_null_geq_obs": int((null >= obs).sum()),
            "elapsed_seconds": dt,
            "seed_labels": labels,
        }
        results["__shuffle_seed"] = args.shuffle_seed
        results["__n_perms"] = args.n_perms
        results["__seed_average"] = True
        results["__effective_batch"] = int(eff_b)
        print(
            f"[seed-averaged] obs={obs:.4f}  null_mean={null.mean():.4f}  "
            f"null_max={null.max():.4f}  obs/null_mean={obs / null.mean():.2f}  "
            f"obs in null-sigma={obs_z:.1f}  p={p:.2e}  ({dt:.1f}s)",
            flush=True,
        )
        print(f"\nSeed-averaged p across {len(labels)} seeds: {p:.2e}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, args.output)
        print(f"Saved to {args.output}")

        import json

        try:
            args.aggregate_json.parent.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, OSError):
            args.aggregate_json = args.output.parent / args.aggregate_json.name
            args.aggregate_json.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate_json.write_text(
            json.dumps(
                {
                    "seed_averaged_p": float(p),
                    "n_seeds": len(labels),
                    "seed_labels": labels,
                    "n_perms": args.n_perms,
                    "shuffle_seed": args.shuffle_seed,
                    "requested_batch": int(args.batch),
                    "effective_batch": int(eff_b),
                },
                indent=2,
            )
        )
        print(f"Aggregate seed-averaged p mirrored to {args.aggregate_json}")
        return

    for seed_idx, (label, rates) in enumerate(rates_per_seed.items()):
        # Independent permutation draws per training seed: Fisher combination
        # downstream assumes independence of the per-seed p-values, which is
        # only valid if the null permutations themselves are independent
        # across seeds. Using a single shuffle_seed for all training seeds
        # would make the combined p anti-conservative.
        per_label_seed = args.shuffle_seed + seed_idx
        print(
            f"[{label}] starting {args.n_perms} perms (batch {args.batch}, "
            f"shuffle_seed={per_label_seed})...",
            flush=True,
        )
        t0 = time.time()
        obs, p, null, obs_z, eff_b = permutation_test_vectorized(
            rates,
            n_permutations=args.n_perms,
            batch=args.batch,
            seed=per_label_seed,
            device=args.device,
        )
        dt = time.time() - t0
        effective_batches[label] = int(eff_b)
        # Save the upper tail of the null distribution for GPD fitting.
        # We keep the top 5% of null samples (5,000 of 100,000) plus the 95th percentile
        # threshold. This is enough to fit a Generalized Pareto Distribution to the
        # exceedances and extrapolate a parametric p-value beyond the empirical floor.
        sort_idx = np.argsort(null)[::-1]
        n_tail = max(5000, int(0.05 * len(null)))
        null_tail = null[sort_idx[:n_tail]]
        null_p95 = float(np.quantile(null, 0.95))
        per_seed = {
            "observed": float(obs),
            "p_value": float(p),
            "n_permutations": args.n_perms,
            "requested_batch": int(args.batch),
            "effective_batch": int(eff_b),
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_max": float(null.max()),
            "null_p95": null_p95,
            "null_p99": float(np.quantile(null, 0.99)),
            "null_p999": float(np.quantile(null, 0.999)),
            "null_tail_top5pct": null_tail.astype(np.float32),  # (5000,) float32
            "obs_in_null_sigma_units": obs_z,
            "n_null_geq_obs": int((null >= obs).sum()),
            "elapsed_seconds": dt,
        }
        results[label] = per_seed
        p_values.append(p)
        print(
            f"[{label}] obs={obs:.4f}  null_mean={null.mean():.4f}  "
            f"null_max={null.max():.4f}  obs/null_mean={obs / null.mean():.2f}  "
            f"obs in null-sigma={obs_z:.1f}  p={p:.2e}  ({dt:.1f}s)",
            flush=True,
        )

    fisher_p = fisher_combine(p_values)
    results["__fisher_combined_p"] = fisher_p
    results["__shuffle_seed"] = args.shuffle_seed
    results["__n_perms"] = args.n_perms
    print(f"\nFisher-combined p across {len(p_values)} seeds: {fisher_p:.2e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.output)
    print(f"Saved to {args.output}")

    # Mirror the seed-aggregate Fisher p to a small JSON for the paper. We do
    # NOT update the per-row manifest here: rates_per_seed is keyed by seed
    # labels, not checkpoint paths, and Fisher-p is intrinsically aggregate.
    import json

    # Tolerate a broken `results/` symlink (e.g. project-level symlink that
    # points at a Mac SSD path on a different host). Falling back next to
    # --output ensures the aggregate JSON still lands somewhere persistent.
    try:
        args.aggregate_json.parent.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, OSError):
        args.aggregate_json = args.output.parent / args.aggregate_json.name
        args.aggregate_json.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_json.write_text(
        json.dumps(
            {
                "fisher_combined_p": fisher_p,
                "n_seeds": len(p_values),
                "per_seed_p": {
                    lbl: float(results[lbl]["p_value"])
                    for lbl in results
                    if not lbl.startswith("__")
                },
                "n_perms": args.n_perms,
                "shuffle_seed": args.shuffle_seed,
                "requested_batch": int(args.batch),
                "effective_batch": effective_batches,
            },
            indent=2,
        )
    )
    print(f"Aggregate Fisher p mirrored to {args.aggregate_json}")
