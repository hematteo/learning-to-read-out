#!/usr/bin/env python3
"""Aggregate recipe-control lifecycle statistics across dictionary-fit seeds.

Input: one ``lifecycle_stats.json`` per crosscoder seed (``analyze_lifecycle.py``
run with ``--seed 0/1/2`` after ``train_control_crosscoder.py --seed 0/1/2``).
Output: a flat per-(seed, condition) CSV plus a JSON with means, sample standard
deviations, ranges, and per-seed checks that the readout LR ordering of the
median peak step and the reorganization step holds (slow -> fast readout LR
is non-increasing) and that ``warmup_short`` matches ``baseline``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rc_common import DEFAULT_CONDITIONS, write_csv

LR_CONDITIONS = ["wu_lr_0p25", "baseline", "wu_lr_4x"]  # slow -> fast readout LR
PROFILES = ["persistent", "transitional", "early-decaying", "late-emerging", "mixed/ambiguous"]
SCALARS = [
    "median_peak_step",
    "median_peak_tau",
    "median_lifespan_snaps",
    "reorg_peak_step",
    "reorg_peak_tau",
    "max_norm_turnover",
]
QUALITY = ["explained_variance", "reconstruction_mse", "mean_l0", "dead_rate"]


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-stat", action="append", required=True, type=Path, help="lifecycle_stats.json (repeat)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="seed ids matching --seed-stat order")
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(len(args.seed_stat)))
    if len(seeds) != len(args.seed_stat):
        raise SystemExit("--seeds must have one entry per --seed-stat")
    by_seed = {seed: json.loads(path.read_text()) for seed, path in zip(seeds, args.seed_stat)}
    conditions = args.conditions
    lr_conditions = [c for c in LR_CONDITIONS if c in conditions]

    rows = []
    for seed in sorted(by_seed):
        for condition in conditions:
            item = by_seed[seed][condition]
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    **{key: item[key] for key in SCALARS},
                    **{f"profile_{name}": item["profile_fractions"][name] for name in PROFILES},
                    **{f"quality_{name}": item["quality"].get(name) for name in QUALITY},
                }
            )
    write_csv(args.out_csv, rows)

    summary: dict = {"seeds": sorted(by_seed), "conditions": {}, "ordering_checks": {}}
    for condition in conditions:
        cs = {}
        for key in SCALARS:
            cs[key] = summarize([float(by_seed[s][condition][key]) for s in sorted(by_seed)])
        for profile in PROFILES:
            cs[f"profile_fraction_{profile}"] = summarize(
                [float(by_seed[s][condition]["profile_fractions"][profile]) for s in sorted(by_seed)]
            )
        for key in QUALITY:
            values = [by_seed[s][condition]["quality"].get(key) for s in sorted(by_seed)]
            if all(v is not None for v in values):
                cs[f"quality_{key}"] = summarize([float(v) for v in values])
        summary["conditions"][condition] = cs

    for seed in sorted(by_seed):
        st = by_seed[seed]
        peak = [float(st[c]["median_peak_step"]) for c in lr_conditions]
        reorg = [float(st[c]["reorg_peak_step"]) for c in lr_conditions]
        check = {
            "lr_conditions_slow_to_fast": lr_conditions,
            "median_peak_steps": peak,
            "median_peak_nonincreasing": all(a >= b for a, b in zip(peak, peak[1:])),
            "reorg_peak_steps": reorg,
            "reorg_peak_nonincreasing": all(a >= b for a, b in zip(reorg, reorg[1:])),
        }
        if "warmup_short" in st and "baseline" in st:
            check["warmup_minus_baseline_median_peak_step"] = float(st["warmup_short"]["median_peak_step"]) - float(
                st["baseline"]["median_peak_step"]
            )
            check["warmup_minus_baseline_reorg_peak_step"] = float(st["warmup_short"]["reorg_peak_step"]) - float(
                st["baseline"]["reorg_peak_step"]
            )
        summary["ordering_checks"][f"seed{seed}"] = check

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["ordering_checks"], indent=2))
    print(f"wrote {args.out_json}\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
