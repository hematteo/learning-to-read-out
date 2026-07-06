"""Build the canonical per-run analysis table.

For every (run_id, seed) discovered on disk, populate one row with run
metadata, paths to rates / decoder norms, scalar diagnostics
(EV, L0, dead_rate), provenance fields (sha256 / metric_version / git_sha),
and pointers to derived artifacts (rates_norm, rotation, rotation_weight,
direction_to_terminal, lifecycle, cusum, cusum_null).

Write the table to `experiments/crosscoders/crosscoder_main/derived/aggregates/analysis_table.parquet` and a JSON manifest
(`experiments/crosscoders/crosscoder_main/derived/aggregates/analysis_table.manifest.json`) that records, for every row,
which derived artifacts are present on disk. Phases 1+ read from this table;
they should never re-derive metrics ad hoc.

Usage:
    .venv/bin/python -m experiments.crosscoders.crosscoder_main.aggregates.build_analysis_table
    .venv/bin/python -m experiments.crosscoders.crosscoder_main.aggregates.build_analysis_table --no-cache-derive
        # skip slow re-derivation; rows missing rates_path stay missing.
    .venv/bin/python -m experiments.crosscoders.crosscoder_main.aggregates.build_analysis_table --derive-artifacts
        # additionally compute rates_norm/rotation/lifecycle/cusum caches.
    .venv/bin/python -m experiments.crosscoders.crosscoder_main.aggregates.build_analysis_table \\
        --derive-artifacts --derive-cusum-null --n-perms 10000
        # also compute (or reuse) the temporal permutation null on rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
from readout.crosscoder.checkpoint_loaders import _load_npy_or_pt, load_run  # noqa: E402
from readout.dynamics import derive as derive_mod  # noqa: E402
from readout.dynamics.discovery import RunRow, list_runs  # noqa: E402
from readout.dynamics.provenance import (  # noqa: E402
    METRIC_VERSION,
    file_sha256,
    git_sha,
)

OUTPUT_PARQUET = REPO / "experiments/crosscoders/crosscoder_main/derived/aggregates/analysis_table.parquet"
OUTPUT_JSON = REPO / "experiments/crosscoders/crosscoder_main/derived/aggregates/analysis_table.manifest.json"


def _scalars_from_run(row: RunRow, derive_if_missing: bool) -> dict:
    """Compute (or load) EV, L0, dead_rate without keeping the full arrays in memory."""
    out = {"EV": None, "L0": None, "dead_rate": None}
    rates = None
    if row.rates_path is not None:
        try:
            obj = _load_npy_or_pt(Path(row.rates_path))
            if isinstance(obj, np.ndarray):
                if obj.ndim == 3 and row.seed_index_in_cache is not None:
                    rates = obj[row.seed_index_in_cache]
                elif obj.ndim == 2:
                    rates = obj
        except Exception as e:
            print(f"  [warn] failed to load rates for {row.run_id} seed {row.seed}: {e}")

    if rates is None and derive_if_missing and row.ckpt_path is not None:
        try:
            arr = load_run(
                run_id=row.run_id,
                seed=row.seed,
                d_sae=row.d_sae,
                ckpt_path=row.ckpt_path,
                rates_cache=row.rates_path,
                norms_cache=row.norms_path,
                seed_index_in_cache=row.seed_index_in_cache,
                keep_decoder_weights=False,
                steps_fallback=row.steps,
                allow_missing_rates=True,
            )
            rates = arr.rates
        except Exception as e:
            print(f"  [warn] derivation failed for {row.run_id} seed {row.seed}: {e}")

    if rates is not None:
        out["dead_rate"] = float((rates.max(axis=0) < 1e-4).mean())

    # EV and L0 come from the training quality dict (the only authoritative
    # source for L0; rates are P(fire), not feature counts per sample). Skip
    # checkpoints stored as directories (T1.3 per-snap SAE collection).
    if row.ckpt_path is not None and Path(row.ckpt_path).is_file():
        try:
            import torch  # local import; heavy

            ck = torch.load(row.ckpt_path, map_location="cpu", weights_only=False)
            q = ck.get("quality", {})
            if isinstance(q, dict):
                ev = q.get("explained_variance") or q.get("ev")
                if isinstance(ev, dict):
                    ev = ev.get("final") or ev.get("mean")
                out["EV"] = float(ev) if ev is not None else None
                l0 = q.get("mean_l0") or q.get("l0")
                out["L0"] = float(l0) if l0 is not None else None
        except Exception as e:
            print(f"  [warn] quality read failed for {row.run_id} seed {row.seed}: {e}")
    return out


def _provenance_for_row(row: RunRow, gsha: str | None) -> dict:
    """Compute file sha256s + (metric_version, git_sha) for one row.

    Hashing is streaming and cheap relative to checkpoint loads, but multi-seed
    npy caches (run3 firing_rates_all_seeds.npy, ~10 MB) hash once per seed.
    Tolerable; cache the results in a future iteration if it becomes a
    bottleneck.
    """
    return {
        "checkpoint_sha256": file_sha256(row.ckpt_path),
        "rates_sha256": file_sha256(row.rates_path),
        "norms_sha256": file_sha256(row.norms_path),
        "metric_version": METRIC_VERSION,
        "git_sha": gsha,
    }


def _artifact_summary(rec: dict) -> dict:
    """Boolean presence map for each plan §2 derived artifact."""
    keys = (
        "rates_norm_path",
        "rotation_path",
        "rotation_weight_path",
        "direction_to_terminal_path",
        "lifecycle_path",
        "cusum_path",
        "cusum_null_path",
    )
    return {k.removesuffix("_path"): rec.get(k) is not None for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-cache-derive",
        action="store_true",
        help="Skip slow re-derivation of rates from checkpoints; rows without "
        "cached rates keep EV/L0/dead_rate as null.",
    )
    ap.add_argument(
        "--derive-artifacts",
        action="store_true",
        help="Compute and cache plan §2 derived artifacts (rates_norm, rotation, "
        "rotation_weight, direction_to_terminal, lifecycle, cusum) per row. "
        "Without this flag, the script only registers existing on-disk caches.",
    )
    ap.add_argument(
        "--no-derive-rotation",
        action="store_true",
        help="Skip rotation/direction-to-terminal derivation (they require "
        "loading the full (K, D, d_model) decoder tensor from each checkpoint).",
    )
    ap.add_argument(
        "--derive-cusum-null",
        action="store_true",
        help="Compute (or reuse the existing 100k cache for) the signed-CUSUM "
        "permutation null on firing rates. Slow: ~30s/row at n_perms=10000.",
    )
    ap.add_argument("--n-perms", type=int, default=10_000)
    ap.add_argument(
        "--force-rederive",
        action="store_true",
        help="Recompute artifacts even when the cache file exists.",
    )
    ap.add_argument(
        "--no-provenance",
        action="store_true",
        help="Skip sha256 hashing; useful for fast smoke runs.",
    )
    ap.add_argument(
        "--output-parquet",
        type=Path,
        default=OUTPUT_PARQUET,
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_JSON,
    )
    ap.add_argument(
        "--filter-run-id",
        default=None,
        help="Comma-separated run_ids to keep (others dropped). Useful for "
        "targeted reruns; the manifest still records them as filtered.",
    )
    args = ap.parse_args()

    rows = list_runs()
    if args.filter_run_id:
        keep = set(args.filter_run_id.split(","))
        rows = [r for r in rows if r.run_id in keep]
    if not rows:
        print("No runs discovered. Check SSD mount and cluster sync.")
        return
    print(f"Discovered {len(rows)} (run_id, seed) rows.")
    by_id: dict[str, int] = {}
    for r in rows:
        by_id[r.run_id] = by_id.get(r.run_id, 0) + 1
    for k, v in sorted(by_id.items()):
        print(f"  {k}: {v} seed(s)")

    derive_scalars = not args.no_cache_derive
    gsha = git_sha()
    derived_root = derive_mod.default_derived_root()
    enriched: list[dict] = []
    artifact_index: list[dict] = []

    for r in rows:
        scalars = _scalars_from_run(r, derive_if_missing=derive_scalars)
        rec = r.to_dict()
        rec.update(scalars)

        if not args.no_provenance:
            rec.update(_provenance_for_row(r, gsha))
        else:
            rec.update(
                {
                    "checkpoint_sha256": None,
                    "rates_sha256": None,
                    "norms_sha256": None,
                    "metric_version": METRIC_VERSION,
                    "git_sha": gsha,
                }
            )

        if args.derive_artifacts:
            try:
                paths = derive_mod.derive_row(
                    r,
                    root=derived_root,
                    derive_rotation=not args.no_derive_rotation,
                    derive_cusum_null=args.derive_cusum_null,
                    n_perms=args.n_perms,
                    force=args.force_rederive,
                )
            except Exception as e:
                print(f"  [warn] derive failed for {r.run_id} seed {r.seed}: {e}")
                paths = derive_mod.existing_paths(r, root=derived_root)
        else:
            paths = derive_mod.existing_paths(r, root=derived_root)

        rec.update(paths.as_dict())
        enriched.append(rec)
        artifact_index.append(
            {
                "run_id": r.run_id,
                "seed": r.seed,
                "artifacts": _artifact_summary(rec),
                "paths": paths.as_dict(),
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "metric_version": METRIC_VERSION,
        "git_sha": gsha,
        "derived_root": str(derived_root),
        "n_rows": len(enriched),
        "rows": enriched,
        "artifact_index": artifact_index,
    }
    with open(args.output_json, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Wrote manifest JSON: {args.output_json}")

    try:
        import pandas as pd

        df = pd.DataFrame(enriched)
        df.to_parquet(args.output_parquet, index=False)
        print(f"Wrote parquet:    {args.output_parquet}  ({len(df)} rows)")
        cols = [
            "run_id",
            "seed",
            "d_sae",
            "matrix_type",
            "EV",
            "L0",
            "dead_rate",
            "lifecycle_path",
            "cusum_path",
            "rotation_path",
        ]
        avail = [c for c in cols if c in df.columns]
        printed = df[avail].copy()
        for c in avail:
            if c.endswith("_path"):
                printed[c] = printed[c].notna()
        print(printed.to_string(index=False))
    except ImportError:
        print("pandas/pyarrow not available; only JSON manifest was written.")


if __name__ == "__main__":
    main()
