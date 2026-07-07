"""Build and persist contrastive (y+/y-) task datasets per task family.

Datasets are tokenizer-specific and saved to
``results/experiments/probes/contrastive_readout_swap/datasets/<model>/``
as JSON-lines, one file per family. Auditable; reused by the swap-grid eval
and the sparse-feature attribution experiment.

Usage:
    uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \\
        --model EleutherAI/pythia-1b --out-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from readout.core.paths import (
    repo_root,  # noqa: E402
    snapshot_path,  # noqa: E402
)
from readout.probes import contrastive_tasks as CT  # noqa: E402

REPO = repo_root()


def _maybe_load_terminal_W_U(model_name: str) -> torch.Tensor | None:
    """Use the terminal-step W_U row norms for matched-norm filtering, if available.

    Only missing/unreadable snapshot files are tolerated (returns None); any
    other failure (bad payload shape, dtype, ...) propagates so it can't
    silently disable the row-norm-matched filtering the README advertises.
    """
    try:
        # Pythia uses step 143000 as terminal.
        p = snapshot_path(model_name, step=143000, kind="wu")
        if p.exists():
            W = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(W, dict):
                W = W.get("W_U") or W.get("weight") or next(iter(W.values()))
            return W.float()
    except (FileNotFoundError, OSError) as e:
        print(f"[warn] failed to read terminal W_U snapshot ({e})", flush=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="EleutherAI/pythia-1b",
        help="HF model name (used only for tokenizer + optional W_U norm match).",
    )
    ap.add_argument(
        "--families",
        nargs="+",
        default=list(CT.TASK_BUILDERS.keys()),
        help=f"Families to build (default = all). Available: {list(CT.TASK_BUILDERS)}",
    )
    ap.add_argument("--n-max-per-family", type=int, default=2000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory; one <family>.jsonl per family will be written here.",
    )
    ap.add_argument(
        "--no-norm-match",
        action="store_true",
        help="Skip W_U row-norm matched filtering (e.g. when terminal W_U is missing).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    W_U = None if args.no_norm_match else _maybe_load_terminal_W_U(args.model)
    if W_U is None and not args.no_norm_match:
        print(
            f"[WARN] no terminal W_U found for {args.model}: the row-norm-matched "
            "distractor filtering advertised in the README is DISABLED for this "
            "build. Pass --no-norm-match to acknowledge, or make the terminal "
            "snapshot available (see docs/DATA.md).",
            flush=True,
        )

    summary: list[dict] = []
    for family in args.families:
        if family not in CT.TASK_BUILDERS:
            print(f"[skip] unknown family {family!r}", flush=True)
            continue
        builder = CT.TASK_BUILDERS[family]
        kwargs = {"n_max": args.n_max_per_family, "W_U_for_norm_match": W_U}
        if "rng_seed" in builder.__code__.co_varnames:
            kwargs["rng_seed"] = args.rng_seed
        examples = builder(tokenizer, **kwargs)

        # Build paired corruption examples for SVA + IOI.
        corrupted: list[CT.Example] = []
        for ex in examples:
            c = CT.build_corruption(ex, tokenizer)
            if c is not None:
                corrupted.append(c)

        out_path = args.out_dir / f"{family}.jsonl"
        CT.save_examples(out_path, examples)
        if corrupted:
            CT.save_examples(args.out_dir / f"{family}_corrupt.jsonl", corrupted)

        # Random-token control distractors as a sidecar manifest.
        rand_distractors = CT.random_distractor_ids(
            tokenizer,
            examples,
            rng_seed=args.rng_seed,
        )
        ctrl_path = args.out_dir / f"{family}_random_distractors.json"
        ctrl_path.write_text(json.dumps(rand_distractors))

        # MCQ-letter families: also emit a label-permutation control where
        # y_minus is a uniformly random non-gold letter token. This is the
        # stronger control for piqa / arc_easy / arc_challenge / sciq.
        label_perm = CT.mcq_label_permutation_distractor_ids(tokenizer, examples, rng_seed=args.rng_seed)
        if label_perm is not None:
            (args.out_dir / f"{family}_label_permutation.json").write_text(json.dumps(label_perm))

        row = {
            "family": family,
            "n": len(examples),
            "n_corrupt": len(corrupted),
            "out": str(out_path),
        }
        summary.append(row)
        print(
            f"[built] {family}: n={len(examples):4d}  corrupt={len(corrupted):4d}  -> {out_path.name}",
            flush=True,
        )

    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "rng_seed": args.rng_seed,
                "n_max_per_family": args.n_max_per_family,
                "norm_match": W_U is not None,
                "families": summary,
            },
            indent=2,
        )
    )
    print(f"[done] wrote {len(summary)} family files to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
