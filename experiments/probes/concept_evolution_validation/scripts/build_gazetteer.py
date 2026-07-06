"""Build the 37-concept gazetteer for Pythia-1B and write audit report.

Outputs:
  configs/preregistration/concepts_v1.json       — {concept: [token_ids]}
  configs/preregistration/concepts_v1_audit.json — per-concept stats

Usage:
  python scripts/build_gazetteer.py
  python scripts/build_gazetteer.py --model EleutherAI/pythia-70m --revision step143000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from readout.probes.concept_gazetteer import (
    audit_gazetteer,
    build_gazetteer,
    save_gazetteer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-1b")
    parser.add_argument("--revision", default="step143000")
    parser.add_argument(
        "--out-dir",
        default="configs/preregistration",
        help="Output directory (relative to repo root).",
    )
    parser.add_argument("--min-tokens", type=int, default=20)
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model} @ {args.revision}")
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    print(f"  vocab_size = {tok.vocab_size}")

    print("Building gazetteer...")
    gaz = build_gazetteer(tok)
    print(f"  built {len(gaz)} concepts")

    print("Auditing gazetteer...")
    report = audit_gazetteer(gaz, tok, min_tokens=args.min_tokens)

    # Print audit table to stdout
    print()
    print(f"{'Concept':<30} {'N tokens':>8}  {'Pass':>5}  Sample tokens")
    print("-" * 100)
    for concept in sorted(gaz):
        r = report[concept]
        sample_str = ", ".join(repr(t) for t in r["sample_tokens"])
        ok = "OK" if r["pass_min_size"] else "FAIL"
        print(f"{concept:<30} {r['n_tokens']:>8}  {ok:>5}  {sample_str}")

    # Failure summary
    failing = [c for c in gaz if not report[c]["pass_min_size"]]
    heavy_overlaps = [c for c in gaz if report[c]["heavy_overlaps"]]
    meta = report["_meta"]

    print()
    print("─── SUMMARY ──────────────────────────────────────────────────────")
    print(f"Total concepts: {meta['n_concepts']}")
    print(
        f"Total unique tokens covered: {meta['total_unique_tokens']} / {meta['vocab_size']}"
    )
    print(f"Vocabulary coverage: {meta['coverage_fraction'] * 100:.1f}%")
    print(f"Concepts failing min-size ({args.min_tokens}): {len(failing)}")
    if failing:
        for c in failing:
            print(f"    ✗ {c} ({report[c]['n_tokens']} tokens)")
    print(f"Concepts with heavy overlap (>30%) with another: {len(heavy_overlaps)}")
    if heavy_overlaps:
        for c in heavy_overlaps:
            for other, shared, frac in report[c]["heavy_overlaps"]:
                print(f"    ⚠ {c} ∩ {other}: {shared} tokens ({frac:.0%} of {c})")

    out_dir = Path(__file__).resolve().parents[1] / args.out_dir
    save_gazetteer(gaz, report, out_dir)
    print()
    print(f"Saved to: {out_dir}/")


if __name__ == "__main__":
    main()
