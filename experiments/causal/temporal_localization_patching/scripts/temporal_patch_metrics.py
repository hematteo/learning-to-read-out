"""CLI driver for the temporal readout-patch metrics.

The analysis library lives in readout.dynamics.temporal_patch (shared with
sparse_feature_causal_tests and the sibling grid/rescue drivers); this script
only parses arguments, resolves the active model config, and calls run().
See the library module docstring for the metric definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import readout.dynamics.temporal_patch as tpl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=["pythia-160m", "pythia-1b"],
        default="pythia-160m",
        help="Which crosscoder to evaluate.",
    )
    ap.add_argument(
        "--d-sae",
        type=int,
        default=24576,
        help="d_sae for pythia-1b (one of 8192/16384/24576). Ignored for pythia-160m (Run-3 fixed at 8192).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=tpl.OUT_DIR_DEFAULT / "smoke")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1 transition, 2 subsets (transition_top_k + matched_random), 3 concepts.",
    )
    ap.add_argument("--max-contexts", type=int, default=256)
    ap.add_argument("--n-match", type=int, default=99)
    ap.add_argument("--n-concept-tokens", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--bootstrap", type=int, default=1)
    ap.add_argument("--h-eval", choices=["target", "base"], default="target")
    ap.add_argument("--no-probe", action="store_true", help="Skip M4 probe-logit transfer.")
    ap.add_argument(
        "--include-kl",
        action="store_true",
        help="Include M6 candidate-set logit-lens KL (appendix).",
    )
    ap.add_argument(
        "--eval-tokens",
        type=Path,
        default=tpl.CORPUS_TOKENS_PYTHIA,
        help="Evaluation-token tensor (ids/scripts/languages dict); see docs/DATA.md.",
    )
    args = ap.parse_args()

    tpl.CORPUS_TOKENS = args.eval_tokens

    # Resolve the active model config (library-module state, as in the grid driver).
    if args.model == "pythia-160m":
        tpl.ACTIVE_CFG = tpl.CFG_PYTHIA_160M
        tpl.ACTIVE_D_SAE = 8192
    elif args.model == "pythia-1b":
        tpl.ACTIVE_CFG = tpl.CFG_PYTHIA_1B
        tpl.ACTIVE_D_SAE = args.d_sae
    tpl._refresh_aliases()

    K = args.top_k
    if args.smoke:
        transitions = [
            tpl.TransitionSpec(
                base_step=512,
                target_step=1000,
                name="t_512_1000",
                subset_specs=[("transition_top_k", K), ("matched_random_top_k", K)],
            ),
        ]
        concepts = ["punctuation", "function_words", "non_latin_scripts"]
    else:
        transitions = [
            tpl.TransitionSpec(
                base_step=512,
                target_step=1000,
                name="t_512_1000",
                subset_specs=[
                    ("transition_top_k", K),
                    ("decay_top_k", K),
                    ("persistent_top_k", K),
                    ("matched_random_top_k", K),
                ],
            ),
            tpl.TransitionSpec(
                base_step=1000,
                target_step=143000,
                name="t_1000_terminal",
                subset_specs=[
                    ("late_specialist_top_k", K),
                    ("late_rise_top_k", K),
                    ("persistent_top_k", K),
                    ("matched_random_top_k", K),
                ],
            ),
        ]
        concepts = tpl.DEFAULT_CONCEPTS

    tpl.run(
        seed=args.seed,
        transitions=transitions,
        concept_names=concepts,
        out_dir=args.out_dir,
        h_step_for_eval=args.h_eval,
        max_contexts=args.max_contexts,
        n_match=args.n_match,
        n_concept_tokens=args.n_concept_tokens,
        fit_probe=not args.no_probe,
        include_kl=args.include_kl,
        bootstrap=args.bootstrap,
    )


if __name__ == "__main__":
    main()
