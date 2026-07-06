"""Off-family specificity matrix for sparse-feature causal pilot runs.

Rows are source feature sets selected by `run_1b_pilot.py`; columns are measured
token-family metrics. The output checks whether family-selected features mostly
damage/recover their own family or whether the intervention is generic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_1b_pilot as PILOT  # noqa: E402

from readout.core.data import write_csv
from readout.core.paths import repo_root  # noqa: E402

REPO = repo_root()

CONCEPTS = ["non_latin_scripts", "punctuation", "digits", "function_words"]


def source_run_dir(root: Path, concept: str, snapshot_step: int) -> Path:
    return root / f"1b_{concept}_step{snapshot_step}_positive"


def read_top_features(root: Path, concept: str, snapshot_step: int) -> list[int]:
    raw = torch.load(
        source_run_dir(root, concept, snapshot_step) / "raw.pt",
        map_location="cpu",
        weights_only=False,
    )
    return [int(x) for x in raw["top_features"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-step", type=int, required=True)
    ap.add_argument("--h-step", type=int, required=True)
    ap.add_argument("--concepts", nargs="+", default=CONCEPTS)
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--max-contexts", type=int, default=128)
    ap.add_argument("--n-concept-tokens", type=int, default=256)
    ap.add_argument("--n-controls", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=PILOT.device_default())
    ap.add_argument(
        "--pilot-root",
        type=Path,
        default=REPO / "results/experiments/sparse_feature_causal_tests",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = (
            args.pilot_root / f"specificity_step{args.snapshot_step}_h{args.h_step}"
        )

    PILOT.configure_tpm()
    pieces = PILOT.load_pieces(args.snapshot_step, args.device)
    step_idx = pieces.steps.index(args.snapshot_step)
    agg = torch.load(PILOT.AGG_PATH, map_location="cpu", weights_only=False)
    rates = agg["rates"].numpy()
    decoder_norms = agg["decoder_norms"].numpy()

    ids, ids_seqs = PILOT.load_eval_sequences()
    tok = PILOT.TPM.load_tokenizer()
    W_snapshot = PILOT.TPM.load_snapshot(args.snapshot_step)
    W_terminal = PILOT.TPM.load_snapshot(143000)
    meta = PILOT.TPM.build_token_meta(tok, ids, V_explicit=W_snapshot.shape[0])
    registry = PILOT.TPM.build_concept_registry(meta)
    pool = PILOT.TPM.build_pools(meta, W_terminal)

    source_features = {
        concept: read_top_features(args.pilot_root, concept, args.snapshot_step)
        for concept in args.concepts
    }

    rows: list[dict] = []
    for measured in args.concepts:
        concept = registry[measured]
        in_C, null_C = PILOT.concept_rows(
            meta, concept, pool, args.n_concept_tokens, args.seed
        )
        row_ids = np.concatenate([in_C, null_C]).astype(np.int64)
        n_in = len(in_C)
        row_recon = PILOT.encode_reconstruct_rows(row_ids, pieces, args.device)
        _h_sel, h_eval = PILOT.collect_h_splits(
            args.h_step, ids_seqs, concept, args.max_contexts, args.seed
        )
        orig = PILOT.concept_mass_from_rows(
            h_eval, W_snapshot[row_ids].to(args.device), n_in, args.device
        )
        full = PILOT.concept_mass_from_rows(
            h_eval, row_recon.W_full, n_in, args.device
        )
        bias = PILOT.concept_mass_from_rows(
            h_eval, row_recon.W_bias, n_in, args.device
        )
        for source in args.concepts:
            for k in args.ks:
                feats = source_features[source][:k]
                W_ablate = PILOT.edited_rows(row_recon, pieces, feats, "ablate")
                W_only = PILOT.edited_rows(row_recon, pieces, feats, "only")
                cm_ablate = PILOT.concept_mass_from_rows(
                    h_eval, W_ablate, n_in, args.device
                )
                cm_only = PILOT.concept_mass_from_rows(
                    h_eval, W_only, n_in, args.device
                )
                rows.extend(
                    [
                        {
                            "snapshot_step": args.snapshot_step,
                            "h_step": args.h_step,
                            "source_concept": source,
                            "measured_concept": measured,
                            "k": k,
                            "condition": "source_ablate",
                            "control_id": -1,
                            "concept_mass": cm_ablate,
                            "drop_from_full": full - cm_ablate,
                            "recovery_from_bias": cm_ablate - bias,
                            "orig_concept_mass": orig,
                            "full_concept_mass": full,
                            "bias_concept_mass": bias,
                        },
                        {
                            "snapshot_step": args.snapshot_step,
                            "h_step": args.h_step,
                            "source_concept": source,
                            "measured_concept": measured,
                            "k": k,
                            "condition": "source_only",
                            "control_id": -1,
                            "concept_mass": cm_only,
                            "drop_from_full": full - cm_only,
                            "recovery_from_bias": cm_only - bias,
                            "orig_concept_mass": orig,
                            "full_concept_mass": full,
                            "bias_concept_mass": bias,
                        },
                    ]
                )
                for ci, ctrl in enumerate(
                    PILOT.matched_controls(
                        feats,
                        rates,
                        decoder_norms,
                        step_idx,
                        args.n_controls,
                        args.seed + k,
                    )
                ):
                    Wc_ablate = PILOT.edited_rows(row_recon, pieces, ctrl, "ablate")
                    cm_c_ablate = PILOT.concept_mass_from_rows(
                        h_eval, Wc_ablate, n_in, args.device
                    )
                    rows.append(
                        {
                            "snapshot_step": args.snapshot_step,
                            "h_step": args.h_step,
                            "source_concept": source,
                            "measured_concept": measured,
                            "k": k,
                            "condition": "matched_ablate",
                            "control_id": ci,
                            "concept_mass": cm_c_ablate,
                            "drop_from_full": full - cm_c_ablate,
                            "recovery_from_bias": cm_c_ablate - bias,
                            "orig_concept_mass": orig,
                            "full_concept_mass": full,
                            "bias_concept_mass": bias,
                        }
                    )
        print(f"[specificity] measured={measured} done", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "specificity.csv", rows)
    manifest = {
        "snapshot_step": args.snapshot_step,
        "h_step": args.h_step,
        "concepts": args.concepts,
        "ks": args.ks,
        "n_controls": args.n_controls,
        "max_contexts": args.max_contexts,
        "n_concept_tokens": args.n_concept_tokens,
        "pilot_root": str(args.pilot_root),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[specificity] wrote {args.out_dir / 'specificity.csv'}", flush=True)


if __name__ == "__main__":
    main()

