"""Matched-token controls for the Pythia-160M WordNet W_U probe audit.

The paper-facing WordNet probe asks whether WordNet supersenses are linearly
available from centered W_U rows over training. This script runs the same probe
pipeline against controls that preserve obvious token covariates:

* metadata-only probe: frequency, token length, row norm, prefix/special flags,
  and coarse token class, without W_U geometry;
* matched-random null: off-family token sets matched per positive token;
* label-shuffle null: token sets sampled from the same matched bins, allowing
  original positives, to approximate a within-bin label permutation.

Frequencies are eval-corpus counts, not Pile training frequencies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from run_wordnet_supersense_probe import (
    LEXNAME_ORDER,
    build_wordnet_concepts,
    centered_rows,
)

from src.core.data import write_csv
from src.core.model_specs import DEFAULT_STEPS_BY_MODEL, MODEL_HF_NAMES
from src.core.paths import repo_root, snapshot_path
from src.core.repro import git_commit
from src.crosscoder.snapshots import load_snapshot
from src.probes.wu_probes_gpu import probe_balanced_accuracy_batched

DEFAULT_EVAL_TOKENS = (
    REPO / "_archive/legacy_crosscoder_160m/intervention/eval_tokens.pt"
)
FALLBACK_EVAL_TOKENS = REPO / "results/derived/eval_corpus/eval_tokens.pt"


@dataclass(frozen=True)
class TokenCovariates:
    text: list[str]
    log_freq: np.ndarray
    freq_bin: np.ndarray
    char_len: np.ndarray
    len_bin: np.ndarray
    has_space: np.ndarray
    is_special: np.ndarray
    coarse_class: np.ndarray
    coarse_class_names: list[str]


def load_eval_counts(path: Path, vocab_size: int) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    ids = data["ids"].detach().cpu().numpy().astype(np.int64)
    ids = ids[(0 <= ids) & (ids < vocab_size)]
    return np.bincount(ids, minlength=vocab_size).astype(np.int64)


def ranked_bins(
    values: np.ndarray, n_bins: int, *, zero_bin: bool = False
) -> np.ndarray:
    values = np.asarray(values)
    bins = np.zeros(len(values), dtype=np.int32)
    if zero_bin:
        mask = values > 0
        start = 1
        n_rank_bins = max(1, n_bins - 1)
    else:
        mask = np.isfinite(values)
        start = 0
        n_rank_bins = n_bins
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return bins
    order = np.argsort(values[idx], kind="mergesort")
    ranks = np.empty(len(idx), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(idx), endpoint=False)
    b = np.floor(ranks * n_rank_bins).astype(np.int32) + start
    out = bins.copy()
    out[idx] = b
    return np.clip(out, 0, n_bins - 1)


def coarse_class_of(text: str, *, special: bool) -> str:
    stripped = text.strip()
    if special:
        return "special"
    if text and all(c.isspace() for c in text):
        return "whitespace"
    if not stripped:
        return "empty"
    if stripped.isascii() and stripped.isalpha():
        if stripped.islower():
            return "ascii_alpha_lower"
        if stripped.isupper():
            return "ascii_alpha_upper"
        return "ascii_alpha_mixed"
    if stripped.isascii() and stripped.isdigit():
        return "ascii_digit"
    if stripped.isascii() and all(not c.isalnum() for c in stripped):
        return "ascii_punct"
    if stripped.isascii():
        return "ascii_other"
    return "non_ascii"


def build_token_covariates(
    tokenizer, counts: np.ndarray, vocab_size: int
) -> TokenCovariates:
    text: list[str] = []
    char_len = np.zeros(vocab_size, dtype=np.int32)
    has_space = np.zeros(vocab_size, dtype=bool)
    is_special = np.zeros(vocab_size, dtype=bool)
    coarse = np.empty(vocab_size, dtype=object)
    special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])

    for tid in range(vocab_size):
        try:
            decoded = tokenizer.decode([tid])
        except Exception:
            decoded = ""
        special = tid in special_ids or decoded.startswith("<|")
        text.append(decoded)
        stripped = decoded.lstrip(" ")
        char_len[tid] = len(stripped)
        has_space[tid] = decoded.startswith(" ")
        is_special[tid] = special
        coarse[tid] = coarse_class_of(decoded, special=special)

    class_names = sorted(str(x) for x in set(coarse.tolist()))
    log_freq = np.log1p(counts).astype(np.float32)
    freq_bin = ranked_bins(counts, 10, zero_bin=True)
    len_bin = np.where(
        char_len <= 1,
        0,
        np.where(
            char_len == 2,
            1,
            np.where(
                char_len == 3,
                2,
                np.where(char_len <= 5, 3, np.where(char_len <= 8, 4, 5)),
            ),
        ),
    ).astype(np.int32)
    return TokenCovariates(
        text=text,
        log_freq=log_freq,
        freq_bin=freq_bin,
        char_len=char_len,
        len_bin=len_bin,
        has_space=has_space,
        is_special=is_special,
        coarse_class=coarse,
        coarse_class_names=class_names,
    )


def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not math.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32)


def metadata_feature_matrix(cov: TokenCovariates, row_norm: np.ndarray) -> torch.Tensor:
    class_to_idx = {name: i for i, name in enumerate(cov.coarse_class_names)}
    one_hot = np.zeros((len(cov.text), len(cov.coarse_class_names)), dtype=np.float32)
    for i, name in enumerate(cov.coarse_class):
        one_hot[i, class_to_idx[str(name)]] = 1.0
    cols = [
        zscore(cov.log_freq),
        zscore(cov.char_len.astype(np.float32)),
        zscore(row_norm.astype(np.float32)),
        cov.has_space.astype(np.float32),
        cov.is_special.astype(np.float32),
        cov.freq_bin.astype(np.float32) / 9.0,
        cov.len_bin.astype(np.float32) / 5.0,
    ]
    mat = np.column_stack(cols + [one_hot]).astype(np.float32)
    return torch.from_numpy(mat)


@dataclass(frozen=True)
class MatchState:
    cov: TokenCovariates
    norm_bin: np.ndarray
    vocab_ids: np.ndarray


def build_match_state(cov: TokenCovariates, row_norm: np.ndarray) -> MatchState:
    return MatchState(
        cov=cov,
        norm_bin=ranked_bins(row_norm, 10, zero_bin=False),
        vocab_ids=np.arange(len(cov.text), dtype=np.int64),
    )


def candidate_mask_for_level(
    state: MatchState,
    tid: int,
    level: int,
    *,
    concept_mask: np.ndarray,
    used: set[int],
    disjoint: bool,
) -> np.ndarray:
    cov = state.cov
    mask = np.ones(len(cov.text), dtype=bool)
    if level <= 0:
        mask &= cov.freq_bin == cov.freq_bin[tid]
        mask &= cov.len_bin == cov.len_bin[tid]
        mask &= state.norm_bin == state.norm_bin[tid]
        mask &= cov.has_space == cov.has_space[tid]
        mask &= cov.is_special == cov.is_special[tid]
        mask &= cov.coarse_class == cov.coarse_class[tid]
    elif level == 1:
        mask &= cov.freq_bin == cov.freq_bin[tid]
        mask &= cov.len_bin == cov.len_bin[tid]
        mask &= state.norm_bin == state.norm_bin[tid]
        mask &= cov.is_special == cov.is_special[tid]
        mask &= cov.coarse_class == cov.coarse_class[tid]
    elif level == 2:
        mask &= cov.freq_bin == cov.freq_bin[tid]
        mask &= cov.len_bin == cov.len_bin[tid]
        mask &= cov.is_special == cov.is_special[tid]
        mask &= cov.coarse_class == cov.coarse_class[tid]
    elif level == 3:
        mask &= np.abs(cov.freq_bin - cov.freq_bin[tid]) <= 1
        mask &= np.abs(cov.len_bin - cov.len_bin[tid]) <= 1
        mask &= cov.is_special == cov.is_special[tid]
        mask &= cov.coarse_class == cov.coarse_class[tid]
    elif level == 4:
        mask &= np.abs(cov.freq_bin - cov.freq_bin[tid]) <= 1
        mask &= np.abs(cov.len_bin - cov.len_bin[tid]) <= 1
        mask &= cov.is_special == cov.is_special[tid]
    elif level == 5:
        mask &= np.abs(cov.freq_bin - cov.freq_bin[tid]) <= 2
        mask &= cov.is_special == cov.is_special[tid]
    else:
        mask &= cov.is_special == cov.is_special[tid]
    if disjoint:
        mask &= ~concept_mask
    if used:
        mask[np.fromiter(used, dtype=np.int64)] = False
    return mask


def sample_matched_set(
    concept_ids: Iterable[int],
    state: MatchState,
    rng: np.random.Generator,
    *,
    disjoint: bool,
) -> tuple[set[int], dict[str, object]]:
    concept = sorted(int(x) for x in concept_ids)
    concept_mask = np.zeros(len(state.cov.text), dtype=bool)
    concept_mask[concept] = True
    used: set[int] = set()
    level_counts = Counter()
    failed = 0

    for tid in rng.permutation(np.asarray(concept, dtype=np.int64)).tolist():
        picked = None
        for level in range(7):
            mask = candidate_mask_for_level(
                state,
                int(tid),
                level,
                concept_mask=concept_mask,
                used=used,
                disjoint=disjoint,
            )
            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                continue
            picked = int(rng.choice(candidates))
            level_counts[level] += 1
            break
        if picked is None:
            failed += 1
            continue
        used.add(picked)

    sampled = used
    overlap = len(sampled.intersection(concept_mask.nonzero()[0].tolist()))
    return sampled, {
        "n_requested": len(concept),
        "n_sampled": len(sampled),
        "failed": failed,
        "overlap_with_source": overlap,
        "exact_frac": level_counts[0] / max(len(concept), 1),
        "relax_mean": (
            sum(level * count for level, count in level_counts.items())
            / max(sum(level_counts.values()), 1)
        ),
    }


def run_probe_chunks(
    rows: torch.Tensor,
    concepts: dict[str, set[int]],
    *,
    seed: int,
    device: str | None,
    max_iter: int,
    chunk_size: int,
) -> dict[str, dict]:
    names = list(concepts)
    out: dict[str, dict] = {}
    for start in range(0, len(names), chunk_size):
        sub = {name: concepts[name] for name in names[start : start + chunk_size]}
        res = probe_balanced_accuracy_batched(
            rows,
            sub,
            seed=seed + start,
            device=device,
            max_iter=max_iter,
        )
        out.update(res)
    return out


def summarize_null(values: list[float], observed: float) -> dict[str, object]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float32)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": "",
            "p50": "",
            "p95": "",
            "max": "",
            "p_ge_observed": "",
        }
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(arr.max()), 6),
        "p_ge_observed": round(
            float((1 + np.sum(arr >= observed)) / (1 + arr.size)), 6
        ),
    }


def parse_steps(args: argparse.Namespace) -> list[int]:
    all_steps = list(DEFAULT_STEPS_BY_MODEL[args.model])
    if args.all_steps:
        return all_steps
    if args.steps:
        requested = [int(s) for s in args.steps.split(",") if s.strip()]
    else:
        requested = [0, 128, 256, 512, 1000, 143000]
    missing = sorted(set(requested) - set(all_steps))
    if missing:
        raise ValueError(f"Requested steps are not in the 32-step schedule: {missing}")
    return requested


def run(args: argparse.Namespace) -> None:
    root = repo_root()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODEL_HF_NAMES[args.model]
    steps = parse_steps(args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=args.local_files_only
    )
    concepts_all, audit_rows = build_wordnet_concepts(
        tokenizer,
        min_word_len=args.min_word_len,
        min_token_chars=args.min_token_chars,
        dominance_threshold=args.dominance_threshold,
    )
    concepts = {
        name: toks
        for name, toks in concepts_all.items()
        if len(toks) >= args.min_probe_tokens
    }
    concepts = {name: concepts[name] for name in LEXNAME_ORDER if name in concepts}

    probe_path = snapshot_path(model_name, steps[0])
    vocab_size = int(load_snapshot(model_name, steps[0]).shape[0])
    if not probe_path.exists():
        raise FileNotFoundError(probe_path)

    eval_tokens = Path(args.eval_tokens)
    if (
        not eval_tokens.exists()
        and eval_tokens == DEFAULT_EVAL_TOKENS
        and FALLBACK_EVAL_TOKENS.exists()
    ):
        eval_tokens = FALLBACK_EVAL_TOKENS
    counts = load_eval_counts(eval_tokens, vocab_size)
    cov = build_token_covariates(tokenizer, counts, vocab_size)

    start = time.time()
    summary_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []

    rng = np.random.default_rng(args.seed)
    for step_i, step in enumerate(steps, start=1):
        print(
            f"[{step_i:02d}/{len(steps):02d}] step {step}: observed + controls",
            flush=True,
        )
        wu = centered_rows(load_snapshot(model_name, step), args.preprocess)
        row_norm = wu.norm(dim=1).detach().cpu().numpy().astype(np.float32)
        state = build_match_state(cov, row_norm)

        observed = probe_balanced_accuracy_batched(
            wu,
            concepts,
            seed=args.seed,
            device=args.device,
            max_iter=args.max_iter,
        )
        meta_x = metadata_feature_matrix(cov, row_norm)
        metadata = probe_balanced_accuracy_batched(
            meta_x,
            concepts,
            seed=args.seed,
            device=args.device,
            max_iter=args.max_iter,
        )

        synthetic: dict[str, set[int]] = {}
        synth_meta: dict[str, tuple[str, str, int, dict[str, object]]] = {}
        for cname, toks in concepts.items():
            for perm_i in range(args.n_perm):
                for kind, disjoint in (
                    ("matched_random", True),
                    ("label_shuffle", False),
                ):
                    draw, q = sample_matched_set(toks, state, rng, disjoint=disjoint)
                    key = f"{cname}__{kind}__perm{perm_i:04d}"
                    if len(draw) >= args.min_probe_tokens:
                        synthetic[key] = draw
                    synth_meta[key] = (cname, kind, perm_i, q)
                    quality_rows.append(
                        {
                            "model": args.model,
                            "step": step,
                            "concept": cname,
                            "control": kind,
                            "perm": perm_i,
                            **q,
                        }
                    )

        null_res = run_probe_chunks(
            wu,
            synthetic,
            seed=args.seed + 10_000 + step,
            device=args.device,
            max_iter=args.max_iter,
            chunk_size=args.chunk_size,
        )
        by_control: dict[tuple[str, str], list[float]] = {
            (cname, "matched_random"): [] for cname in concepts
        }
        by_control.update({(cname, "label_shuffle"): [] for cname in concepts})
        for key, vals in null_res.items():
            cname, kind, perm_i, q = synth_meta[key]
            acc = float(vals["balanced_accuracy"])
            by_control[(cname, kind)].append(acc)
            null_rows.append(
                {
                    "model": args.model,
                    "step": step,
                    "concept": cname,
                    "control": kind,
                    "perm": perm_i,
                    "balanced_accuracy": round(acc, 6),
                    "best_C": vals["best_C"],
                    "n_positive": vals["n_positive"],
                    "exact_frac": round(float(q["exact_frac"]), 6),
                    "relax_mean": round(float(q["relax_mean"]), 6),
                    "failed": q["failed"],
                    "overlap_with_source": q["overlap_with_source"],
                }
            )

        for cname in concepts:
            obs_acc = float(observed[cname]["balanced_accuracy"])
            meta_acc = float(metadata[cname]["balanced_accuracy"])
            rand = summarize_null(by_control[(cname, "matched_random")], obs_acc)
            shuf = summarize_null(by_control[(cname, "label_shuffle")], obs_acc)
            summary_rows.append(
                {
                    "model": args.model,
                    "step": step,
                    "concept": cname,
                    "pos": cname.split(".", 1)[0],
                    "n_positive": observed[cname]["n_positive"],
                    "observed_acc": round(obs_acc, 6),
                    "metadata_acc": round(meta_acc, 6),
                    "observed_minus_metadata": round(obs_acc - meta_acc, 6),
                    "matched_random_n": rand["n"],
                    "matched_random_mean": rand["mean"],
                    "matched_random_p95": rand["p95"],
                    "matched_random_max": rand["max"],
                    "matched_random_p_ge_observed": rand["p_ge_observed"],
                    "label_shuffle_n": shuf["n"],
                    "label_shuffle_mean": shuf["mean"],
                    "label_shuffle_p95": shuf["p95"],
                    "label_shuffle_max": shuf["max"],
                    "label_shuffle_p_ge_observed": shuf["p_ge_observed"],
                }
            )
        del wu, meta_x

    write_csv(out_dir / "wordnet_matched_control_summary_160m.csv", summary_rows)
    write_csv(out_dir / "wordnet_matched_control_null_samples_160m.csv", null_rows)
    write_csv(out_dir / "wordnet_matched_control_match_quality_160m.csv", quality_rows)
    write_csv(out_dir / "wordnet_matched_control_audit_160m.csv", audit_rows)
    (out_dir / "wordnet_matched_control_metadata_160m.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "model_name": model_name,
                "steps": steps,
                "n_probed": len(concepts),
                "n_perm": args.n_perm,
                "preprocess": args.preprocess,
                "eval_tokens": str(eval_tokens),
                "frequency_source": "eval-corpus counts, not training-corpus frequency",
                "controls": ["metadata_only", "matched_random", "label_shuffle"],
                "seed": args.seed,
                "device": args.device,
                "max_iter": args.max_iter,
                "git_commit": git_commit(),
                "elapsed_s": round(time.time() - start, 2),
            },
            indent=2,
        )
    )
    print(f"[done] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pythia-160m", choices=["pythia-160m"])
    parser.add_argument(
        "--out-dir",
        default="experiments/probes/concept_evolution_validation/derived/wordnet_matched_controls_160m",
    )
    parser.add_argument(
        "--steps", default=None, help="Comma-separated subset of 32 schedule steps."
    )
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--n-perm", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--min-word-len", type=int, default=3)
    parser.add_argument("--min-token-chars", type=int, default=3)
    parser.add_argument("--dominance-threshold", type=float, default=0.6)
    parser.add_argument("--min-probe-tokens", type=int, default=20)
    parser.add_argument(
        "--preprocess", choices=["none", "center", "center_scale"], default="center"
    )
    parser.add_argument("--eval-tokens", type=Path, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
