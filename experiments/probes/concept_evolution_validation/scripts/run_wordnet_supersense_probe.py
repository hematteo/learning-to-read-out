"""Run W_U probes for the complete WordNet supersense inventory.

This is the paper-facing replacement for the older hand-selected WordNet subset:
all WordNet lexicographer files are built first, then categories are either
probed or reported as skipped based only on token support.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
import torch
from transformers import AutoTokenizer

from readout.core.data import write_csv
from readout.core.model_specs import DEFAULT_STEPS_BY_MODEL, MODEL_HF_NAMES
from readout.core.paths import repo_root, snapshot_path
from readout.core.repro import git_commit
from readout.crosscoder.snapshots import load_snapshot
from readout.probes.wu_probes_gpu import probe_balanced_accuracy_batched

REPO = repo_root()

try:
    from nltk.corpus import wordnet as wn
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script needs NLTK WordNet. Run with: "
        "uv run --with nltk python experiments/probes/concept_evolution_validation/"
        "scripts/run_wordnet_supersense_probe.py"
    ) from exc


LEXNAME_ORDER = [
    "adj.all",
    "adj.pert",
    "adj.ppl",
    "adv.all",
    "noun.Tops",
    "noun.act",
    "noun.animal",
    "noun.artifact",
    "noun.attribute",
    "noun.body",
    "noun.cognition",
    "noun.communication",
    "noun.event",
    "noun.feeling",
    "noun.food",
    "noun.group",
    "noun.location",
    "noun.motive",
    "noun.object",
    "noun.person",
    "noun.phenomenon",
    "noun.plant",
    "noun.possession",
    "noun.process",
    "noun.quantity",
    "noun.relation",
    "noun.shape",
    "noun.state",
    "noun.substance",
    "noun.time",
    "verb.body",
    "verb.change",
    "verb.cognition",
    "verb.communication",
    "verb.competition",
    "verb.consumption",
    "verb.contact",
    "verb.creation",
    "verb.emotion",
    "verb.motion",
    "verb.perception",
    "verb.possession",
    "verb.social",
    "verb.stative",
    "verb.weather",
]

# Filename suffix per model. Used to keep the 160M outputs unchanged while
# letting the 1B run write to its own files.
MODEL_SUFFIX = {
    "pythia-160m": "160m",
    "pythia-1b": "1b",
}


def model_suffix(model: str) -> str:
    if model not in MODEL_SUFFIX:
        raise KeyError(f"Add {model!r} to MODEL_SUFFIX")
    return MODEL_SUFFIX[model]


def atomic_write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def wordnet_lexnames() -> list[str]:
    # NLTK exposes the canonical lexname list on the corpus reader.
    names = list(getattr(wn, "_lexnames"))
    unknown = sorted(set(names) - set(LEXNAME_ORDER))
    missing = sorted(set(LEXNAME_ORDER) - set(names))
    if unknown or missing:
        raise RuntimeError(
            f"Unexpected WordNet lexnames unknown={unknown} missing={missing}"
        )
    return list(LEXNAME_ORDER)


def clean_lemma(name: str, *, min_word_len: int) -> str | None:
    word = name.replace("_", " ").lower()
    if " " in word:
        return None
    if len(word) < min_word_len:
        return None
    if not word.isalpha():
        return None
    return word


def dominant_lexname_by_word(
    min_word_len: int, dominance_threshold: float
) -> dict[str, str]:
    per_word: dict[str, Counter[str]] = {}
    for synset in wn.all_synsets():
        lexname = synset.lexname()
        for lemma in synset.lemmas():
            word = clean_lemma(lemma.name(), min_word_len=min_word_len)
            if word is None:
                continue
            per_word.setdefault(word, Counter())[lexname] += 1

    out: dict[str, str] = {}
    for word, counts in per_word.items():
        lexname, count = counts.most_common(1)[0]
        if count / sum(counts.values()) >= dominance_threshold:
            out[word] = lexname
    return out


def token_variants(word: str) -> list[str]:
    return [word, " " + word, word.capitalize(), " " + word.capitalize(), word.upper()]


def token_is_clean(tokenizer, token_id: int, *, min_token_chars: int) -> bool:
    decoded = tokenizer.decode([token_id])
    stripped = decoded.strip()
    if len(stripped) < min_token_chars:
        return False
    if not stripped.isascii():
        return False
    if not stripped.isalpha():
        return False
    # Prefer whole-token forms. Bare all-lowercase fragments such as "ing" can
    # pass BPE as single tokens, but without a leading-space variant they are
    # not clean word rows for WordNet probing.
    if decoded.startswith(" "):
        return True
    return stripped[0].isupper() or stripped.isupper()


def map_words_to_token_ids(
    tokenizer, words: Iterable[str], *, min_token_chars: int
) -> set[int]:
    token_ids: set[int] = set()
    for word in words:
        for variant in token_variants(word):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1 and token_is_clean(
                tokenizer, ids[0], min_token_chars=min_token_chars
            ):
                token_ids.add(int(ids[0]))
    return token_ids


def build_wordnet_concepts(
    tokenizer,
    *,
    min_word_len: int,
    min_token_chars: int,
    dominance_threshold: float,
) -> tuple[dict[str, set[int]], list[dict[str, object]]]:
    lexnames = wordnet_lexnames()
    word_to_lexname = dominant_lexname_by_word(min_word_len, dominance_threshold)
    words_by_lexname: dict[str, list[str]] = {name: [] for name in lexnames}
    for word, lexname in word_to_lexname.items():
        words_by_lexname[lexname].append(word)

    concepts: dict[str, set[int]] = {}
    audit_rows: list[dict[str, object]] = []
    for lexname in lexnames:
        words = sorted(words_by_lexname[lexname])
        token_ids = map_words_to_token_ids(
            tokenizer, words, min_token_chars=min_token_chars
        )
        decoded = [tokenizer.decode([tid]) for tid in sorted(token_ids)[:12]]
        concepts[lexname] = token_ids
        audit_rows.append(
            {
                "concept": lexname,
                "pos": lexname.split(".", 1)[0],
                "n_dominant_lemmas": len(words),
                "n_token_ids": len(token_ids),
                "sample_tokens": " | ".join(repr(s) for s in decoded),
            }
        )
    return concepts, audit_rows


def centered_rows(wu: torch.Tensor, mode: str) -> torch.Tensor:
    wu = wu.float()
    if mode == "none":
        return wu
    centered = wu - wu.mean(dim=0, keepdim=True)
    if mode == "center":
        return centered
    if mode == "center_scale":
        scale = torch.sqrt(centered.pow(2).sum(dim=1).mean()).clamp(min=1e-8)
        return centered / scale
    raise ValueError(f"unknown preprocess mode {mode!r}")


def sustained_emergence(
    trace: dict[int, float], *, min_abs: float, min_gain: float, persist: int
) -> int | None:
    steps = sorted(trace)
    if not steps:
        return None
    threshold = max(min_abs, trace[steps[0]] + min_gain)
    for i, step in enumerate(steps):
        window = steps[i : i + persist]
        if len(window) < min(2, persist):
            continue
        if all(trace[s] >= threshold for s in window):
            return step
    return None


def max_adjacent_gain(trace: dict[int, float]) -> tuple[int | None, int | None, float]:
    steps = sorted(trace)
    if len(steps) < 2:
        return None, None, float("nan")
    best = (steps[0], steps[1], trace[steps[1]] - trace[steps[0]])
    for a, b in zip(steps[:-1], steps[1:]):
        gain = trace[b] - trace[a]
        if gain > best[2]:
            best = (a, b, gain)
    return best


def run(args: argparse.Namespace) -> None:
    root = repo_root()
    suffix = model_suffix(args.model)
    out_rel = args.out_dir or (
        f"experiments/probes/concept_evolution_validation/derived/wordnet_supersense_{suffix}"
    )
    out_dir = root / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODEL_HF_NAMES[args.model]
    steps = list(DEFAULT_STEPS_BY_MODEL[args.model])
    if args.limit_steps:
        steps = steps[: args.limit_steps]

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
    skipped = sorted(set(concepts_all) - set(concepts))

    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(step: int) -> Path:
        return cache_dir / f"step{step}.json"

    cached_steps = [s for s in steps if cache_path(s).exists()]
    pending_steps = [s for s in steps if not cache_path(s).exists()]
    print(
        f"[resume] {len(cached_steps)}/{len(steps)} steps already cached; "
        f"{len(pending_steps)} pending. First pending: "
        f"{pending_steps[0] if pending_steps else 'none'}",
        flush=True,
    )

    start = time.time()
    trajectories: dict[str, dict[int, dict[str, float]]] = {
        name: {} for name in concepts
    }
    for i, step in enumerate(steps, start=1):
        cp = cache_path(step)
        if cp.exists():
            cached = json.loads(cp.read_text())
            for concept, vals in cached.items():
                if concept in concepts:
                    trajectories[concept][step] = vals
            print(
                f"[{i:02d}/{len(steps):02d}] step {step} cached, skipping", flush=True
            )
            continue
        print(f"[{i:02d}/{len(steps):02d}] loading/probing step {step}", flush=True)
        missing = snapshot_path(model_name, step)
        if not missing.exists():
            raise FileNotFoundError(missing)
        wu = centered_rows(load_snapshot(model_name, step), args.preprocess)
        res = probe_balanced_accuracy_batched(
            wu,
            concepts,
            seed=args.seed,
            device=args.device,
            max_iter=args.max_iter,
        )
        for concept, vals in res.items():
            trajectories[concept][step] = vals
        atomic_write_json(cp, {c: v for c, v in res.items()})
        del wu

    trajectory_rows: list[dict[str, object]] = []
    for concept in LEXNAME_ORDER:
        if concept not in concepts:
            continue
        for step in steps:
            vals = trajectories[concept][step]
            trajectory_rows.append(
                {
                    "model": args.model,
                    "concept": concept,
                    "pos": concept.split(".", 1)[0],
                    "step": step,
                    "balanced_accuracy": round(float(vals["balanced_accuracy"]), 6),
                    "best_C": vals["best_C"],
                    "n_positive": vals["n_positive"],
                }
            )

    by_support = {r["concept"]: r for r in audit_rows}
    summary_rows: list[dict[str, object]] = []
    for concept in LEXNAME_ORDER:
        support = by_support[concept]
        if concept in concepts:
            trace = {
                step: float(trajectories[concept][step]["balanced_accuracy"])
                for step in steps
                if np.isfinite(float(trajectories[concept][step]["balanced_accuracy"]))
            }
            emergence = sustained_emergence(
                trace,
                min_abs=args.emergence_min_abs,
                min_gain=args.emergence_min_gain,
                persist=args.emergence_persist,
            )
            a, b, gain = max_adjacent_gain(trace)
            first_step = min(trace)
            final_step = max(trace)
            summary_rows.append(
                {
                    "model": args.model,
                    "concept": concept,
                    "pos": concept.split(".", 1)[0],
                    "n_dominant_lemmas": support["n_dominant_lemmas"],
                    "n_token_ids": support["n_token_ids"],
                    "probed": 1,
                    "emergence_step": emergence if emergence is not None else "",
                    "max_gain_pair": f"{a}->{b}" if a is not None else "",
                    "max_gain": round(gain, 6) if np.isfinite(gain) else "",
                    "step0_acc": round(trace[first_step], 6),
                    "final_acc": round(trace[final_step], 6),
                    "total_gain": round(trace[final_step] - trace[first_step], 6),
                }
            )
        else:
            summary_rows.append(
                {
                    "model": args.model,
                    "concept": concept,
                    "pos": concept.split(".", 1)[0],
                    "n_dominant_lemmas": support["n_dominant_lemmas"],
                    "n_token_ids": support["n_token_ids"],
                    "probed": 0,
                    "emergence_step": "",
                    "max_gain_pair": "",
                    "max_gain": "",
                    "step0_acc": "",
                    "final_acc": "",
                    "total_gain": "",
                }
            )

    pos_rows: list[dict[str, object]] = []
    for pos in ["noun", "verb", "adj", "adv"]:
        rows = [r for r in summary_rows if r["pos"] == pos and r["probed"] == 1]
        emerged = [int(r["emergence_step"]) for r in rows if r["emergence_step"] != ""]
        pos_rows.append(
            {
                "model": args.model,
                "pos": pos,
                "n_concepts": len(rows),
                "n_emerged": len(emerged),
                "median_emergence_step": int(median(emerged)) if emerged else "",
                "frac_max_gain_256_1000": round(
                    sum(
                        1
                        for r in rows
                        if isinstance(r["max_gain_pair"], str)
                        and r["max_gain_pair"]
                        in {"128->256", "256->512", "512->1000", "1000->2000"}
                    )
                    / len(rows),
                    3,
                )
                if rows
                else "",
                "median_final_acc": round(
                    median(float(r["final_acc"]) for r in rows), 6
                )
                if rows
                else "",
            }
        )

    write_csv(out_dir / f"wordnet_supersense_audit_{suffix}.csv", audit_rows)
    write_csv(
        out_dir / f"wordnet_supersense_probe_trajectory_{suffix}.csv", trajectory_rows
    )
    write_csv(out_dir / f"wordnet_supersense_probe_summary_{suffix}.csv", summary_rows)
    write_csv(out_dir / f"wordnet_supersense_probe_pos_summary_{suffix}.csv", pos_rows)

    serializable_trajectories = {
        concept: {str(step): vals for step, vals in sorted(trace.items())}
        for concept, trace in trajectories.items()
    }
    atomic_write_json(
        out_dir / f"wordnet_supersense_probe_trajectory_{suffix}.json",
        serializable_trajectories,
    )
    atomic_write_json(
        out_dir / f"wordnet_supersense_probe_metadata_{suffix}.json",
        {
            "model": args.model,
            "model_name": model_name,
            "steps": steps,
            "n_all_wordnet_lexnames": len(LEXNAME_ORDER),
            "n_probed": len(concepts),
            "skipped_for_support": skipped,
            "min_probe_tokens": args.min_probe_tokens,
            "min_word_len": args.min_word_len,
            "min_token_chars": args.min_token_chars,
            "dominance_threshold": args.dominance_threshold,
            "preprocess": args.preprocess,
            "seed": args.seed,
            "device": args.device,
            "max_iter": args.max_iter,
            "git_commit": git_commit(),
            "elapsed_s": round(time.time() - start, 2),
        },
    )

    print(f"[done] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="pythia-160m", choices=["pythia-160m", "pythia-1b"]
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output dir; defaults to derived/wordnet_supersense_<suffix>/ keyed on --model",
    )
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
    parser.add_argument("--emergence-min-abs", type=float, default=0.55)
    parser.add_argument("--emergence-min-gain", type=float, default=0.05)
    parser.add_argument("--emergence-persist", type=int, default=3)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
