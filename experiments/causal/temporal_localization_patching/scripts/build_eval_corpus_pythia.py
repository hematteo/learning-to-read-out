"""Build the multilingual held-out eval corpus with the Pythia tokenizer.

This is the builder of the released
``evaluation/eval-corpus/eval_tokens.pt`` (the default ``--eval-tokens`` of
the swap grids): nine 20,000-character slices streamed from the
``wikimedia/wikipedia`` 2023-11-01 dumps (en, ru, zh, ja, th, ar, hi, ko, bn),
concatenated in that order, tokenized with the Pythia BPE (shared by every
Pythia size), and tagged per token with its script and language.
``scripts/extract/build_eval_corpus_olmo.py`` is the OLMo-2 counterpart.

Output schema:
    ids               (N,) long
    scripts           list[str] length N
    languages         list[str] length N
    per_lang_counts   dict[str, int]
    script_counts     dict[str, int]

The raw text (dict[lang, str]) is saved to ``--text-output`` and can be passed
back as ``--fallback-corpus`` to re-tokenize without network access.

Usage:
    uv run python experiments/causal/temporal_localization_patching/scripts/build_eval_corpus_pythia.py
    uv run python experiments/causal/temporal_localization_patching/scripts/build_eval_corpus_pythia.py \\
        --offline --fallback-corpus path/to/eval_corpus_text_pythia.pt   # no network
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from readout.core.paths import repo_root
from readout.probes.token_scripts import script_of

REPO = repo_root()
DEFAULT_OUT = REPO / "results/experiments/causal/temporal_localization_patching/eval_tokens_pythia.pt"
DEFAULT_TEXT_OUT = REPO / "results/experiments/causal/temporal_localization_patching/eval_corpus_text_pythia.pt"
TOKENIZER = "EleutherAI/pythia-160m"
TARGET_CHARS_PER_LANG = 20_000
LANGUAGES = ["en", "ru", "zh", "ja", "th", "ar", "hi", "ko", "bn"]


def try_wikipedia_streaming(lang: str, target_chars: int) -> str | None:
    """First articles (>=200 chars, up to 50) of the 2023-11-01 dump, cut to ``target_chars``."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{lang}",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"  [{lang}] wikipedia streaming failed: {type(e).__name__}: {e}")
        return None
    chunks: list[str] = []
    total = 0
    n_articles = 0
    try:
        for ex in ds:
            text = ex.get("text", "")
            if not text:
                continue
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) < 200:
                continue
            chunks.append(text)
            total += len(text)
            n_articles += 1
            if total >= target_chars or n_articles >= 50:
                break
    except Exception as e:
        print(f"  [{lang}] wikipedia stream iter failed: {type(e).__name__}: {e}")
        if not chunks:
            return None
    if not chunks:
        return None
    print(f"  [{lang}] wikipedia: {n_articles} articles, {total:,} chars")
    return ("\n\n".join(chunks))[:target_chars]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"where to write the token file (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--text-output",
        type=Path,
        default=DEFAULT_TEXT_OUT,
        help=f"where to write the raw dict[lang, str] (default: {DEFAULT_TEXT_OUT})",
    )
    ap.add_argument(
        "--fallback-corpus",
        type=Path,
        default=None,
        help="optional torch-saved dict[lang, str] used to supplement languages Wikipedia streaming misses",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="skip Wikipedia streaming entirely (requires --fallback-corpus)",
    )
    args = ap.parse_args()

    if args.offline and args.fallback_corpus is None:
        ap.error("--offline requires --fallback-corpus")

    corpus: dict[str, str] = {}
    if args.offline:
        print("--offline set; skipping wikipedia.")
    else:
        print("Trying wikipedia streaming per language...")
        for lang in LANGUAGES:
            text = try_wikipedia_streaming(lang, TARGET_CHARS_PER_LANG)
            if text:
                corpus[lang] = text

    if args.fallback_corpus is not None:
        if not args.fallback_corpus.exists():
            ap.error(f"--fallback-corpus not found: {args.fallback_corpus}")
        fallback = torch.load(args.fallback_corpus, weights_only=False)
        for lang in LANGUAGES:
            if (lang not in corpus or len(corpus[lang]) < 500) and lang in fallback:
                print(f"  supplementing {lang} from {args.fallback_corpus.name}")
                corpus[lang] = fallback[lang]

    # Concatenation order fixes token positions (and so the swap grids' windows).
    corpus = {lang: corpus[lang] for lang in LANGUAGES if lang in corpus}
    missing = [lang for lang in LANGUAGES if lang not in corpus]
    if not corpus:
        print(
            "ERROR: no language could be built. Check network access, or pass "
            "--fallback-corpus with a pre-built dict[lang, str]."
        )
        return 1
    if missing:
        print(
            f"\nWARNING: {len(missing)}/{len(LANGUAGES)} languages missing "
            f"({', '.join(missing)}); the corpus will not match the released eval_tokens.pt."
        )

    print(f"\nFinal corpus: {len(corpus)} languages")
    for lang, text in corpus.items():
        print(f"  {lang}: {len(text):,} chars")

    print(f"\nTokenizing with {TOKENIZER} ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    all_ids: list[int] = []
    all_scripts: list[str] = []
    all_lang: list[str] = []
    per_lang_counts: dict[str, int] = {}
    for lang, text in corpus.items():
        ids = tok.encode(text, add_special_tokens=False)
        for tid in ids:
            all_ids.append(tid)
            all_scripts.append(script_of(tok.decode([tid])))
            all_lang.append(lang)
        per_lang_counts[lang] = len(ids)

    print(f"\n  total tokens: {len(all_ids):,}")
    for lang, n in per_lang_counts.items():
        print(f"    {lang}: {n:,} tokens")

    script_counts = Counter(all_scripts)
    print("\n  Script distribution:")
    for s, n in script_counts.most_common():
        print(f"    {s:<15} {n:>7,}  ({100 * n / len(all_ids):.1f}%)")

    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corpus, args.text_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ids": torch.tensor(all_ids, dtype=torch.long),
            "scripts": all_scripts,
            "languages": all_lang,
            "per_lang_counts": per_lang_counts,
            "script_counts": dict(script_counts),
        },
        args.output,
    )
    print(f"\nSaved -> {args.output} (text -> {args.text_output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
