"""Tokenize a multilingual eval corpus with the OLMo-2 tokenizer.

OLMo-2 counterpart of the legacy Pythia eval-corpus builder: same nine
languages, but tokenized with the OLMo-2 tokenizer so target IDs are valid for
OLMo's vocabulary.

Strategy: stream Wikipedia (~20K chars/lang -> ~2K tokens/lang per language);
optionally supplement missing languages from a pre-built raw corpus dict
(--fallback-corpus, a torch-saved dict[lang, str]). The Pythia counterpart
produces ~174K tokens; aim for the same order of magnitude so per-concept
counts are comparable across models.

Output schema matches the Pythia eval_tokens.pt:
    ids               (N,) long
    scripts           list[str] length N
    languages         list[str] length N
    per_lang_counts   dict[str, int]
    script_counts     dict[str, int]

Usage:
    uv run python scripts/extract/build_eval_corpus_olmo.py
    uv run python scripts/extract/build_eval_corpus_olmo.py \\
        --offline --fallback-corpus path/to/corpus.pt   # no network
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    REPO
    / "results/experiments/causal/temporal_localization_patching/eval_tokens_olmo.pt"
)
MODEL = "allenai/OLMo-2-1124-7B"
TARGET_CHARS_PER_LANG = 20_000
LANGUAGES = ["en", "ru", "zh", "ja", "th", "ar", "hi", "ko", "bn"]


def script_of(token_text: str) -> str:
    if not token_text:
        return "empty"
    s = token_text.lstrip(" ")
    if not s:
        return "space"
    cp = ord(s[0])
    if cp < 128:
        return "ASCII"
    if 0x0400 <= cp <= 0x04FF:
        return "Cyrillic"
    if 0x0E00 <= cp <= 0x0E7F:
        return "Thai"
    if 0x0600 <= cp <= 0x06FF:
        return "Arabic"
    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    if 0x0980 <= cp <= 0x09FF:
        return "Bengali"
    if 0x4E00 <= cp <= 0x9FFF:
        return "CJK"
    if 0x3040 <= cp <= 0x30FF:
        return "Japanese"
    if 0xAC00 <= cp <= 0xD7AF:
        return "Korean"
    return "other_unicode"


def try_wikipedia_streaming(lang: str, target_chars: int) -> str | None:
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
        print(f"  [{lang}] wiki stream failed: {type(e).__name__}: {e}")
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
        print(f"  [{lang}] wiki iter failed: {type(e).__name__}: {e}")
        if not chunks:
            return None
    if not chunks:
        return None
    print(f"  [{lang}] wiki: {n_articles} articles, {total:,} chars")
    return ("\n\n".join(chunks))[:target_chars]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"where to write eval_tokens_olmo.pt (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--fallback-corpus",
        type=Path,
        default=None,
        help="optional torch-saved dict[lang, str] used to supplement languages "
        "Wikipedia streaming misses",
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

    # Supplement missing/short languages from the optional fallback corpus so
    # downstream per-language counts stay aligned.
    if args.fallback_corpus is not None:
        if not args.fallback_corpus.exists():
            ap.error(f"--fallback-corpus not found: {args.fallback_corpus}")
        fallback = torch.load(args.fallback_corpus, weights_only=False)
        for lang in LANGUAGES:
            if (lang not in corpus or len(corpus[lang]) < 500) and lang in fallback:
                print(f"  supplementing {lang} from {args.fallback_corpus.name}")
                corpus[lang] = fallback[lang]

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
            f"({', '.join(missing)}); per-language counts will not be comparable "
            "to runs built with the full language set."
        )

    print(f"\nFinal corpus: {len(corpus)} languages")
    for lang, text in corpus.items():
        print(f"  {lang}: {len(text):,} chars")

    print(f"\nTokenizing with {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)

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
    print(f"\nSaved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
