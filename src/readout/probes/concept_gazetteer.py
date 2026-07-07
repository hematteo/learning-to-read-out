"""Concept gazetteer: build pre-registered token-ID sets per concept for SAE anchoring.

Produces 37 concepts across four families:
- WordNet supersenses (20 concepts)
- Morphological patterns (7 concepts; 3 dropped after v1 audit)
- Syntactic / closed-class (5 concepts)
- Register / domain (5 concepts)

For each concept we map to Pythia token IDs via single-token BPE encoding
with leading-space and capitalisation variants. Concepts are *not* disjoint
by construction; overlap is audited in the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from nltk.corpus import wordnet as wn

# ── WordNet supersense selection ────────────────────────────────────────────

WORDNET_SUPERSENSES: list[str] = [
    "noun.animal",
    "noun.artifact",
    "noun.body",
    "noun.cognition",
    "noun.communication",
    "noun.food",
    "noun.group",
    "noun.location",
    "noun.person",
    "noun.plant",
    "noun.quantity",
    "noun.state",
    "noun.substance",
    "noun.time",
    "verb.change",
    "verb.communication",
    "verb.contact",
    "verb.motion",
    "verb.possession",
    "verb.social",
]


def _primary_supersense_map(min_word_len: int = 4) -> dict[str, str]:
    """Map word → its dominant (most frequent) supersense across all WordNet
    synsets, computed ONCE. Words are kept only if they unambiguously belong
    to one supersense >= 60% of the time.

    Filters:
      - alphabetic only, length >= min_word_len
      - must have a single dominant supersense (≥60% of synset occurrences)
    """
    from collections import Counter

    per_word_ss: dict[str, Counter] = {}
    for syn in wn.all_synsets():
        ss = syn.lexname()
        for lemma in syn.lemmas():
            form = lemma.name().replace("_", " ")
            if " " in form or not form.isalpha():
                continue
            w = form.lower()
            if len(w) < min_word_len:
                continue
            per_word_ss.setdefault(w, Counter())[ss] += 1

    dominant: dict[str, str] = {}
    for w, ctr in per_word_ss.items():
        top_ss, top_count = ctr.most_common(1)[0]
        total = sum(ctr.values())
        if top_count / total >= 0.6:
            dominant[w] = top_ss
    return dominant


# Cached lazily on first call
_DOMINANT_CACHE: dict[str, str] | None = None


def _wordnet_words(supersense: str, min_word_len: int = 4) -> set[str]:
    """Words whose DOMINANT WordNet supersense is `supersense`.

    Filters applied:
      - alphabetic only, length >= min_word_len
      - word's primary supersense (≥60% of its synset occurrences) is the target
    """
    global _DOMINANT_CACHE
    if _DOMINANT_CACHE is None:
        _DOMINANT_CACHE = _primary_supersense_map(min_word_len=min_word_len)
    return {w for w, ss in _DOMINANT_CACHE.items() if ss == supersense}


# ── Morphological patterns ──────────────────────────────────────────────────

# Each is a predicate on the DECODED token (with leading space stripped).
MORPHOLOGICAL_PATTERNS: dict[str, Callable[[str], bool]] = {
    # Plural nouns: end in 's', not 'ss', length >= 4, alphabetic
    "plural_nouns": lambda w: (
        len(w) >= 4
        and w.isalpha()
        and w.endswith("s")
        and not w.endswith("ss")
        and not w.endswith("is")
        and not w.endswith("us")
    ),
    # Past tense / past participle: end in 'ed', length >= 5
    "past_tense_ed": lambda w: len(w) >= 5 and w.isalpha() and w.endswith("ed"),
    # Present participle / gerund: end in 'ing', length >= 5
    "gerund_ing": lambda w: len(w) >= 5 and w.isalpha() and w.endswith("ing"),
    # Abstract nominalisations: -tion
    "nominalisation_tion": lambda w: len(w) >= 6 and w.isalpha() and w.endswith("tion"),
    # Abstract nominalisations: -ness
    "nominalisation_ness": lambda w: len(w) >= 6 and w.isalpha() and w.endswith("ness"),
    # Adverbs: -ly
    "adverb_ly": lambda w: len(w) >= 5 and w.isalpha() and w.endswith("ly"),
    # Capability / quality: -ity
    "abstract_ity": lambda w: len(w) >= 6 and w.isalpha() and w.endswith("ity"),
    # Note: comparative_er, superlative_est, agent_er were dropped after v1 audit —
    # the -er/-est/-or suffix heuristic mis-classifies too many non-comparative words
    # (other, after, inter), and the three patterns self-overlap >72%.
}


# ── Syntactic / closed-class word lists ─────────────────────────────────────

SYNTACTIC_LISTS: dict[str, list[str]] = {
    "articles": ["the", "a", "an"],
    "pronouns": [
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "this",
        "that",
        "these",
        "those",
        "who",
        "whom",
        "which",
        "what",
        "whose",
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "yourselves",
        "themselves",
    ],
    "conjunctions": [
        "and",
        "or",
        "but",
        "yet",
        "so",
        "nor",
        "for",
        "because",
        "although",
        "though",
        "while",
        "whereas",
        "if",
        "unless",
        "until",
        "since",
        "when",
        "whenever",
        "where",
        "wherever",
        "as",
        "than",
    ],
    "prepositions": [
        "in",
        "on",
        "at",
        "to",
        "from",
        "with",
        "without",
        "by",
        "for",
        "of",
        "about",
        "into",
        "onto",
        "upon",
        "over",
        "under",
        "above",
        "below",
        "between",
        "among",
        "through",
        "across",
        "around",
        "behind",
        "beside",
        "during",
        "before",
        "after",
        "since",
        "until",
        "against",
        "toward",
        "towards",
        "near",
        "beyond",
    ],
    "auxiliaries": [
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "done",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "must",
        "ought",
    ],
}


# ── Register / domain lists ─────────────────────────────────────────────────

REGISTER_LISTS: dict[str, list[str]] = {
    "numeric_words": [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "tenth",
        "twice",
        "double",
        "triple",
    ],
    "code_keywords_python": [
        "def",
        "return",
        "import",
        "class",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "break",
        "continue",
        "pass",
        "try",
        "except",
        "finally",
        "raise",
        "with",
        "as",
        "yield",
        "lambda",
        "global",
        "nonlocal",
        "from",
        "True",
        "False",
        "None",
        "and",
        "or",
        "not",
        "in",
        "is",
        "print",
        "self",
    ],
}

# Digit tokens and punctuation are computed directly from the tokenizer vocab.


# ── Token ID mapping ────────────────────────────────────────────────────────


def _single_token_ids(tokenizer, word: str) -> list[int]:
    """Return token IDs for all surface variants of `word` that encode to a
    single token. Variants: bare, leading-space, leading-space + capitalised,
    capitalised, upper-case."""
    variants = [
        word,
        " " + word,
        " " + word.capitalize(),
        word.capitalize(),
        word.upper(),
    ]
    ids: set[int] = set()
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) == 1:
            ids.add(toks[0])
    return sorted(ids)


def _map_word_set_to_tokens(tokenizer, words: set[str]) -> set[int]:
    token_ids: set[int] = set()
    for w in words:
        token_ids.update(_single_token_ids(tokenizer, w))
    return token_ids


def _digit_tokens(tokenizer) -> set[int]:
    """Tokens whose decoded form is purely digit characters."""
    out: set[int] = set()
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid]).strip()
        if s and s.isdigit():
            out.add(tid)
    return out


def _punct_tokens(tokenizer) -> set[int]:
    """Tokens whose decoded form consists entirely of ASCII punctuation."""
    out: set[int] = set()
    punct = set(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid]).strip()
        if not s:
            continue
        if all(c in punct for c in s):
            out.add(tid)
    return out


def _non_latin_tokens(tokenizer) -> set[int]:
    """Tokens whose decoded form contains non-ASCII alphabetic characters."""
    out: set[int] = set()
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid])
        if any(ord(c) > 127 and c.isalpha() for c in s):
            out.add(tid)
    return out


def _morphological_tokens(tokenizer, predicate: Callable[[str], bool]) -> set[int]:
    """Tokens matching a morphological predicate on their stripped lowercased form."""
    out: set[int] = set()
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid])
        # Skip tokens that are empty, pure punctuation, or contain non-ASCII
        stripped = s.strip()
        if not stripped or not stripped.isascii():
            continue
        if predicate(stripped.lower()):
            out.add(tid)
    return out


# ── Main builder ────────────────────────────────────────────────────────────


def build_gazetteer(tokenizer) -> dict[str, list[int]]:
    """Build the full 37-concept gazetteer as concept_name → sorted token IDs."""
    gaz: dict[str, list[int]] = {}

    # A. WordNet supersenses (20)
    for ss in WORDNET_SUPERSENSES:
        words = _wordnet_words(ss)
        tids = _map_word_set_to_tokens(tokenizer, words)
        gaz[ss] = sorted(tids)

    # B. Morphological patterns (7)
    for name, pred in MORPHOLOGICAL_PATTERNS.items():
        tids = _morphological_tokens(tokenizer, pred)
        gaz[name] = sorted(tids)

    # C. Syntactic / closed-class (5)
    for name, wlist in SYNTACTIC_LISTS.items():
        tids = _map_word_set_to_tokens(tokenizer, set(wlist))
        gaz[name] = sorted(tids)

    # D. Register / domain (5)
    for name, wlist in REGISTER_LISTS.items():
        tids = _map_word_set_to_tokens(tokenizer, set(wlist))
        gaz[name] = sorted(tids)
    gaz["digit_tokens"] = sorted(_digit_tokens(tokenizer))
    gaz["punctuation"] = sorted(_punct_tokens(tokenizer))
    gaz["non_latin"] = sorted(_non_latin_tokens(tokenizer))

    return gaz


# ── Audit ───────────────────────────────────────────────────────────────────


def audit_gazetteer(
    gaz: dict[str, list[int]],
    tokenizer,
    min_tokens: int = 20,
) -> dict[str, dict]:
    """Per-concept audit: size, top-5 sample tokens, overlap with other concepts."""
    report: dict[str, dict] = {}
    all_ids = set()
    for ids in gaz.values():
        all_ids.update(ids)

    for concept, ids in gaz.items():
        sample = [tokenizer.decode([tid]) for tid in ids[:5]]
        # Overlap with other concepts
        overlaps = []
        for other, other_ids in gaz.items():
            if other == concept:
                continue
            shared = len(set(ids) & set(other_ids))
            if shared > 0 and shared / max(len(ids), 1) > 0.3:
                overlaps.append((other, shared, round(shared / len(ids), 2)))
        report[concept] = {
            "n_tokens": len(ids),
            "sample_tokens": sample,
            "heavy_overlaps": overlaps,  # concepts sharing >30% of this concept's tokens
            "pass_min_size": len(ids) >= min_tokens,
        }
    report["_meta"] = {
        "n_concepts": len(gaz),
        "total_unique_tokens": len(all_ids),
        "vocab_size": tokenizer.vocab_size,
        "coverage_fraction": round(len(all_ids) / tokenizer.vocab_size, 3),
    }
    return report


def save_gazetteer(
    gaz: dict[str, list[int]],
    report: dict[str, dict],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "concepts_v1.json").open("w") as f:
        json.dump(gaz, f, indent=2)
    with (out_dir / "concepts_v1_audit.json").open("w") as f:
        json.dump(report, f, indent=2)
