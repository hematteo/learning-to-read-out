"""Temporal readout-patch metrics for proto-token families.

This is the headline behavioral pipeline for the W_U lifecycle paper. It
replaces multilingual NLL with row/readout-local, proto-token-specific,
matched-confound metrics applied to a temporally-patched W_U.

==============================================================================
1. Snapshot-local activations and the temporal-patch operation
==============================================================================
The cross-snapshot crosscoder has snapshot-local encoder/decoder pieces
(W_E[t], b_E[t], W_D[t], scale[t], mean[t]) and a SHARED log-jumprelu
threshold. Snapshot-local activations are computed by re-encoding the
snapshot's W_U through that snapshot's pieces:

    a_k(t)[v]  =  encode_snapshot_local(W_U_t, pieces_t)[v, k]
                =  jumprelu( (W_U_t[v] - mean_t)/scale_t @ W_E[t][:,k] + b_E[t,k] )

The rank-1 readout contribution of feature k at snapshot t is therefore:

    R_k(t)[v, :]  =  a_k(t)[v] * D_k(t) * scale_t      # (V, d_model)

where D_k(t) = W_D[t, k] (one decoder vector per snapshot). Per-snapshot
preprocessing scale enters because the crosscoder reconstructs the
*standardized* W_U; we map back to model-space units.

The temporal patch over a feature subset S, snapshots (base, target):

    delta_S         =  sum_{k in S} [ R_k(target) - R_k(base) ]
    W_U_patch_in    =  W_U_base    + delta_S
    W_U_patch_out   =  W_U_target  - delta_S

`patch_in` is the "necessity-of-the-rest, sufficiency-of-S" framing:
"early-checkpoint W_U with only S patched up to its target form."
`patch_out` is the "necessity-of-S" framing: "late-checkpoint W_U
with only S rolled back to its base form." A clean lifecycle claim
about S predicts both: M(patch_in) > M(base) and M(patch_out) < M(target).

==============================================================================
2. Headline metric stack (revised 2026-05-01)
==============================================================================
Headline (main text):
  M1. Concept-mass logit ratio (primary).
  M2. Matched MRR / rank within per-target candidate set.
  M5. Row-geometry recovery (mechanism-facing).

Appendix:
  M4. Probe-logit transfer.
  M6. Candidate-set logit-lens KL.

For each (transition, subset, concept), every metric reports BOTH:

    raw_delta_in  = M(W_patch_in)  - M(W_base)        # forward effect
    raw_delta_out = M(W_target)    - M(W_patch_out)   # backward effect
    recovery_in   = raw_delta_in  / (M(target) - M(base))
    recovery_out  = raw_delta_out / (M(target) - M(base))

Recovery values are FLAGGED as `low_denom` when |M(target)-M(base)| < eps_M
(per-metric floor; default 0.01 nats for log-space metrics, 0.005 for cosine).
The PRIMARY statistical test is the paired raw effect of `S = real subset`
versus `S = matched_random subset`, not the recovery ratio itself.

==============================================================================
3. Family selectivity (M3) — forward + backward separately
==============================================================================
For target concept C and matched off-target concepts C' (averaged):

    selectivity_in  (C, S) = effect_in (C, S) - mean_{C'} effect_in (C', S)
    selectivity_out (C, S) = effect_out(C, S) - mean_{C'} effect_out(C', S)

Reported per (subset, transition, concept) row.

==============================================================================
4. Proto-token concept registry (no longer script-only)
==============================================================================
Default concepts:

    space_newline       — whitespace and newline tokens
    punctuation         — pure-ASCII punctuation
    digits              — numeric tokens
    ascii_morphology    — short lowercase ASCII subword pieces (suffixes)
    function_words      — closed-class English function words
    code_math_latex     — common code/math/LaTeX glyphs and keywords
    non_latin_scripts   — secondary; collapses Cyrillic+Arabic+Thai+...

Each ConceptDef carries (member_mask, coarse_class set, null_class_pool).
Matched-distractor sampling is per-concept-aware: distractors for a
script-concept must be in different scripts; distractors for punctuation
must be outside the punctuation set, ideally same-coarse-class.

==============================================================================
5. Pre-registered feature subsets (ranked once, reused across metrics)
==============================================================================
For each (base, target) transition, the following subsets are constructed
*once* per seed and cached in the manifest before any metric is touched:

    transition_top_k        rank by CUSUM peaked in [base, target]
    decay_top_k             rank by (rate_base - rate_target), positive
    persistent_top_k        rank by min(rate_base, rate_target)
    late_rise_top_k         rank by (rate_target - max_{<base} rate)
    late_specialist_top_k   peak_step >= late_cut, rank by terminal decoder norm
    matched_random_top_k    matched-control sampled jointly on
                            (terminal decoder norm, base firing rate),
                            excluded from any of the above

The control is the matched_random subset; the headline test for "subset S
matters" is the paired effect of S - matched_random_top_k.

==============================================================================
6. Outputs
==============================================================================
results/temporal_patch_metrics/<tag>/
    manifest.json       pre-registration: concepts, subsets, transitions, seeds
    summary.csv         long-form: one row per (transition, subset, concept)
    selectivity.csv     forward+backward selectivity per (transition, subset, concept)
    paired_vs_random.csv  paired Δ(real - matched_random) per metric
    raw.pt              full record dict + per-context arrays for re-analysis
"""

from __future__ import annotations

import csv
import json
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from readout.core.data import get_device
from readout.core.hf_revisions import resolve_revision
from readout.core.paths import repo_root, ssd_path
from readout.crosscoder import inference
from readout.crosscoder.checkpoints import load_checkpoint
from readout.crosscoder.snapshots import load_snapshot as _load_canonical_snapshot
from readout.probes.token_scripts import script_of

_REPO = repo_root()

DEVICE = get_device()  # cuda > mps > cpu, as documented in the README
# Sequence length of the eval corpora (both the released Pythia corpus and
# scripts/extract/build_eval_corpus_olmo.py chunk to this).
SEQ_LEN = 512
# Sequences per forward pass when building the hLN cache (archived
# intervention pipeline value; small because K x V x d activations dominate).
HLN_FORWARD_BATCH = 4


def encode_snapshot_local(W_U: torch.Tensor, pieces: dict) -> torch.Tensor:
    """Snapshot-LOCAL feature activations a_k(t) from one snapshot's pieces.

    Preprocess the raw W_U with snapshot t's (mean, scale), take that
    snapshot's encoder head alone, and gate on the raw pre-activation (the
    rate convention — no decoder-norm rescale; historically the archived
    02_intervention.encode). ``pieces`` comes from
    :func:`extract_pieces_for_steps`.
    """
    x_proc = (W_U - pieces["mean_i"]) / pieces["scale_i"]
    pre = x_proc @ pieces["W_E"] + pieces["b_E"]
    acts, _fired = inference.jumprelu_feature_acts(pre, pieces["thr"])
    return acts


# ---------------------------------------------------------------------------
# Per-model configurations (resolve at run start).
# SSD root is env-driven so the same code works on local SSD + cluster nfs.
# ---------------------------------------------------------------------------
SSD = ssd_path()
# Canonical (Pythia-tokenized) eval corpus — ships with the released HF
# artifacts (byte-identical to the archived original); override with
# --eval-tokens (see docs/DATA.md).
CORPUS_TOKENS_PYTHIA = SSD / "hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt"
# OLMo-tokenized variant; regenerate with scripts/extract/build_eval_corpus_olmo.py
# (this is that builder's default --output). Selected by temporal_patch_grid.py
# for --model olmo-2-7b.
CORPUS_TOKENS_OLMO = _REPO / "results/experiments/causal/temporal_localization_patching/eval_tokens_olmo.pt"
# Alias read by the library call sites; main() (and temporal_patch_grid.py)
# reassigns it per run.
CORPUS_TOKENS = CORPUS_TOKENS_PYTHIA
OUT_DIR_DEFAULT = _REPO / "results/temporal_patch_metrics"


@dataclass
class ModelCfg:
    """Everything the loader needs for a given model+run."""

    model_name: str
    ckpt_template: str  # absolute path with {seed} placeholder
    aggregates_template: str | None  # if set, sidecar .pt path (1B); else use npy fan-out (160M)
    npy_dir: Path | None  # 160M only: dir with *_all_seeds.npy
    hLN_cache_dir: Path  # where Pythia hLN per snapshot is cached
    steps_canonical: list[int]  # the 32 GE-style snapshots


CFG_PYTHIA_160M = ModelCfg(
    model_name="EleutherAI/pythia-160m",
    ckpt_template=str(
        SSD
        / "hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d8192/seed{seed}.safetensors"
    ),
    aggregates_template=None,
    npy_dir=SSD / "derived/rates/wu-d8192-multiseed",
    hLN_cache_dir=SSD / "readout_edit_timing",
    steps_canonical=[
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        14000,
        21000,
        27000,
        34000,
        47000,
        61000,
        75000,
        89000,
        102000,
        116000,
        130000,
        143000,
    ],
)

CFG_PYTHIA_1B = ModelCfg(
    model_name="EleutherAI/pythia-1b",
    ckpt_template=str(
        SSD
        / "hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d{d_sae}/seed{seed}.safetensors"
    ),
    aggregates_template=str(SSD / "derived/aggregates/aggregates_dsae{d_sae}_seed{seed}.pt"),
    npy_dir=None,
    hLN_cache_dir=SSD / "readout_edit_timing_pythia1b",
    steps_canonical=[
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        14000,
        21000,
        27000,
        34000,
        47000,
        61000,
        75000,
        89000,
        102000,
        116000,
        130000,
        143000,
    ],
)

# Pythia-6.9B and OLMo-2-7B configs resolve the released artifacts where they
# exist (the 6.9B template matches the default-lambda run; the selected sparse
# run is seed0-sparse.safetensors in the same directory). The temporal_patch_grid pipeline
# only needs steps_canonical, model_name, and hLN_cache_dir; subset / aggregate
# helpers will fail loudly if invoked on these configs without ckpts in place.
CFG_PYTHIA_6_9B = ModelCfg(
    model_name="EleutherAI/pythia-6.9b",
    ckpt_template=str(
        SSD
        / "hf_release/parameter-trajectory-crosscoders/pythia-6.9b/W_U/cross-snapshot-32/d{d_sae}/seed{seed}.safetensors"
    ),
    aggregates_template=str(SSD / "derived/aggregates/aggregates_pythia-6.9b_d{d_sae}_seed{seed}.pt"),
    npy_dir=None,
    hLN_cache_dir=SSD / "hln_cache/temporal_patch_grid/pythia-6.9b",
    steps_canonical=[
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        14000,
        21000,
        27000,
        34000,
        47000,
        61000,
        75000,
        89000,
        102000,
        116000,
        130000,
        143000,
    ],
)

# OLMo-2-7B's published step list (matches experiments/crosscoders/crosscoder_olmo/scripts/extract_wu_olmo.py).
CFG_OLMO_2_7B = ModelCfg(
    model_name="allenai/OLMo-2-1124-7B",
    ckpt_template=str(SSD / "olmo2_7b/wu_cc_olmo7b_dsae{d_sae}_seed{seed}.pt"),
    aggregates_template=str(SSD / "olmo2_7b/aggregates_dsae{d_sae}_seed{seed}.pt"),
    npy_dir=None,
    hLN_cache_dir=SSD / "hln_cache/temporal_patch_grid/olmo-2-7b",
    steps_canonical=[
        150,
        600,
        850,
        900,
        1000,
        2000,
        3000,
        14000,
        928000,
    ],
)

# Resolved at run start. All loaders read from this. Defaults to 160M
# so existing CLI invocations stay backwards-compatible.
ACTIVE_CFG: ModelCfg = CFG_PYTHIA_160M
ACTIVE_D_SAE: int = 8192

# Aliases that read from the active config so all the per-step indexing code
# in select_feature_subset / build_temporal_patch / build_matched_random_control
# keeps working unchanged.
GE_STEPS_32 = ACTIVE_CFG.steps_canonical  # backwards-compat name
MODEL_NAME = ACTIVE_CFG.model_name  # backwards-compat name
RNG_SEED = 0


def _refresh_aliases():
    """Re-bind GE_STEPS_32 / MODEL_NAME after ACTIVE_CFG is reassigned."""
    global GE_STEPS_32, MODEL_NAME
    GE_STEPS_32 = ACTIVE_CFG.steps_canonical
    MODEL_NAME = ACTIVE_CFG.model_name


# Recovery denominator floors (per-metric). Values below trigger a low_denom flag.
RECOVERY_FLOOR = {
    "concept_mass": 0.01,
    "margin": 0.05,
    "mrr": 0.005,
    "probe_gap": 0.001,
    "row_geom": 0.005,
}

# Late-rise / late-specialist cutoff: features that PEAK at or after this step
# count as "late specialists" for the 1000->143000 transition.
LATE_PEAK_CUTOFF = 5000

# ---------------------------------------------------------------------------
# Coarse-class taxonomy (per-token Unicode coarse classification)
# ---------------------------------------------------------------------------
NON_LATIN_SCRIPTS = {
    "Cyrillic",
    "Arabic",
    "Thai",
    "Devanagari",
    "CJK",
    "Japanese",
    "Korean",
    "Bengali",
}


def coarse_class_of(text: str) -> str:
    """Coarse Unicode class for a tokenizer-decoded string. Used as the join
    key for matched-distractor pools: distractors for a target token are
    sampled from the SAME coarse class as the target whenever possible."""
    s = text.lstrip(" ")
    if not s:
        return "whitespace"
    cp = ord(s[0])
    if cp < 128:
        if all(c.isspace() for c in s):
            return "ascii_whitespace"
        if all(c.isdigit() for c in s):
            return "ascii_digit"
        if all((not c.isalnum()) and (not c.isspace()) for c in s):
            return "ascii_punct"
        if all(c.isalpha() and c.islower() for c in s):
            return "ascii_letter_lower"
        if all(c.isalpha() and c.isupper() for c in s):
            return "ascii_letter_upper"
        if any(c.isalpha() for c in s):
            return "ascii_letter_mixed"
        return "ascii_other"
    sc = script_of(text)
    if sc in NON_LATIN_SCRIPTS:
        return f"script_{sc}"
    return "other_unicode"


# ---------------------------------------------------------------------------
# Proto-token concept membership predicates
# ---------------------------------------------------------------------------
ENGLISH_FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "as",
    "into",
    "about",
    "between",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "because",
    "while",
    "though",
    "although",
    "however",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "our",
    "we",
    "you",
    "your",
    "i",
    "me",
    "my",
    "not",
    "no",
    "yes",
    "so",
    "very",
    "just",
    "also",
    "only",
    "even",
    "still",
    "ever",
    "never",
    "always",
    "often",
    "sometimes",
}
ASCII_MORPH_SUFFIXES = {
    "ing",
    "ed",
    "ly",
    "tion",
    "ness",
    "ment",
    "able",
    "ible",
    "er",
    "est",
    "ity",
    "ous",
    "ive",
    "al",
    "ic",
    "ish",
    "ize",
    "ise",
    "ful",
    "less",
    "ward",
    "wise",
    "ship",
    "hood",
    "dom",
    "ation",
    "s",
    "es",
    "en",
}
CODE_MATH_LATEX_TOKENS = {
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    "<",
    ">",
    "==",
    "!=",
    "<=",
    ">=",
    "&&",
    "||",
    "::",
    "->",
    "=>",
    ";",
    "++",
    "--",
    "**",
    "def",
    "return",
    "import",
    "from",
    "class",
    "elif",
    "function",
    "var",
    "let",
    "const",
    "true",
    "false",
    "null",
    "None",
    "True",
    "False",
    "self",
    "this",
    "new",
    "static",
    "void",
    "int",
    "float",
    "str",
    "bool",
    "print",
    "len",
    "range",
    "list",
    "dict",
    "tuple",
    "\\frac",
    "\\sum",
    "\\int",
    "\\alpha",
    "\\beta",
    "\\gamma",
    "\\delta",
    "\\theta",
    "\\lambda",
    "\\mu",
    "\\sigma",
    "\\pi",
    "\\begin",
    "\\end",
    "\\textbf",
    "\\emph",
    "\\cite",
    "\\ref",
}


def _is_member_space_newline(text: str) -> bool:
    return bool(text) and all(c.isspace() for c in text)


def _is_member_punctuation(text: str) -> bool:
    s = text.lstrip(" ")
    return bool(s) and all((c in string.punctuation) for c in s)


def _is_member_digits(text: str) -> bool:
    s = text.lstrip(" ")
    return bool(s) and all(c.isdigit() for c in s) and ord(s[0]) < 128


def _is_member_ascii_morphology(text: str) -> bool:
    # Morphological subword pieces are short, lowercase, ASCII, no leading
    # space (= subword continuations in BPE), and either in a small whitelist
    # of common suffixes or sufficiently short (1-3 chars).
    if text.startswith(" "):
        return False
    s = text
    if not s or len(s) > 5:
        return False
    if not all(c.isalpha() and c.islower() and ord(c) < 128 for c in s):
        return False
    return s in ASCII_MORPH_SUFFIXES or len(s) <= 3


def _is_member_function_word(text: str) -> bool:
    s = text.lstrip(" ").lower()
    return s in ENGLISH_FUNCTION_WORDS


def _is_member_code_math_latex(text: str) -> bool:
    s = text.lstrip(" ")
    return s in CODE_MATH_LATEX_TOKENS


def _is_member_non_latin_script(text: str) -> bool:
    return script_of(text) in NON_LATIN_SCRIPTS


CONCEPT_PREDICATES: dict[str, callable] = {
    "space_newline": _is_member_space_newline,
    "punctuation": _is_member_punctuation,
    "digits": _is_member_digits,
    "ascii_morphology": _is_member_ascii_morphology,
    "function_words": _is_member_function_word,
    "code_math_latex": _is_member_code_math_latex,
    "non_latin_scripts": _is_member_non_latin_script,
}

# Per-concept allowed coarse classes for the matched-distractor pool.
# `same_class_required=True` forces the matched distractor to share coarse
# class with the *target token* (used for non_latin_scripts: must be a
# different script, but still a script). For other concepts, the matched
# distractor should preferentially share coarse class with the target,
# with relaxation if the cell is empty.
CONCEPT_DISTRACTOR_RULES: dict[str, dict] = {
    "space_newline": {
        "prefer_same_class": True,
        "fallback_classes": ["ascii_other", "ascii_punct"],
    },
    "punctuation": {"prefer_same_class": True, "fallback_classes": ["ascii_other"]},
    "digits": {
        "prefer_same_class": True,
        "fallback_classes": ["ascii_letter_lower", "ascii_letter_mixed"],
    },
    "ascii_morphology": {
        "prefer_same_class": True,
        "fallback_classes": ["ascii_letter_lower"],
    },
    "function_words": {
        "prefer_same_class": True,
        "fallback_classes": ["ascii_letter_lower"],
    },
    "code_math_latex": {
        "prefer_same_class": True,
        "fallback_classes": ["ascii_punct", "ascii_letter_lower"],
    },
    "non_latin_scripts": {
        "prefer_same_class": False,
        "fallback_classes": ["ascii_letter_lower", "ascii_letter_mixed", "ascii_other"],
    },
}


# ---------------------------------------------------------------------------
# Vocab metadata
# ---------------------------------------------------------------------------
@dataclass
class TokenMeta:
    V: int
    text: list[str]
    coarse_class: np.ndarray  # (V,) str
    has_space_prefix: np.ndarray  # (V,) bool
    char_len: np.ndarray  # (V,) int
    log_freq: np.ndarray  # (V,) float
    freq_decile: np.ndarray  # (V,) int [0,10)
    len_bucket: np.ndarray  # (V,) int [0,5)


def build_token_meta(tok, corpus_ids: torch.Tensor, V_explicit: int) -> TokenMeta:
    V = V_explicit
    counts = np.bincount(corpus_ids.numpy(), minlength=V).astype(np.int64)
    log_freq = np.log1p(counts).astype(np.float32)

    text = [None] * V
    coarse_class = np.empty(V, dtype=object)
    has_space = np.zeros(V, dtype=bool)
    char_len = np.zeros(V, dtype=np.int32)
    for v in range(V):
        try:
            t = tok.decode([int(v)])
        except Exception:
            t = ""
        text[v] = t
        coarse_class[v] = coarse_class_of(t)
        has_space[v] = t.startswith(" ")
        char_len[v] = len(t.lstrip(" "))

    pos_mask = counts > 0
    if pos_mask.sum() > 0:
        edges = np.quantile(log_freq[pos_mask], np.linspace(0, 1, 11))
        edges[0] = -np.inf
        edges[-1] = np.inf
        freq_decile = np.digitize(log_freq, edges[1:-1])
    else:
        freq_decile = np.zeros(V, dtype=np.int64)
    freq_decile = np.clip(freq_decile, 0, 9).astype(np.int32)

    len_bucket = np.where(
        char_len <= 1,
        0,
        np.where(char_len == 2, 1, np.where(char_len == 3, 2, np.where(char_len <= 5, 3, 4))),
    ).astype(np.int32)

    return TokenMeta(
        V=V,
        text=text,
        coarse_class=coarse_class,
        has_space_prefix=has_space,
        char_len=char_len,
        log_freq=log_freq,
        freq_decile=freq_decile,
        len_bucket=len_bucket,
    )


# ---------------------------------------------------------------------------
# Concept registry
# ---------------------------------------------------------------------------
@dataclass
class ConceptDef:
    name: str
    member_mask: np.ndarray  # (V,) bool
    rule: dict  # CONCEPT_DISTRACTOR_RULES[name]


def build_concept_registry(meta: TokenMeta) -> dict[str, ConceptDef]:
    out: dict[str, ConceptDef] = {}
    for name, pred in CONCEPT_PREDICATES.items():
        mask = np.array([bool(pred(t)) for t in meta.text], dtype=bool)
        out[name] = ConceptDef(name=name, member_mask=mask, rule=CONCEPT_DISTRACTOR_RULES[name])
    return out


# ---------------------------------------------------------------------------
# Distractor pools (keyed by coarse_class + matched stats)
# ---------------------------------------------------------------------------
@dataclass
class PoolState:
    pools: dict[tuple, np.ndarray]  # (class, fd, lb, sp, nd) -> ids
    norm_decile: np.ndarray  # (V,) int [0,10)


def build_pools(meta: TokenMeta, target_W_U: torch.Tensor) -> PoolState:
    row_norm = target_W_U.norm(dim=1).numpy()
    edges = np.quantile(row_norm, np.linspace(0, 1, 11))
    edges[0] = -np.inf
    edges[-1] = np.inf
    nd = np.clip(np.digitize(row_norm, edges[1:-1]), 0, 9).astype(np.int32)
    pools: dict[tuple, list[int]] = {}
    for v in range(meta.V):
        key = (
            meta.coarse_class[v],
            int(meta.freq_decile[v]),
            int(meta.len_bucket[v]),
            bool(meta.has_space_prefix[v]),
            int(nd[v]),
        )
        pools.setdefault(key, []).append(v)
    pools = {k: np.array(vs, dtype=np.int64) for k, vs in pools.items()}
    return PoolState(pools=pools, norm_decile=nd)


def sample_matched_distractors(
    target_id: int,
    concept: ConceptDef,
    meta: TokenMeta,
    pool: PoolState,
    n_match: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_match off-concept distractors matched on
    (coarse_class, freq decile, len bucket, space-prefix, norm decile),
    with progressive bin relaxation. Falls back to fallback_classes if the
    target's coarse class can't yield enough off-concept tokens."""
    fd = int(meta.freq_decile[target_id])
    lb = int(meta.len_bucket[target_id])
    sp = bool(meta.has_space_prefix[target_id])
    nd = int(pool.norm_decile[target_id])
    target_class = meta.coarse_class[target_id]

    # Class candidates in priority order.
    if concept.rule["prefer_same_class"]:
        class_priority = [target_class] + list(concept.rule["fallback_classes"])
    else:
        class_priority = list(concept.rule["fallback_classes"]) + [target_class]
    seen_classes = set()
    class_priority = [c for c in class_priority if not (c in seen_classes or seen_classes.add(c))]

    relax_orders = [
        ([0], [0], [0]),
        ([0], [0], [-1, 1]),
        ([0], [-1, 1], [-1, 0, 1]),
        ([-1, 1], [-1, 0, 1], [-1, 0, 1]),
    ]

    seen: set[int] = {target_id}
    candidates: list[int] = []
    for cl in class_priority:
        for fd_off, lb_off, nd_off in relax_orders:
            for df in fd_off:
                for dl in lb_off:
                    for dn in nd_off:
                        key = (
                            cl,
                            max(0, min(9, fd + df)),
                            max(0, min(4, lb + dl)),
                            sp,
                            max(0, min(9, nd + dn)),
                        )
                        arr = pool.pools.get(key)
                        if arr is None:
                            continue
                        for c in arr.tolist():
                            if c in seen:
                                continue
                            if concept.member_mask[c]:
                                continue
                            seen.add(c)
                            candidates.append(c)
            if len(candidates) >= n_match * 2:
                break
        if len(candidates) >= n_match:
            break

    # Last-resort: any off-concept token (rare).
    if len(candidates) < n_match:
        for v in range(meta.V):
            if v in seen or concept.member_mask[v]:
                continue
            candidates.append(v)
            if len(candidates) >= n_match * 4:
                break

    if not candidates:
        return np.array([], dtype=np.int64)
    pick = rng.choice(len(candidates), size=min(n_match, len(candidates)), replace=False)
    return np.asarray([candidates[i] for i in pick], dtype=np.int64)


def build_concept_null(
    concept: ConceptDef,
    meta: TokenMeta,
    pool: PoolState,
    in_C_sample: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One matched off-concept distractor per concept-member token; deduped."""
    null_ids: list[int] = []
    seen: set[int] = set()
    for v in in_C_sample:
        d = sample_matched_distractors(int(v), concept, meta, pool, n_match=1, rng=rng)
        if d.size and int(d[0]) not in seen:
            null_ids.append(int(d[0]))
            seen.add(int(d[0]))
    if not null_ids:
        # Hard fallback: any off-concept token, frequency-weighted.
        off = np.where(~concept.member_mask)[0]
        w = meta.log_freq[off] + 1e-3
        w = w / w.sum()
        size = min(len(in_C_sample), len(off))
        null_ids = rng.choice(off, size=size, replace=False, p=w).tolist()
    return np.asarray(null_ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Loading (config-driven; uses ACTIVE_CFG)
# ---------------------------------------------------------------------------
def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ACTIVE_CFG.model_name)


def load_snapshot(step: int) -> torch.Tensor:
    return _load_canonical_snapshot(ACTIVE_CFG.model_name, step)


def load_crosscoder(seed: int) -> tuple[dict, dict, list[int]]:
    """Returns (state_dict, preprocess_stats, steps).

    For 1B (run5_pythia1b_capsweep) the stats live in the aggregates sidecar
    distributed as a released artifact (see docs/DATA.md). For 160M they lived
    in the legacy Run-3 .pt; the released safetensors sidecars ship none, so
    they are rebuilt deterministically from the raw snapshots on load
    (bitwise means, ulp-level scales — see
    readout.crosscoder.inference.reconstruct_preprocess_stats).
    """
    ckpt_path = Path(ACTIVE_CFG.ckpt_template.format(seed=seed, d_sae=ACTIVE_D_SAE))
    cp = load_checkpoint(ckpt_path)
    if ACTIVE_CFG.aggregates_template is not None:
        agg_path = Path(
            ACTIVE_CFG.aggregates_template.format(
                seed=seed,
                d_sae=ACTIVE_D_SAE,
            )
        )
        if not agg_path.exists():
            raise FileNotFoundError(
                f"Aggregates sidecar missing: {agg_path}. The per-model "
                f"aggregates are distributed as released artifacts; see docs/DATA.md "
                f"to obtain "
                f"derived/aggregates/aggregates_*_d{ACTIVE_D_SAE}_seed{seed}.pt."
            )
        agg = torch.load(agg_path, map_location="cpu", weights_only=False)
        return cp.state_dict, agg["preprocess_stats"], list(agg["steps"])
    ps = cp.preprocess_stats
    steps = cp.steps or ACTIVE_CFG.steps_canonical
    if ps is None:
        ps = inference.reconstruct_preprocess_stats(load_snapshot(s) for s in steps)
    return cp.state_dict, ps, steps


def load_aggregate_stats(seed: int) -> dict:
    """Per-feature lifecycle statistics used by select_feature_subset:
    rates (K, D), decoder_norms (K, D), cusum_max (D,), peak_steps (D,).

    For 160M these come from the per-seed *_all_seeds.npy fan-out (loaded
    once across all seeds; we slice [seed]). For 1B from the sidecar.
    """
    if ACTIVE_CFG.aggregates_template is not None:
        agg_path = Path(
            ACTIVE_CFG.aggregates_template.format(
                seed=seed,
                d_sae=ACTIVE_D_SAE,
            )
        )
        agg = torch.load(agg_path, map_location="cpu", weights_only=False)
        return {
            "rates": agg["rates"].numpy(),
            "norms": agg["decoder_norms"].numpy(),
            "cusum": agg["cusum_max"].numpy(),
            "peak_steps": agg["peak_steps"].numpy(),
        }
    npy = ACTIVE_CFG.npy_dir
    return {
        "rates": np.load(npy / "firing_rates_all_seeds.npy")[seed],
        "norms": np.load(npy / "decoder_norms_all_seeds.npy")[seed],
        "cusum": np.load(npy / "cusum_max_all_seeds.npy")[seed],
        "peak_steps": np.load(npy / "peak_steps_all_seeds.npy")[seed],
    }


def _resolve_readout_module(model):
    """Return the unembedding module (forward-pre-hook source for h_LN)."""
    if hasattr(model, "embed_out"):  # Pythia / GPT-NeoX
        return model.embed_out
    if hasattr(model, "lm_head"):  # OLMo / LLaMA-style
        return model.lm_head
    raise AttributeError(f"{type(model).__name__} has neither .embed_out nor .lm_head")


def _cache_hidden_generic(model, ids_seqs: torch.Tensor) -> torch.Tensor:
    """Capture the input to the readout module via a forward pre-hook.

    Works for both embed_out (Pythia) and lm_head (OLMo / LLaMA) without
    tying to a specific class.
    """
    captured: list[torch.Tensor] = []

    def hook(module, inputs):
        captured.append(inputs[0].detach().cpu())

    readout = _resolve_readout_module(model)
    handle = readout.register_forward_pre_hook(hook)
    bsz = HLN_FORWARD_BATCH
    print(
        f"Forwarding {type(model).__name__} on {ids_seqs.shape[0]} sequences (batch={bsz})...",
        flush=True,
    )
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, ids_seqs.shape[0], bsz):
            batch = ids_seqs[start : start + bsz].to(DEVICE)
            _ = model(batch)
    handle.remove()
    print(f"  done in {time.time() - t0:.1f}s", flush=True)
    return torch.cat(captured, dim=0)


# Cache for HF branch enumeration so we don't re-hit the API per step.
_resolve_revision = resolve_revision  # canonical resolver: readout.core.hf_revisions


def cache_or_build_hLN(step: int, ids_seqs: torch.Tensor) -> dict:
    """Load cached h_LN at this step; if missing, run the model forward and save.

    Generic over Pythia (embed_out / step<N>) and OLMo (lm_head /
    stage1-step<N>-tokens<T>B). The revision-resolution map is cached per
    process to keep HF API hits to one per model."""
    cache_dir = ACTIVE_CFG.hLN_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"hLN_step{step}.pt"
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=False)
    revision = _resolve_revision(ACTIVE_CFG.model_name, step)
    print(
        f"  [h_LN cache] forwarding {ACTIVE_CFG.model_name} {revision} -> {p.name}",
        flush=True,
    )
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    # OLMo-2-7B at fp32 is ~28 GB — load in bf16 on CUDA, fp32 on MPS/CPU.
    use_bf16 = "olmo" in ACTIVE_CFG.model_name.lower() and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        ACTIVE_CFG.model_name,
        revision=revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    h_LN = _cache_hidden_generic(model, ids_seqs).float()
    readout = _resolve_readout_module(model)
    W_U_orig = readout.weight.detach().cpu().to(torch.float32).clone()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    out = {"h_LN": h_LN, "W_U_orig": W_U_orig, "revision": revision}
    torch.save(out, p)
    print(f"  [h_LN cache] done in {time.time() - t0:.1f}s", flush=True)
    return out


# Back-compat alias used by older callers.
def load_hLN_cache(step: int) -> dict:
    p = ACTIVE_CFG.hLN_cache_dir / f"hLN_step{step}.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"Hidden-state cache missing: {p}. Either pre-populate via "
            f"scripts/extract/build_hln_cache.py or use cache_or_build_hLN()."
        )
    return torch.load(p, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Temporal-patch construction
# ---------------------------------------------------------------------------
@dataclass
class TemporalPatch:
    base_step: int
    target_step: int
    feature_subset: list[int]
    W_U_base: torch.Tensor
    W_U_target: torch.Tensor
    W_U_patch_in: torch.Tensor
    W_U_patch_out: torch.Tensor


def decode_subset_at(feat_acts: torch.Tensor, pieces: dict, subset_idx: list[int]) -> torch.Tensor:
    """sum_{k in S} a_k(t)[v] * D_k(t) * scale_t  ->  (V, d_model).

    `feat_acts = encode_snapshot_local(W_U_t, pieces_t)` provides the snapshot-LOCAL
    activations a_k(t)[v]. The decoder W_D[t, k] and preprocessing scale[t]
    are also per-snapshot. Sign convention matches `differential_W_U` in
    `02_intervention.py`."""
    if not subset_idx:
        return torch.zeros(feat_acts.shape[0], pieces["W_D"].shape[1])
    idx = torch.tensor(subset_idx, dtype=torch.long)
    a = feat_acts[:, idx]  # (V, |S|), a_k(t)
    d = pieces["W_D"][idx]  # (|S|, d_model), D_k(t)
    return (a @ d) * pieces["scale_i"]  # (V, d_model), scale_t-rescaled


def extract_pieces_for_steps(seed: int, needed_steps: list[int]) -> dict[int, dict]:
    """Load the crosscoder, slice out only the per-snapshot pieces we will use,
    CLONE so they're independent of the parent state_dict, then drop the rest.

    Critical for memory on Pythia-1B at d_sae=24576: the full state_dict is
    ~13 GB (W_E+W_D each (32, 24576, 2048) fp32). We only need 2-3 snapshots'
    worth of pieces in practice. Returns {step: pieces_dict}."""
    import gc

    sd, ps, _ = load_crosscoder(seed)
    canonical = ACTIVE_CFG.steps_canonical
    out: dict[int, dict] = {}
    log_thr = sd["activation_function.log_jumprelu_threshold"].exp().float().clone()
    for s in needed_steps:
        idx = canonical.index(s)
        out[s] = {
            "W_E": sd["W_E"][idx].float().clone(),
            "W_D": sd["W_D"][idx].float().clone(),
            "b_E": sd["b_E"][idx].float().clone(),
            "b_D": sd["b_D"][idx].float().clone(),
            "thr": log_thr,  # shared across snapshots
            "mean_i": ps["mean"][idx].squeeze(0).float().clone(),
            "scale_i": ps["scale"][idx].squeeze().float().clone(),
        }
    del sd, ps
    gc.collect()
    return out


def build_temporal_patch(
    pieces_b: dict,
    pieces_t: dict,
    W_U_b: torch.Tensor,
    W_U_t: torch.Tensor,
    base_step: int,
    target_step: int,
    feature_subset: list[int],
) -> TemporalPatch:
    """Build a TemporalPatch from pre-extracted pieces and pre-loaded W_U slabs.

    Caller is responsible for managing the lifetime of pieces_b/pieces_t and
    the W_U slabs — typically extract once per (base, target) snapshot in
    `run()` and reuse across all subsets/concepts."""
    a_b = encode_snapshot_local(W_U_b, pieces_b)  # a_k(b)
    a_t = encode_snapshot_local(W_U_t, pieces_t)  # a_k(t)
    R_b = decode_subset_at(a_b, pieces_b, feature_subset)
    R_t = decode_subset_at(a_t, pieces_t, feature_subset)
    delta_S = R_t - R_b
    return TemporalPatch(
        base_step=base_step,
        target_step=target_step,
        feature_subset=list(feature_subset),
        W_U_base=W_U_b,
        W_U_target=W_U_t,
        W_U_patch_in=W_U_b + delta_S,
        W_U_patch_out=W_U_t - delta_S,
    )


# ---------------------------------------------------------------------------
# Pre-registered feature subsets
# ---------------------------------------------------------------------------
def select_feature_subset(
    name: str,
    top_k: int,
    base_step: int,
    target_step: int,
    rates_seed: np.ndarray,
    decoder_norms_seed: np.ndarray,
    cusum_seed: np.ndarray,
    peak_steps_seed: np.ndarray,
    rng: np.random.Generator,
    excluded: set[int] | None = None,
) -> list[int]:
    """Return list of feature ids selected by the named ranking rule.

    rates_seed: (32, D) firing rates per snapshot
    decoder_norms_seed: (32, D)
    cusum_seed: (D,) max-CUSUM statistic per feature
    peak_steps_seed: (D,) the step at which each feature peaks (in step units)
    """
    excluded = excluded or set()
    bi = GE_STEPS_32.index(base_step)
    ti = GE_STEPS_32.index(target_step)

    if name == "transition_top_k":
        # Top-k by CUSUM magnitude, restricted to features whose peak step lies
        # in the (base, target] window.
        peak = peak_steps_seed
        window = (peak > base_step) & (peak <= target_step)
        score = np.where(window, cusum_seed, -np.inf)
        order = np.argsort(-score)
    elif name == "decay_top_k":
        score = rates_seed[bi] - rates_seed[ti]
        score = np.where(score > 0, score, -np.inf)
        order = np.argsort(-score)
    elif name == "persistent_top_k":
        score = np.minimum(rates_seed[bi], rates_seed[ti])
        order = np.argsort(-score)
    elif name == "late_rise_top_k":
        # rate at target minus max rate over indices [0..bi].
        max_pre = rates_seed[: bi + 1].max(axis=0)
        score = rates_seed[ti] - max_pre
        score = np.where(score > 0, score, -np.inf)
        order = np.argsort(-score)
    elif name == "late_specialist_top_k":
        # Peak at or after LATE_PEAK_CUTOFF; rank by terminal decoder norm.
        is_late = peak_steps_seed >= LATE_PEAK_CUTOFF
        terminal = decoder_norms_seed[GE_STEPS_32.index(143000)]
        score = np.where(is_late, terminal, -np.inf)
        order = np.argsort(-score)
    elif name == "matched_random_top_k":
        # This is a CONTROL — built later by `build_matched_random_control`,
        # which needs to know the real subset(s) it's matched against. Returning
        # an empty list here so the driver can route around it.
        return []
    else:
        raise ValueError(f"unknown subset name: {name}")

    out = []
    for idx in order:
        if int(idx) in excluded:
            continue
        out.append(int(idx))
        if len(out) >= top_k:
            break
    return out


def build_matched_random_control(
    target_subset: list[int],
    top_k: int,
    rates_seed: np.ndarray,
    decoder_norms_seed: np.ndarray,
    base_step: int,
    exclude_pool: set[int],
    rng: np.random.Generator,
    n_norm_bins: int = 10,
    n_rate_bins: int = 10,
) -> list[int]:
    """Sample top_k features matched jointly on (terminal decoder norm, base
    firing rate) to `target_subset`, with NN fallback when bins are empty."""
    if not target_subset:
        return []
    bi = GE_STEPS_32.index(base_step)
    ti = GE_STEPS_32.index(143000)
    norm = decoder_norms_seed[ti]
    rate = rates_seed[bi]
    D = len(norm)
    norm_bins = np.quantile(norm, np.linspace(0, 1, n_norm_bins + 1))
    rate_bins = np.quantile(rate, np.linspace(0, 1, n_rate_bins + 1))
    nb = np.clip(np.digitize(norm, norm_bins[1:-1]), 0, n_norm_bins - 1)
    rb = np.clip(np.digitize(rate, rate_bins[1:-1]), 0, n_rate_bins - 1)

    available = np.ones(D, dtype=bool)
    for j in exclude_pool:
        available[j] = False
    for j in target_subset:
        available[j] = False
    std_n = float(norm.std()) + 1e-12
    std_r = float(rate.std()) + 1e-12

    sampled: list[int] = []
    for j in target_subset[:top_k]:
        cand_mask = available & (nb == nb[j]) & (rb == rb[j])
        cand = np.where(cand_mask)[0]
        if cand.size == 0:
            dist = np.sqrt(((norm - norm[j]) / std_n) ** 2 + ((rate - rate[j]) / std_r) ** 2)
            dist[~available] = np.inf
            cand = np.argsort(dist)[:50]
        pick = int(rng.choice(cand))
        sampled.append(pick)
        available[pick] = False
    return sampled


# ---------------------------------------------------------------------------
# Hidden-state harvesting
# ---------------------------------------------------------------------------
def collect_concept_contexts(
    h_LN: torch.Tensor,
    ids_seqs: torch.Tensor,
    concept: ConceptDef,
    max_per_concept: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, np.ndarray]:
    """Pick contexts where the next-token id is in concept.member_mask.
    Returns h at the *predictive* position (t-1). Each per-context candidate
    set is later built by sample_matched_distractors() on the per-target id."""
    N, T, D = h_LN.shape
    targets = ids_seqs[:, 1:].numpy()  # (N, T-1)
    in_C = concept.member_mask[targets]  # (N, T-1) bool
    flat = in_C.reshape(-1)
    if flat.sum() == 0:
        return torch.empty(0, D), np.array([], dtype=np.int64)
    h_pred = h_LN[:, :-1, :].reshape(-1, D)  # (N*(T-1), d_model)
    tgt_flat = targets.reshape(-1).astype(np.int64)
    sel = np.where(flat)[0]
    if len(sel) > max_per_concept:
        sel = rng.choice(sel, size=max_per_concept, replace=False)
    sel.sort()
    return h_pred[sel].clone(), tgt_flat[sel]


# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
def concept_mass_score(h: torch.Tensor, W_U: torch.Tensor, in_C: np.ndarray, null_C: np.ndarray) -> torch.Tensor:
    if h.numel() == 0:
        return torch.empty(0)
    Wc = W_U[torch.as_tensor(in_C, dtype=torch.long)].to(h.device)
    Wn = W_U[torch.as_tensor(null_C, dtype=torch.long)].to(h.device)
    lc = torch.logsumexp(h @ Wc.T, dim=-1)
    ln = torch.logsumexp(h @ Wn.T, dim=-1)
    return (lc - ln).cpu()


def matched_token_margin(
    h: torch.Tensor,
    W_U: torch.Tensor,
    target_ids: np.ndarray,
    distractor_ids: np.ndarray,
) -> torch.Tensor:
    if h.numel() == 0:
        return torch.empty(0)
    Wt = W_U[torch.as_tensor(target_ids, dtype=torch.long)].to(h.device)
    Wd = W_U[torch.as_tensor(distractor_ids, dtype=torch.long)].to(h.device)
    return ((h * Wt).sum(-1) - (h * Wd).sum(-1)).cpu()


def rank_within_candidate_set(
    h: torch.Tensor,
    W_U: torch.Tensor,
    target_ids: np.ndarray,
    candidate_ids: list[np.ndarray],
) -> dict:
    n = len(target_ids)
    if n == 0:
        return {
            "rank": np.empty(0, np.int64),
            "rrank": np.empty(0),
            "top1": np.empty(0, bool),
            "top5": np.empty(0, bool),
            "set_size": np.empty(0, np.int64),
        }
    ranks = np.empty(n, np.int64)
    rranks = np.empty(n, np.float32)
    top1 = np.empty(n, bool)
    top5 = np.empty(n, bool)
    set_size = np.empty(n, np.int64)
    for i in range(n):
        cand = np.concatenate([[int(target_ids[i])], candidate_ids[i]])
        Wc = W_U[torch.as_tensor(cand, dtype=torch.long)].to(h.device)
        scores = (h[i : i + 1] @ Wc.T).squeeze(0).cpu().numpy()
        order = np.argsort(-scores)
        rk = int(np.where(order == 0)[0][0]) + 1
        ranks[i] = rk
        rranks[i] = 1.0 / rk
        top1[i] = rk == 1
        top5[i] = rk <= 5
        set_size[i] = len(cand)
    return {
        "rank": ranks,
        "rrank": rranks,
        "top1": top1,
        "top5": top5,
        "set_size": set_size,
    }


def row_geometry_recovery(
    W_patch: torch.Tensor,
    W_target: torch.Tensor,
    W_base: torch.Tensor,
    in_C: np.ndarray,
    off_C: np.ndarray,
) -> dict:
    def per_set(idx):
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        cp = F.cosine_similarity(W_patch[idx_t], W_target[idx_t], dim=-1)
        cb = F.cosine_similarity(W_base[idx_t], W_target[idx_t], dim=-1)
        return float((cp - cb).mean()), float(cp.mean()), float(cb.mean())

    in_d, in_cp, in_cb = per_set(in_C)
    off_d, off_cp, off_cb = per_set(off_C)
    return {
        "in_C_recovery": in_d,
        "in_C_cos_patch": in_cp,
        "in_C_cos_base": in_cb,
        "off_C_recovery": off_d,
        "off_C_cos_patch": off_cp,
        "off_C_cos_base": off_cb,
        "selectivity": in_d - off_d,
    }


def fit_logreg_probe(
    W: torch.Tensor,
    in_C: np.ndarray,
    off_C: np.ndarray,
    l2: float = 1.0,
    n_iter: int = 200,
) -> tuple[torch.Tensor, float]:
    idx = np.concatenate([in_C, off_C])
    y = np.concatenate([np.ones(len(in_C)), np.zeros(len(off_C))]).astype(np.float32)
    X = W[torch.as_tensor(idx, dtype=torch.long)].float()
    yt = torch.from_numpy(y)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=n_iter, history_size=20)

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = F.binary_cross_entropy_with_logits(z, yt) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), float(b.detach())


def probe_logit_transfer(W_eval: torch.Tensor, w: torch.Tensor, b: float, in_C: np.ndarray, off_C: np.ndarray) -> dict:
    z_in = W_eval[torch.as_tensor(in_C, dtype=torch.long)].float() @ w + b
    z_off = W_eval[torch.as_tensor(off_C, dtype=torch.long)].float() @ w + b
    return {
        "probe_gap": float(z_in.mean() - z_off.mean()),
        "probe_mean_in": float(z_in.mean()),
        "probe_mean_off": float(z_off.mean()),
    }


def logit_lens_kl_on_candidates(
    h: torch.Tensor,
    W_a: torch.Tensor,
    W_b: torch.Tensor,
    candidate_ids: list[np.ndarray],
    target_ids: np.ndarray,
) -> torch.Tensor:
    n = len(target_ids)
    if n == 0:
        return torch.empty(0)
    out = torch.empty(n)
    for i in range(n):
        cand = np.concatenate([[int(target_ids[i])], candidate_ids[i]])
        idx = torch.as_tensor(cand, dtype=torch.long)
        sa = (h[i : i + 1] @ W_a[idx].T.to(h.device)).squeeze(0).float()
        sb = (h[i : i + 1] @ W_b[idx].T.to(h.device)).squeeze(0).float()
        pa = F.log_softmax(sa, dim=-1)
        pb = F.log_softmax(sb, dim=-1)
        out[i] = (pa.exp() * (pa - pb)).sum()
    return out


# ---------------------------------------------------------------------------
# Recovery framing (raw + ratio + low-denom flag)
# ---------------------------------------------------------------------------
def recovery_dict(
    m_base: float,
    m_target: float,
    m_patch_in: float,
    m_patch_out: float,
    metric_key: str,
) -> dict:
    """Returns raw deltas + recovery + low_denom flag for one metric.
    metric_key indexes RECOVERY_FLOOR for the per-metric denominator threshold."""
    floor = RECOVERY_FLOOR.get(metric_key, 1e-6)
    denom = m_target - m_base
    raw_in = m_patch_in - m_base
    raw_out = m_target - m_patch_out
    low = abs(denom) < floor
    return {
        "raw_delta_in": raw_in,
        "raw_delta_out": raw_out,
        "spread_target_minus_base": denom,
        "recovery_in": (raw_in / denom) if not low else float("nan"),
        "recovery_out": (raw_out / denom) if not low else float("nan"),
        "low_denom": bool(low),
    }


# ---------------------------------------------------------------------------
# Per-(transition, subset, concept) evaluation
# ---------------------------------------------------------------------------
def evaluate_concept_under_patch(
    concept: ConceptDef,
    h_LN: torch.Tensor,
    ids_seqs: torch.Tensor,
    meta: TokenMeta,
    pool: PoolState,
    patch: TemporalPatch,
    rng: np.random.Generator,
    max_contexts: int,
    n_match: int,
    n_concept_tokens: int,
    fit_probe: bool,
    include_kl: bool,
    bootstrap: int = 1,
) -> dict:
    """One row: (concept × patch). Reports raw deltas, recovery, flags."""
    h, target_ids = collect_concept_contexts(h_LN, ids_seqs, concept, max_per_concept=max_contexts, rng=rng)
    if len(target_ids) == 0:
        return {"concept": concept.name, "n_contexts": 0}

    # Concept-token sample for log-sum-exp aggregation.
    in_C_all = np.where(concept.member_mask)[0]
    if len(in_C_all) > n_concept_tokens:
        # Frequency-stratified deterministic sample by concept seed.
        det_rng = np.random.default_rng(hash(("concept_sample", concept.name)) % 2**32)
        w = meta.log_freq[in_C_all] + 1e-3
        w = w / w.sum()
        in_C = np.sort(det_rng.choice(in_C_all, size=n_concept_tokens, replace=False, p=w))
    else:
        in_C = in_C_all

    null_C = build_concept_null(concept, meta, pool, in_C, rng)
    off_C_for_geom = build_concept_null(concept, meta, pool, in_C, rng)

    distractors = [
        sample_matched_distractors(int(t), concept, meta, pool, n_match=n_match, rng=rng) for t in target_ids
    ]
    single_d = np.array(
        [d[0] if d.size > 0 else int(target_ids[i]) for i, d in enumerate(distractors)],
        dtype=np.int64,
    )

    rec = {
        "concept": concept.name,
        "base_step": patch.base_step,
        "target_step": patch.target_step,
        "n_features_S": len(patch.feature_subset),
        "n_contexts": int(len(target_ids)),
        "n_concept_tokens": int(len(in_C)),
        "n_null_tokens": int(len(null_C)),
    }

    # ---- M1: concept-mass logit ratio
    cm_b = concept_mass_score(h, patch.W_U_base, in_C, null_C).mean().item()
    cm_t = concept_mass_score(h, patch.W_U_target, in_C, null_C).mean().item()
    cm_pi = concept_mass_score(h, patch.W_U_patch_in, in_C, null_C).mean().item()
    cm_po = concept_mass_score(h, patch.W_U_patch_out, in_C, null_C).mean().item()
    rec.update(
        {
            "concept_mass_base": cm_b,
            "concept_mass_target": cm_t,
            "concept_mass_patch_in": cm_pi,
            "concept_mass_patch_out": cm_po,
        }
    )
    cm_rec = recovery_dict(cm_b, cm_t, cm_pi, cm_po, "concept_mass")
    rec.update({f"concept_mass_{k}": v for k, v in cm_rec.items()})

    # Bootstrap concept-token resamples for the headline metric.
    if bootstrap > 1 and len(in_C_all) > n_concept_tokens:
        boot = []
        for bi in range(bootstrap):
            brng = np.random.default_rng(hash(("boot", concept.name, bi)) % 2**32)
            w = meta.log_freq[in_C_all] + 1e-3
            w = w / w.sum()
            sample = np.sort(brng.choice(in_C_all, size=n_concept_tokens, replace=False, p=w))
            null_b = build_concept_null(concept, meta, pool, sample, brng)
            cmb_b = concept_mass_score(h, patch.W_U_base, sample, null_b).mean().item()
            cmb_t = concept_mass_score(h, patch.W_U_target, sample, null_b).mean().item()
            cmb_pi = concept_mass_score(h, patch.W_U_patch_in, sample, null_b).mean().item()
            cmb_po = concept_mass_score(h, patch.W_U_patch_out, sample, null_b).mean().item()
            boot.append((cmb_b, cmb_t, cmb_pi, cmb_po))
        boot = np.array(boot)
        rec.update(
            {
                "boot_concept_mass_base_std": float(boot[:, 0].std(ddof=1)),
                "boot_concept_mass_target_std": float(boot[:, 1].std(ddof=1)),
                "boot_concept_mass_patch_in_std": float(boot[:, 2].std(ddof=1)),
                "boot_concept_mass_patch_out_std": float(boot[:, 3].std(ddof=1)),
            }
        )

    # ---- M2a: matched-token margin
    mg_b = matched_token_margin(h, patch.W_U_base, target_ids, single_d).mean().item()
    mg_t = matched_token_margin(h, patch.W_U_target, target_ids, single_d).mean().item()
    mg_pi = matched_token_margin(h, patch.W_U_patch_in, target_ids, single_d).mean().item()
    mg_po = matched_token_margin(h, patch.W_U_patch_out, target_ids, single_d).mean().item()
    rec.update(
        {
            "margin_base": mg_b,
            "margin_target": mg_t,
            "margin_patch_in": mg_pi,
            "margin_patch_out": mg_po,
        }
    )
    mg_rec = recovery_dict(mg_b, mg_t, mg_pi, mg_po, "margin")
    rec.update({f"margin_{k}": v for k, v in mg_rec.items()})

    # ---- M2b: rank-in-candidate-set (MRR is the headline reduction)
    for tag, W in [
        ("base", patch.W_U_base),
        ("target", patch.W_U_target),
        ("patch_in", patch.W_U_patch_in),
        ("patch_out", patch.W_U_patch_out),
    ]:
        rk = rank_within_candidate_set(h, W, target_ids, distractors)
        rec[f"mrr_{tag}"] = float(rk["rrank"].mean())
        rec[f"top1_{tag}"] = float(rk["top1"].mean())
        rec[f"top5_{tag}"] = float(rk["top5"].mean())
        rec[f"mean_rank_{tag}"] = float(rk["rank"].mean())
    mrr_rec = recovery_dict(
        rec["mrr_base"],
        rec["mrr_target"],
        rec["mrr_patch_in"],
        rec["mrr_patch_out"],
        "mrr",
    )
    rec.update({f"mrr_{k}": v for k, v in mrr_rec.items()})

    # ---- M5: row-geometry (in_C vs off_C, both directions)
    rg_in = row_geometry_recovery(patch.W_U_patch_in, patch.W_U_target, patch.W_U_base, in_C, off_C_for_geom)
    rg_out = row_geometry_recovery(patch.W_U_patch_out, patch.W_U_base, patch.W_U_target, in_C, off_C_for_geom)
    rec.update({f"row_geom_in_{k}": v for k, v in rg_in.items()})
    rec.update({f"row_geom_out_{k}": v for k, v in rg_out.items()})

    # ---- M4: probe-logit transfer (appendix)
    if fit_probe:
        wp, bp = fit_logreg_probe(patch.W_U_target, in_C, off_C_for_geom)
        gaps = {}
        for tag, W in [
            ("base", patch.W_U_base),
            ("target", patch.W_U_target),
            ("patch_in", patch.W_U_patch_in),
            ("patch_out", patch.W_U_patch_out),
        ]:
            r = probe_logit_transfer(W, wp, bp, in_C, off_C_for_geom)
            rec[f"probe_gap_{tag}"] = r["probe_gap"]
            gaps[tag] = r["probe_gap"]
        pg_rec = recovery_dict(
            gaps["base"],
            gaps["target"],
            gaps["patch_in"],
            gaps["patch_out"],
            "probe_gap",
        )
        rec.update({f"probe_gap_{k}": v for k, v in pg_rec.items()})

    # ---- M6: candidate-set logit-lens KL (appendix)
    if include_kl:
        kl_in = (
            logit_lens_kl_on_candidates(h, patch.W_U_patch_in, patch.W_U_target, distractors, target_ids).mean().item()
        )
        kl_out = (
            logit_lens_kl_on_candidates(h, patch.W_U_patch_out, patch.W_U_target, distractors, target_ids).mean().item()
        )
        kl_base = (
            logit_lens_kl_on_candidates(h, patch.W_U_base, patch.W_U_target, distractors, target_ids).mean().item()
        )
        rec.update(
            {
                "kl_patch_in_vs_target": kl_in,
                "kl_patch_out_vs_target": kl_out,
                "kl_base_vs_target": kl_base,
            }
        )

    return rec


# ---------------------------------------------------------------------------
# Forward + backward selectivity (M3)
# ---------------------------------------------------------------------------
def family_selectivity(records: list[dict]) -> list[dict]:
    """For each (transition, subset, concept C), compute:

        effect_in (C, S)  = M_C(W_patch_in)  - M_C(W_base)
        effect_out(C, S)  = M_C(W_target)    - M_C(W_patch_out)
        sel_in (C, S)     = effect_in (C, S) - mean_{C' off-target} effect_in (C', S)
        sel_out(C, S)     = effect_out(C, S) - mean_{C' off-target} effect_out(C', S)

    Effect is on the headline (concept-mass) metric.
    """
    by_S = {}
    for r in records:
        if r.get("n_contexts", 0) == 0:
            continue
        key = (
            r["base_step"],
            r["target_step"],
            r["n_features_S"],
            r.get("subset_name", ""),
        )
        by_S.setdefault(key, {})[r["concept"]] = r
    out = []
    for key, recs in by_S.items():
        for cat_t, r_t in recs.items():
            others_in = [r["concept_mass_patch_in"] - r["concept_mass_base"] for c, r in recs.items() if c != cat_t]
            others_out = [r["concept_mass_target"] - r["concept_mass_patch_out"] for c, r in recs.items() if c != cat_t]
            if not others_in:
                continue
            eff_in = r_t["concept_mass_patch_in"] - r_t["concept_mass_base"]
            eff_out = r_t["concept_mass_target"] - r_t["concept_mass_patch_out"]
            out.append(
                {
                    "base_step": key[0],
                    "target_step": key[1],
                    "n_features_S": key[2],
                    "subset_name": key[3],
                    "concept": cat_t,
                    "effect_in_on_C": eff_in,
                    "effect_out_on_C": eff_out,
                    "effect_in_off_C_mean": float(np.mean(others_in)),
                    "effect_out_off_C_mean": float(np.mean(others_out)),
                    "selectivity_in": eff_in - float(np.mean(others_in)),
                    "selectivity_out": eff_out - float(np.mean(others_out)),
                }
            )
    return out


def paired_vs_random(records: list[dict], real_subset_names: list[str]) -> list[dict]:
    """For each (transition, concept), compute paired Δ(real - matched_random)
    on the headline metrics. Real-vs-random is the PRIMARY statistical test."""
    by_key = {}
    for r in records:
        if r.get("n_contexts", 0) == 0:
            continue
        key = (r["base_step"], r["target_step"], r["concept"])
        by_key.setdefault(key, {})[r.get("subset_name", "")] = r
    out = []
    metrics_in = [
        ("concept_mass", "concept_mass_patch_in", "concept_mass_base"),
        ("margin", "margin_patch_in", "margin_base"),
        ("mrr", "mrr_patch_in", "mrr_base"),
    ]
    metrics_out = [
        ("concept_mass", "concept_mass_target", "concept_mass_patch_out"),
        ("margin", "margin_target", "margin_patch_out"),
        ("mrr", "mrr_target", "mrr_patch_out"),
    ]
    for key, by_subset in by_key.items():
        rand = by_subset.get("matched_random_top_k")
        if rand is None:
            continue
        for sname in real_subset_names:
            real = by_subset.get(sname)
            if real is None:
                continue
            row = {
                "base_step": key[0],
                "target_step": key[1],
                "concept": key[2],
                "subset_name": sname,
            }
            for label, k_pi, k_b in metrics_in:
                row[f"paired_in_{label}"] = (real[k_pi] - real[k_b]) - (rand[k_pi] - rand[k_b])
            for label, k_t, k_po in metrics_out:
                row[f"paired_out_{label}"] = (real[k_t] - real[k_po]) - (rand[k_t] - rand[k_po])
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def append_csv(path: Path, row: dict):
    keys = list(row.keys())
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class TransitionSpec:
    base_step: int
    target_step: int
    name: str
    subset_specs: list[tuple[str, int]] = field(default_factory=list)
    # list of (subset_name, top_k)


def run(
    seed: int,
    transitions: list[TransitionSpec],
    concept_names: list[str],
    out_dir: Path,
    h_step_for_eval: str = "target",
    max_contexts: int = 256,
    n_match: int = 99,
    n_concept_tokens: int = 256,
    fit_probe: bool = True,
    include_kl: bool = False,
    bootstrap: int = 1,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[seed {seed}] model={ACTIVE_CFG.model_name} d_sae={ACTIVE_D_SAE} loading aggregates",
        flush=True,
    )
    agg = load_aggregate_stats(seed)
    rates_seed, norms_seed = agg["rates"], agg["norms"]
    cusum_seed, peak_seed = agg["cusum"], agg["peak_steps"]

    tok = load_tokenizer()
    eval_data = torch.load(CORPUS_TOKENS, weights_only=False)
    ids = eval_data["ids"]
    n_full = (len(ids) // SEQ_LEN) * SEQ_LEN
    ids_seqs = ids[:n_full].view(-1, SEQ_LEN)

    # Vocab size from any snapshot.
    V_explicit = load_snapshot(143000).shape[0]
    print("  building token meta + concept registry...", flush=True)
    meta = build_token_meta(tok, ids[:n_full], V_explicit=V_explicit)
    concepts = build_concept_registry(meta)
    for n in concept_names:
        if n not in concepts:
            raise KeyError(f"unknown concept: {n}; have {list(concepts)}")

    # Pre-registration manifest written BEFORE any metric.
    manifest = {
        "seed": seed,
        "transitions": [
            {
                "name": ts.name,
                "base_step": ts.base_step,
                "target_step": ts.target_step,
                "subsets": [{"name": n, "top_k": k} for n, k in ts.subset_specs],
            }
            for ts in transitions
        ],
        "concepts": concept_names,
        "concept_member_counts": {n: int(concepts[n].member_mask.sum()) for n in concept_names},
        "h_step_for_eval": h_step_for_eval,
        "max_contexts": max_contexts,
        "n_match": n_match,
        "n_concept_tokens": n_concept_tokens,
        "fit_probe": fit_probe,
        "include_kl": include_kl,
        "bootstrap": bootstrap,
        "recovery_floor": RECOVERY_FLOOR,
        "late_peak_cutoff": LATE_PEAK_CUTOFF,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Resolve subsets per transition (seeded RNG used for matched-random).
    rng_root = np.random.default_rng(RNG_SEED + seed)
    resolved_subsets: dict[tuple[int, int], dict[str, list[int]]] = {}
    for ts in transitions:
        per_ts: dict[str, list[int]] = {}
        # First pass: real subsets.
        for name, k in ts.subset_specs:
            if name == "matched_random_top_k":
                continue
            per_ts[name] = select_feature_subset(
                name,
                k,
                ts.base_step,
                ts.target_step,
                rates_seed,
                norms_seed,
                cusum_seed,
                peak_seed,
                rng_root,
                excluded=set(),
            )
        # Second pass: matched_random matched against the union of real subsets.
        union_real = sorted({i for ids_ in per_ts.values() for i in ids_})
        for name, k in ts.subset_specs:
            if name != "matched_random_top_k":
                continue
            per_ts[name] = build_matched_random_control(
                target_subset=union_real,
                top_k=k,
                rates_seed=rates_seed,
                decoder_norms_seed=norms_seed,
                base_step=ts.base_step,
                exclude_pool=set(union_real),
                rng=rng_root,
            )
        resolved_subsets[(ts.base_step, ts.target_step)] = per_ts
    # Persist subset ids in manifest.
    (out_dir / "subsets.json").write_text(
        json.dumps(
            {f"{b}->{t}": {n: ids_ for n, ids_ in per_ts.items()} for (b, t), per_ts in resolved_subsets.items()},
            indent=2,
        )
    )

    summary_path = out_dir / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    raw_records: list[dict] = []
    rng = np.random.default_rng(RNG_SEED + 100 * seed)

    # Extract pieces only for the snapshots we will actually patch on, then
    # drop the full crosscoder state_dict — at d_sae=24576 it is ~13 GB.
    needed_steps = sorted({s for ts in transitions for s in (ts.base_step, ts.target_step)})
    print(
        f"  extracting pieces for snapshots {needed_steps} (will drop full sd)",
        flush=True,
    )
    pieces_by_step = extract_pieces_for_steps(seed, needed_steps)
    W_U_by_step = {s: load_snapshot(s) for s in needed_steps}

    for ts in transitions:
        h_step = ts.target_step if h_step_for_eval == "target" else ts.base_step
        cached = cache_or_build_hLN(h_step, ids_seqs)
        h_LN = cached["h_LN"]
        for name, _k in ts.subset_specs:
            S = resolved_subsets[(ts.base_step, ts.target_step)][name]
            if not S:
                print(f"\n[{ts.name} / {name}] empty subset — skipping", flush=True)
                continue
            print(f"\n[{ts.name} / {name}: {len(S)} features]  head={S[:5]}", flush=True)
            patch = build_temporal_patch(
                pieces_by_step[ts.base_step],
                pieces_by_step[ts.target_step],
                W_U_by_step[ts.base_step],
                W_U_by_step[ts.target_step],
                ts.base_step,
                ts.target_step,
                S,
            )
            pool = build_pools(meta, patch.W_U_target)
            for cn in concept_names:
                t0 = time.time()
                rec = evaluate_concept_under_patch(
                    concept=concepts[cn],
                    h_LN=h_LN,
                    ids_seqs=ids_seqs,
                    meta=meta,
                    pool=pool,
                    patch=patch,
                    rng=rng,
                    max_contexts=max_contexts,
                    n_match=n_match,
                    n_concept_tokens=n_concept_tokens,
                    fit_probe=fit_probe,
                    include_kl=include_kl,
                    bootstrap=bootstrap,
                )
                rec["seed"] = seed
                rec["transition"] = ts.name
                rec["subset_name"] = name
                rec["h_eval_step"] = h_step
                raw_records.append(rec)
                append_csv(summary_path, rec)
                if rec.get("n_contexts", 0) > 0:
                    cm_in = rec["concept_mass_recovery_in"]
                    cm_out = rec["concept_mass_recovery_out"]
                    flag = " (low_denom)" if rec.get("concept_mass_low_denom") else ""
                    raw_in = rec["concept_mass_raw_delta_in"]
                    raw_out = rec["concept_mass_raw_delta_out"]
                    print(
                        f"  {cn:<17} n={rec['n_contexts']:>4} | "
                        f"CM raw_in={raw_in:+.3f}  raw_out={raw_out:+.3f}  "
                        f"rec_in={cm_in:+.3f}  rec_out={cm_out:+.3f}{flag}  "
                        f"[{time.time() - t0:.1f}s]",
                        flush=True,
                    )
                else:
                    print(f"  {cn:<17} skipped (no contexts)", flush=True)
            # Free the per-subset patch (4 W_U slabs ~1.6 GB at d_model=2048)
            # before the next subset builds its own.
            del patch, pool
            import gc

            gc.collect()

    # M3 selectivity (forward + backward) per (transition, subset, concept)
    sel_records = family_selectivity(raw_records)
    write_csv(out_dir / "selectivity.csv", sel_records)

    # Primary test: paired effect of real subset vs matched_random
    real_subset_names = [name for ts in transitions for name, _k in ts.subset_specs if name != "matched_random_top_k"]
    real_subset_names = sorted(set(real_subset_names))
    pvr = paired_vs_random(raw_records, real_subset_names)
    write_csv(out_dir / "paired_vs_random.csv", pvr)

    torch.save(
        {
            "records": raw_records,
            "selectivity": sel_records,
            "paired_vs_random": pvr,
            "manifest": manifest,
            "subsets": resolved_subsets,
        },
        out_dir / "raw.pt",
    )
    print(
        f"\nDone. Wrote {len(raw_records)} per-(transition,subset,concept) records.",
        flush=True,
    )
    return raw_records, sel_records, pvr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_CONCEPTS = [
    "space_newline",
    "punctuation",
    "digits",
    "ascii_morphology",
    "function_words",
    "code_math_latex",
    "non_latin_scripts",
]
