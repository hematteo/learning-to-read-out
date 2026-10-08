"""Single-token, two-choice contrastive task families for readout-swap analysis.

Each task family yields a list of ``Example``s after filtering for:
  - both ``y_plus`` and ``y_minus`` answers tokenize to a single tokenizer-token,
  - matched leading-space form and casing,
  - finite usable examples per family (capped by `n_max`).

Builders are tokenizer-aware. Caller is responsible for picking the right
tokenizer (Pythia uses GPT-NeoX BPE; OLMo / LLaMA use SentencePiece-derived
BPE — leading-space form differs).

Used by:
  - reproduce/availability/build_task_datasets.py
  - reproduce/localization/run_feature_attribution.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


@dataclass
class Example:
    family: str
    prompt: str
    prompt_ids: list[int]  # tokenized prompt (no answer)
    y_plus: int  # correct single-token id
    y_minus: int  # contrastive single-token id
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------
def _single_token_id(tokenizer, text: str, *, leading_space: bool = True) -> int | None:
    """Return the single token id for ``text``; None if it is not single-token.

    If leading_space=True (default for answer tokens in the middle of running text),
    we tokenize " text" and require exactly one token. Mirrors GPT-2 BPE convention.
    """
    s = (" " + text.lstrip()) if leading_space else text
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        return None
    return int(ids[0])


def _row_norm(W_U: torch.Tensor, idx: int) -> float:
    return float(W_U[idx].float().norm().item())


def _filter_pair(
    tokenizer,
    pos: str,
    neg: str,
    *,
    leading_space: bool = True,
    W_U_for_norm_match: torch.Tensor | None = None,
    norm_tol: float = 0.5,  # accept rows whose log-ratio is within tol
) -> tuple[int, int] | None:
    """Tokenize pos/neg, enforce single-token form and (optional) row-norm match."""
    pid = _single_token_id(tokenizer, pos, leading_space=leading_space)
    nid = _single_token_id(tokenizer, neg, leading_space=leading_space)
    if pid is None or nid is None or pid == nid:
        # pid == nid would make the contrastive margin identically zero.
        return None
    if W_U_for_norm_match is not None:
        a = _row_norm(W_U_for_norm_match, pid)
        b = _row_norm(W_U_for_norm_match, nid)
        if a > 0 and b > 0:
            ratio = abs(np.log(a / b))
            if ratio > norm_tol:
                return None
    return pid, nid


# ---------------------------------------------------------------------------
# Task families
# ---------------------------------------------------------------------------
def _build_sva_pairs() -> list[tuple[str, str, str]]:
    """Subject-verb agreement minimal pairs.

    Returns triples (prompt, y_plus, y_minus). Plural subject -> "are"/"is",
    singular subject -> "is"/"are". Templates are deterministic and audit-able.
    """
    nouns_pl = [
        "keys",
        "books",
        "doors",
        "cars",
        "students",
        "teachers",
        "papers",
        "tables",
        "windows",
        "boxes",
        "ships",
        "roads",
        "leaves",
        "letters",
        "bells",
        "clocks",
        "rooms",
        "cups",
        "lamps",
        "shoes",
    ]
    nouns_sg = [
        "key",
        "book",
        "door",
        "car",
        "student",
        "teacher",
        "paper",
        "table",
        "window",
        "box",
        "ship",
        "road",
        "leaf",
        "letter",
        "bell",
        "clock",
        "room",
        "cup",
        "lamp",
        "shoe",
    ]
    attractors = [
        "to the cabinet",
        "near the wall",
        "by the table",
        "on the shelf",
        "in the corner",
        "next to the lamp",
        "above the desk",
        "below the window",
    ]
    plural = [(f"The {n} {atr}", "are", "is") for n in nouns_pl for atr in attractors]
    singular = [(f"The {n} {atr}", "is", "are") for n in nouns_sg for atr in attractors]
    # Interleave the two number classes so that any truncated prefix (build_sva
    # caps at n_max) stays balanced instead of returning all-plural ("are") first.
    out: list[tuple[str, str, str]] = []
    for pl, sg in zip(plural, singular):
        out.append(pl)
        out.append(sg)
    out.extend(plural[len(singular) :])
    out.extend(singular[len(plural) :])
    return out


def build_sva(tokenizer, *, n_max: int = 2000, W_U_for_norm_match=None) -> list[Example]:
    out: list[Example] = []
    pair = _filter_pair(tokenizer, "are", "is", W_U_for_norm_match=W_U_for_norm_match)
    if pair is None:
        return out
    are_id, is_id = pair
    for prompt, pos, neg in _build_sva_pairs():
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        y_p = are_id if pos == "are" else is_id
        y_m = is_id if neg == "is" else are_id
        out.append(
            Example(
                family="sva",
                prompt=prompt,
                prompt_ids=ids,
                y_plus=y_p,
                y_minus=y_m,
                meta={"pos": pos, "neg": neg},
            )
        )
        if len(out) >= n_max:
            break
    return out


def build_sva_across_pp(tokenizer, *, n_max: int = 2000, W_U_for_norm_match=None) -> list[Example]:
    """Harder SVA with a prepositional-phrase attractor and varied verbs."""
    from readout.probes.sva import generate_sva_across_pp

    out: list[Example] = []
    for item in generate_sva_across_pp(n=max(n_max, 1000), seed=0):
        pair = _filter_pair(
            tokenizer,
            item.verb_sg.strip(),
            item.verb_pl.strip(),
            W_U_for_norm_match=W_U_for_norm_match,
        )
        if pair is None:
            continue
        verb_sg_id, verb_pl_id = pair
        for prompt, pos_id, neg_id, number in [
            (item.clean_text, verb_sg_id, verb_pl_id, "singular"),
            (item.corrupted_text, verb_pl_id, verb_sg_id, "plural"),
        ]:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            out.append(
                Example(
                    family="sva_across_pp",
                    prompt=prompt,
                    prompt_ids=ids,
                    y_plus=pos_id,
                    y_minus=neg_id,
                    meta={
                        "number": number,
                        "verb_sg": item.verb_sg.strip(),
                        "verb_pl": item.verb_pl.strip(),
                    },
                )
            )
            if len(out) >= n_max:
                return out
    return out


def _induction_token_pool(tokenizer, *, k: int = 200) -> list[int]:
    """Pick `k` low-rare-but-not-too-common single tokens with a leading space.

    We use a static curated word list to avoid pulling massive frequency tables.
    """
    # NOTE: the default k=200 is unreachable — this list has ~80 unique words
    # (six duplicates: engine, river, anchor, hammer, needle, valley, deduped
    # by the `tid not in ids` check below), and single-token filtering shrinks
    # the pool further. Both quirks are kept as-is for dataset stability.
    words = [
        "apple",
        "river",
        "music",
        "stone",
        "candle",
        "engine",
        "novel",
        "winter",
        "garden",
        "bridge",
        "letter",
        "puzzle",
        "harbor",
        "forest",
        "doctor",
        "ribbon",
        "marble",
        "planet",
        "mirror",
        "valley",
        "school",
        "cookie",
        "mountain",
        "tunnel",
        "cabin",
        "bottle",
        "violet",
        "circle",
        "dragon",
        "feather",
        "ladder",
        "pencil",
        "rabbit",
        "saddle",
        "thunder",
        "anchor",
        "burner",
        "cellar",
        "diamond",
        "engine",
        "flower",
        "guitar",
        "hammer",
        "island",
        "jacket",
        "kitten",
        "lemon",
        "magnet",
        "needle",
        "oxygen",
        "pillow",
        "queen",
        "rocket",
        "silver",
        "tiger",
        "umbrella",
        "violin",
        "window",
        "yellow",
        "zebra",
        "anchor",
        "basket",
        "carrot",
        "domain",
        "eagle",
        "festival",
        "ginger",
        "hammer",
        "iceberg",
        "jaguar",
        "kettle",
        "lantern",
        "morning",
        "needle",
        "ocean",
        "palace",
        "quartz",
        "river",
        "shadow",
        "ticket",
        "unicorn",
        "valley",
        "willow",
        "zircon",
    ]
    ids: list[int] = []
    for w in words:
        tid = _single_token_id(tokenizer, w, leading_space=True)
        if tid is not None and tid not in ids:
            ids.append(tid)
        if len(ids) >= k:
            break
    return ids


def build_induction(
    tokenizer,
    *,
    n_max: int = 1000,
    rng_seed: int = 0,
    W_U_for_norm_match=None,
) -> list[Example]:
    """Induction: prompt 'A B ... A' should predict B.

    Each example uses two distinct single-tokens A, B from a pool. The
    contrastive distractor is another pool token of similar frequency class.

    Caveat: the "{a}{b}{a}{b}{a}" prompt is re-encoded from decoded text and
    only ``len(ids) >= 4`` is checked, so BPE merges across concatenation
    boundaries can silently break the induction premise (the encoded prompt
    need not contain the literal A B A B A token sequence) for some pairs.
    """
    pool = _induction_token_pool(tokenizer, k=200)
    if len(pool) < 4:
        return []
    rng = np.random.default_rng(rng_seed)
    out: list[Example] = []
    seen: set[tuple[int, int]] = set()
    # Hard cap on the rejection-sampling loop so an exhausted pair pool can't hang.
    max_pairs = len(pool) * (len(pool) - 1)
    max_attempts = max(8 * n_max, 4 * max_pairs)
    attempts = 0
    while len(out) < n_max and attempts < max_attempts:
        attempts += 1
        a, b, c = rng.choice(pool, size=3, replace=False)
        a, b, c = int(a), int(b), int(c)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        # Render: "<A><B> ... <A>", expect <B>. Decode to text.
        a_str = tokenizer.decode([a])
        b_str = tokenizer.decode([b])
        prompt_text = f"{a_str}{b_str}{a_str}{b_str}{a_str}"
        ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(ids) < 4:
            continue
        if W_U_for_norm_match is not None:
            ra = _row_norm(W_U_for_norm_match, b)
            rc = _row_norm(W_U_for_norm_match, c)
            if ra > 0 and rc > 0 and abs(np.log(ra / rc)) > 0.5:
                continue
        out.append(
            Example(
                family="induction",
                prompt=prompt_text,
                prompt_ids=ids,
                y_plus=b,
                y_minus=c,
                meta={"a": a, "b": b, "distractor": c},
            )
        )
    return out


def _ioi_names(tokenizer) -> list[int]:
    candidates = [
        "John",
        "Mary",
        "Tom",
        "Lucy",
        "Mark",
        "Anna",
        "Sara",
        "Paul",
        "Eric",
        "Lisa",
        "Dave",
        "Kate",
        "Mike",
        "Jane",
        "Adam",
        "Emma",
        "Carl",
        "Nora",
        "Ryan",
        "Beth",
        "Jack",
        "Ruth",
        "Sean",
        "Iris",
    ]
    ids = []
    for name in candidates:
        tid = _single_token_id(tokenizer, name, leading_space=True)
        if tid is not None:
            ids.append(tid)
    return ids


def build_ioi(
    tokenizer,
    *,
    n_max: int = 1000,
    rng_seed: int = 0,
    W_U_for_norm_match=None,
) -> list[Example]:
    """Indirect-object identification (IOI).

    'Alice and Bob went to the store. Alice gave the book to' -> ' Bob'
    Distractor is the other named entity (' Alice').
    """
    names = _ioi_names(tokenizer)
    if len(names) < 4:
        return []
    rng = np.random.default_rng(rng_seed)
    out: list[Example] = []
    seen: set[tuple[int, int]] = set()
    objects = ["book", "ball", "pen", "letter", "key", "ring", "cup", "card"]
    # Hard cap so an exhausted (name, name, object) pair pool can't hang.
    max_pairs = len(names) * (len(names) - 1) * len(objects)
    max_attempts = max(8 * n_max, 4 * max_pairs)
    attempts = 0
    while len(out) < n_max and attempts < max_attempts:
        attempts += 1
        a_id, b_id = rng.choice(names, size=2, replace=False)
        a_id, b_id = int(a_id), int(b_id)
        if (a_id, b_id) in seen:
            continue
        seen.add((a_id, b_id))
        a_str = tokenizer.decode([a_id]).strip()
        b_str = tokenizer.decode([b_id]).strip()
        obj = rng.choice(objects)
        # Subject is A, recipient is B; A is the indirect-object distractor.
        prompt = f"{a_str} and {b_str} went to the store. {a_str} gave the {obj} to"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if W_U_for_norm_match is not None:
            ra = _row_norm(W_U_for_norm_match, b_id)
            rc = _row_norm(W_U_for_norm_match, a_id)
            if ra > 0 and rc > 0 and abs(np.log(ra / rc)) > 0.5:
                continue
        out.append(
            Example(
                family="ioi",
                prompt=prompt,
                prompt_ids=ids,
                y_plus=b_id,
                y_minus=a_id,
                meta={"subj": a_str, "recipient": b_str, "obj": obj},
            )
        )
    return out


def build_ioi_role_balanced(
    tokenizer,
    *,
    n_max: int = 1000,
    rng_seed: int = 0,
    W_U_for_norm_match=None,
) -> list[Example]:
    """Balanced IOI where the target is role binding, not recipient identity.

    For each ordered name pair/object we emit both role orders with a balancing
    filler mention:

    - subject first:    ``Alice and Bob ... Bob waited. Alice gave ... to`` -> `` Bob``
    - recipient first: ``Alice and Bob ... Alice waited. Bob gave ... to`` -> `` Alice``

    For a fixed first/second name pair, the two classes have the same
    orderless token bag and the same name counts. A probe must use sequence
    structure rather than learning that "the second name is the answer" or
    that "the less frequent name is the answer".
    """
    names = _ioi_names(tokenizer)
    if len(names) < 4:
        return []

    rng = np.random.default_rng(rng_seed)
    objects = ["book", "ball", "pen", "letter", "key", "ring", "cup", "card"]
    pair_object_specs: list[tuple[int, int, str]] = []
    for i, a_id in enumerate(names):
        for b_id in names[i + 1 :]:
            for obj in objects:
                pair_object_specs.append((int(a_id), int(b_id), obj))
    rng.shuffle(pair_object_specs)

    out: list[Example] = []
    for a_id, b_id, obj in pair_object_specs:
        if len(out) + 2 > n_max:
            break
        if W_U_for_norm_match is not None:
            ra = _row_norm(W_U_for_norm_match, a_id)
            rb = _row_norm(W_U_for_norm_match, b_id)
            if ra > 0 and rb > 0 and abs(np.log(ra / rb)) > 0.5:
                continue

        a_str = tokenizer.decode([a_id]).strip()
        b_str = tokenizer.decode([b_id]).strip()
        pair_key = "::".join(sorted((a_str, b_str)))

        # Recipient second: A is subject, B is indirect object.
        prompt = f"{a_str} and {b_str} went to the store. {b_str} waited. {a_str} gave the {obj} to"
        out.append(
            Example(
                family="ioi_role_balanced",
                prompt=prompt,
                prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
                y_plus=b_id,
                y_minus=a_id,
                meta={
                    "subj": a_str,
                    "recipient": b_str,
                    "obj": obj,
                    "first_name": a_str,
                    "second_name": b_str,
                    "recipient_position": "second",
                    "unordered_pair": pair_key,
                },
            )
        )

        # Recipient first: B is subject, A is indirect object.
        prompt = f"{a_str} and {b_str} went to the store. {a_str} waited. {b_str} gave the {obj} to"
        out.append(
            Example(
                family="ioi_role_balanced",
                prompt=prompt,
                prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
                y_plus=a_id,
                y_minus=b_id,
                meta={
                    "subj": b_str,
                    "recipient": a_str,
                    "obj": obj,
                    "first_name": a_str,
                    "second_name": b_str,
                    "recipient_position": "first",
                    "unordered_pair": pair_key,
                },
            )
        )

    return out


def build_numeric_greater_than(
    tokenizer,
    *,
    n_max: int = 1000,
    rng_seed: int = 0,
    W_U_for_norm_match=None,
) -> list[Example]:
    """Greater-than: '17 is greater than' -> ' 16' (the smaller one).

    For tokenizers that tokenize 1-2 digit numbers as a single token (Pythia
    does for many of them), this is a clean contrastive pair.
    """
    rng = np.random.default_rng(rng_seed)
    pool = []
    for n in range(2, 100):
        tid = _single_token_id(tokenizer, str(n), leading_space=True)
        if tid is not None:
            pool.append((n, tid))
    if len(pool) < 4:
        return []
    out: list[Example] = []
    seen: set[tuple[int, int]] = set()
    # Hard cap so an exhausted ordered-pair pool can't hang.
    max_pairs = len(pool) * (len(pool) - 1)
    max_attempts = max(8 * n_max, 4 * max_pairs)
    attempts = 0
    while len(out) < n_max and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(len(pool), size=2, replace=False)
        a_n, a_id = pool[i]
        b_n, b_id = pool[j]
        # Dedup on the unordered pair: the emitted example only depends on
        # {big, small}, so (17,16) and (16,17) are the same example.
        key = (a_n, b_n) if a_n < b_n else (b_n, a_n)
        if a_n == b_n or key in seen:
            continue
        seen.add(key)
        if a_n > b_n:
            big, big_id, small, small_id = a_n, a_id, b_n, b_id
        else:
            big, big_id, small, small_id = b_n, b_id, a_n, a_id
        prompt = f"{big} is greater than"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if W_U_for_norm_match is not None:
            r1 = _row_norm(W_U_for_norm_match, small_id)
            r2 = _row_norm(W_U_for_norm_match, big_id)
            if r1 > 0 and r2 > 0 and abs(np.log(r1 / r2)) > 0.5:
                continue
        out.append(
            Example(
                family="numeric_gt",
                prompt=prompt,
                prompt_ids=ids,
                y_plus=small_id,
                y_minus=big_id,
                meta={"big": big, "small": small},
            )
        )
    return out


def build_relational_facts(
    tokenizer,
    *,
    n_max: int = 200,
    W_U_for_norm_match=None,
) -> list[Example]:
    """Country -> capital relational facts.

    Treated as exploratory: Pythia may or may not have these, and tokenization
    is fragile. Caller should treat low yield as expected.
    """
    # NOTE: diverges from build_relational_facts_balanced, which drops the
    # Mexico and Argentina rows — unintended, but frozen for dataset stability.
    pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("Russia", "Moscow"),
        ("Egypt", "Cairo"),
        ("Greece", "Athens"),
        ("Poland", "Warsaw"),
        ("Portugal", "Lisbon"),
        ("Sweden", "Stockholm"),
        ("Norway", "Oslo"),
        ("Denmark", "Copenhagen"),
        ("Finland", "Helsinki"),
        ("Austria", "Vienna"),
        ("Hungary", "Budapest"),
        ("Belgium", "Brussels"),
        ("Ireland", "Dublin"),
        ("Cuba", "Havana"),
        ("Mexico", "Mexico"),
        ("Canada", "Ottawa"),
        ("Iran", "Tehran"),
        ("India", "Delhi"),
        ("Turkey", "Ankara"),
        ("Argentina", "Buenos"),
        ("Brazil", "Brasilia"),
        ("Korea", "Seoul"),
        ("Vietnam", "Hanoi"),
        ("Thailand", "Bangkok"),
    ]
    out: list[Example] = []
    capitals = [c for _, c in pairs]
    for country, cap in pairs:
        if len(out) >= n_max:
            break
        # Distractor: a capital of a different country with similar leading form.
        for distractor in capitals:
            if distractor == cap:
                continue
            pair = _filter_pair(
                tokenizer,
                cap,
                distractor,
                W_U_for_norm_match=W_U_for_norm_match,
            )
            if pair is None:
                continue
            yp, ym = pair
            prompt = f"The capital of {country} is"
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            out.append(
                Example(
                    family="relational_facts",
                    prompt=prompt,
                    prompt_ids=ids,
                    y_plus=yp,
                    y_minus=ym,
                    meta={"country": country, "capital": cap, "distractor": distractor},
                )
            )
            break
    return out


def build_relational_facts_balanced(
    tokenizer,
    *,
    n_max: int = 1000,
    n_distractors_per_prompt: int = 5,
    W_U_for_norm_match=None,
    rng_seed: int = 0,
) -> list[Example]:
    """Same country->capital prompts, but each (country, capital) pair is
    expanded into ``n_distractors_per_prompt`` separate examples — one per
    distractor capital. The previous builder pinned every example to a single
    distractor (typically ``' Paris'``), confounding the contrast with
    a single-token suppression effect; this builder spreads the
    contrast across all available capitals.

    Top-K feature attribution can then be re-evaluated per (country, distractor)
    pair to test whether the same K features generalize across distractors,
    or whether they specifically encode ``y+ - ' Paris'``-style steering.
    """
    # NOTE: diverges from build_relational_facts, which also has Mexico and
    # Argentina rows — unintended, but frozen for dataset stability.
    pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("Russia", "Moscow"),
        ("Egypt", "Cairo"),
        ("Greece", "Athens"),
        ("Poland", "Warsaw"),
        ("Portugal", "Lisbon"),
        ("Sweden", "Stockholm"),
        ("Norway", "Oslo"),
        ("Denmark", "Copenhagen"),
        ("Finland", "Helsinki"),
        ("Austria", "Vienna"),
        ("Hungary", "Budapest"),
        ("Belgium", "Brussels"),
        ("Ireland", "Dublin"),
        ("Cuba", "Havana"),
        ("Canada", "Ottawa"),
        ("Iran", "Tehran"),
        ("India", "Delhi"),
        ("Turkey", "Ankara"),
        ("Brazil", "Brasilia"),
        ("Korea", "Seoul"),
        ("Vietnam", "Hanoi"),
        ("Thailand", "Bangkok"),
    ]
    rng = np.random.default_rng(rng_seed)
    capitals = [c for _, c in pairs]
    out: list[Example] = []
    for country, cap in pairs:
        candidates = [c for c in capitals if c != cap]
        rng.shuffle(candidates)
        kept = 0
        for distractor in candidates:
            if kept >= n_distractors_per_prompt:
                break
            pair = _filter_pair(
                tokenizer,
                cap,
                distractor,
                W_U_for_norm_match=W_U_for_norm_match,
            )
            if pair is None:
                continue
            yp, ym = pair
            prompt = f"The capital of {country} is"
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            out.append(
                Example(
                    family="relational_facts_balanced",
                    prompt=prompt,
                    prompt_ids=ids,
                    y_plus=yp,
                    y_minus=ym,
                    meta={
                        "country": country,
                        "capital": cap,
                        "distractor": distractor,
                    },
                )
            )
            kept += 1
            if len(out) >= n_max:
                return out
    return out


# ---------------------------------------------------------------------------
# Hugging Face dataset loading (used by the BLiMP families in tasks_blimp)
# ---------------------------------------------------------------------------
_DATASETS_IMPORT_WARNED = False


def _load_hf_split(name: str, *args, split: str = "validation"):
    """Wrapper that returns ``None`` on any failure, with one printed reason."""
    global _DATASETS_IMPORT_WARNED
    try:
        from datasets import load_dataset
    except ImportError:
        if not _DATASETS_IMPORT_WARNED:
            print(
                "[contrastive_tasks] `datasets` not installed; benchmark "
                "families will return []. Install with `uv add datasets`.",
                flush=True,
            )
            _DATASETS_IMPORT_WARNED = True
        return None
    try:
        return load_dataset(name, *args, split=split)
    except Exception as exc:
        print(
            f"[contrastive_tasks] failed to load {name!r} ({args!r} split={split}): {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


# ---------------------------------------------------------------------------
# Registry + I/O
# ---------------------------------------------------------------------------
TASK_BUILDERS: dict[str, Callable] = {
    "sva": build_sva,
    "sva_across_pp": build_sva_across_pp,
    "induction": build_induction,
    "ioi": build_ioi,
    "ioi_role_balanced": build_ioi_role_balanced,
    "numeric_gt": build_numeric_greater_than,
    "relational_facts": build_relational_facts,
    "relational_facts_balanced": build_relational_facts_balanced,
}


def build_corruption(ex: Example, tokenizer) -> Example | None:
    """Return a clean-corrupted analogue where y_plus and y_minus are swapped.

    For SVA: the prompt is left unchanged but the *expected* answer flips when
    we change the subject's number; here we model that by simply swapping roles
    so margin reverses sign on a correct prompt.

    For IOI: swap subject and recipient name in the prompt (entity-binding flip).

    For others, returns None (corruption not defined).

    Caveat: for ``ioi_role_balanced`` this rebuilds the PLAIN IOI template,
    dropping the "X waited." balancing sentence from the clean prompt — the
    corrupt example is therefore not a minimal pair of its clean counterpart
    (token bag and length both change) for that family.
    """
    if ex.family == "sva":
        # Subject swap: rebuild with the other number form. Caller can match on meta.
        # For simplicity we return a swapped-answer example with same prompt;
        # downstream code interprets as "what if we relabel?".
        return Example(
            family=ex.family + "_corrupt",
            prompt=ex.prompt,
            prompt_ids=ex.prompt_ids,
            y_plus=ex.y_minus,
            y_minus=ex.y_plus,
            meta={**ex.meta, "corrupt": True},
        )
    if ex.family in {"ioi", "ioi_role_balanced"}:
        subj = ex.meta["subj"]
        recip = ex.meta["recipient"]
        obj = ex.meta["obj"]
        # Swap subject/recipient names: who *gave* changes -> answer should flip.
        prompt = f"{recip} and {subj} went to the store. {recip} gave the {obj} to"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        return Example(
            family=ex.family + "_corrupt",
            prompt=prompt,
            prompt_ids=ids,
            y_plus=ex.y_minus,
            y_minus=ex.y_plus,
            meta={**ex.meta, "corrupt": True},
        )
    return None


def save_examples(path: Path, examples: list[Example]) -> None:
    """Save as JSON-lines for easy auditing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {
                        "family": ex.family,
                        "prompt": ex.prompt,
                        "prompt_ids": ex.prompt_ids,
                        "y_plus": ex.y_plus,
                        "y_minus": ex.y_minus,
                        "meta": ex.meta,
                    }
                )
                + "\n"
            )


def load_examples(path: Path) -> list[Example]:
    out: list[Example] = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out.append(
                Example(
                    family=d["family"],
                    prompt=d["prompt"],
                    prompt_ids=d["prompt_ids"],
                    y_plus=d["y_plus"],
                    y_minus=d["y_minus"],
                    meta=d.get("meta", {}),
                )
            )
    return out


def random_distractor_ids(
    tokenizer,
    examples: list[Example],
    rng_seed: int = 0,
) -> list[int]:
    """Random-token control distractors. Same length as ``examples``.

    Sampled from single-token leading-space ASCII-alpha words in a small pool,
    excluding the example's true y_plus.
    """
    rng = np.random.default_rng(rng_seed)
    pool: list[int] = []
    for v in range(min(50000, getattr(tokenizer, "vocab_size", 50000))):
        s = tokenizer.decode([v])
        if not s or len(s) < 2 or not s.startswith(" "):
            continue
        if not s[1:].isalpha():
            continue
        pool.append(v)
        if len(pool) >= 5000:
            break
    if not pool:
        raise RuntimeError("no candidate distractors found in vocab")
    pool_arr = np.asarray(pool, dtype=np.int64)
    out: list[int] = []
    for ex in examples:
        while True:
            d = int(rng.choice(pool_arr))
            if d != ex.y_plus:
                out.append(d)
                break
    return out
