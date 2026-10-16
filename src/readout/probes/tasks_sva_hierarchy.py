"""Subject-verb agreement (SVA) hierarchy for the availability/expression panel.

Three agreement levels above plain ``sva`` (``contrastive_tasks.build_sva``), each
a single-token contrast between the head-agreeing verb form (``y_plus``) and the
attractor-agreeing form (``y_minus``):

  - ``sva_across_pp``:  "The {head} {prep} {pp_noun}"
  - ``sva_subject_rc``: "The {head} that {verb_embed} the {attractor}"
  - ``sva_object_rc``:  "The {head} that the {attractor} {verb_embed}"

In every level the attractor's grammatical number is drawn independently of the
head's (50/50; ``sva_subject_rc`` also has the earlier always-mismatched variant,
``balance_attractor=False``, used for the published Pythia-1B dataset), so the head number cannot be read off the attractor token at the
read position; about half the items are the hard number-disagreeing case, and
``meta["attractor_number"]`` records which. ``meta["head_lemma"]`` is the
number-invariant group key used for held-out-lemma probe splits, and
``meta["head_number"]`` is the probed feature.

``sva_across_pp`` here is the panel version: it differs from
``readout.probes.sva.generate_sva_across_pp`` (used by the swap-grid and
attribution experiments, whose PP noun number is unconstrained and unrecorded) in
drawing the PP-noun number explicitly and recording it. The relative-clause levels
follow Marvin & Linzen (2018).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from readout.probes.contrastive_tasks import Example, _filter_pair
from readout.probes.sva import PP_PREP, SUBJECT_NOUNS_PL, SUBJECT_NOUNS_SG, VERB_PAIRS

# Row-norm tolerance (|log ratio|) applied when a terminal W_U is passed.
NORM_TOL = 0.2

# ``readout.probes.sva.PP_NOUN_PHRASES`` split by grammatical number (same order;
# SG + PL == PP_NOUN_PHRASES).
PP_NOUN_PHRASES_SG = [
    "the table",
    "the window",
    "the door",
    "the garden",
    "the river",
    "the school",
    "the office",
    "the street",
    "the building",
    "the kitchen",
    "the museum",
    "the library",
    "the bridge",
    "the entrance",
    "the fountain",
]
PP_NOUN_PHRASES_PL = [
    "the children",
    "the parents",
    "the teachers",
    "the lawyers",
    "the actors",
    "the engineers",
    "the dancers",
    "the artists",
]


@dataclass
class AcrossPPItem:
    clean_text: str  # singular-head prompt
    corrupted_text: str  # plural-head prompt (independent PP noun)
    verb_sg: str  # leading-space singular verb, e.g. " is"
    verb_pl: str
    head_lemma: str  # number-invariant head key (singular form)
    attr_num_clean: str  # PP-noun number in the singular-head prompt
    attr_num_corr: str  # PP-noun number in the plural-head prompt


def generate_sva_across_pp_balanced(n: int = 1000, seed: int = 0) -> list[AcrossPPItem]:
    """Across-PP pairs with the PP-noun number drawn independently per member."""
    rng = random.Random(seed)
    items: list[AcrossPPItem] = []
    while len(items) < n:
        i = rng.randrange(len(SUBJECT_NOUNS_SG))
        prep = rng.choice(PP_PREP)
        verb_sg, verb_pl = rng.choice(VERB_PAIRS)
        sg_subj = SUBJECT_NOUNS_SG[i]
        pl_subj = SUBJECT_NOUNS_PL[i]
        an_c = "singular" if rng.random() < 0.5 else "plural"
        an_k = "singular" if rng.random() < 0.5 else "plural"
        pp_c = rng.choice(PP_NOUN_PHRASES_SG if an_c == "singular" else PP_NOUN_PHRASES_PL)
        pp_k = rng.choice(PP_NOUN_PHRASES_SG if an_k == "singular" else PP_NOUN_PHRASES_PL)
        # Prompt ends right before the verb, no period.
        items.append(
            AcrossPPItem(
                clean_text=f"The {sg_subj} {prep} {pp_c}",
                corrupted_text=f"The {pl_subj} {prep} {pp_k}",
                verb_sg=" " + verb_sg,
                verb_pl=" " + verb_pl,
                head_lemma=sg_subj,
                attr_num_clean=an_c,
                attr_num_corr=an_k,
            )
        )
    return items


def build_sva_across_pp_balanced(tokenizer, *, n_max: int = 2000, W_U_for_norm_match=None) -> list[Example]:
    """``sva_across_pp`` with independent PP-noun number (seed fixed at 0)."""
    out: list[Example] = []
    for item in generate_sva_across_pp_balanced(n=max(n_max, 1000), seed=0):
        pair = _filter_pair(
            tokenizer,
            item.verb_sg.strip(),
            item.verb_pl.strip(),
            W_U_for_norm_match=W_U_for_norm_match,
            norm_tol=NORM_TOL,
        )
        if pair is None:
            continue
        verb_sg_id, verb_pl_id = pair
        for prompt, pos_id, neg_id, number, attractor_number in [
            (item.clean_text, verb_sg_id, verb_pl_id, "singular", item.attr_num_clean),
            (item.corrupted_text, verb_pl_id, verb_sg_id, "plural", item.attr_num_corr),
        ]:
            out.append(
                Example(
                    family="sva_across_pp",
                    prompt=prompt,
                    prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
                    y_plus=pos_id,
                    y_minus=neg_id,
                    meta={
                        "number": number,
                        "head_number": number,
                        "attractor_number": attractor_number,
                        "head_lemma": item.head_lemma,
                        "verb_sg": item.verb_sg.strip(),
                        "verb_pl": item.verb_pl.strip(),
                    },
                )
            )
            if len(out) >= n_max:
                return out
    return out


# ---------------------------------------------------------------------------
# Relative clauses
# ---------------------------------------------------------------------------
# Animate head/attractor nouns, index-aligned singular/plural.
RC_NOUNS_SG = [
    "author",
    "senator",
    "officer",
    "farmer",
    "pilot",
    "teacher",
    "student",
    "doctor",
    "lawyer",
    "painter",
    "dancer",
    "singer",
    "skater",
    "banker",
    "driver",
    "soldier",
    "writer",
    "reader",
    "runner",
    "swimmer",
]
RC_NOUNS_PL = [n + "s" for n in RC_NOUNS_SG]
_RC_PL_TO_SG = dict(zip(RC_NOUNS_PL, RC_NOUNS_SG))

# Embedded-clause verbs (in the prompt, not the contrast), index-aligned 3sg / bare.
EMBED_VERBS_SG = ["admires", "watches", "knows", "likes", "hates", "follows"]
EMBED_VERBS_PL = ["admire", "watch", "know", "like", "hate", "follow"]

# Main-verb contrast pairs (3sg, bare); y_plus agrees with the head.
MAIN_VERB_PAIRS = [
    ("likes", "like"),
    ("runs", "run"),
    ("knows", "know"),
    ("swims", "swim"),
    ("writes", "write"),
    ("reads", "read"),
]


def _subject_rc_prompt(*, head_surface, attractor_surface, attractor_number, embed_sg, embed_pl) -> str:
    # The head is the embedded verb's subject, so the embedded verb agrees with it.
    embed = embed_pl if head_surface in _RC_PL_TO_SG else embed_sg
    return f"The {head_surface} that {embed} the {attractor_surface}"


def _object_rc_prompt(*, head_surface, attractor_surface, attractor_number, embed_sg, embed_pl) -> str:
    # The attractor is the embedded verb's subject, so the embedded verb agrees with it.
    embed = embed_pl if attractor_number == "plural" else embed_sg
    return f"The {head_surface} that the {attractor_surface} {embed}"


def _build_rc(
    family: str,
    rc_type: str,
    prompt_fn,
    tokenizer,
    *,
    n_max: int,
    rng_seed: int,
    W_U_for_norm_match,
    balance_attractor: bool = True,
):
    """Cross head lemma x main-verb pair x embedded verb, both head numbers per cell.

    The attractor is the next lemma in ``RC_NOUNS_SG`` (cyclic). With
    ``balance_attractor`` its number is an independent fair draw for each arm;
    otherwise it always mismatches the head (no draws).
    """
    verb_ids: dict[tuple[str, str], tuple[int, int]] = {}
    for sg_form, pl_form in MAIN_VERB_PAIRS:
        pair = _filter_pair(tokenizer, sg_form, pl_form, W_U_for_norm_match=W_U_for_norm_match, norm_tol=NORM_TOL)
        if pair is not None:
            verb_ids[(sg_form, pl_form)] = pair
    if not verb_ids:
        return []

    rng = np.random.default_rng(rng_seed)
    # Drawn (and unused) so the attractor-number draws below keep their stream.
    rng.permutation(len(EMBED_VERBS_SG))

    out: list[Example] = []
    for i, (head_sg, head_pl) in enumerate(zip(RC_NOUNS_SG, RC_NOUNS_PL)):
        attr_j = (i + 1) % len(RC_NOUNS_SG)
        attr_sg, attr_pl = RC_NOUNS_SG[attr_j], RC_NOUNS_PL[attr_j]
        for (sg_form, pl_form), (sg_id, pl_id) in verb_ids.items():
            for embed_sg, embed_pl in zip(EMBED_VERBS_SG, EMBED_VERBS_PL):
                if balance_attractor:
                    an_sg = "singular" if rng.integers(2) else "plural"
                    an_pl = "singular" if rng.integers(2) else "plural"
                else:
                    an_sg, an_pl = "plural", "singular"
                for head_surface, head_number, an, y_plus, y_minus in [
                    (head_sg, "singular", an_sg, sg_id, pl_id),
                    (head_pl, "plural", an_pl, pl_id, sg_id),
                ]:
                    attractor_surface = attr_sg if an == "singular" else attr_pl
                    prompt = prompt_fn(
                        head_surface=head_surface,
                        attractor_surface=attractor_surface,
                        attractor_number=an,
                        embed_sg=embed_sg,
                        embed_pl=embed_pl,
                    )
                    out.append(
                        Example(
                            family=family,
                            prompt=prompt,
                            prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
                            y_plus=y_plus,
                            y_minus=y_minus,
                            meta={
                                "rc_type": rc_type,
                                "head_lemma": head_sg,
                                "head_surface": head_surface,
                                "head_number": head_number,
                                "attractor_lemma": attr_sg,
                                "attractor_surface": attractor_surface,
                                "attractor_number": an,
                                "verb_sg": sg_form,
                                "verb_pl": pl_form,
                                "number": head_number,
                            },
                        )
                    )
                    if len(out) >= n_max:
                        return out
    return out


def build_sva_subject_rc(
    tokenizer, *, n_max: int = 2000, rng_seed: int = 0, W_U_for_norm_match=None, balance_attractor: bool = True
) -> list[Example]:
    """Subject relative clause: "The {head} that {verb_embed} the {attractor}"."""
    return _build_rc(
        "sva_subject_rc",
        "subject_rc",
        _subject_rc_prompt,
        tokenizer,
        n_max=n_max,
        rng_seed=rng_seed,
        W_U_for_norm_match=W_U_for_norm_match,
        balance_attractor=balance_attractor,
    )


def build_sva_object_rc(tokenizer, *, n_max: int = 2000, rng_seed: int = 0, W_U_for_norm_match=None) -> list[Example]:
    """Object relative clause: "The {head} that the {attractor} {verb_embed}"."""
    return _build_rc(
        "sva_object_rc",
        "object_rc",
        _object_rc_prompt,
        tokenizer,
        n_max=n_max,
        rng_seed=rng_seed,
        W_U_for_norm_match=W_U_for_norm_match,
    )
