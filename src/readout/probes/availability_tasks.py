"""Task registry for the availability/expression panel (paper Section 4.3, Appendix G).

``AVAILABILITY_TASK_BUILDERS`` maps the 20 panel families to their builders, and
``build_availability_corruption`` is the panel's counterfactual pairing. Built with
``build_task_datasets.py --task-set availability --no-norm-match`` these reproduce
the published Pythia datasets byte for byte.

Where a family also exists in ``contrastive_tasks.TASK_BUILDERS`` the panel
version differs as follows (the swap-grid registry is left unchanged):

  - ``sva``: same items, plus ``meta`` ``noun`` / ``attractor`` / ``number`` (the
    probe label and lemma group key need them).
  - ``sva_across_pp``: PP-noun number drawn independently of the head and
    recorded (``tasks_sva_hierarchy``).
  - corruptions: ``sva`` swaps the head noun's number in the prompt (the
    swap-grid version only relabels), ``numeric_gt`` swaps candidate order, and
    IOI name swaps additionally record ``corruption_kind``.
"""

from __future__ import annotations

from typing import Callable

from readout.probes import contrastive_tasks as CT
from readout.probes.contrastive_tasks import Example
from readout.probes.tasks_blimp import build_blimp_anaphor_agreement, build_blimp_det_noun_agreement, build_blimp_npi
from readout.probes.tasks_controls import build_copy_control, build_fixed_token_control
from readout.probes.tasks_function_vectors import FV_TASK_BUILDERS
from readout.probes.tasks_numeric import (
    build_numeric_greater_than_v2,
    build_numeric_less_than_v2,
    build_numeric_truefalse_v2,
)
from readout.probes.tasks_sva_hierarchy import build_sva_across_pp_balanced, build_sva_object_rc, build_sva_subject_rc

# Plural -> singular for the plain-``sva`` nouns (contrastive_tasks._build_sva_pairs).
SVA_PL_TO_SG = {
    "keys": "key",
    "books": "book",
    "doors": "door",
    "cars": "car",
    "students": "student",
    "teachers": "teacher",
    "papers": "paper",
    "tables": "table",
    "windows": "window",
    "boxes": "box",
    "ships": "ship",
    "roads": "road",
    "leaves": "leaf",
    "letters": "letter",
    "bells": "bell",
    "clocks": "clock",
    "rooms": "room",
    "cups": "cup",
    "lamps": "lamp",
    "shoes": "shoe",
}
SVA_SG_TO_PL = {v: k for k, v in SVA_PL_TO_SG.items()}


def build_sva_with_meta(tokenizer, *, n_max: int = 2000, W_U_for_norm_match=None) -> list[Example]:
    """``contrastive_tasks.build_sva`` items with head noun, attractor and number in ``meta``."""
    out = CT.build_sva(tokenizer, n_max=n_max, W_U_for_norm_match=W_U_for_norm_match)
    for ex in out:
        _, noun, attractor = ex.prompt.split(" ", 2)  # "The {noun} {attractor}"
        ex.meta = {
            **ex.meta,
            "noun": noun,
            "attractor": attractor,
            "number": "plural" if ex.meta["pos"] == "are" else "singular",
        }
    return out


def build_availability_corruption(ex: Example, tokenizer) -> Example | None:
    """Counterfactual prompt for ``sva``, IOI and ``numeric_gt``; None for other families."""
    if ex.family == "sva":
        noun, attractor = ex.meta.get("noun"), ex.meta.get("attractor")
        if noun in SVA_PL_TO_SG:
            swapped, number = SVA_PL_TO_SG[noun], "singular"
        elif noun in SVA_SG_TO_PL:
            swapped, number = SVA_SG_TO_PL[noun], "plural"
        else:
            return None
        prompt = f"The {swapped} {attractor}"
        return Example(
            family=ex.family + "_corrupt",
            prompt=prompt,
            prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
            y_plus=ex.y_minus,
            y_minus=ex.y_plus,
            meta={**ex.meta, "corrupt": True, "noun": swapped, "number": number, "corruption_kind": "noun_number_swap"},
        )
    if ex.family in {"ioi", "ioi_role_balanced"}:
        c = CT.build_corruption(ex, tokenizer)
        c.meta = {**c.meta, "corruption_kind": "name_swap"}
        return c
    if ex.family.startswith("numeric_gt"):
        first, second = ex.meta.get("first"), ex.meta.get("second")
        if first is None or second is None:
            return None
        prompt = f"Which is greater, {second} or {first}? Answer:"
        return Example(
            family=ex.family + "_corrupt",
            prompt=prompt,
            prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
            y_plus=ex.y_plus,
            y_minus=ex.y_minus,
            meta={
                **ex.meta,
                "first": second,
                "second": first,
                "answer_position": "second" if ex.meta.get("answer_position") == "first" else "first",
                "corrupt": True,
                "corruption_kind": "candidate_position_swap",
            },
        )
    return None


AVAILABILITY_TASK_BUILDERS: dict[str, Callable] = {
    "sva": build_sva_with_meta,
    "sva_across_pp": build_sva_across_pp_balanced,
    "sva_subject_rc": build_sva_subject_rc,
    "sva_object_rc": build_sva_object_rc,
    "ioi_role_balanced": CT.build_ioi_role_balanced,
    "copy_control": build_copy_control,
    "fixed_token_control": build_fixed_token_control,
    "blimp_det_noun_agreement": build_blimp_det_noun_agreement,
    "blimp_anaphor_agreement": build_blimp_anaphor_agreement,
    "blimp_npi": build_blimp_npi,
    **FV_TASK_BUILDERS,
    "induction": CT.build_induction,
    "numeric_gt_v2": build_numeric_greater_than_v2,
    "numeric_lt_v2": build_numeric_less_than_v2,
    "numeric_tf_v2": build_numeric_truefalse_v2,
    "relational_facts": CT.build_relational_facts,
    "relational_facts_balanced": CT.build_relational_facts_balanced,
}

# Family groups as run for the paper (Pythia-1B split them across two runs).
AGREEMENT_FAMILIES = [
    "sva",
    "sva_across_pp",
    "sva_subject_rc",
    "sva_object_rc",
    "ioi_role_balanced",
    "copy_control",
    "fixed_token_control",
]
BREADTH_FAMILIES = [
    "blimp_det_noun_agreement",
    "blimp_anaphor_agreement",
    "blimp_npi",
    "fv_antonym",
    "fv_country_capital",
    "fv_en_fr",
    "fv_present_past",
    "induction",
    "numeric_gt_v2",
    "numeric_lt_v2",
    "numeric_tf_v2",
    "relational_facts",
    "relational_facts_balanced",
]
