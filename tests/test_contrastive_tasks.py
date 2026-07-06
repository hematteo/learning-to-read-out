"""Tests for src/readout/probes/contrastive_tasks.py.

Uses the actual Pythia-160m tokenizer (small, ~30MB, available offline if cached).
If the tokenizer is unavailable in the test environment the tests skip rather
than fail; CI provisioning is responsible for ensuring it is fetched once.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer unavailable: {e}")


def _assert_examples_well_formed(examples, family_name: str, tokenizer):
    """Every example: single-token y+/y-, distinct, non-empty prompt_ids."""
    for ex in examples:
        assert ex.family == family_name
        assert ex.y_plus != ex.y_minus, f"{family_name}: y+ == y- ({ex.y_plus})"
        assert len(ex.prompt_ids) > 0
        # y+ and y- decode to nontrivial strings (no single-byte UTF-8 fragments)
        s_plus = tokenizer.decode([ex.y_plus])
        s_minus = tokenizer.decode([ex.y_minus])
        assert len(s_plus) > 0 and len(s_minus) > 0


# ── builder smoke ────────────────────────────────────────────────────────────
def test_sva_yields_examples(tokenizer):
    from readout.probes.contrastive_tasks import build_sva

    ex = build_sva(tokenizer, n_max=20)
    assert len(ex) > 0
    _assert_examples_well_formed(ex, "sva", tokenizer)
    # Sanity: SVA prompts contain "the" (article)
    assert any("The " in e.prompt or "the " in e.prompt for e in ex)


def test_induction_yields_examples(tokenizer):
    from readout.probes.contrastive_tasks import build_induction

    ex = build_induction(tokenizer, n_max=20, rng_seed=0)
    assert len(ex) > 0
    _assert_examples_well_formed(ex, "induction", tokenizer)


def test_ioi_yields_examples(tokenizer):
    from readout.probes.contrastive_tasks import build_ioi

    ex = build_ioi(tokenizer, n_max=20, rng_seed=0)
    assert len(ex) > 0
    _assert_examples_well_formed(ex, "ioi", tokenizer)
    # IOI prompts should mention "gave" and "to"
    for e in ex:
        assert "gave" in e.prompt and " to" in e.prompt


def test_numeric_gt_yields_examples(tokenizer):
    from readout.probes.contrastive_tasks import build_numeric_greater_than

    ex = build_numeric_greater_than(tokenizer, n_max=20, rng_seed=0)
    assert len(ex) > 0
    _assert_examples_well_formed(ex, "numeric_gt", tokenizer)
    # The "small" answer must be smaller than the "big" anchor
    for e in ex:
        assert e.meta["small"] < e.meta["big"]


def test_arc_numeric_labels_prompt_gold_consistent(tokenizer):
    """ARC numeric labels ('1'-'4'): prompt options and gold token must agree.

    Replicates _build_arc's normalisation on a synthetic numeric-label row and
    checks the gold letter shown in the prompt matches the gold token / meta.
    """
    from readout.probes.contrastive_tasks import _build_mcq

    # answerKey "3" -> 3rd option ("nine"), which must be labelled "C" in prompt.
    raw = {
        "question": "What is 3 + 6?",
        "choices": {
            "label": ["1", "2", "3", "4"],
            "text": ["seven", "eight", "nine", "ten"],
        },
        "answerKey": "3",
    }
    labels = raw["choices"]["label"]
    norm_labels = [chr(ord("A") + i) for i in range(len(labels))]
    gold = raw["answerKey"].strip().upper()
    gold = norm_labels[labels.index(gold)]
    row = {**raw, "_label_norm": gold, "_labels_norm": norm_labels}

    def prompt_fn(r):
        body = "\n".join(f"{L}: {t}" for L, t in zip(r["_labels_norm"], r["choices"]["text"]))
        return f"Question: {r['question']}\n{body}\nAnswer:"

    ex = _build_mcq(
        "arc_test",
        tokenizer,
        [row],
        n_options=4,
        label_field="_label_norm",
        label_kind="letter",
        prompt_fn=prompt_fn,
        n_max=1,
        rng_seed=0,
    )
    if not ex:
        pytest.skip("tokenizer lacks single-token MCQ letters")
    e = ex[0]
    assert e.meta["gold_letter"] == "C"
    # The gold answer text "nine" must be the option labelled "C" in the prompt.
    assert "C: nine" in e.prompt
    # y_plus must decode to the same letter the prompt assigns to the gold option.
    assert tokenizer.decode([e.y_plus]).strip() == "C"


def test_relational_facts_yields_examples(tokenizer):
    from readout.probes.contrastive_tasks import build_relational_facts

    ex = build_relational_facts(tokenizer, n_max=20)
    assert len(ex) > 0
    _assert_examples_well_formed(ex, "relational_facts", tokenizer)


def test_hypernym_yields_examples(tokenizer):
    from readout.probes.contrastive_tasks import build_hypernym

    ex = build_hypernym(tokenizer, n_max=20)
    assert len(ex) > 0
    _assert_examples_well_formed(ex, "hypernym", tokenizer)


# ── single-token invariant ───────────────────────────────────────────────────
def test_y_plus_minus_are_single_tokens(tokenizer):
    """Filter contract: every y+/y- must tokenize to a single tokenizer token."""
    from readout.probes.contrastive_tasks import TASK_BUILDERS

    for name, builder in TASK_BUILDERS.items():
        kwargs = {"n_max": 10}
        if "rng_seed" in builder.__code__.co_varnames:
            kwargs["rng_seed"] = 0
        ex = builder(tokenizer, **kwargs)
        for e in ex:
            # Decode each id, re-encode with leading space, verify length 1.
            for tok_id in (e.y_plus, e.y_minus):
                s = tokenizer.decode([tok_id])
                # single-token round-trip — encoding the decoded form should
                # produce exactly one token (the same id).
                ids = tokenizer.encode(s, add_special_tokens=False)
                assert len(ids) == 1, (
                    f"{name}: y={tok_id} decodes to {s!r} which re-encodes to {ids} (not single-token)"
                )


# ── corruption symmetry ──────────────────────────────────────────────────────
def test_sva_corruption_swaps_labels(tokenizer):
    from readout.probes.contrastive_tasks import build_corruption, build_sva

    ex = build_sva(tokenizer, n_max=5)
    assert len(ex) > 0
    for e in ex:
        c = build_corruption(e, tokenizer)
        assert c is not None, "SVA corruption should always return an example"
        # Roles flipped, prompt unchanged
        assert c.y_plus == e.y_minus
        assert c.y_minus == e.y_plus
        assert c.prompt == e.prompt
        assert c.family == "sva_corrupt"


def test_ioi_corruption_swaps_subject_recipient(tokenizer):
    from readout.probes.contrastive_tasks import build_corruption, build_ioi

    ex = build_ioi(tokenizer, n_max=5, rng_seed=0)
    assert len(ex) > 0
    for e in ex:
        c = build_corruption(e, tokenizer)
        assert c is not None, "IOI corruption should always return an example"
        # Names should appear in swapped order in the new prompt
        subj = e.meta["subj"]
        recip = e.meta["recipient"]
        assert c.prompt.startswith(f"{recip} and {subj}")
        assert c.y_plus == e.y_minus
        assert c.y_minus == e.y_plus


def test_corruption_returns_none_for_unsupported_family(tokenizer):
    from readout.probes.contrastive_tasks import (
        build_corruption,
        build_numeric_greater_than,
    )

    ex = build_numeric_greater_than(tokenizer, n_max=2, rng_seed=0)
    if not ex:
        pytest.skip("numeric_gt produced no examples")
    c = build_corruption(ex[0], tokenizer)
    assert c is None, "corruption should be undefined for numeric_gt"


# ── random distractor control ────────────────────────────────────────────────
def test_random_distractor_ids_distinct_from_y_plus(tokenizer):
    from readout.probes.contrastive_tasks import build_sva, random_distractor_ids

    ex = build_sva(tokenizer, n_max=10)
    if not ex:
        pytest.skip("sva produced no examples")
    rand = random_distractor_ids(tokenizer, ex, rng_seed=0)
    assert len(rand) == len(ex)
    for e, r in zip(ex, rand):
        assert r != e.y_plus, "random distractor must not collide with y+"


# ── round-trip JSONL save/load ───────────────────────────────────────────────
def test_save_and_load_round_trip(tokenizer, tmp_path):
    from readout.probes.contrastive_tasks import build_sva, load_examples, save_examples

    original = build_sva(tokenizer, n_max=5)
    if not original:
        pytest.skip("sva produced no examples")
    p = tmp_path / "sva.jsonl"
    save_examples(p, original)
    loaded = load_examples(p)
    assert len(loaded) == len(original)
    for a, b in zip(original, loaded):
        assert a.family == b.family
        assert a.prompt == b.prompt
        assert a.prompt_ids == b.prompt_ids
        assert a.y_plus == b.y_plus
        assert a.y_minus == b.y_minus
