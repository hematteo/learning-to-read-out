"""Availability-panel task builders (readout.probes.availability_tasks and siblings).

Uses the Pythia tokenizer (skips offline, as test_contrastive_tasks.py). No model,
no network data: BLiMP is exercised through its pair reducer and the
function-vector builder through a synthetic JSON file.
"""

from __future__ import annotations

import collections
import json

import pytest

pytest.importorskip("transformers")

from readout.probes import availability_tasks as AT
from readout.probes import sva as SVA
from readout.probes import tasks_sva_hierarchy as H
from readout.probes.availability_expression import derive_group_key, feature_label
from readout.probes.contrastive_tasks import save_examples


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer unavailable: {e}")


def _as_dict(ex) -> dict:
    return {"family": ex.family, "prompt": ex.prompt, "meta": ex.meta}


def _check_well_formed(examples, tokenizer):
    for ex in examples:
        assert ex.y_plus != ex.y_minus
        assert ex.prompt_ids == tokenizer.encode(ex.prompt, add_special_tokens=False)
        for tid in (ex.y_plus, ex.y_minus):
            assert tokenizer.encode(tokenizer.decode([tid]), add_special_tokens=False) == [tid]


def test_pp_noun_split_matches_shared_list():
    assert H.PP_NOUN_PHRASES_SG + H.PP_NOUN_PHRASES_PL == SVA.PP_NOUN_PHRASES
    assert H.RC_NOUNS_PL == [n + "s" for n in H.RC_NOUNS_SG]


@pytest.mark.parametrize("builder", [H.build_sva_subject_rc, H.build_sva_object_rc])
def test_relative_clause_items_well_formed_and_balanced(builder, tokenizer):
    ex = builder(tokenizer, n_max=2000)
    assert len(ex) == 1200  # 20 lemmas x 5 single-token main-verb pairs x 6 embedded verbs x 2 numbers
    _check_well_formed(ex, tokenizer)
    heads = collections.Counter(e.meta["head_number"] for e in ex)
    assert heads["singular"] == heads["plural"] == 600
    # Attractor number is independent of head number: all four cells populated, near 50/50.
    cells = collections.Counter((e.meta["head_number"], e.meta["attractor_number"]) for e in ex)
    assert len(cells) == 4 and min(cells.values()) > 250
    for e in ex:
        m = e.meta
        assert m["attractor_lemma"] != m["head_lemma"]
        assert m["head_surface"] == (m["head_lemma"] if m["head_number"] == "singular" else m["head_lemma"] + "s")
        # y_plus is the head-agreeing main verb (3sg form for a singular head).
        agree, other = (m["verb_sg"], m["verb_pl"]) if m["head_number"] == "singular" else (m["verb_pl"], m["verb_sg"])
        assert tokenizer.decode([e.y_plus]).strip() == agree and tokenizer.decode([e.y_minus]).strip() == other
        assert feature_label(_as_dict(e)) == m["head_number"]
        assert derive_group_key(_as_dict(e)) == m["head_lemma"]
    words = ex[0].prompt.split()
    if builder is H.build_sva_subject_rc:
        # "The {head} that {embed} the {attractor}": embedded verb agrees with the head.
        assert words[3] in (H.EMBED_VERBS_SG if ex[0].meta["head_number"] == "singular" else H.EMBED_VERBS_PL)
    else:
        # "The {head} that the {attractor} {embed}": embedded verb agrees with the attractor.
        ok = H.EMBED_VERBS_SG if ex[0].meta["attractor_number"] == "singular" else H.EMBED_VERBS_PL
        assert words[-1] in ok


def test_subject_rc_mismatched_variant(tokenizer):
    ex = H.build_sva_subject_rc(tokenizer, balance_attractor=False)
    assert all(e.meta["attractor_number"] != e.meta["head_number"] for e in ex)


def test_across_pp_balanced(tokenizer):
    ex = H.build_sva_across_pp_balanced(tokenizer, n_max=2000)
    assert len(ex) == 2000
    _check_well_formed(ex, tokenizer)
    for clean, corr in zip(ex[::2], ex[1::2]):  # singular / plural members of one item
        assert clean.meta["head_number"] == "singular" and corr.meta["head_number"] == "plural"
        assert clean.meta["head_lemma"] == corr.meta["head_lemma"]
        assert (clean.y_plus, clean.y_minus) == (corr.y_minus, corr.y_plus)
        for e in (clean, corr):
            pp = " ".join(e.prompt.split()[-2:])
            assert pp in (H.PP_NOUN_PHRASES_SG if e.meta["attractor_number"] == "singular" else H.PP_NOUN_PHRASES_PL)
    agree = sum(e.meta["attractor_number"] == e.meta["head_number"] for e in ex) / len(ex)
    assert 0.4 < agree < 0.6


def test_sva_meta_and_counterfactual_corruption(tokenizer):
    ex = AT.build_sva_with_meta(tokenizer)
    assert len(ex) == 320
    for e in ex:
        assert e.prompt == f"The {e.meta['noun']} {e.meta['attractor']}"
        assert e.meta["number"] == ("plural" if e.meta["pos"] == "are" else "singular")
        c = AT.build_availability_corruption(e, tokenizer)
        assert c.family == "sva_corrupt" and c.prompt != e.prompt
        assert (c.y_plus, c.y_minus) == (e.y_minus, e.y_plus) and c.meta["number"] != e.meta["number"]


def test_numeric_v2_balanced(tokenizer):
    from readout.probes.tasks_numeric import (
        build_numeric_greater_than_v2,
        build_numeric_less_than_v2,
        build_numeric_truefalse_v2,
    )

    for builder, fam in [(build_numeric_greater_than_v2, "numeric_gt"), (build_numeric_less_than_v2, "numeric_lt")]:
        ex = builder(tokenizer, n_max=400)
        assert len(ex) == 400
        _check_well_formed(ex, tokenizer)
        pos = collections.Counter(e.meta["answer_position"] for e in ex)
        assert pos["first"] == pos["second"] == 200
        for e in ex:
            assert e.family == fam
            ans = e.meta["big"] if fam == "numeric_gt" else e.meta["small"]
            assert tokenizer.decode([e.y_plus]).strip() == str(ans)
            assert (e.meta["first"] == ans) == (e.meta["answer_position"] == "first")
        pairs = collections.Counter((e.meta["small"], e.meta["big"]) for e in ex)
        assert set(pairs.values()) == {2}  # each unordered pair once, in both orders
    tf = build_numeric_truefalse_v2(tokenizer, n_max=400)
    assert collections.Counter(e.meta["label"] for e in tf) == {"true": 200, "false": 200}
    gt = build_numeric_greater_than_v2(tokenizer, n_max=10)
    c = AT.build_availability_corruption(gt[0], tokenizer)
    assert c.meta["answer_position"] != gt[0].meta["answer_position"] and c.y_plus == gt[0].y_plus


def test_controls(tokenizer):
    from readout.probes.tasks_controls import build_copy_control, build_fixed_token_control

    copy = build_copy_control(tokenizer, n_max=200)
    assert len(copy) == 200
    for e in copy:
        assert e.prompt_ids[-1] == e.y_plus and e.y_minus not in e.prompt_ids
        assert feature_label(_as_dict(e)) is None
    fixed = build_fixed_token_control(tokenizer, n_max=200)
    assert len({(e.y_plus, e.y_minus) for e in fixed}) == 1 and len({e.prompt for e in fixed}) == 200


def test_blimp_pair_reduction(tokenizer):
    from readout.probes.tasks_blimp import reduce_pair_to_contrast

    prefix, pos, yp, ym = reduce_pair_to_contrast(
        tokenizer, "Many girls insulted these men.", "Many girls insulted this men."
    )
    assert tokenizer.decode(prefix) == "Many girls insulted"
    assert tokenizer.decode([yp]) == " these" and tokenizer.decode([ym]) == " this"
    assert reduce_pair_to_contrast(tokenizer, "A dog ran.", "A dog ran quickly away.") is None  # no interior diff
    assert reduce_pair_to_contrast(tokenizer, "The cat sat.", "A dog ran.") is None  # tails differ


def test_function_vector_builder(tokenizer, tmp_path, monkeypatch):
    from readout.probes.tasks_function_vectors import FV_TASK_BUILDERS

    words = ["hot", "cold", "big", "small", "up", "down", "fast", "slow", "good", "bad", "old", "new", "high", "low"]
    rows = [{"input": a, "output": b} for a, b in zip(words[::2], words[1::2])]
    rows += [{"input": b, "output": a} for a, b in zip(words[::2], words[1::2])]
    (tmp_path / "antonym.json").write_text(json.dumps(rows))
    monkeypatch.setenv("FV_DATA_DIR", str(tmp_path))
    ex = FV_TASK_BUILDERS["fv_antonym"](tokenizer, n_max=100)
    assert len(ex) == len(rows)
    for e in ex:
        assert tokenizer.decode([e.y_plus]).strip() == e.meta["output"]
        assert e.meta["output"] != e.meta["distractor_output"]
        assert e.meta["query_index"] not in e.meta["demo_indices"] and len(e.meta["demo_indices"]) == 10
        assert e.prompt.endswith("\n" + e.meta["input"] + ":")
        assert derive_group_key(_as_dict(e)) == e.meta["input"]
    monkeypatch.setenv("FV_DATA_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        FV_TASK_BUILDERS["fv_antonym"](tokenizer)


def test_registry_is_deterministic(tokenizer, tmp_path):
    fams = ["sva", "sva_across_pp", "sva_subject_rc", "sva_object_rc", "numeric_gt_v2", "copy_control"]
    assert set(fams) <= set(AT.AVAILABILITY_TASK_BUILDERS)
    assert set(AT.AGREEMENT_FAMILIES + AT.BREADTH_FAMILIES) == set(AT.AVAILABILITY_TASK_BUILDERS)
    for fam in fams:
        a, b = tmp_path / f"{fam}_a.jsonl", tmp_path / f"{fam}_b.jsonl"
        save_examples(a, AT.AVAILABILITY_TASK_BUILDERS[fam](tokenizer, n_max=300))
        save_examples(b, AT.AVAILABILITY_TASK_BUILDERS[fam](tokenizer, n_max=300))
        assert a.read_bytes() == b.read_bytes()
