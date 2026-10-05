"""Lemma-grouped WordNet probe splits: readout.probes.wu_probes_gpu group-aware folds and the
lemma_groups / split_stats helpers of run_wordnet_supersense_probe.py. No data, no GPU."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from readout.core.paths import repo_root
from readout.probes import wu_probes_gpu as wpg

SCRIPTS = repo_root() / "experiments" / "probes" / "concept_evolution_validation" / "scripts"


@pytest.fixture(scope="module")
def probe_script():
    pytest.importorskip("nltk")
    mp = pytest.MonkeyPatch()
    mp.syspath_prepend(str(SCRIPTS))
    try:
        yield importlib.import_module("run_wordnet_supersense_probe")
    finally:
        mp.undo()


def _toy(V=90, n_groups=30, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), V // n_groups)  # 3 variants per "lemma"
    rng.shuffle(groups)
    pos_groups_a = set(range(0, 12))
    pos_groups_b = set(range(12, 20))
    concepts = {
        "a": {i for i in range(V) if groups[i] in pos_groups_a},
        "b": {i for i in range(V) if groups[i] in pos_groups_b},
    }
    return V, groups, concepts


def test_grouped_folds_never_split_a_group():
    V, groups, concepts = _toy()
    folds = wpg.pack_concept_folds(V, concepts, n_folds=3, seed=0, groups=groups)
    inner_tr, inner_va, _, outer_tr, outer_te, _, y_mat, valid = folds
    assert valid.all()
    for ci in range(len(concepts)):
        tr, te = outer_tr[ci] > 0, outer_te[ci] > 0
        assert not (tr & te).any() and (tr | te).all()
        assert set(groups[tr]).isdisjoint(groups[te])
        assert abs(te.sum() / V - 1 / 3) < 0.1
        assert y_mat[ci][te].sum() > 0  # stratification keeps positives in test
        for fi in range(3):
            itr, iva = inner_tr[ci, fi] > 0, inner_va[ci, fi] > 0
            assert not (iva & te).any() and not (itr & te).any()
            assert set(groups[itr]).isdisjoint(groups[iva])


def test_row_folds_unchanged_without_groups():
    V, _, concepts = _toy()
    a = wpg.pack_concept_folds(V, concepts, n_folds=3, seed=0)
    b = wpg._pack_concept_folds(V, concepts, 3, 0)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_probe_accepts_precomputed_folds():
    V, groups, concepts = _toy()
    X = torch.randn(V, 5, generator=torch.Generator().manual_seed(0))
    folds = wpg.pack_concept_folds(V, concepts, seed=0, groups=groups)
    via_folds = wpg.probe_balanced_accuracy_batched(X, concepts, seed=0, device="cpu", max_iter=20, folds=folds)
    via_groups = wpg.probe_balanced_accuracy_batched(X, concepts, seed=0, device="cpu", max_iter=20, groups=groups)
    assert via_folds == via_groups


class _Tok:
    def __init__(self, pieces):
        self.pieces = pieces

    def __len__(self):
        return len(self.pieces)

    def decode(self, ids):
        return self.pieces[ids[0]]


def test_lemma_groups_and_leakage_stats(probe_script):
    tok = _Tok([" dog", "Dog", "DOG", " cat", "cat", " tree"])
    groups = probe_script.lemma_groups(tok, 8)  # 2 padded rows beyond the tokenizer
    assert groups[0] == groups[1] == groups[2] and groups[3] == groups[4]
    assert len({groups[0], groups[3], groups[5], groups[6], groups[7]}) == 5
    # Row-style split that puts " dog" in train and "Dog" in test: one leaked test positive.
    V = 8
    train = np.array([1, 0, 1, 1, 0, 1, 1, 0], dtype=np.float32)
    concepts = {"noun.animal": {0, 1, 2, 3, 4}}
    y = np.zeros((1, V), dtype=np.float32)
    y[0, [0, 1, 2, 3, 4]] = 1
    folds = (None, None, None, train[None], 1 - train[None], None, y, np.array([True]))
    (row,) = probe_script.split_stats(folds, concepts, groups)
    assert row["n_positive"] == 5 and row["n_positive_lemmas"] == 2
    assert row["n_test_positive"] == 2  # "Dog", "cat"
    assert row["n_test_pos_with_train_sibling"] == 2  # both have a same-lemma row in train
    assert row["n_pos_lemmas_split_across"] == 2
