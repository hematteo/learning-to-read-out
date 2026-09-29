"""readout.probes.availability_probe: separable data clears the nulls; infeasible cells skip."""

from __future__ import annotations

import math

import numpy as np

from readout.probes.availability_probe import availability_probe, bootstrap_ci


def _blobs(n_per_class=60, d=5, sep=4.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.normal(0, 1, (n_per_class, d)), rng.normal(sep, 1, (n_per_class, d))])
    y = np.repeat([0, 1], n_per_class)
    return X, y


def test_separable_target_beats_both_nulls():
    X, y = _blobs()
    out = availability_probe(X, y, n_splits=5, seed=0)
    assert out["skip_reason"] == "" and out["n"] == len(y) and out["n_classes"] == 2
    assert out["probe_acc"] > 0.95
    assert out["probe_acc_label_shuffle"] < 0.75
    assert out["probe_acc_random_label"] < 0.75
    assert out["probe_acc_ci_lo"] <= out["probe_acc"] <= out["probe_acc_ci_hi"]
    assert len(out["fold_acc"]) == 5


def test_deterministic_given_seed():
    X, y = _blobs(sep=1.0, seed=3)
    a = availability_probe(X, y, seed=7)
    b = availability_probe(X, y, seed=7)
    assert a["probe_acc"] == b["probe_acc"] and a["fold_acc"] == b["fold_acc"]
    assert a["probe_acc_label_shuffle"] == b["probe_acc_label_shuffle"]


def test_skips_when_infeasible():
    X, y = _blobs()
    one_class = availability_probe(X, np.zeros_like(y))
    assert math.isnan(one_class["probe_acc"]) and "fewer than 2" in one_class["skip_reason"]
    y_rare = y.copy()
    y_rare[:] = 0
    y_rare[:3] = 1  # 3 members < n_splits
    rare = availability_probe(X, y_rare, n_splits=5)
    assert math.isnan(rare["probe_acc"]) and "StratifiedKFold" in rare["skip_reason"]


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([0.5, 0.6, 0.7, 0.8], seed=0)
    assert lo <= 0.65 <= hi
    assert all(math.isnan(v) for v in bootstrap_ci([0.5]))
