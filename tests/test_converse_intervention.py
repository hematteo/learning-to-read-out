"""run_converse_intervention.py (contrastive_task_feature_rescue) on synthetic tensors.

Loaded with the experiment's ``scripts/`` dir on ``sys.path`` (it imports the
sibling ``run_feature_attribution`` for the per-family split RNG), as in
``test_recipe_control_lifecycle.py``. No data, no GPU.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from readout.core.paths import repo_root

SCRIPTS = repo_root() / "experiments" / "causal" / "contrastive_task_feature_rescue" / "scripts"


@pytest.fixture(scope="module")
def conv():
    mp = pytest.MonkeyPatch()
    mp.syspath_prepend(str(SCRIPTS))
    try:
        yield importlib.import_module("run_converse_intervention")
    finally:
        mp.undo()


def test_project_out_removes_span_only(conv):
    g = torch.Generator().manual_seed(0)
    h = torch.randn(7, 5, generator=g)  # (N, d)
    D = torch.randn(2, 5, generator=g)  # (K, d)
    hp = conv.project_out(h, D)
    torch.testing.assert_close(hp @ D.T, torch.zeros(7, 2), atol=1e-5, rtol=0)
    torch.testing.assert_close(conv.project_out(hp, D), hp, atol=1e-5, rtol=1e-5)  # idempotent
    # The removed part lies in span(D): h - hp = c @ D for some c.
    c = torch.linalg.lstsq(D.T, (h - hp).T).solution  # (K, N)
    torch.testing.assert_close(c.T @ D, h - hp, atol=1e-4, rtol=1e-4)


def test_held_out_indices_match_attribution_split(conv):
    fam, n_total, n_attr = "sva", 20, 10
    split = {"n_total": n_total, "n_attr": n_attr, "n_eval": 10, "use_split": True, "split_seed": 0}
    idx_b = conv.held_out_indices(fam, split)
    # Same derivation as run_feature_attribution.py's per-family split.
    perm = conv.stable_rng(0, "split", fam).permutation(n_total)
    np.testing.assert_array_equal(idx_b, np.sort(perm[n_attr:]))
    assert set(idx_b).isdisjoint(perm[:n_attr])
    # Different family -> different (independent) split.
    assert not np.array_equal(conv.held_out_indices("ioi", split), idx_b)
    fallback = dict(split, use_split=False, n_attr=n_total, n_eval=n_total)
    np.testing.assert_array_equal(conv.held_out_indices(fam, fallback), np.arange(n_total))
    with pytest.raises(ValueError):
        conv.held_out_indices(fam, dict(split, n_eval=9))


def test_converse_family_orthogonal_directions_leave_margins(conv):
    g = torch.Generator().manual_seed(1)
    d, V, N = 6, 9, 8
    W = torch.randn(V, d, generator=g)
    h = torch.randn(N, d, generator=g)
    h[:, 4:] = 0.0  # hidden states live in the first four coordinates
    yp = torch.randint(0, V, (N,), generator=g)
    ym = (yp + 1) % V
    D_top = torch.zeros(2, d)
    D_top[0, 4], D_top[1, 5] = 1.0, 1.0  # directions orthogonal to every h
    out = conv.converse_family(h, yp, ym, D_top, [1, 2, 8], W_native_t=W, W_native_s=W)
    assert sorted(out["per_K"]) == [1, 2]  # K is capped at the number of saved rows
    base = out["baseline"]["native_t"]
    for rec in out["per_K"].values():
        assert rec["summary_proj_native_t"]["accuracy"] == base["accuracy"]
        assert rec["logit_proj_native_t"]["dmargin"] == pytest.approx(0.0, abs=1e-5)
        assert rec["h_residual_frac"] == pytest.approx(1.0)


def test_converse_family_removing_the_answer_direction(conv):
    d, V = 3, 2
    W = torch.eye(V, d)  # y=0 reads coordinate 0, y=1 reads coordinate 1
    h = torch.tensor([[2.0, 1.0, 0.5], [3.0, 0.5, 1.0]])
    yp, ym = torch.tensor([0, 0]), torch.tensor([1, 1])
    out = conv.converse_family(h, yp, ym, torch.tensor([[1.0, 0.0, 0.0]]), [1], W_native_t=W, W_native_s=W)
    assert out["baseline"]["native_t"]["accuracy"] == 1.0
    rec = out["per_K"][1]
    assert rec["summary_proj_native_t"]["accuracy"] == 0.0  # y+ logit removed, y- untouched
    assert rec["logit_proj_native_t"]["dy_plus"] == pytest.approx(-2.5)
    assert rec["logit_proj_native_t"]["dy_minus"] == pytest.approx(0.0)
