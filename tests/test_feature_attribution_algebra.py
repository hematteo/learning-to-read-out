"""Tests for the per-feature attribution and ablate/preserve algebra in
experiments/causal/contrastive_task_feature_rescue/scripts/run_feature_attribution.py.

These tests build a tiny synthetic crosscoder state (no real model required)
and verify two invariants:

  1. The ablate / preserve interventions partition the reconstructed margin:
       mean_margin(ablate(top)) + mean_margin(preserve(top)) ==
       mean_margin(recon) + mean_margin(bias_only_W)

     where `bias_only_W` is the constant baseline `b_D * scale + mean`
     contributed by both interventions. (This baseline cancels in the
     contrastive margin, so the identity simplifies to
     mean_margin(ablate) + mean_margin(preserve) == mean_margin(recon)
     for the contrastive case.)

  2. per_feature_attribution computed for ALL features sums to the
     reconstructed margin (modulo the bias, which cancels in the contrast).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_attribution_module():
    repo = Path(__file__).resolve().parents[1]
    p = repo / "experiments/causal/contrastive_task_feature_rescue/scripts/run_feature_attribution.py"
    spec = importlib.util.spec_from_file_location("rfa", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["rfa"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rfa():
    return _load_attribution_module()


def _toy_crosscoder(V=12, d=4, D=6, K_snap=3, seed=0):
    """Construct a synthetic crosscoder state_dict + accumulated preact."""
    torch.manual_seed(seed)
    sd = {
        "W_E": torch.randn(K_snap, d, D) * 0.3,
        "b_E": torch.randn(K_snap, D) * 0.1,
        "W_D": torch.randn(K_snap, D, d) * 0.4,
        "b_D": torch.randn(K_snap, d) * 0.05,
        # JumpReLU threshold; small enough that some features fire.
        "activation_function.log_jumprelu_threshold": torch.full((D,), float(np.log(0.05))),
    }
    # Hand-roll an "encoder preactivation" tensor (V, D) — the joint encoder
    # output before JumpReLU, as produced by the main script's accumulation.
    preact = torch.randn(V, D) * 0.5
    return sd, preact


def test_ablate_preserve_partition_recon_margin(rfa):
    """For any subset S, mean_margin(ablate_S) + mean_margin(preserve_S) ==
    mean_margin(recon) when evaluated on the contrastive margin.

    Why: the bias and mean baselines (which appear in both `ablate` and
    `preserve`) cancel in the contrast h·(W[y+] - W[y-]).
    """
    V, d, D, K_snap = 16, 5, 8, 3
    sd, preact = _toy_crosscoder(V=V, d=d, D=D, K_snap=K_snap, seed=1)
    step_idx = 1

    # Build the per-step feature_acts and a "stats" payload with scalar scale.
    feature_acts, _ = rfa.encode_at_step(sd, preact, step_idx, device="cpu")
    stats_step = {
        "scale": torch.tensor([[2.0]]),  # per-snapshot scalar (center_scale)
        "mean": torch.zeros(d),
    }
    W_recon = rfa.reconstruct_W(feature_acts, sd, step_idx, stats_step, device="cpu")

    # Pick an arbitrary subset to ablate / preserve.
    rng = np.random.default_rng(7)
    K_sub = 3
    subset = rng.choice(D, size=K_sub, replace=False)
    subset_t = torch.from_numpy(subset).long()
    contrib_sub = rfa.feature_contribution(feature_acts, sd, step_idx, stats_step, subset_t, device="cpu")
    complement = torch.from_numpy(np.setdiff1d(np.arange(D), subset)).long()
    contrib_comp = rfa.feature_contribution(feature_acts, sd, step_idx, stats_step, complement, device="cpu")

    # W_ablate keeps {bias, complement}; W_preserve keeps {bias, subset}.
    W_ablate = W_recon - contrib_sub
    W_preserve = W_recon - contrib_comp

    # Random h, contrastive y+/y-.
    torch.manual_seed(42)
    N = 20
    h = torch.randn(N, d)
    y_plus = torch.randint(0, V, (N,))
    y_minus = (y_plus + torch.randint(1, V, (N,))) % V  # different from y+

    m_recon = rfa.margins_with_W(h, W_recon, y_plus, y_minus)
    m_ab = rfa.margins_with_W(h, W_ablate, y_plus, y_minus)
    m_pres = rfa.margins_with_W(h, W_preserve, y_plus, y_minus)

    # The "double-counted bias" identity:
    #   ablate + preserve = recon + (constant baseline contribution to the
    #   margin) where the constant baseline is the same in both interventions.
    # In the contrastive margin the constant baseline cancels, so:
    #   ablate_margin + preserve_margin == recon_margin    + 0
    # but each of those margins ALSO has the bias subtracted twice. Concretely:
    #   recon            = bias + sum_f contrib_f          (W_recon column form)
    #   ablate(S)        = bias + sum_{f notin S} contrib_f
    #   preserve(S)      = bias + sum_{f in S} contrib_f
    #   ablate + preserve = 2*bias + recon  (column-wise)
    # In the contrastive margin h·(W[y+] - W[y-]) the bias cancels in EACH
    # term, so:
    #   margin(ablate) + margin(preserve) == margin(recon)  exactly.
    # Identity in margin space, per example:
    assert torch.allclose(m_ab + m_pres, m_recon, atol=1e-5), (
        f"ablate+preserve != recon (per-example): max diff {(m_ab + m_pres - m_recon).abs().max().item()}"
    )


def test_per_feature_attribution_sums_to_recon_margin(rfa):
    """sum_f attr_f(over examples mean) == mean(recon margin) for the
    contrastive case (constants cancel)."""
    V, d, D, K_snap = 16, 5, 8, 3
    sd, preact = _toy_crosscoder(V=V, d=d, D=D, K_snap=K_snap, seed=2)
    step_idx = 0

    feature_acts, _ = rfa.encode_at_step(sd, preact, step_idx, device="cpu")
    stats_step = {"scale": torch.tensor([[1.5]]), "mean": torch.zeros(d)}
    W_recon = rfa.reconstruct_W(feature_acts, sd, step_idx, stats_step, device="cpu")

    torch.manual_seed(11)
    N = 24
    h = torch.randn(N, d)
    y_plus = torch.randint(0, V, (N,))
    y_minus = (y_plus + 1 + torch.randint(0, V - 1, (N,))) % V

    W_D_step = sd["W_D"][step_idx]
    attrs = rfa.per_feature_attribution(h, feature_acts, W_D_step, stats_step["scale"], y_plus, y_minus)
    # Sum across all features should equal the mean recon margin.
    margin_recon = rfa.margins_with_W(h, W_recon, y_plus, y_minus)
    assert attrs.sum().item() == pytest.approx(margin_recon.float().mean().item(), abs=1e-4)


def test_attribution_sign_matches_manual(rfa):
    """For a hand-picked example with a specific feature firing only on y+,
    that feature's attribution must be positive."""
    V, d, D = 6, 3, 4
    # Hand-build feature_acts: feature 0 fires only on vocab row 1 (= y+).
    feature_acts = torch.zeros(V, D)
    feature_acts[1, 0] = 0.5  # y+ row
    # Decoder direction for feature 0: along +x.
    W_D = torch.zeros(D, d)
    W_D[0] = torch.tensor([1.0, 0.0, 0.0])
    # Hidden state along +x.
    h = torch.tensor([[1.0, 0.0, 0.0]])  # (1, d)
    y_plus = torch.tensor([1])
    y_minus = torch.tensor([2])
    scale = torch.tensor([[1.0]])

    attrs = rfa.per_feature_attribution(h, feature_acts, W_D, scale, y_plus, y_minus)
    # attr_0 = (a_{1,0} - a_{2,0}) * (h · D_0) * scale = (0.5 - 0) * 1.0 * 1.0
    assert attrs[0].item() == pytest.approx(0.5, abs=1e-6)
    # Other features have all-zero feature_acts, so attribution is zero.
    assert torch.allclose(attrs[1:], torch.zeros(D - 1), atol=1e-7)


def test_matched_random_features_excludes_target(rfa):
    """Sanity: matched control never overlaps with the target set."""
    D = 32
    rng = np.random.default_rng(0)
    decoder_norms = rng.uniform(0.1, 2.0, size=D)
    rates = rng.uniform(0.001, 0.5, size=D)
    target = rng.choice(D, size=8, replace=False)
    ctrl = rfa.matched_random_features(target, decoder_norms, rates, rng)
    assert len(ctrl) == len(target)
    assert len(set(ctrl) & set(target)) == 0


def test_matched_random_features_sign_filter(rfa):
    """When the sign-matched pool is smaller than K we must fall back, but
    when it has enough candidates the result must respect the filter."""
    D = 32
    rng = np.random.default_rng(0)
    decoder_norms = rng.uniform(0.1, 2.0, size=D)
    rates = rng.uniform(0.001, 0.5, size=D)
    pos_mask = np.zeros(D, dtype=bool)
    pos_mask[10:30] = True  # 20 positive features
    target = np.array([10, 11, 12, 13, 14])  # all in positive pool
    ctrl = rfa.matched_random_features(target, decoder_norms, rates, rng, sign_filter=pos_mask)
    assert all(pos_mask[i] for i in ctrl), "sign-matched control violated mask"
    assert len(set(ctrl) & set(target.tolist())) == 0


def test_matched_random_features_falls_back_when_pool_too_small(rfa):
    """If the sign-matched pool < K (after target exclusion), fall back to
    the unfiltered pool. The control should still be size K and disjoint
    from target, but may include features outside the sign filter."""
    D = 32
    rng = np.random.default_rng(0)
    decoder_norms = rng.uniform(0.1, 2.0, size=D)
    rates = rng.uniform(0.001, 0.5, size=D)
    pos_mask = np.zeros(D, dtype=bool)
    pos_mask[:10] = True  # 10 positive features
    target = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # 8 of the 10 positives
    # Asking for 8 sign-matched controls outside target leaves only 2 — fallback fires.
    ctrl = rfa.matched_random_features(target, decoder_norms, rates, rng, sign_filter=pos_mask)
    assert len(ctrl) == len(target)
    assert len(set(ctrl) & set(target.tolist())) == 0
    # Some controls will be outside the sign filter (proves fallback fired).
    assert any(not pos_mask[i] for i in ctrl)
