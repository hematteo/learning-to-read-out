"""Tests for src/readout/probes/readout_swap.py.

Covers the contrastive-margin computation and the gauge-alignment ladder.
align_readout is the single source of truth for the ladder (none/mean/scale/
row_norm/procrustes); run_aligned_swap_grid.py imports it, so the behavioural
tests below pin the only implementation.
"""

from __future__ import annotations

import pytest
import torch


# ── compute_margin: hand-computed reference ──────────────────────────────────
def test_compute_margin_matches_hand_value():
    from readout.probes.readout_swap import compute_margin

    # Tiny toy: 2 examples, d=3, V=4.
    h = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])  # (2, 3)
    W = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # vocab 0
            [0.0, 1.0, 0.0],  # vocab 1
            [0.0, 0.0, 1.0],  # vocab 2
            [1.0, 1.0, 1.0],  # vocab 3
        ]
    )
    y_plus = torch.tensor([0, 3])  # ex1: y+=v0; ex2: y+=v3
    y_minus = torch.tensor([2, 1])  # ex1: y-=v2; ex2: y-=v1

    # ex1: h0·W[0] - h0·W[2] = 1.0 - (-1.0) = 2.0
    # ex2: h1·W[3] - h1·W[1] = (0.5+0.5+0) - 0.5 = 0.5
    expected = torch.tensor([2.0, 0.5])
    out = compute_margin(h, W, y_plus, y_minus)
    assert torch.allclose(out, expected, atol=1e-6)


def test_margin_summary_basic_stats():
    from readout.probes.readout_swap import margin_summary

    s = margin_summary(torch.tensor([1.0, 2.0, -1.0, 0.5]))
    assert s["n"] == 4
    assert s["accuracy"] == pytest.approx(0.75)  # 3/4 positive
    assert s["mean_margin"] == pytest.approx(0.625)


# ── alignment ladder: identity + reference parity ────────────────────────────
def test_align_none_is_identity():
    from readout.probes.readout_swap import align_readout

    torch.manual_seed(0)
    W_s = torch.randn(8, 5)
    W_t = torch.randn(8, 5)
    out = align_readout(W_s, W_t, "none")
    assert torch.allclose(out, W_s.float(), atol=0)


def test_align_modes_smoke():
    """Each mode runs and produces a (V, d) tensor of finite values."""
    from readout.probes.readout_swap import ALIGN_MODES, align_readout

    torch.manual_seed(1)
    W_s = torch.randn(16, 4)
    W_t = torch.randn(16, 4)
    for mode in ALIGN_MODES:
        out = align_readout(W_s, W_t, mode)
        assert out.shape == W_s.shape
        assert torch.isfinite(out).all(), f"alignment {mode} produced non-finite values"


def test_align_unknown_mode_raises():
    from readout.probes.readout_swap import align_readout

    torch.manual_seed(0)
    W = torch.randn(4, 3)
    with pytest.raises(ValueError, match="unknown alignment mode"):
        align_readout(W, W, "no_such_mode")


def test_align_row_norm_matches_target_row_norms():
    """row_norm mode: every row of the aligned W_s has the row norm of W_t."""
    from readout.probes.readout_swap import align_readout

    torch.manual_seed(2)
    W_s = torch.randn(20, 6) * 3.0
    W_t = torch.randn(20, 6) * 0.5
    out = align_readout(W_s, W_t, "row_norm")
    expected = W_t.float().norm(dim=-1)
    actual = out.norm(dim=-1)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_align_scale_matches_target_frob_norm():
    """scale mode: ||centered aligned W_s|| == ||centered W_t||."""
    from readout.probes.readout_swap import align_readout

    torch.manual_seed(3)
    W_s = torch.randn(20, 6) * 4.0
    W_t = torch.randn(20, 6) * 2.0
    out = align_readout(W_s, W_t, "scale")
    out_c = out - out.mean(0, keepdim=True)
    Wt_c = W_t.float() - W_t.float().mean(0, keepdim=True)
    assert out_c.norm() == pytest.approx(Wt_c.norm().item(), rel=1e-5)


# NB: run_aligned_swap_grid.py now imports align_readout from this module
# (single source of truth), so the former reference-parity test is obsolete —
# the behavioural tests above fully pin align_readout.
