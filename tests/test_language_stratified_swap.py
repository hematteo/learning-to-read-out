"""Block layout, bootstrap, and argmin-probability helpers of
experiments/causal/language_stratified_readout_swap/scripts/run_language_stratified_swap.py.

CPU-only, synthetic data: no model, corpus, or snapshot is loaded.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def lss():
    repo = Path(__file__).resolve().parents[1]
    p = repo / "experiments/causal/language_stratified_readout_swap/scripts/run_language_stratified_swap.py"
    spec = importlib.util.spec_from_file_location("lss", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["lss"] = m
    spec.loader.exec_module(m)
    return m


def _two_language_corpus(seq_len=8, n_seqs=3, boundary=11):
    """Flat corpus of n_seqs * seq_len + 3 tokens; en before `boundary`, ru after."""
    n = n_seqs * seq_len + 3  # trailing partial window is truncated
    ids = torch.arange(n, dtype=torch.long) % 7
    languages = ["en"] * boundary + ["ru"] * (n - boundary)
    return ids, languages


# ── per-language boundary drop ──────────────────────────────────────────────


def test_layout_drops_only_cross_language_targets(lss):
    seq_len, block = 8, 4
    ids, languages = _two_language_corpus(seq_len=seq_len, boundary=11)
    layout = lss.build_block_layout(ids, languages, seq_len=seq_len, block_tokens=block)

    assert layout.ids_seqs.shape == (3, seq_len)
    n_targets = 3 * (seq_len - 1)
    # Token 11 (seq 1, pos 3) is the first ru token; its context (token 10) is en.
    assert layout.n_cross_language_targets == 1
    assert layout.language_block_ids[1, 2].item() == -1  # target at flat position 11
    assert (layout.language_block_ids >= 0).sum().item() == n_targets - 1

    counts = layout.counts
    n_all = int(counts[slice(*layout.language_slices["all"])].sum())
    n_en = int(counts[slice(*layout.language_slices["en"])].sum())
    n_ru = int(counts[slice(*layout.language_slices["ru"])].sum())
    assert n_all == n_targets
    assert n_en + n_ru == n_targets - 1
    # en targets: flat positions 1..10 minus window starts 0 and 8 -> 9 scored.
    assert n_en == 9
    assert list(layout.language_slices) == ["all", "en", "ru"]


def test_window_starts_are_never_targets(lss):
    """A language change at a window start is not a scored transition."""
    seq_len = 8
    ids, languages = _two_language_corpus(seq_len=seq_len, boundary=8)
    layout = lss.build_block_layout(ids, languages, seq_len=seq_len, block_tokens=5)
    assert layout.n_cross_language_targets == 0
    assert (layout.language_block_ids >= 0).all()


def test_layout_blocks_are_consecutive_and_sized(lss):
    ids, languages = _two_language_corpus(seq_len=8, boundary=11)
    layout = lss.build_block_layout(ids, languages, seq_len=8, block_tokens=4)
    start, stop = layout.language_slices["all"]
    assert layout.counts[start:stop].tolist() == [4, 4, 4, 4, 4, 1]
    flat = layout.all_block_ids.reshape(-1)
    assert torch.equal(flat, torch.arange(flat.numel()) // 4)
    assert layout.block_language == ["all"] * 6 + ["en"] * 3 + ["ru"] * 3
    assert layout.block_index == [0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2]


def test_layout_rejects_bad_inputs(lss):
    ids, languages = _two_language_corpus()
    with pytest.raises(ValueError, match="length mismatch"):
        lss.build_block_layout(ids, languages[:-1], seq_len=8, block_tokens=4)
    with pytest.raises(ValueError, match="positive"):
        lss.build_block_layout(ids, languages, seq_len=8, block_tokens=0)


# ── block NLL sums ──────────────────────────────────────────────────────────


def test_block_sums_match_direct_nll(lss):
    torch.manual_seed(0)
    seq_len, d_model, vocab = 8, 5, 7
    ids, languages = _two_language_corpus(seq_len=seq_len, boundary=11)
    layout = lss.build_block_layout(ids, languages, seq_len=seq_len, block_tokens=4)
    h_ln = torch.randn(3, seq_len, d_model)  # (n_seqs, seq_len, d_model)
    readout = torch.randn(vocab, d_model)  # (V, d_model)

    sums = lss.evaluate_readout_by_block(h_ln, readout, layout, device="cpu", batch_seqs=2)

    logp = torch.log_softmax((h_ln @ readout.T)[:, :-1], dim=-1).double()
    nll = -logp.gather(-1, layout.ids_seqs[:, 1:].unsqueeze(-1)).squeeze(-1)  # (n_seqs, seq_len - 1)
    start, stop = layout.language_slices["all"]
    assert sums[start:stop].sum().item() == pytest.approx(nll.sum().item(), rel=1e-6)
    for lang in ("en", "ru"):
        start, stop = layout.language_slices[lang]
        mask = torch.isin(layout.language_block_ids, torch.arange(start, stop))
        assert sums[start:stop].sum().item() == pytest.approx(nll[mask].sum().item(), rel=1e-6)


# ── paired block bootstrap ──────────────────────────────────────────────────


def test_bootstrap_constant_ratio_has_degenerate_interval(lss):
    den = np.array([4.0, 2.0, 7.0, 1.0])
    lo, hi = lss.bootstrap_mean(0.5 * den, den, reps=300, seed=0)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_bootstrap_matches_reference_resampling(lss):
    rng = np.random.default_rng(1)
    num, den = rng.normal(size=9), rng.integers(1, 5, size=9).astype(float)
    reps, seed = 600, 3  # 600 spans three 256-rep chunks
    ref_rng = np.random.default_rng(seed)
    draws = np.concatenate(
        [ref_rng.integers(0, 9, size=(min(256, reps - s), 9)) for s in range(0, reps, 256)]
    )  # (reps, n_blocks)
    ref = num[draws].sum(1) / den[draws].sum(1)
    lo, hi = lss.bootstrap_mean(num, den, reps=reps, seed=seed)
    assert (lo, hi) == pytest.approx(tuple(np.quantile(ref, [0.025, 0.975])))
    assert lo <= num.sum() / den.sum() <= hi
    assert (lo, hi) == lss.bootstrap_mean(num, den, reps=reps, seed=seed)


def test_bootstrap_rejects_bad_inputs(lss):
    with pytest.raises(ValueError):
        lss.bootstrap_mean(np.zeros(0), np.zeros(0), reps=10, seed=0)
    with pytest.raises(ValueError):
        lss.bootstrap_mean(np.zeros(3), np.ones(2), reps=10, seed=0)


def test_stable_seed_is_fixed(lss):
    # Seeds must not depend on Python's salted hash(); pin two values.
    assert lss.stable_seed(0, 256, 1000, "all") == 9434941
    assert lss.stable_seed(1, 256, 0, "en") == 1258628
    assert lss.stable_seed(0, 256, 1000, "en") != lss.stable_seed(0, 256, 1000, "ru")


def test_summarize_native_cell_is_zero_delta(lss):
    ids, languages = _two_language_corpus(seq_len=8, boundary=11)
    layout = lss.build_block_layout(ids, languages, seq_len=8, block_tokens=4)
    sums = torch.rand(len(layout.block_language), dtype=torch.float64)
    rows = lss.summarize_cell(
        model="m",
        seed=0,
        h_step=5,
        s_step=5,
        cell_sums=sums,
        native_sums=sums,
        layout=layout,
        bootstrap_reps=50,
        bootstrap_seed=0,
    )
    assert [r["language"] for r in rows] == ["all", "en", "ru"]
    for r in rows:
        assert r["delta_nll"] == 0.0
        assert r["delta_nll_ci_low"] == 0.0 and r["delta_nll_ci_high"] == 0.0


# ── argmin probability ──────────────────────────────────────────────────────


def test_argmin_probability_dominant_readout(lss):
    counts = np.array([3.0, 5.0, 2.0, 4.0])
    base = np.array([6.0, 9.0, 5.0, 8.0])
    sums = np.stack([base + counts, base, base + 2 * counts])  # readout 1 wins every block
    p = lss.argmin_probabilities(sums, counts, reps=200, rng=np.random.default_rng(0))
    assert p.tolist() == [0.0, 1.0, 0.0]


def test_argmin_probability_ties_go_to_first(lss):
    counts = np.ones(5)
    sums = np.tile(np.arange(5.0), (2, 1))
    p = lss.argmin_probabilities(sums, counts, reps=64, rng=np.random.default_rng(0))
    assert p.tolist() == [1.0, 0.0]


def test_argmin_probability_matches_reference(lss):
    rng = np.random.default_rng(2)
    counts = rng.integers(50, 256, size=12).astype(float)
    # Two readouts close in mean NLL, so resampling splits the wins.
    sums = np.stack([counts * (3.0 + rng.normal(0, 0.2, 12)), counts * (3.0 + rng.normal(0, 0.2, 12))])
    reps, seed = 300, 7  # 300 spans three 128-rep chunks
    ref_rng = np.random.default_rng(seed)
    wins = np.zeros(2)
    for s in range(0, reps, 128):
        for draw in ref_rng.integers(0, 12, size=(min(128, reps - s), 12)):
            wins[np.argmin(sums[:, draw].sum(1) / counts[draw].sum())] += 1
    p = lss.argmin_probabilities(sums, counts, reps=reps, rng=np.random.default_rng(seed))
    assert p == pytest.approx(wins / reps)
    assert p.sum() == pytest.approx(1.0)
    assert 0.0 < p[0] < 1.0


# ── reference parity ────────────────────────────────────────────────────────


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_reference_parity_reads_grid_and_aligned_summaries(lss, tmp_path):
    ours = [
        {"language": "all", "h_eval_step": 256, "snapshot_step": 1000, "nll": 4.8},
        {"language": "en", "h_eval_step": 256, "snapshot_step": 1000, "nll": 7.0},
    ]
    grid = _write(
        tmp_path / "summary_global.csv",
        [{"model": "m", "h_eval_step": 256, "snapshot_step": 1000, "nll": 4.8 + 2e-6}],
    )
    out = lss.reference_parity(grid, ours, model="m")
    assert out["status"] == "pass" and out["n_overlapping_cells"] == 1

    aligned = _write(
        tmp_path / "summary.csv",
        [
            {"model": "m", "alignment": "scale", "h_eval_step": 256, "snapshot_step": 1000, "nll": 1.0},
            {"model": "m", "alignment": "none", "h_eval_step": 256, "snapshot_step": 1000, "nll": 4.7},
        ],
    )
    out = lss.reference_parity(aligned, ours, model="m")
    assert out["status"] == "fail"
    assert out["max_abs_nll_difference"] == pytest.approx(0.1)

    other_model = lss.reference_parity(grid, ours, model="other")
    assert other_model["status"] == "not_checked"
    assert lss.reference_parity(tmp_path / "missing.csv", ours, model="m")["status"] == "not_checked"
