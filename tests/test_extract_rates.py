"""Parity tests for readout.crosscoder.extract_rates against the
readout.crosscoder.inference helpers, on a tiny synthetic checkpoint.

Covers the three real entry points the CLI dispatches to:
  - compute_rates            (legacy per-head pre-activation, --legacy-per-head)
  - compute_rates_canonical  (joint pre-activation, decoder-norm-effective
                              threshold — matches llamascopium encode)
  - _maybe_compute_preprocess_stats (stats passthrough + rebuild-from-snapshots)

Everything is offline: snapshots live in a tmp cache dir under the canonical
``{slug}_step{step}_wu.pt`` filenames.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from readout.crosscoder import inference as inf
from readout.crosscoder.extract_rates import (
    _maybe_compute_preprocess_stats,
    compute_rates,
    compute_rates_canonical,
)
from readout.crosscoder.wu_adapter import preprocess_snapshots

K, V, D_MODEL, D_SAE = 3, 48, 8, 16
STEPS = [0, 1000, 143000]
MODEL = "EleutherAI/pythia-160m"
SLUG = MODEL.replace("/", "_")


def _toy_sd(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "W_E": torch.randn(K, D_MODEL, D_SAE, generator=g) * 0.3,
        "b_E": torch.randn(K, D_SAE, generator=g) * 0.1,
        "W_D": torch.randn(K, D_SAE, D_MODEL, generator=g) * 0.4,
        "b_D": torch.randn(K, D_MODEL, generator=g) * 0.05,
        "activation_function.log_jumprelu_threshold": torch.log(torch.rand(D_SAE, generator=g) * 0.2 + 0.05),
    }


@pytest.fixture
def snap_dir(tmp_path: Path) -> Path:
    g = torch.Generator().manual_seed(7)
    d = tmp_path / "cache"
    d.mkdir()
    for s in STEPS:
        torch.save(torch.randn(V, D_MODEL, generator=g), d / f"{SLUG}_step{s}_wu.pt")
    return d


def _save_ckpt(tmp_path: Path, sd, name="cc.pt", **extra) -> Path:
    p = tmp_path / name
    torch.save({"state_dict": sd, "steps": STEPS, "model_name": MODEL, **extra}, p)
    return p


def _load_snaps(snap_dir: Path) -> torch.Tensor:
    return torch.stack([torch.load(snap_dir / f"{SLUG}_step{s}_wu.pt", weights_only=True) for s in STEPS])


def test_compute_rates_matches_inference_per_head_form(tmp_path, snap_dir):
    sd = _toy_sd()
    ckpt = _save_ckpt(tmp_path, sd)
    rates, steps = compute_rates(ckpt, snap_dir)
    assert steps == STEPS and rates.shape == (K, D_SAE)

    snaps = _load_snaps(snap_dir)
    thr = inf.jumprelu_threshold(sd)
    for k in range(K):
        # Legacy per-head convention: this head's pre-activation (with its own
        # bias) gated against the global threshold, no decoder-norm rescale.
        pre_k = inf.encoder_preacts(sd, snaps[k], k) + sd["b_E"][k].float()
        expected = (pre_k > thr).float().mean(dim=0)
        assert torch.equal(rates[k], expected), f"per-head rate parity fails at k={k}"


def test_compute_rates_canonical_matches_inference_joint_form(tmp_path, snap_dir):
    sd = _toy_sd(seed=1)
    ckpt = _save_ckpt(tmp_path, sd)
    rates, steps = compute_rates_canonical(ckpt, snap_dir)
    assert steps == STEPS and rates.shape == (K, D_SAE)

    snaps = _load_snaps(snap_dir)
    joint = sum(inf.encoder_preacts(sd, snaps[k], k) for k in range(K)) + inf.encoder_bias_total(sd)
    thr = inf.jumprelu_threshold(sd)
    for k in range(K):
        _, fired = inf.jumprelu_feature_acts(joint, thr, inf.decoder_norms(sd, k))
        expected = fired.float().mean(dim=0)
        assert torch.allclose(rates[k], expected, atol=1e-7), f"canonical rate parity fails at k={k}"


def test_preprocess_stats_passthrough_changes_rates_consistently(tmp_path, snap_dir):
    """A ckpt with embedded center_scale stats must gate on the PREPROCESSED
    inputs; parity is checked by preprocessing manually first."""
    sd = _toy_sd(seed=2)
    raw = _load_snaps(snap_dir)
    proc, stats = preprocess_snapshots(raw, mode="center_scale")
    ckpt = _save_ckpt(tmp_path, sd, name="cc_stats.pt", preprocess_stats=stats)

    rates, _ = compute_rates(ckpt, snap_dir)
    thr = inf.jumprelu_threshold(sd)
    for k in range(K):
        pre_k = inf.encoder_preacts(sd, proc[k], k) + sd["b_E"][k].float()
        expected = (pre_k > thr).float().mean(dim=0)
        assert torch.allclose(rates[k], expected, atol=1e-7)

    # Sanity: preprocessing genuinely changed the answer vs the raw path.
    raw_rates, _ = compute_rates(_save_ckpt(tmp_path, sd, name="cc_raw.pt"), snap_dir)
    assert not torch.equal(rates, raw_rates)


def test_stats_rebuilt_from_mode_match_training_and_explicit_stats(tmp_path, snap_dir):
    """No embedded stats + preprocess_mode='center_scale' rebuilds the exact
    training-time stats from the snapshots, so rates equal the explicit-stats
    checkpoint bitwise."""
    sd = _toy_sd(seed=3)
    raw = _load_snaps(snap_dir)
    _, train_stats = preprocess_snapshots(raw, mode="center_scale")

    rebuilt = _maybe_compute_preprocess_stats(None, "center_scale", SLUG, snap_dir, STEPS)
    assert rebuilt["mode"] == "center_scale"
    assert torch.equal(rebuilt["mean"], train_stats["mean"])
    assert torch.equal(rebuilt["scale"], train_stats["scale"])

    ck_mode = _save_ckpt(tmp_path, sd, name="cc_mode.pt", preprocess_mode="center_scale")
    ck_stats = _save_ckpt(tmp_path, sd, name="cc_stats.pt", preprocess_stats=train_stats)
    for fn in (compute_rates, compute_rates_canonical):
        r_mode, _ = fn(ck_mode, snap_dir)
        r_stats, _ = fn(ck_stats, snap_dir)
        assert torch.equal(r_mode, r_stats), f"{fn.__name__}: rebuilt-stats path diverges from explicit stats"


def test_maybe_compute_preprocess_stats_center_and_none_modes(snap_dir):
    raw = _load_snaps(snap_dir)
    stats = _maybe_compute_preprocess_stats(None, "center", SLUG, snap_dir, STEPS)
    assert stats["mode"] == "center"
    assert torch.allclose(stats["mean"], raw.mean(dim=1, keepdim=True))
    assert torch.equal(stats["scale"], torch.ones(K, 1, 1))

    assert _maybe_compute_preprocess_stats(None, None, SLUG, snap_dir, STEPS) is None
    assert _maybe_compute_preprocess_stats(None, "none", SLUG, snap_dir, STEPS) is None
    sentinel = {"mode": "center_scale"}
    assert _maybe_compute_preprocess_stats(sentinel, "center_scale", SLUG, snap_dir, STEPS) is sentinel
    with pytest.raises(ValueError, match="Unknown preprocess mode"):
        _maybe_compute_preprocess_stats(None, "whiten", SLUG, snap_dir, STEPS)
