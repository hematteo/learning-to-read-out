"""Pins for the shipped replacements of the archived intervention helpers.

encode_snapshot_local and script_of replaced the non-shipped
_archive/legacy_crosscoder_160m helpers; on the real release data the swap was
verified bitwise (helper outputs and the end-to-end smoke pipeline). These
tests pin the same contracts on synthetic data so CI holds them.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import torch

from readout.core.repro import seed_everything
from readout.dynamics.temporal_patch import (
    CFG_PYTHIA_160M,
    ModelCfg,
    PatchContext,
    _stable_seed,
    encode_snapshot_local,
    select_feature_subset,
)
from readout.probes.token_scripts import script_of


def _legacy_encode(W_U, pieces):
    """Verbatim archived 02_intervention.encode."""
    x_proc = (W_U - pieces["mean_i"]) / pieces["scale_i"]
    pre = x_proc @ pieces["W_E"] + pieces["b_E"]
    return (pre > pieces["thr"]).float() * pre


def test_encode_snapshot_local_matches_archived_formula():
    seed_everything(0)
    V, d, D = 64, 8, 32
    pieces = {
        "W_E": torch.randn(d, D) * 0.3,
        "b_E": torch.randn(D) * 0.1,
        "thr": torch.rand(D) * 0.2,
        "mean_i": torch.randn(d) * 0.5,
        "scale_i": torch.tensor(2.0),
    }
    W_U = torch.randn(V, d)
    a_new = encode_snapshot_local(W_U, pieces)
    a_old = _legacy_encode(W_U, pieces)
    assert torch.equal(a_new, a_old)
    assert (a_new != 0).any() and (a_new == 0).any()  # gate actually active


def test_script_of_branches():
    cases = {
        "": "empty",
        "   ": "space",
        " the": "ASCII",
        "!": "ASCII",
        " Привет": "Cyrillic",
        "ไทย": "Thai",
        "مرحبا": "Arabic",
        "नमस्ते": "Devanagari",
        "বাংলা": "Bengali",
        "中文": "CJK",
        "ひらがな": "Japanese",
        "한국어": "Korean",
        "Ωμέγα": "other_unicode",  # Greek falls through to the catch-all
    }
    for text, expected in cases.items():
        assert script_of(text) == expected, f"{text!r} -> {script_of(text)} != {expected}"


# ── _stable_seed: process-stable RNG seeding ─────────────────────────────────


def test_stable_seed_pinned_values():
    """crc32-based seeds are pinned exactly: any change to the scheme silently
    resamples every 'deterministic' concept/bootstrap draw."""
    assert _stable_seed("concept_sample", "her") == 4120451028
    assert _stable_seed("eval", "capitals") == 3663789081
    assert _stable_seed("boot", "induction", 3) == 1740180821
    # 32-bit range and type (np.random.default_rng requires a plain int seed).
    s = _stable_seed("anything")
    assert isinstance(s, int) and 0 <= s < 2**32


def test_stable_seed_is_pythonhashseed_independent():
    """Builtin hash() is salted per process; _stable_seed must not be. Two
    subprocesses with different PYTHONHASHSEED must print identical seeds."""
    code = (
        "from readout.dynamics.temporal_patch import _stable_seed; "
        "print(_stable_seed('eval', 'capitals'), _stable_seed('boot', 'induction', 3))"
    )
    outs = []
    for hashseed in ("0", "424242"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
        assert r.returncode == 0, r.stderr[-1500:]
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] == "3663789081 1740180821"


# ── select_feature_subset: scarcity guard ────────────────────────────────────


def _tiny_ctx(steps):
    cfg = ModelCfg(
        model_name=CFG_PYTHIA_160M.model_name,
        ckpt_template=CFG_PYTHIA_160M.ckpt_template,
        aggregates_template=None,
        npy_dir=None,
        hLN_cache_dir=CFG_PYTHIA_160M.hLN_cache_dir,
        steps_canonical=steps,
    )
    return PatchContext(cfg=cfg, d_sae=8)


def test_select_feature_subset_scarcity_returns_only_eligible(capsys):
    """With fewer eligible features than top_k, the subset contains ONLY the
    eligible ones — never -inf ineligible padding."""
    steps = [0, 1000, 143000]
    ctx = _tiny_ctx(steps)
    D = 8
    rates = np.full((3, D), 0.5)
    # decay_top_k eligibility: rate(base) - rate(target) > 0. Only 2 and 5 decay.
    rates[0, 2], rates[2, 2] = 0.9, 0.1  # decay 0.8
    rates[0, 5], rates[2, 5] = 0.6, 0.3  # decay 0.3
    rates[0, 1], rates[2, 1] = 0.1, 0.9  # rises: ineligible

    out = select_feature_subset(
        ctx,
        "decay_top_k",
        top_k=4,
        base_step=0,
        target_step=143000,
        rates_seed=rates,
        decoder_norms_seed=np.ones((3, D)),
        cusum_seed=np.zeros(D),
        peak_steps_seed=np.zeros(D),
    )
    assert out == [2, 5], f"expected only the eligible features (best first), got {out}"
    assert "2/4 eligible" in capsys.readouterr().out

    # excluded shrinks the eligible pool further, still without padding.
    out_excl = select_feature_subset(
        ctx,
        "decay_top_k",
        top_k=4,
        base_step=0,
        target_step=143000,
        rates_seed=rates,
        decoder_norms_seed=np.ones((3, D)),
        cusum_seed=np.zeros(D),
        peak_steps_seed=np.zeros(D),
        excluded={2},
    )
    assert out_excl == [5]


def test_select_feature_subset_no_eligible_features_returns_empty():
    steps = [0, 1000, 143000]
    ctx = _tiny_ctx(steps)
    D = 8
    rates = np.full((3, D), 0.5)  # flat: nothing decays
    out = select_feature_subset(
        ctx,
        "decay_top_k",
        top_k=3,
        base_step=0,
        target_step=143000,
        rates_seed=rates,
        decoder_norms_seed=np.ones((3, D)),
        cusum_seed=np.zeros(D),
        peak_steps_seed=np.zeros(D),
    )
    assert out == []
