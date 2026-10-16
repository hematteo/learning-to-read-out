"""Lifecycle-stability port (tab:app-lifecycle-stability) and the late-start profile rows.

Covers readout.dynamics.lifecycle.profile_fractions, the release-safetensors path of
readout.crosscoder.extract_rates.compute_rates_canonical, and the helpers of
feature_lifecycle_trajectories/scripts/lifecycle_stability.py and
olmo_matched_checkpoint_window/scripts/build_late_start_vs_olmo_figures.py. The
experiment scripts are loaded with their scripts dir on ``sys.path`` (the precedent of
test_recipe_control_lifecycle.py). CPU only, no data: tiny synthetic tensors.
"""

from __future__ import annotations

import csv
import importlib
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from readout.core.model_specs import DEFAULT_STEPS_32
from readout.core.paths import repo_root
from readout.crosscoder.extract_rates import compute_rates_canonical
from readout.crosscoder.inference import reconstruct_preprocess_stats
from readout.dynamics.lifecycle import PROFILE_ORDER, classify_profiles_refined, profile_fractions

LIFECYCLE_SCRIPTS = repo_root() / "experiments/lifecycle/feature_lifecycle_trajectories/scripts"
LATE_START_SCRIPTS = repo_root() / "experiments/ablations/olmo_matched_checkpoint_window/scripts"
STEPS = np.asarray(DEFAULT_STEPS_32, dtype=np.int64)
K = STEPS.size


def _load(scripts_dir, name):
    mp = pytest.MonkeyPatch()
    mp.syspath_prepend(str(scripts_dir))
    try:
        return importlib.import_module(name)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def ls():
    return _load(LIFECYCLE_SCRIPTS, "lifecycle_stability")


@pytest.fixture(scope="module")
def late():
    return _load(LATE_START_SCRIPTS, "build_late_start_vs_olmo_figures")


def _profile_norms() -> np.ndarray:
    """(K, 5) decoder norms: persistent, early-decaying, late-emerging, inactive, noisy."""
    t = np.linspace(0.0, 1.0, K)
    norms = np.empty((K, 5), dtype=np.float32)
    norms[:, 0] = 0.8  # persistent
    norms[:, 1] = 0.5 * np.clip(1.2 - 2.0 * t, 0.02, 1.0)  # early-decaying
    norms[:, 2] = 0.3 * np.clip(2.0 * t - 0.8, 0.02, 1.0)  # late-emerging
    norms[:, 3] = 0.005  # peak below the 0.01 activity threshold
    norms[:, 4] = 0.2 + 0.1 * np.sin(np.arange(K))
    return norms


def test_profile_fractions_classifies_designed_profiles():
    norms = _profile_norms()
    out = profile_fractions(norms, STEPS)
    assert out["n_active"] == 4
    assert set(PROFILE_ORDER) <= set(out)
    assert sum(out[name] for name in PROFILE_ORDER) == pytest.approx(1.0)
    for name in ("persistent", "early_decay", "late_emerge"):
        assert out[name] >= 0.25, name
    # Same result as classifying the fp16-rounded peak-normalized active trajectories by hand.
    active = norms.max(axis=0) > 0.01
    traj = (norms[:, active] / norms[:, active].max(axis=0)).astype(np.float16).astype(np.float32)
    profile = classify_profiles_refined(traj, STEPS)["profile"]
    assert out["early_decay"] == float((profile == "early_decay").mean())
    # float64 input gives the float32 answer (the late-start blob is read as float64).
    assert profile_fractions(norms.astype(np.float64), STEPS) == out


def test_profile_fractions_restricted_schedule_renormalizes():
    norms = _profile_norms()
    keep = STEPS >= 256
    out = profile_fractions(norms[keep], STEPS[keep])
    assert out["median_peak_step"] >= 256
    assert out["n_active"] == 4


def test_build_fits_covers_the_released_pythia_dictionaries(ls):
    fits = ls.build_fits()
    # 21 Pythia fits in tab:app-lifecycle-stability; the selected 6.9B seed0-sparse one is validate's.
    assert len(fits) == 20
    assert len({f.key for f in fits}) == 20
    by_stage = {s: [f for f in fits if f.stage == s] for s in ("pythia1b", "pythia160m", "lambda", "pythia69b")}
    assert [len(v) for v in by_stage.values()] == [3, 9, 7, 1]
    keys = {f.key for f in fits}
    # The selected dictionaries keep the window script's null-seed keys.
    assert {"pythia160m_d24576", "pythia1b_d24576"} <= keys
    assert "pythia160m_d8192_lam0p40_seed0" in keys and "pythia160m_d8192_lam1p35_seed2" in keys
    assert all(f.aggregate is not None for f in by_stage["pythia1b"] + by_stage["pythia69b"])
    assert all(f.checkpoint.name.endswith(".safetensors") for f in fits)


def test_release_decoder_arrays_matches_full_tensor_geometry(ls, tmp_path):
    lc = _load(LIFECYCLE_SCRIPTS, "lifecycle_common")
    g = torch.Generator().manual_seed(0)
    W_D = torch.randn(5, 8, 3, generator=g)  # (K, D, d)
    path = tmp_path / "cc.safetensors"
    save_file({"W_D": W_D}, str(path))
    norms, rotation, cos_terminal = ls.release_decoder_arrays(path, chunk=3)  # D not a multiple of chunk
    ref_rotation, ref_cos = lc._geometry_from_decoder(W_D)
    np.testing.assert_array_equal(norms, torch.linalg.norm(W_D, dim=-1).numpy())
    np.testing.assert_array_equal(rotation, ref_rotation)
    np.testing.assert_array_equal(cos_terminal, ref_cos)


def test_window_summary_is_deterministic(ls):
    rng = np.random.default_rng(0)
    D = 16
    norms = rng.uniform(0.05, 1.0, size=(K, D)).astype(np.float32)
    rotation = rng.uniform(0.0, 0.5, size=(K - 1, D)).astype(np.float32)
    rates = rng.uniform(0.0, 0.1, size=(K, D)).astype(np.float32)
    cos_terminal = rng.uniform(0.5, 1.0, size=(K, D)).astype(np.float32)
    a = ls.window_summary("toy", "toy", norms, rotation, rates, cos_terminal, STEPS)
    b = ls.window_summary("toy", "toy", norms, rotation, rates, cos_terminal, STEPS)
    assert a == b
    assert a["window_start"] < a["window_end"]
    assert 0.0 < a["null_p"] <= 1.0


def test_compute_rates_canonical_reads_release_safetensors(tmp_path):
    steps, V, d, D = [0, 1000, 143000], 12, 4, 6
    model, slug = "EleutherAI/pythia-160m", "EleutherAI_pythia-160m"
    g = torch.Generator().manual_seed(3)
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    for s in steps:
        torch.save(torch.randn(V, d, generator=g), snaps / f"{slug}_step{s}_wu.pt")
    sd = {
        "W_E": torch.randn(len(steps), d, D, generator=g),
        "b_E": torch.randn(len(steps), D, generator=g) * 0.1,
        "W_D": torch.randn(len(steps), D, d, generator=g),
        "activation_function.log_jumprelu_threshold": torch.log(torch.rand(D, generator=g) + 0.1),
    }
    pt = tmp_path / "cc.pt"
    torch.save({"state_dict": sd, "steps": steps, "model_name": model, "preprocess_mode": "center_scale"}, pt)
    st = tmp_path / "seed0.safetensors"
    save_file(sd, str(st))
    (tmp_path / "seed0.config.json").write_text(
        json.dumps({"steps": steps, "model_name": model, "preprocess_mode": "center_scale"})
    )

    rates_pt, steps_pt = compute_rates_canonical(pt, snaps)
    rates_st, steps_st = compute_rates_canonical(st, snaps)
    assert steps_pt == steps_st == steps
    assert torch.equal(rates_pt, rates_st)

    stats = reconstruct_preprocess_stats(torch.load(snaps / f"{slug}_step{s}_wu.pt") for s in steps)
    rates_override, _ = compute_rates_canonical(st, snaps, preprocess_stats=stats)
    assert torch.equal(rates_override, rates_st)


def test_late_start_profile_rows(late, tmp_path):
    norms = _profile_norms()
    selected_path = tmp_path / "pythia1b_d24576_decoder_norms.npy"
    np.save(selected_path, norms)
    late_steps = np.asarray(late.PYTHIA_LATE_START_STEPS)
    late_norms = np.ones((late_steps.size, 3))
    rows = late.compute_late_start_profile_fractions(late_norms, late_steps, (tmp_path,), selected_path)
    keep = STEPS >= 256
    assert rows[0] == {"dictionary": "selected d24576", **profile_fractions(norms, STEPS)}
    assert rows[1] == {
        "dictionary": "selected d24576, steps >= 256 (not refitted)",
        **profile_fractions(norms[keep], STEPS[keep]),
    }
    assert rows[2]["persistent"] == 1.0
    with (tmp_path / "late_start_profile_fractions.csv").open(newline="") as f:
        written = list(csv.DictReader(f))
    assert [r["dictionary"] for r in written] == [r["dictionary"] for r in rows]
