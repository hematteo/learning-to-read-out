"""Population lifecycle diagnostics and bootstrap window counts (feature_lifecycle_trajectories).

build_population_diagnostics.summarize_run backs
fig:app-selected-population-lifecycle-diagnostics; find_reorganization_steps.bootstrap_window_counts
backs the resample columns of tab:app-reorganization-window-stats. Both scripts are loaded
with their scripts dir on ``sys.path``. CPU only, no data.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from readout.core.paths import repo_root

SCRIPTS = repo_root() / "experiments/lifecycle/feature_lifecycle_trajectories/scripts"


def _load(name):
    mp = pytest.MonkeyPatch()
    mp.syspath_prepend(str(SCRIPTS))
    try:
        return importlib.import_module(name)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def bpd():
    return _load("build_population_diagnostics")


@pytest.fixture(scope="module")
def find():
    return _load("find_reorganization_steps")


def test_summarize_run_peak_counts_and_mass(bpd):
    steps = np.array([0, 10, 100, 1000])
    norms = np.array(
        [
            [1.0, 0.1, 0.2, 0.001],
            [0.5, 0.2, 0.4, 0.002],
            [0.2, 0.9, 0.4, 0.003],
            [0.1, 0.3, 0.1, 0.001],
        ],
        dtype=np.float32,
    )  # (K, D); feature 3 never exceeds the 0.01 activity threshold
    payload, peak_rows, mass_rows = bpd.summarize_run(bpd.Run("toy", "Toy", steps), norms)

    np.testing.assert_array_equal(payload["active_feature_ids"], [0, 1, 2])
    # Peaks: feature 0 at step 0, feature 1 at 100, feature 2 first reaches 0.4 at step 10.
    np.testing.assert_array_equal(payload["peak_step"], [0, 100, 10])
    assert [(r["step"], r["feature_count_fraction"]) for r in peak_rows] == [
        (0, pytest.approx(1 / 3)),
        (10, pytest.approx(1 / 3)),
        (100, pytest.approx(1 / 3)),
    ]
    total = norms[:, :3].sum(axis=1)
    np.testing.assert_allclose(payload["total_decoder_norm_mass"], total)
    np.testing.assert_allclose(payload["mass_normalized_to_model_max"], total / total.max())
    assert payload["mass_normalized_to_model_max"].max() == pytest.approx(1.0)
    assert sum(r["mass_fraction_over_snapshots"] for r in mass_rows) == pytest.approx(1.0)
    assert {r["n_active_features"] for r in mass_rows} == {3}


def test_early_peak_rows(bpd):
    steps = np.array([0, 128, 1000, 3000])
    norms = np.array(
        [
            [1.0, 0.5, 0.1, 0.001],
            [0.8, 1.0, 0.2, 0.002],
            [0.4, 0.5, 1.0, 0.003],
            [0.2, 0.1, 0.5, 0.001],
        ],
        dtype=np.float32,
    )  # (K, D); peaks at 0, 128, 1000; feature 3 inactive
    (window,), decay = bpd.early_peak_rows(bpd.Run("toy", "Toy", steps), norms)

    assert window["fraction_active_peaking_in_window"] == pytest.approx(1 / 3)
    assert window["n_active_features"] == 3
    # features 0 and 1 peak by step 128; rho is each feature's norm over its own max
    assert [r["n_early_peaking"] for r in decay] == [2, 2, 2, 2]
    np.testing.assert_allclose([r["mean_rho_early_peaking"] for r in decay], [0.75, 0.9, 0.45, 0.15], rtol=1e-6)


def test_bootstrap_window_counts(find):
    summary = [
        {"model": "a", "label": "A", "primary_start_step": 512, "primary_end_step": 1621},
        {"model": "b", "label": "B", "primary_start_step": 9000, "primary_end_step": 28463},
    ]
    windows = [(512, 1621)] * 5 + [(1000, 3164)] * 2 + [(14000, 44274)] * 3
    boot = [{"model": "a", "start_step": s, "end_step": e} for s, e in windows]
    boot += [{"model": "b", "start_step": 900, "end_step": 2848}] * 4  # selected window never picked

    rows = find.bootstrap_window_counts(summary, boot)
    a = [(r["start_step"], r["end_step"], r["n_resamples"], r["selected_window"]) for r in rows if r["model"] == "a"]
    assert a == [(512, 1621, 5, True), (14000, 44274, 3, False), (1000, 3164, 2, False)]
    b = [(r["start_step"], r["n_resamples"], r["selected_window"]) for r in rows if r["model"] == "b"]
    assert b == [(9000, 0, True), (900, 4, False)]
    assert {r["n_bootstraps"] for r in rows if r["model"] == "a"} == {10}
