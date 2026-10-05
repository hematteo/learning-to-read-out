"""aggregate_causal_tests (sparse_feature_causal_tests) against hand-computed tables.

The aggregation backs fig:app-sparse-feature-causal-curves and
fig:app-sparse-feature-causal-specificity-k32. The script is loaded with its
scripts dir on ``sys.path``. CPU only, no data.
"""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

from readout.core.paths import repo_root

SCRIPTS = repo_root() / "experiments/causal/sparse_feature_causal_tests/scripts"


@pytest.fixture(scope="module")
def agg():
    mp = pytest.MonkeyPatch()
    mp.syspath_prepend(str(SCRIPTS))
    try:
        return importlib.import_module("aggregate_causal_tests")
    finally:
        mp.undo()


def _pilot_run(tmp_path, concept="digits", step=1000):
    run = tmp_path / f"1b_{concept}_step{step}_positive"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"concept": concept, "snapshot_step": step, "n_in_C": 256, "n_null_C": 256})
    )
    base = {"h_step": 1000, "k": 8, "orig_concept_mass": 3.0, "full_concept_mass": 2.5, "bias_concept_mass": 0.5}
    rows = [
        {**base, "condition": "baseline", "control_id": -1, "drop_from_full": 0.0, "recovery_from_bias": 0.0},
        {
            **base,
            "condition": "top_positive_attr_ablate",
            "control_id": -1,
            "drop_from_full": 2.0,
            "recovery_from_bias": 0.0,
        },
        {
            **base,
            "condition": "top_positive_attr_only",
            "control_id": -1,
            "drop_from_full": 0.0,
            "recovery_from_bias": 1.5,
        },
    ]
    for ci, (drop, rec) in enumerate([(0.1, 0.2), (0.3, 0.0), (-0.1, 0.1)]):
        rows.append(
            {**base, "condition": "matched_ablate", "control_id": ci, "drop_from_full": drop, "recovery_from_bias": 0.0}
        )
        rows.append(
            {**base, "condition": "matched_only", "control_id": ci, "drop_from_full": 0.0, "recovery_from_bias": rec}
        )
    pd.DataFrame(rows).to_csv(run / "summary.csv", index=False)
    return run


def test_family_rows(agg, tmp_path):
    (row,) = agg.family_rows(_pilot_run(tmp_path), "label")
    assert (row["concept"], row["snapshot_step"], row["h_step"], row["k"]) == ("digits", 1000, 1000, 8)
    assert row["top_drop"] == 2.0
    assert row["matched_drop_mean"] == pytest.approx(0.1)
    assert (row["matched_drop_min"], row["matched_drop_max"]) == (-0.1, 0.3)
    assert row["top_minus_matched_mean"] == pytest.approx(1.9)
    assert row["top_minus_matched_max"] == pytest.approx(1.7)
    assert row["top_only_minus_matched_mean"] == pytest.approx(1.4)
    assert row["recon_gap_orig_minus_full"] == pytest.approx(0.5)
    assert row["full_minus_bias"] == pytest.approx(2.0)


def _specificity(concepts, k=32):
    rows = []
    for i, src in enumerate(concepts):
        for j, meas in enumerate(concepts):
            source_drop = 5.0 if i == j else float(j)
            rows.append(
                {
                    "source_concept": src,
                    "measured_concept": meas,
                    "k": k,
                    "condition": "source_ablate",
                    "control_id": -1,
                    "drop_from_full": source_drop,
                }
            )
            for ci, d in enumerate([0.0, 1.0]):
                rows.append(
                    {
                        "source_concept": src,
                        "measured_concept": meas,
                        "k": k,
                        "condition": "matched_ablate",
                        "control_id": ci,
                        "drop_from_full": d,
                    }
                )
    return pd.DataFrame(rows)


def test_control_corrected_drop_and_diagonal(agg):
    concepts = ["a", "b", "c"]
    m = agg.control_corrected_drop(_specificity(concepts), 32, concepts)
    assert list(m.index) == concepts and list(m.columns) == concepts
    # source drop minus the mean matched drop (0.5)
    assert m.loc["a", "a"] == pytest.approx(4.5)
    assert m.loc["a", "c"] == pytest.approx(1.5)
    rows = {r["source_concept"]: r for r in agg.diagonal_rows(m, "run", 32)}
    assert rows["a"]["max_offdiag"] == pytest.approx(1.5)
    assert rows["a"]["mean_offdiag"] == pytest.approx(1.0)
    assert rows["a"]["diag_minus_max_offdiag"] == pytest.approx(3.0)
    assert all(r["diag_rank"] == 1 for r in rows.values())
