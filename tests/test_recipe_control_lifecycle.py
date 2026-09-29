"""analyze_lifecycle.py (pretraining_recipe_control) on a synthetic trajectory crosscoder.

The script lives in an experiment ``scripts/`` dir (a sibling import of ``rc_common``),
so it is loaded with the scripts dir on ``sys.path`` (pytest's monkeypatch), the
precedent set by ``test_check_layout.py``. No data, no GPU: a hand-built decoder
tensor with one feature per designed profile is written as the crosscoder payload.
"""

from __future__ import annotations

import csv
import importlib
import json

import numpy as np
import pytest
import torch

from readout.core.paths import repo_root

SCRIPTS = repo_root() / "experiments" / "ablations" / "pretraining_recipe_control" / "scripts"
STEPS = [0, 1, 4, 16, 64, 256, 1024]
K, D, d, V = len(STEPS), 8, 3, 10


@pytest.fixture(scope="module")
def al(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.syspath_prepend(str(SCRIPTS))
    try:
        yield importlib.import_module("analyze_lifecycle")
    finally:
        mp.undo()


def _synthetic_payload():
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(D, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.zeros((K, D))
    norms[:, 0] = 1.0  # persistent: flat
    norms[:, 1] = [1.0, 1.0, 0.9, 0.5, 0.2, 0.05, 0.05]  # early-decaying
    norms[:, 2] = [0.02, 0.02, 0.05, 0.1, 0.4, 0.9, 1.0]  # late-emerging
    norms[:, 3] = 0.0  # inactive (zero decoder throughout)
    for f in range(4, D):
        norms[:, f] = rng.uniform(0.2, 1.0, size=K)
    W_D = torch.tensor(norms[:, :, None] * directions[None, :, :], dtype=torch.float32)  # (K, D, d)
    W_E = torch.tensor(rng.normal(size=(K, d, D)), dtype=torch.float32)
    b_E = torch.zeros(K, D)
    thr = torch.full((D,), np.log(0.1), dtype=torch.float32)
    return dict(
        state_dict={"W_D": W_D, "W_E": W_E, "b_E": b_E, "activation_function.log_jumprelu_threshold": thr},
        steps=STEPS,
        condition="baseline",
        seed=0,
        quality={"explained_variance": 0.9, "mean_l0": 3.0, "dead_rate": 0.0, "reconstruction_mse": 0.1},
        preprocess_stats=None,
    )


def test_profiles_and_lifecycle_statistics(al):
    r = al.analyse_condition(_synthetic_payload(), "baseline", sample_size=500, seed=0)
    assert r["n_active"] == D - 1 and 3 not in r["active"].tolist()
    assert r["labels"][3] == "inactive"
    assert r["labels"][0] == "persistent"
    assert r["labels"][1] == "early-decaying"
    assert r["labels"][2] == "late-emerging"
    assert r["peak_step"][2] == 1024 and r["birth_step"][2] == 256 and r["lifespan"][2] == 2
    assert r["birth_step"][3] == -1 and r["lifespan"][3] == 0
    assert r["dnorm_turn"].shape == r["rot_turn"].shape == (K - 1,)
    assert r["reorg_peak_step"] in STEPS[1:]
    assert set(r["prof_counts"]) == set(al.PROFILES) and sum(r["prof_counts"].values()) == r["n_active"]
    assert np.all(np.diff(r["peak_step"][r["sample"]]) >= 0)  # display sample sorted by peak step
    assert r["total_mass"].max() == pytest.approx(1.0)


def test_firing_rates_use_checkpoint_wu(al, tmp_path):
    root = tmp_path / "runs"
    for st in STEPS:
        dd = root / "baseline" / "ckpts" / f"step{st}"
        dd.mkdir(parents=True)
        torch.manual_seed(st)
        torch.save(
            {
                "embed_out.weight": torch.randn(V, d).half(),
                "gpt_neox.final_layer_norm.weight": torch.ones(d).half(),
                "gpt_neox.final_layer_norm.bias": torch.zeros(d).half(),
            },
            dd / "model_fp16.pt",
        )
    rates = al.firing_rates(_synthetic_payload(), root, "baseline")
    assert rates.shape == (K, D)
    assert np.all((rates >= 0) & (rates <= 1))
    assert np.all(rates[:, 3] == 0)  # zero decoder norm -> infinite effective threshold -> never fires


def test_main_persists_every_metric_file(al, tmp_path, monkeypatch):
    cc_dir = tmp_path / "cc"
    cc_dir.mkdir()
    torch.save(_synthetic_payload(), cc_dir / "cc_baseline_d8_seed0.pt")
    root = tmp_path / "runs"
    (root / "baseline" / "ckpts" / "step1024").mkdir(parents=True)
    (root / "baseline" / "config.json").write_text(json.dumps({"readout_lr_mult": 1.0, "warmup_steps": 1430}))
    (root / "baseline" / "ckpts" / "step1024" / "metrics.json").write_text(json.dumps({"val_loss": 3.3}))
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_lifecycle.py",
            "--cc-dir",
            str(cc_dir),
            "--ckpt-root",
            str(root),
            "--conditions",
            "baseline",
            "--d-sae",
            "8",
            "--out-dir",
            str(out),
            "--no-rates",
        ],
    )
    al.main()
    for name in (
        "lifecycle_stats.json",
        "lifecycle_summary.csv",
        "median_trajectories.csv",
        "peak_step_population.csv",
        "reorg_window.csv",
        "feature_lifecycles_baseline.csv",
        "feature_trajectories.pt",
        "provenance_analyze_lifecycle.json",
    ):
        assert (out / name).is_file(), name
    stats = json.loads((out / "lifecycle_stats.json").read_text())["baseline"]
    assert stats["n_active"] == D - 1 and stats["val_loss_final"] == 3.3 and stats["readout_lr_mult"] == 1.0
    assert stats["quality"]["explained_variance"] == 0.9
    with (out / "lifecycle_summary.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and float(rows[0]["median_peak_step"]) == stats["median_peak_step"]
    with (out / "reorg_window.csv").open(newline="") as fh:
        assert len(list(csv.DictReader(fh))) == K - 1
    traj = torch.load(out / "feature_trajectories.pt", weights_only=False)["baseline"]
    assert traj["rho"].shape == (K, D) and traj["rates"] is None and len(traj["profile"]) == D
