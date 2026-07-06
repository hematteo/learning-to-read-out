"""Coverage for the most-imported IO/infra modules that previously had none:
snapshot loaders, checkpoint loaders, the LR schedule, the resumability
primitives, and the production permutation null.
"""

from __future__ import annotations

import csv
import json

import pytest
import torch

from readout.baselines.permutation_test_fast import (
    cusum_max_feature_batched,
    cusum_max_feature_single,
    permutation_test_vectorized,
)
from readout.core.resume import (
    aggregate_json_shards,
    append_manifest,
    atomic_write_json,
    atomic_write_torch,
    iter_undone,
)
from readout.crosscoder._schedules import lr_multiplier
from readout.crosscoder.checkpoints import load_checkpoint, unwrap_ckpt
from readout.crosscoder.snapshots import load_snapshot, load_snapshot_at, load_snapshots

# ── _schedules.lr_multiplier ─────────────────────────────────────


def test_lr_multiplier_constant_without_warmup_or_decay():
    assert all(lr_multiplier(s, 100, 0.0, 0.0) == 1.0 for s in (0, 50, 99))


def test_lr_multiplier_warmup_ramp_and_plateau():
    # 10% warmup over 100 steps -> steps 0..9 ramp (s+1)/10, then 1.0.
    assert lr_multiplier(0, 100, 0.1, 0.0) == pytest.approx(0.1)
    assert lr_multiplier(9, 100, 0.1, 0.0) == pytest.approx(1.0)
    assert lr_multiplier(10, 100, 0.1, 0.0) == 1.0
    assert lr_multiplier(50, 100, 0.1, 0.0) == 1.0


def test_lr_multiplier_decay_tail():
    # 20% decay over 100 steps -> decay starts at step 80, hits 0 at step 100.
    assert lr_multiplier(79, 100, 0.0, 0.2) == 1.0
    assert lr_multiplier(80, 100, 0.0, 0.2) == pytest.approx(1.0)
    assert lr_multiplier(90, 100, 0.0, 0.2) == pytest.approx(0.5)
    assert lr_multiplier(100, 100, 0.0, 0.2) == 0.0


# ── snapshots ────────────────────────────────────────────────────


def _fake_ssd(monkeypatch, tmp_path, model="EleutherAI/pythia-160m", steps=(0, 1000)):
    monkeypatch.setenv("UM_SSD_ROOT", str(tmp_path))
    snap_dir = tmp_path / "snapshots" / "pythia-160m"
    snap_dir.mkdir(parents=True)
    tensors = {}
    for i, s in enumerate(steps):
        W = torch.full((6, 4), float(i), dtype=torch.float32)
        # exercise both on-disk conventions: raw tensor and {"W_U": ...} dict
        payload = W if i % 2 == 0 else {"W_U": W}
        torch.save(payload, snap_dir / f"EleutherAI_pythia-160m_step{s}_wu.pt")
        tensors[s] = W
    return snap_dir, tensors


def test_load_snapshot_unwraps_and_stacks(monkeypatch, tmp_path):
    snap_dir, tensors = _fake_ssd(monkeypatch, tmp_path)
    one = load_snapshot("EleutherAI/pythia-160m", 1000)
    assert torch.equal(one, tensors[1000])
    stack = load_snapshots("EleutherAI/pythia-160m", [0, 1000])
    assert stack.shape == (2, 6, 4)
    assert torch.equal(stack[0], tensors[0]) and torch.equal(stack[1], tensors[1000])


def test_load_snapshot_dtype_and_dir_override(monkeypatch, tmp_path):
    snap_dir, _ = _fake_ssd(monkeypatch, tmp_path)
    x = load_snapshot("EleutherAI/pythia-160m", 0, dtype=torch.float64)
    assert x.dtype == torch.float64
    # dir_override resolves the canonical filename inside another directory
    alt = tmp_path / "elsewhere"
    alt.mkdir()
    (alt / "EleutherAI_pythia-160m_step0_wu.pt").write_bytes(
        (snap_dir / "EleutherAI_pythia-160m_step0_wu.pt").read_bytes()
    )
    y = load_snapshot("EleutherAI/pythia-160m", 0, dir_override=alt)
    assert torch.equal(x.float(), y)


def test_load_snapshot_at_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_snapshot_at(tmp_path / "nope.pt")


# ── checkpoints ──────────────────────────────────────────────────


def _sd():
    return {"W_D": torch.randn(2, 3, 4), "b_D": torch.randn(2, 4)}


def test_load_checkpoint_legacy_flat_pt(tmp_path):
    p = tmp_path / "legacy.pt"
    torch.save(
        {
            "state_dict": _sd(),
            "steps": [0, 143000],
            "model_name": "EleutherAI/pythia-160m",
            "training": {"lr": 5e-5},
            "quality": {"explained_variance": 0.9},
        },
        p,
    )
    ck = load_checkpoint(p)
    assert ck.steps == [0, 143000]
    assert ck.model_name == "EleutherAI/pythia-160m"
    assert ck.training == {"lr": 5e-5}
    assert ck.quality["explained_variance"] == 0.9
    assert set(ck.state_dict) == {"W_D", "b_D"}


def test_unwrap_ckpt_new_format_nested_config():
    ck = {
        "state_dict": _sd(),
        "config": {"steps": [0, 1], "model_name": "m", "config": {"d_sae": 8}},
    }
    flat = unwrap_ckpt(ck)
    assert flat["steps"] == [0, 1] and flat["model_name"] == "m"
    assert flat["config"] == {"d_sae": 8}
    # old-format checkpoints (metadata already at top level) pass through
    old = {"state_dict": _sd(), "steps": [0], "model_name": "m", "config": {"sae_type": "crosscoder"}}
    assert unwrap_ckpt(old) is old


def test_load_checkpoint_safetensors_with_sidecar(tmp_path):
    from safetensors.torch import save_file

    p = tmp_path / "cc.safetensors"
    save_file({k: v.contiguous() for k, v in _sd().items()}, str(p))
    (tmp_path / "cc.config.json").write_text(
        json.dumps({"steps": [0, 1000], "model_name": "m", "training": {"seed": 0}})
    )
    ck = load_checkpoint(p)
    assert ck.steps == [0, 1000] and ck.model_name == "m" and ck.raw is None
    assert torch.is_tensor(ck.state_dict["W_D"])


# ── resume primitives ────────────────────────────────────────────


def test_atomic_writes_roundtrip(tmp_path):
    j = tmp_path / "a" / "x.json"
    atomic_write_json(j, {"k": 1})
    assert json.loads(j.read_text()) == {"k": 1}
    t = tmp_path / "a" / "x.pt"
    atomic_write_torch(t, {"w": torch.ones(2)})
    assert torch.equal(torch.load(t, weights_only=False)["w"], torch.ones(2))
    # no stray .tmp files left behind
    assert not list((tmp_path / "a").glob("*.tmp"))


def test_iter_undone_skips_existing_shards(tmp_path, capsys):
    def shard(i):
        return tmp_path / f"s{i}.json"

    atomic_write_json(shard(1), {})
    todo = list(iter_undone([1, 2, 3], shard, label="step"))
    assert todo == [2, 3]
    assert "1/3" in capsys.readouterr().out


def test_append_manifest_and_aggregate_shards(tmp_path):
    m = tmp_path / "manifest.csv"
    append_manifest(m, {"step": 2, "ev": 0.5})
    append_manifest(m, {"step": 1, "ev": 0.7})
    rows = list(csv.DictReader(m.open()))
    assert [r["step"] for r in rows] == ["2", "1"]  # append-only, header once

    shards = tmp_path / "shards"
    atomic_write_json(shards / "b.json", {"step": 2, "ev": 0.5, "extra": "x"})
    atomic_write_json(shards / "a.json", {"step": 1, "ev": 0.7})
    out = tmp_path / "agg.csv"
    n = aggregate_json_shards(shards, out, key="step")
    assert n == 2
    agg = list(csv.DictReader(out.open()))
    assert [r["step"] for r in agg] == ["1", "2"]  # sorted by key
    assert set(agg[0]) == {"step", "ev", "extra"}  # union of fieldnames


# ── permutation null (production statistic) ──────────────────────


def test_cusum_batched_matches_single_and_is_batch_invariant():
    # K=32 is the production snapshot count. (For K<=8 the robust-floor MAD
    # baseline degenerates to zero and the statistic is NaN — never hit by the
    # shipped schedules, whose smallest K is 9.)
    torch.manual_seed(0)
    rates = torch.rand(32, 64)  # (K, D)
    single = cusum_max_feature_single(rates)
    batch = cusum_max_feature_batched(rates.unsqueeze(0).repeat(5, 1, 1))
    assert batch.shape == (5,)
    assert torch.isfinite(batch).all()
    assert torch.allclose(batch, torch.full((5,), single), atol=1e-5)


def test_permutation_test_seeded_and_calibrated_under_null():
    torch.manual_seed(0)
    # Null data: iid rates with no temporal structure -> p should not be tiny.
    rates = torch.rand(32, 64)
    obs1, p1, null1, z1, _ = permutation_test_vectorized(rates, n_permutations=200, batch=100, seed=7)
    obs2, p2, null2, z2, _ = permutation_test_vectorized(rates, n_permutations=200, batch=100, seed=7)
    assert obs1 == obs2 and p1 == p2 and (null1 == null2).all()  # same seed -> identical
    assert p1 > 0.01  # permutation-invariant data must not look significant
    _, p3, _, _, _ = permutation_test_vectorized(rates, n_permutations=200, batch=100, seed=8)
    assert p3 > 0.01
