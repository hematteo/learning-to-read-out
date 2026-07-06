"""Tests for src/readout/crosscoder/checkpoint_loaders.py.

Covers the cache-priority logic in `load_run`, the per-snap directory
fallback, and the WU_CACHE_DIR resolver — without depending on
${UM_SSD_ROOT}/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from readout.core.paths import ssd_path
from readout.crosscoder.checkpoint_loaders import (
    RunArrays,
    _decoder_norms_from_ckpt,
    _decoder_norms_from_per_snap_dir,
    _load_npy_or_pt,
    _resolve_wu_cache_dir,
    load_run,
)

K = 4
D = 16
D_MODEL = 8
STEPS = [0, 100, 1000, 143000]


# ── _load_npy_or_pt ────────────────────────────────────────────


def test_load_npy_or_pt_returns_none_on_missing(tmp_path: Path):
    assert _load_npy_or_pt(tmp_path / "no.npy") is None


def test_load_npy_or_pt_reads_npy(tmp_path: Path):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    p = tmp_path / "a.npy"
    np.save(p, arr)
    out = _load_npy_or_pt(p)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)


def test_load_npy_or_pt_reads_pt(tmp_path: Path):
    payload = {"rates_per_seed": {"foo": torch.randn(K, D)}}
    p = tmp_path / "a.pt"
    torch.save(payload, p)
    out = _load_npy_or_pt(p)
    assert isinstance(out, dict)
    assert "rates_per_seed" in out


def test_load_npy_or_pt_rejects_unknown_extension(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="unsupported"):
        _load_npy_or_pt(p)


# ── _decoder_norms_from_ckpt ───────────────────────────────────


def _make_ckpt(tmp_path: Path, *, with_steps: bool = True) -> Path:
    W_D = torch.randn(K, D, D_MODEL)
    payload = {
        "state_dict": {"W_D": W_D},
        "model_name": "EleutherAI/pythia-160m",
    }
    if with_steps:
        payload["steps"] = STEPS
    p = tmp_path / "ckpt.pt"
    torch.save(payload, p)
    return p


def test_decoder_norms_from_ckpt_uses_embedded_steps(tmp_path: Path):
    p = _make_ckpt(tmp_path, with_steps=True)
    norms, w_d, steps = _decoder_norms_from_ckpt(p)
    assert norms.shape == (K, D)
    assert w_d.shape == (K, D, D_MODEL)
    assert steps == STEPS
    # Norms should match recomputed L2 over last axis.
    expected = torch.linalg.norm(torch.from_numpy(w_d), dim=-1).numpy()
    np.testing.assert_allclose(norms, expected, rtol=1e-5)


def test_decoder_norms_from_ckpt_falls_back_to_steps_when_missing(tmp_path: Path):
    p = _make_ckpt(tmp_path, with_steps=False)
    _, _, steps = _decoder_norms_from_ckpt(p, steps_fallback=STEPS)
    assert steps == STEPS


def test_decoder_norms_from_ckpt_raises_on_missing_steps_no_fallback(tmp_path: Path):
    p = _make_ckpt(tmp_path, with_steps=False)
    with pytest.raises(KeyError, match="steps"):
        _decoder_norms_from_ckpt(p)


def test_decoder_norms_from_ckpt_rejects_wrong_length_fallback(tmp_path: Path):
    p = _make_ckpt(tmp_path, with_steps=False)
    with pytest.raises(KeyError, match="steps"):
        _decoder_norms_from_ckpt(p, steps_fallback=[0, 1])  # K=4, len(fallback)=2


# ── _decoder_norms_from_per_snap_dir ───────────────────────────


def test_decoder_norms_from_per_snap_dir_stacks_files(tmp_path: Path):
    """Per-snap SAE layout: one .pt per snapshot, decoder shape (D, d)."""
    for s in STEPS:
        torch.save(
            {"state_dict": {"W_D": torch.randn(D, D_MODEL)}},
            tmp_path / f"wu_sae_dsae{D}_step{s}.pt",
        )
    norms, w_d, steps = _decoder_norms_from_per_snap_dir(tmp_path, steps=STEPS, d_sae=D)
    assert w_d.shape == (K, D, D_MODEL)
    assert norms.shape == (K, D)
    assert steps == STEPS


def test_decoder_norms_from_per_snap_dir_skips_missing(tmp_path: Path):
    """Some steps may be absent on disk; loader returns the available subset."""
    for s in (STEPS[0], STEPS[-1]):
        torch.save(
            {"state_dict": {"W_D": torch.randn(D, D_MODEL)}},
            tmp_path / f"wu_sae_dsae{D}_step{s}.pt",
        )
    norms, w_d, steps = _decoder_norms_from_per_snap_dir(tmp_path, steps=STEPS, d_sae=D)
    assert steps == [STEPS[0], STEPS[-1]]
    assert w_d.shape == (2, D, D_MODEL)


def test_decoder_norms_from_per_snap_dir_raises_when_empty(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no per-snap"):
        _decoder_norms_from_per_snap_dir(tmp_path, steps=STEPS, d_sae=D)


# ── load_run cache priority ────────────────────────────────────


@pytest.fixture
def synthetic_run(tmp_path: Path):
    """Build a checkpoint + matching rates/norms cache files."""
    W_D = torch.randn(K, D, D_MODEL)
    ckpt = tmp_path / "run.pt"
    torch.save(
        {
            "state_dict": {"W_D": W_D},
            "steps": STEPS,
            "model_name": "EleutherAI/pythia-160m",
        },
        ckpt,
    )
    norms = torch.linalg.norm(W_D, dim=-1).numpy().astype(np.float32)
    rates = np.random.RandomState(0).rand(K, D).astype(np.float32)

    rates_pt = tmp_path / "rates.pt"
    torch.save({"rates_per_seed": {"run": torch.from_numpy(rates)}}, rates_pt)

    norms_npy = tmp_path / "norms.npy"
    np.save(norms_npy, norms)

    return {
        "ckpt": ckpt,
        "rates_pt": rates_pt,
        "norms_npy": norms_npy,
        "rates": rates,
        "norms": norms,
        "W_D": W_D.numpy().astype(np.float32),
    }


def test_load_run_uses_cached_rates_and_norms(synthetic_run):
    out = load_run(
        run_id="run",
        seed=0,
        d_sae=D,
        ckpt_path=synthetic_run["ckpt"],
        rates_cache=synthetic_run["rates_pt"],
        norms_cache=synthetic_run["norms_npy"],
    )
    assert isinstance(out, RunArrays)
    assert out.rates is not None
    np.testing.assert_array_equal(out.rates, synthetic_run["rates"])
    np.testing.assert_array_equal(out.decoder_norms, synthetic_run["norms"])
    assert out.steps == STEPS
    # Decoder weights are dropped by default to avoid OOM.
    assert out.decoder_weights is None


def test_load_run_keep_decoder_weights_returns_full_tensor(synthetic_run):
    out = load_run(
        run_id="run",
        seed=0,
        d_sae=D,
        ckpt_path=synthetic_run["ckpt"],
        rates_cache=synthetic_run["rates_pt"],
        norms_cache=synthetic_run["norms_npy"],
        keep_decoder_weights=True,
    )
    assert out.decoder_weights is not None
    assert out.decoder_weights.shape == (K, D, D_MODEL)
    np.testing.assert_allclose(out.decoder_weights, synthetic_run["W_D"], rtol=1e-5)


def test_load_run_falls_back_to_ckpt_when_norms_cache_missing(synthetic_run):
    """If no cached norms, derive them from the checkpoint's W_D."""
    out = load_run(
        run_id="run",
        seed=0,
        d_sae=D,
        ckpt_path=synthetic_run["ckpt"],
        rates_cache=synthetic_run["rates_pt"],
        norms_cache=None,
    )
    np.testing.assert_allclose(out.decoder_norms, synthetic_run["norms"], rtol=1e-5)


def test_load_run_rejects_d_sae_mismatch(synthetic_run):
    with pytest.raises(ValueError, match="d_sae mismatch"):
        load_run(
            run_id="run",
            seed=0,
            d_sae=D + 1,  # caller asserts wrong dim
            ckpt_path=synthetic_run["ckpt"],
            rates_cache=synthetic_run["rates_pt"],
            norms_cache=synthetic_run["norms_npy"],
        )


def test_load_run_multi_seed_npy_requires_seed_index(synthetic_run, tmp_path: Path):
    """Multi-seed rate caches are (S, K, D); caller must select the row."""
    rates_3d = np.stack([synthetic_run["rates"], synthetic_run["rates"] * 2.0], axis=0)
    multi = tmp_path / "rates_multi.npy"
    np.save(multi, rates_3d)
    with pytest.raises(ValueError, match="seed_index_in_cache"):
        load_run(
            run_id="run",
            seed=0,
            d_sae=D,
            ckpt_path=synthetic_run["ckpt"],
            rates_cache=multi,
            norms_cache=synthetic_run["norms_npy"],
        )
    # With seed_index_in_cache it should pick the right row.
    out = load_run(
        run_id="run",
        seed=1,
        d_sae=D,
        ckpt_path=synthetic_run["ckpt"],
        rates_cache=multi,
        norms_cache=synthetic_run["norms_npy"],
        seed_index_in_cache=1,
    )
    np.testing.assert_allclose(out.rates, synthetic_run["rates"] * 2.0, rtol=1e-5)


def test_load_run_raises_when_ckpt_missing_and_caches_incomplete(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_run(
            run_id="run",
            seed=0,
            d_sae=D,
            ckpt_path=tmp_path / "no.pt",
            rates_cache=None,
            norms_cache=None,
        )


# ── _resolve_wu_cache_dir ──────────────────────────────────────


def test_resolve_wu_cache_dir_uses_env(monkeypatch, tmp_path: Path):
    """WU_CACHE_DIR overrides the fallback search."""
    monkeypatch.setenv("WU_CACHE_DIR", str(tmp_path))
    assert _resolve_wu_cache_dir(tmp_path / "ignored.pt") == tmp_path


def test_resolve_wu_cache_dir_raises_when_no_default(monkeypatch, tmp_path: Path):
    """No env var and no default snapshot dir on disk → FileNotFoundError."""
    monkeypatch.delenv("WU_CACHE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # cwd-fallback won't have cluster_data/wu_snapshots
    # Point the scratch fallback at a nonexistent tmp dir so the negative case
    # is deterministic regardless of the host.
    monkeypatch.setenv("CLUSTER_SCRATCH", str(tmp_path / "scratch"))

    # The remaining candidate is the SSD layout; if the test machine genuinely
    # has it, skip — this test is about the negative case.
    ssd = ssd_path("wu_crosscoder", "snapshots")
    if ssd.exists():
        pytest.skip("test machine has a default snapshot dir; negative case n/a")

    with pytest.raises(FileNotFoundError, match="WU_CACHE_DIR"):
        _resolve_wu_cache_dir(tmp_path / "any.pt")
