"""Tests for src/readout/core/model_specs.py — the canonical model and step registry."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from readout.core.model_specs import (
    DEFAULT_STEPS_32,
    DEFAULT_STEPS_BY_MODEL,
    MODEL_HF_NAMES,
    OLMO_2_7B,
    OLMO_STEPS_32,
    PYTHIA_1B,
    PYTHIA_6_9B,
    PYTHIA_160M,
    SPECS,
    auto_steps_for,
    default_snap_dir,
    iter_snapshots,
    snap_path_for,
)

# ── step schedules ──────────────────────────────────────────────


def test_default_steps_32_is_canonical_pythia_schedule():
    assert len(DEFAULT_STEPS_32) == 32
    assert DEFAULT_STEPS_32[0] == 0
    assert DEFAULT_STEPS_32[-1] == 143000
    assert DEFAULT_STEPS_32 == sorted(DEFAULT_STEPS_32)
    assert len(set(DEFAULT_STEPS_32)) == 32


def test_default_steps_32_matches_wu_adapter():
    """Trajectory crosscoders are trained from wu_adapter.DEFAULT_STEPS;
    the analysis-side copy must stay in sync or downstream rate caches
    silently disagree with checkpoint contents."""
    from readout.crosscoder.wu_adapter import DEFAULT_STEPS

    assert list(DEFAULT_STEPS_32) == list(DEFAULT_STEPS)


def test_olmo_steps_32_is_olmo_specific():
    assert len(OLMO_STEPS_32) == 32
    assert OLMO_STEPS_32[0] == 150  # OLMo-2 first checkpoint, not 0
    assert OLMO_STEPS_32[-1] == 928000
    assert OLMO_STEPS_32 == sorted(OLMO_STEPS_32)
    # Late OLMo grid extends beyond Pythia's 143000 ceiling.
    assert max(OLMO_STEPS_32) > max(DEFAULT_STEPS_32)


def test_default_steps_by_model_covers_all_specs():
    assert set(DEFAULT_STEPS_BY_MODEL.keys()) == set(SPECS.keys())
    for label in ("pythia-160m", "pythia-1b", "pythia-6.9b"):
        assert DEFAULT_STEPS_BY_MODEL[label] is DEFAULT_STEPS_32
    assert DEFAULT_STEPS_BY_MODEL["olmo-2-7b"] is OLMO_STEPS_32


# ── ModelSpec / SPECS ───────────────────────────────────────────


def test_model_spec_is_frozen():
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError or AttributeError
        PYTHIA_1B.d_model = 9999  # type: ignore[misc]


def test_specs_keyed_by_short_label():
    assert SPECS["pythia-160m"] is PYTHIA_160M
    assert SPECS["pythia-1b"] is PYTHIA_1B
    assert SPECS["pythia-6.9b"] is PYTHIA_6_9B
    assert SPECS["olmo-2-7b"] is OLMO_2_7B


def test_pythia_specs_have_correct_d_model():
    """Catch accidental d_model swaps that would silently break shape checks."""
    assert PYTHIA_160M.d_model == 768
    assert PYTHIA_1B.d_model == 2048
    assert PYTHIA_6_9B.d_model == 4096
    assert OLMO_2_7B.d_model == 4096


def test_pythia_specs_share_vocab_size():
    """Pythia tokenizer is shared across sizes; OLMo uses its own."""
    assert PYTHIA_160M.vocab == PYTHIA_1B.vocab == 50304
    assert PYTHIA_6_9B.vocab == 50432  # padded
    assert OLMO_2_7B.vocab == 100352  # different tokenizer


def test_model_hf_names_round_trip_through_specs():
    for label, hf in MODEL_HF_NAMES.items():
        assert SPECS[label].name == hf


# ── snap_path_for / auto_steps_for ──────────────────────────────


@pytest.fixture
def fake_snap_dir(tmp_path: Path) -> Path:
    """Build a minimal canonical snapshot layout under tmp_path."""
    layout = tmp_path / "pythia-160m"
    layout.mkdir()
    for step in (0, 1000, 143000):
        (layout / f"EleutherAI_pythia-160m_step{step}_wu.pt").write_bytes(b"")
    # Sprinkle in a macOS metadata file to make sure auto_steps_for skips it.
    (layout / "._EleutherAI_pythia-160m_step9999_wu.pt").write_bytes(b"")
    # And a we-suffix file to make sure matrix=W_U doesn't pick it up.
    (layout / "EleutherAI_pythia-160m_step1_we.pt").write_bytes(b"")
    return tmp_path


def test_snap_path_for_finds_per_model_subdir(fake_snap_dir: Path):
    p = snap_path_for("EleutherAI/pythia-160m", 1000, snap_dir=fake_snap_dir)
    assert p.exists()
    assert p.name == "EleutherAI_pythia-160m_step1000_wu.pt"


def test_snap_path_for_falls_back_to_flat_layout(tmp_path: Path):
    """Some old caches keep all snapshots flat under snap_dir/, no per-model subdir."""
    (tmp_path / "EleutherAI_pythia-1b_step0_wu.pt").write_bytes(b"")
    p = snap_path_for("EleutherAI/pythia-1b", 0, snap_dir=tmp_path)
    assert p.parent == tmp_path  # flat layout, not per-model


def test_snap_path_for_raises_on_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        snap_path_for("EleutherAI/pythia-160m", 999999, snap_dir=tmp_path)


def test_snap_path_for_we_matrix_distinguishes_suffix(fake_snap_dir: Path):
    p = snap_path_for("EleutherAI/pythia-160m", 1, matrix="W_E", snap_dir=fake_snap_dir)
    assert p.name.endswith("_we.pt")


def test_auto_steps_for_returns_sorted_unique_ints(fake_snap_dir: Path):
    steps = auto_steps_for(PYTHIA_160M, snap_dir=fake_snap_dir)
    assert steps == [0, 1000, 143000]
    assert all(isinstance(s, int) for s in steps)


def test_auto_steps_for_skips_macos_dotfiles(fake_snap_dir: Path):
    """Verify the .9999 dotfile in the fixture is not picked up."""
    steps = auto_steps_for(PYTHIA_160M, snap_dir=fake_snap_dir)
    assert 9999 not in steps


def test_auto_steps_for_we_matrix(fake_snap_dir: Path):
    steps = auto_steps_for(PYTHIA_160M, matrix="W_E", snap_dir=fake_snap_dir)
    assert steps == [1]


def test_auto_steps_for_empty_dir_returns_empty_list(tmp_path: Path):
    assert auto_steps_for(PYTHIA_160M, snap_dir=tmp_path) == []


# ── iter_snapshots ──────────────────────────────────────────────


def test_iter_snapshots_yields_step_tensor_pairs(tmp_path: Path):
    layout = tmp_path / "pythia-160m"
    layout.mkdir()
    # Real .pt with a small tensor so load_snapshot_at can read it.
    torch.manual_seed(0)
    for step in (0, 1000):
        torch.save(
            torch.randn(8, PYTHIA_160M.d_model, dtype=torch.float32),
            layout / f"EleutherAI_pythia-160m_step{step}_wu.pt",
        )
    pairs = list(iter_snapshots(PYTHIA_160M, steps=[0, 1000], snap_dir=tmp_path))
    assert [s for s, _ in pairs] == [0, 1000]
    assert all(t.shape == (8, PYTHIA_160M.d_model) for _, t in pairs)
    assert all(t.dtype == torch.float32 for _, t in pairs)


# ── default_snap_dir ────────────────────────────────────────────


def test_default_snap_dir_honours_wu_snap_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WU_SNAP_DIR", str(tmp_path))
    assert default_snap_dir() == tmp_path


def test_default_snap_dir_falls_back_to_ssd_root(monkeypatch):
    monkeypatch.delenv("WU_SNAP_DIR", raising=False)
    monkeypatch.setenv("UM_SSD_ROOT", "/tmp/fake-ssd")
    assert default_snap_dir() == Path("/tmp/fake-ssd/snapshots")
