"""Tests for src/core/paths.py."""

from __future__ import annotations

import pytest

from src.core.paths import (
    PROJECT,
    aggregate_path,
    crosscoder_main_derived_dir,
    model_short,
    model_slug,
    release_path,
    repo_root,
    snapshot_dir,
    snapshot_path,
    ssd_root,
)

# ── repo_root ──────────────────────────────────────────────────


def test_repo_root_finds_pyproject_or_git():
    root = repo_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists() or (root / ".git").exists()


def test_repo_root_matches_project_constant():
    """The hand-rolled PROJECT (parents[2]) and dynamic repo_root should agree
    while paths.py stays at src/core/paths.py."""
    assert repo_root() == PROJECT


# ── ssd_root / model_short / model_slug ────────────────────────


def test_ssd_root_honours_um_ssd_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("UM_SSD_ROOT", str(tmp_path))
    assert ssd_root() == tmp_path


def test_ssd_root_default_when_unset(monkeypatch):
    monkeypatch.delenv("UM_SSD_ROOT", raising=False)
    assert ssd_root() == PROJECT / "local_snapshots"


def test_model_short_known_models():
    assert model_short("EleutherAI/pythia-160m") == "pythia-160m"
    assert model_short("EleutherAI/pythia-1b") == "pythia-1b"
    assert model_short("EleutherAI/pythia-6.9b") == "pythia-6.9b"
    assert model_short("allenai/OLMo-2-1124-7B") == "OLMo-2-1124-7B"


def test_model_short_unknown_raises():
    with pytest.raises(KeyError, match="Unknown"):
        model_short("not/a-real-model")


def test_model_slug_replaces_slash():
    assert model_slug("EleutherAI/pythia-1b") == "EleutherAI_pythia-1b"
    assert model_slug("allenai/OLMo-2-1124-7B") == "allenai_OLMo-2-1124-7B"


# ── snapshot_path / snapshot_dir ───────────────────────────────


def test_snapshot_path_canonical_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("UM_SSD_ROOT", str(tmp_path))
    p = snapshot_path("EleutherAI/pythia-160m", 1000, kind="wu")
    assert p == tmp_path / "snapshots/pythia-160m/EleutherAI_pythia-160m_step1000_wu.pt"


def test_snapshot_path_we_kind(monkeypatch, tmp_path):
    monkeypatch.setenv("UM_SSD_ROOT", str(tmp_path))
    p = snapshot_path("EleutherAI/pythia-1b", 0, kind="we")
    assert p.name.endswith("_step0_we.pt")


def test_snapshot_path_rejects_unknown_kind(monkeypatch, tmp_path):
    monkeypatch.setenv("UM_SSD_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="kind"):
        snapshot_path("EleutherAI/pythia-160m", 0, kind="xx")


def test_snapshot_dir_per_model(monkeypatch, tmp_path):
    monkeypatch.setenv("UM_SSD_ROOT", str(tmp_path))
    assert snapshot_dir("EleutherAI/pythia-1b") == tmp_path / "snapshots/pythia-1b"


# ── release_path / aggregate_path ──────────────────────────────


def test_release_path_default_snapshots(monkeypatch, tmp_path):
    monkeypatch.setenv("UM_SSD_ROOT", str(tmp_path))
    p = release_path("pythia-1b", "W_U", dim=24576, seed=0)
    assert p == (
        tmp_path
        / "hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors"
    )


def test_release_path_custom_snapshot_count(monkeypatch, tmp_path):
    monkeypatch.setenv("UM_SSD_ROOT", str(tmp_path))
    p = release_path("pythia-160m", "W_E", dim=8192, seed=2, snapshots=16)
    assert "cross-snapshot-16" in str(p)
    assert p.name == "seed2.safetensors"


def test_aggregate_path(monkeypatch, tmp_path):
    monkeypatch.setenv("UM_SSD_ROOT", str(tmp_path))
    p = aggregate_path("pythia-6.9b_d32768_seed0")
    assert p == tmp_path / "derived/aggregates/aggregates_pythia-6.9b_d32768_seed0.pt"


def test_crosscoder_main_derived_dir_under_repo():
    p = crosscoder_main_derived_dir()
    assert p.is_relative_to(PROJECT)
    assert p.name == "derived"
