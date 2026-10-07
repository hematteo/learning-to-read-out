"""Regression tests for scripts/audit/check_layout.py on a tmp_path fake repo.

Two audit behaviors are pinned here because both regressed silently before:

  1. experiment leaf dirs are discovered STRUCTURALLY (any non-hidden,
     non-underscore ``experiments/<topic>/<id>/`` dir), so every experiment is
     checked against the manifest instead of vanishing from discovery.
  2. check 10 flags ``scripts/`` files importing the ``scripts`` or
     ``experiments`` packages (the entry layer depends only on src/).

The script is import-only-safe (module-level REPO/EXPERIMENTS_YAML constants,
no side effects), so it is loaded via importlib — the precedent set by
tests/test_feature_attribution_algebra.py — and the constants monkeypatched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_check_layout():
    repo = Path(__file__).resolve().parents[1]
    p = repo / "scripts/audit/check_layout.py"
    spec = importlib.util.spec_from_file_location("check_layout", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["check_layout"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def cl():
    return _load_check_layout()


def _point_at_fake_repo(monkeypatch, cl, repo: Path):
    """Retarget the module-level path constants at a tmp fake repo."""
    monkeypatch.setattr(cl, "REPO", repo)
    monkeypatch.setattr(cl, "EXPERIMENTS_YAML", repo / "experiments.yaml")
    monkeypatch.setattr(cl, "PAPER_TEX", repo / "paper" / "main.tex")
    monkeypatch.setattr(cl, "README", repo / "README.md")


# ── experiment discovery is structural ─────────────────────────────


def test_experiment_dirs_are_discovered_structurally(tmp_path, monkeypatch, cl):
    """Any non-underscore experiments/<topic>/<id>/ dir is a leaf, documented or not."""
    (tmp_path / "experiments" / "lifecycle" / "exp_a").mkdir(parents=True)
    (tmp_path / "experiments" / "lifecycle" / "exp_b" / "scripts").mkdir(parents=True)
    # Underscore-prefixed dirs are internal and stay exempt.
    (tmp_path / "experiments" / "lifecycle" / "_wip").mkdir()
    _point_at_fake_repo(monkeypatch, cl, tmp_path)
    assert [p.name for p in cl._experiment_dirs()] == ["exp_a", "exp_b"]


# ── check 10: scripts/ must not import scripts.* / experiments.* ───────


def test_scripts_forbidden_imports_trip_check_10(tmp_path, monkeypatch, cl):
    sdir = tmp_path / "scripts" / "train"
    sdir.mkdir(parents=True)
    (sdir / "bad.py").write_text("import scripts.foo\nfrom experiments.foo import bar\n")
    _point_at_fake_repo(monkeypatch, cl, tmp_path)

    issues = cl.check_scripts_imports()
    assert len(issues) == 2, f"expected both forbidden imports flagged, got {issues}"
    assert all("[check-10]" in i and "bad.py" in i for i in issues)
    assert any("'scripts'" in i for i in issues)
    assert any("'experiments'" in i for i in issues)


def test_scripts_word_boundary_and_readout_imports_pass(tmp_path, monkeypatch, cl):
    """`import scriptsutil` / `import experimentskit` are NOT the forbidden
    packages (word boundary), and src/ imports are always allowed."""
    sdir = tmp_path / "scripts" / "eval"
    sdir.mkdir(parents=True)
    (sdir / "good.py").write_text("import readout\nimport scriptsutil\nfrom experimentskit import x\n")
    _point_at_fake_repo(monkeypatch, cl, tmp_path)
    assert cl.check_scripts_imports() == []
