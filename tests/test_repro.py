"""Reproducibility tests for src/core/repro.py (CPU-only).

Substantiates the seeding/provenance claim: matched seeds reproduce RNG
draws across torch, NumPy, and Python `random`, and `log_run_provenance`
emits the documented keys.
"""

from __future__ import annotations

import random
import re

import numpy as np
import torch


def test_seed_everything_reproduces_torch():
    from src.core.repro import seed_everything

    seed_everything(0)
    a = torch.randn(4)
    seed_everything(0)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_seed_everything_reproduces_numpy():
    from src.core.repro import seed_everything

    seed_everything(0)
    a = np.random.rand(4)
    seed_everything(0)
    b = np.random.rand(4)
    assert np.array_equal(a, b)


def test_seed_everything_reproduces_python_random():
    from src.core.repro import seed_everything

    seed_everything(0)
    a = random.random()
    seed_everything(0)
    b = random.random()
    assert a == b


def test_seed_everything_same_sequence_matches():
    """Two seed-bracketed draws of the SAME multi-source sequence agree."""
    from src.core.repro import seed_everything

    def draw():
        return (
            torch.randn(4),
            np.random.rand(4).copy(),
            random.random(),
        )

    seed_everything(0)
    t1, n1, r1 = draw()
    seed_everything(0)
    t2, n2, r2 = draw()
    assert torch.equal(t1, t2)
    assert np.array_equal(n1, n2)
    assert r1 == r2


def test_different_seed_gives_different_draw():
    from src.core.repro import seed_everything

    seed_everything(0)
    a = torch.randn(4)
    seed_everything(1)
    b = torch.randn(4)
    assert not torch.equal(a, b)


def test_log_run_provenance_keys_without_seed():
    from src.core.repro import log_run_provenance

    info = log_run_provenance()
    assert isinstance(info, dict)
    for key in ("python", "platform", "git_hash", "git_dirty"):
        assert key in info, f"missing provenance key: {key}"
    assert "seed" not in info


def test_log_run_provenance_includes_seed_when_passed():
    from src.core.repro import log_run_provenance

    info = log_run_provenance(seed=123)
    assert info["seed"] == 123
    for key in ("python", "platform", "git_hash", "git_dirty"):
        assert key in info


def test_git_commit_returns_none_or_sha():
    from src.core.repro import git_commit

    h = git_commit()
    assert h is None or re.fullmatch(r"[0-9a-f]{40}", h), h
