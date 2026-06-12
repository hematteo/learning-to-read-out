"""Validate the committed crosscoder run configs (CPU-only, no data/GPU).

These are the settings-of-record under ``configs/runs/`` (see
``configs/README.md``). Each file must parse, identify its model, and carry
numeric hyperparameters as numbers — guarding against a stray quote or typo
silently corrupting the reproducibility record.
"""

from __future__ import annotations

import glob
import numbers
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNS_DIR = _REPO_ROOT / "configs" / "runs"
_RUN_CONFIGS = sorted(glob.glob(str(_RUNS_DIR / "*.yaml")))

# Hyperparameter keys that, when present, must be numeric (int/float).
_NUMERIC_KEYS = (
    "d_sae",
    "expansion_factor",
    "n_snapshots",
    "batch_size",
    "lr",
    "n_epochs",
    "seed",
    "l1_coefficient",
    "tanh_stretch",
    "init_threshold",
    "jumprelu_lr_factor",
    "decoder_transpose_init",
    "l1_warmup_fraction",
    "lr_warmup_fraction",
    "lr_decay_fraction",
)


def test_run_configs_exist():
    assert _RUN_CONFIGS, f"no run configs found under {_RUNS_DIR}"


@pytest.mark.parametrize("path", _RUN_CONFIGS, ids=lambda p: Path(p).name)
def test_run_config_parses_and_has_required_fields(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    assert isinstance(cfg, dict), f"{path} did not parse to a mapping"
    for key in ("name", "model"):
        assert key in cfg, f"{path} missing required key: {key}"
        assert isinstance(cfg[key], str) and cfg[key], (
            f"{path}: {key} must be a non-empty string"
        )


@pytest.mark.parametrize("path", _RUN_CONFIGS, ids=lambda p: Path(p).name)
def test_run_config_numeric_hyperparams_are_numbers(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for key in _NUMERIC_KEYS:
        if key in cfg:
            assert isinstance(cfg[key], numbers.Number) and not isinstance(
                cfg[key], bool
            ), f"{path}: {key} should be numeric, got {type(cfg[key]).__name__}"
