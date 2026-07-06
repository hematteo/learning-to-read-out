"""End-to-end smoke of the training CLI, fully offline.

The unit suite exercises the library; nothing exercised the argparse wiring of
scripts/train/train_crosscoder.py itself. Pre-seeding the snapshot cache with
tiny synthetic W_U tensors at the canonical filenames lets the real CLI run
train -> save -> (reload) on CPU in a few seconds with no network, and pins
the README's determinism claim at both the CLI and library level.
"""

from __future__ import annotations

import subprocess
import sys

import torch

from readout.core.paths import repo_root
from readout.core.repro import seed_everything
from readout.crosscoder.checkpoints import load_checkpoint
from readout.crosscoder.wu_adapter import train

MODEL = "EleutherAI/pythia-160m"
STEPS = (0, 1000)
V, D_MODEL = 64, 16


def _seed_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    g = torch.Generator().manual_seed(0)
    for s in STEPS:
        W = torch.randn(V, D_MODEL, generator=g)
        torch.save(W, cache / f"EleutherAI_pythia-160m_step{s}_wu.pt")
    return cache


def _run_cli(cache, out, seed=0):
    cmd = [
        sys.executable,
        str(repo_root() / "scripts/train/train_crosscoder.py"),
        "--model",
        MODEL,
        "--steps",
        *[str(s) for s in STEPS],
        "--cache-dir",
        str(cache),
        "--expansion-factor",
        "4.0",
        "--n-epochs",
        "2",
        "--batch-size",
        "32",
        "--device",
        "cpu",
        "--seed",
        str(seed),
        "--output",
        str(out),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def test_train_cli_end_to_end_offline_and_deterministic(tmp_path):
    cache = _seed_cache(tmp_path)

    r1 = _run_cli(cache, tmp_path / "cc_a.pt")
    assert r1.returncode == 0, f"CLI failed:\n{r1.stdout[-1500:]}\n{r1.stderr[-1500:]}"

    ck = load_checkpoint(tmp_path / "cc_a.pt")
    assert ck.steps == list(STEPS)
    assert ck.model_name == MODEL
    assert {"W_E", "b_E", "W_D", "b_D"} <= set(ck.state_dict)
    assert ck.state_dict["W_D"].shape == (len(STEPS), 4 * D_MODEL, D_MODEL)
    q = ck.quality
    assert q is not None and torch.isfinite(torch.tensor(q["explained_variance"]))
    assert ck.raw is not None and "provenance" in ck.raw and "git_commit" in ck.raw

    # Same seed through the full CLI -> bitwise-identical weights.
    r2 = _run_cli(cache, tmp_path / "cc_b.pt")
    assert r2.returncode == 0, r2.stderr[-1500:]
    sd_a, sd_b = ck.state_dict, load_checkpoint(tmp_path / "cc_b.pt").state_dict
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"CLI runs diverge at {k} despite equal seeds"


def test_library_train_same_seed_bitwise_identical():
    seed_everything(0)
    snaps = torch.randn(2, V, D_MODEL)

    def _train(seed):
        cc = train(
            snaps.clone(), expansion_factor=4.0, n_epochs=2, batch_size=32, device="cpu", seed=seed, log_every=1000
        )
        return {k: v.detach().clone() for k, v in cc.state_dict().items()}

    a, b = _train(0), _train(0)
    for k in a:
        assert torch.equal(a[k], b[k]), f"same-seed training diverges at {k}"
    c = _train(1)
    assert any(not torch.equal(a[k], c[k]) for k in a), "different seeds produced identical weights"
