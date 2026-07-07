"""Resume-exactness regression for readout.crosscoder.wu_adapter.train.

The repo's stated invariant: a run interrupted at a rolling checkpoint and
resumed from it finishes with weights BITWISE identical to an uninterrupted
run — which requires the rolling checkpoint to carry optimizer (Adam moment)
state and the shuffle generator state, not just weights. Fully offline, CPU,
synthetic snapshots (same pattern as tests/test_cli_train_smoke.py).
"""

from __future__ import annotations

import torch

from readout.core.repro import seed_everything
from readout.crosscoder.wu_adapter import train

V, D_MODEL = 64, 16
N_EPOCHS = 4


def _snaps():
    seed_everything(0)
    return torch.randn(2, V, D_MODEL)


def _train(snaps, n_epochs, **kw):
    cc = train(
        snaps.clone(),
        expansion_factor=4.0,
        n_epochs=n_epochs,
        batch_size=32,
        device="cpu",
        seed=0,
        log_every=1000,
        **kw,
    )
    return {k: v.detach().clone() for k, v in cc.state_dict().items()}


def test_rolling_checkpoint_carries_optimizer_state_and_resume_is_bitwise(tmp_path):
    snaps = _snaps()
    ckpt = tmp_path / "rolling.pt"

    # Uninterrupted reference run.
    sd_full = _train(snaps, N_EPOCHS)

    # "Interrupted" run: stops after 2 of 4 epochs, leaving a rolling ckpt.
    _train(snaps, N_EPOCHS // 2, ckpt_every_epochs=1, ckpt_path=ckpt)
    assert ckpt.exists() and not ckpt.with_suffix(".pt.tmp").exists()

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["epoch"] == N_EPOCHS // 2 - 1
    assert "gen_state" in payload
    # (a) The checkpoint payload must contain optimizer state (Adam moments):
    # without it, resume restarts the moments from zero and is NOT exact.
    opt = payload.get("optimizer")
    assert opt is not None, "rolling checkpoint is missing optimizer state"
    assert opt["state"], "optimizer state dict is empty"
    moment_tensors = [v for st in opt["state"].values() for v in st.values() if isinstance(v, torch.Tensor)]
    assert moment_tensors, "optimizer state carries no moment tensors"
    assert all(t.device.type == "cpu" for t in moment_tensors)

    # Resumed run: auto-detects the rolling ckpt and continues epochs 2..3.
    sd_resumed = _train(snaps, N_EPOCHS, ckpt_every_epochs=1, ckpt_path=ckpt)

    # (b) Bitwise equality with the uninterrupted run.
    assert set(sd_resumed) == set(sd_full)
    for k in sd_full:
        assert torch.equal(sd_resumed[k], sd_full[k]), f"resume diverges from uninterrupted run at {k}"
