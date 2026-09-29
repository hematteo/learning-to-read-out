"""CPU tests for the recipe-control checkpoint loaders and the weight swap oracle.

A tiny random GPTNeoX (2 layers, d=16, vocab 64) is saved in the trainer's
``<root>/<cond>/ckpts/step<N>/model_fp16.pt`` layout; no network, no data.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from readout.probes.readout_swap import swap_cell_metrics
from readout.probes.recipe_control_models import (
    WU_KEY,
    build_control_model,
    condition_config,
    condition_steps,
    load_body,
    load_condition_wu,
    load_readout,
    load_state_dict,
    resolve_step,
    state_dict_path,
    val_loss_at,
)
from readout.probes.weight_swap import eval_bpt, get_weight, swap_weight

TINY = dict(num_hidden_layers=2, hidden_size=16, num_attention_heads=2, intermediate_size=32)
VOCAB = 64
SEQ = 12
STEPS = (0, 4, 8)


def _save_arm(root, cond, mult, *, seed, drop_key=None):
    cdir = root / cond
    (cdir / "ckpts").mkdir(parents=True)
    (cdir / "config.json").write_text(json.dumps({"model_size": TINY, "readout_lr_mult": mult, "warmup_steps": 3}))
    for i, st in enumerate(STEPS):
        torch.manual_seed(seed + st)
        model = build_control_model(TINY, SEQ, vocab_size=VOCAB)
        sd = {k: v.detach().half() for k, v in model.state_dict().items()}
        if drop_key:
            sd.pop(drop_key)
        d = cdir / "ckpts" / f"step{st}"
        d.mkdir()
        torch.save(sd, d / "model_fp16.pt")
        if i == 0:  # step 0 has metrics.json; step 4 only a metrics.csv row; step 8 neither
            (d / "metrics.json").write_text(json.dumps({"step": st, "val_loss": 1.5}))
    (cdir / "metrics.csv").write_text("step,val_loss\n4,2.5\n")


@pytest.fixture(scope="module")
def ckpt_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    _save_arm(root, "baseline", 1.0, seed=0)
    _save_arm(root, "wu_lr_4x", 4.0, seed=100)
    _save_arm(root, "broken", 1.0, seed=200, drop_key="gpt_neox.layers.0.attention.query_key_value.weight")
    return root


def test_steps_are_discovered_and_resolved(ckpt_root):
    assert condition_steps(ckpt_root, "baseline") == list(STEPS)
    assert resolve_step(ckpt_root, "baseline", "final") == 8
    assert resolve_step(ckpt_root, "baseline", 4) == 4
    assert resolve_step(ckpt_root, "baseline", 5) == 4  # nearest on a miss
    assert condition_config(ckpt_root, "wu_lr_4x")["readout_lr_mult"] == 4.0
    with pytest.raises(FileNotFoundError):
        condition_steps(ckpt_root, "missing_arm")


def test_load_body_matches_saved_weights_and_runs(ckpt_root):
    model = load_body(ckpt_root, "baseline", 4)  # model_size from config.json
    sd = load_state_dict(ckpt_root, "baseline", 4)
    assert state_dict_path(ckpt_root, "baseline", 4).is_file()
    assert torch.equal(model.embed_out.weight.detach(), sd[WU_KEY].float())
    assert not model.training and model.embed_out.weight.dtype == torch.float32
    x = torch.randint(0, VOCAB, (2, SEQ))
    out = model(x, labels=x)
    assert torch.isfinite(out.loss)


def test_load_body_refuses_missing_trained_param(ckpt_root):
    with pytest.raises(RuntimeError, match="trained params absent"):
        load_body(ckpt_root, "broken", 0)


def test_load_readout_and_stacked_wu(ckpt_root):
    rd = load_readout(ckpt_root, "baseline", 8)
    assert rd["wu"].shape == (VOCAB, TINY["hidden_size"])
    assert rd["ln_w"].shape == rd["ln_b"].shape == (TINY["hidden_size"],)
    snaps, steps = load_condition_wu(ckpt_root, "baseline")
    assert snaps.shape == (len(STEPS), VOCAB, TINY["hidden_size"]) and steps == list(STEPS)
    assert torch.equal(snaps[-1], rd["wu"])
    sub, sub_steps = load_condition_wu(ckpt_root, "baseline", steps=[0, 8])
    assert sub.shape[0] == 2 and sub_steps == [0, 8]


def test_val_loss_lookup_json_then_csv_then_none(ckpt_root):
    assert val_loss_at(ckpt_root, "baseline", 0) == 1.5
    assert val_loss_at(ckpt_root, "baseline", 4) == 2.5
    assert val_loss_at(ckpt_root, "baseline", 8) is None


def test_swap_weight_and_eval_bpt(ckpt_root):
    model = load_body(ckpt_root, "baseline", 8)
    torch.manual_seed(1)
    tokens = torch.randint(0, VOCAB, (5, SEQ))
    # eval_bpt is the token-weighted mean CE in bits, batch-size invariant.
    with torch.no_grad():
        logits = model(tokens).logits[:, :-1, :].reshape(-1, VOCAB)
        nats = torch.nn.functional.cross_entropy(logits, tokens[:, 1:].reshape(-1)).item()
    assert eval_bpt(model, tokens, batch_size=2) == pytest.approx(nats / math.log(2), rel=1e-5)
    assert eval_bpt(model, tokens, batch_size=5) == pytest.approx(nats / math.log(2), rel=1e-5)

    new = load_readout(ckpt_root, "wu_lr_4x", 8)["wu"]
    orig = get_weight(model, "wu").detach().clone()
    swap_weight(model, new, target="wu")
    assert torch.equal(get_weight(model, "wu").detach(), new)
    swap_weight(model, orig, target="wu")
    assert torch.equal(get_weight(model, "wu").detach(), orig)
    with pytest.raises(ValueError, match="shape mismatch"):
        swap_weight(model, new[:8], target="wu")
    with pytest.raises(ValueError, match="target"):
        get_weight(model, "lm_head")


def test_fast_swap_kernel_matches_physical_swap_oracle(ckpt_root):
    """swap_cell_metrics on cached hidden states == embed_out swap + eval_bpt (nats)."""
    body = load_body(ckpt_root, "baseline", 8)
    W_native = load_readout(ckpt_root, "baseline", 8)["wu"]
    W_other = load_readout(ckpt_root, "wu_lr_4x", 8)["wu"]
    torch.manual_seed(2)
    ids = torch.randint(0, VOCAB, (4, SEQ))
    with torch.no_grad():
        h = body.gpt_neox(input_ids=ids).last_hidden_state.float()  # (N, T, d), post-final-LN
    fast = swap_cell_metrics(h, W_other, W_native, ids, device="cpu", batch_seqs=3)
    assert fast["n_tokens"] == 4 * (SEQ - 1)
    swap_weight(body, W_other)
    oracle_nats = eval_bpt(body, ids, batch_size=2) * math.log(2)
    assert fast["nll"] == pytest.approx(oracle_nats, abs=1e-5)
    swap_weight(body, W_native)
    assert fast["nll_native"] == pytest.approx(eval_bpt(body, ids, batch_size=4) * math.log(2), abs=1e-5)
    # the identity graft is the native model exactly
    same = swap_cell_metrics(h, W_native, W_native, ids, device="cpu", batch_seqs=2)
    assert same["delta_nll"] == pytest.approx(0.0, abs=1e-9)
    assert same["kl_to_native"] == pytest.approx(0.0, abs=1e-6)
    assert same["top1_agreement"] == 1.0 and same["centered_logit_r2"] == pytest.approx(1.0)
