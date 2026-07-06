"""Equivalence tests for readout.crosscoder.inference.

Two anchors:
1. The llamascopium Crosscoder module itself — the helpers must reproduce the
   production encode/decode from a raw state dict, snapshot by snapshot.
2. The historical per-script form ``(pre * norm) * fired / norm`` that the
   experiment scripts used before deduplication — the direct ``pre * fired``
   form must agree on the fired mask exactly and on activations numerically.
"""

from __future__ import annotations

import torch

from readout.core.repro import seed_everything
from readout.crosscoder import inference as inf
from readout.crosscoder.wu_adapter import batch_iter, build_crosscoder, preprocess_snapshots

K, V, D_MODEL, EF = 4, 64, 16, 4.0


def _tiny_crosscoder():
    seed_everything(0)
    cc = build_crosscoder(K, D_MODEL, EF, device="cpu")
    cc.eval()
    snaps = torch.randn(K, V, D_MODEL)
    return cc, snaps


def _module_forward(cc, snaps):
    batch = next(batch_iter(snaps, cc.cfg.hook_points, batch_size=V, shuffle=False))
    x, enc_kw, dec_kw = cc.prepare_input(batch)
    with torch.no_grad():
        acts, hidden_pre = cc.encode(x, return_hidden_pre=True, **enc_kw)
        recon = cc.decode(acts, **dec_kw)
    return acts, hidden_pre, recon


def _helper_pre(sd, snaps):
    pre = torch.zeros(snaps.shape[1], sd["W_E"].shape[-1])
    for k in range(K):
        pre += inf.encoder_preacts(sd, snaps[k], k)
    return pre + inf.encoder_bias_total(sd)


def test_helpers_reproduce_module_forward():
    """encoder_preacts/jumprelu_feature_acts/decode_step == Crosscoder forward."""
    cc, snaps = _tiny_crosscoder()
    acts, hidden_pre, recon = _module_forward(cc, snaps)
    sd = {k: v.detach().clone() for k, v in cc.state_dict().items()}

    pre = _helper_pre(sd, snaps)
    # The shared encoder gives one joint pre-activation, replicated per head.
    assert torch.allclose(pre, hidden_pre[:, 0, :].float(), atol=1e-5)

    thr = inf.jumprelu_threshold(sd)
    for k in range(K):
        dn = inf.decoder_norms(sd, k)
        a_k, fired = inf.jumprelu_feature_acts(pre, thr, dn)
        assert torch.allclose(a_k, acts[:, k, :].float(), atol=1e-5), f"acts diverge at k={k}"
        assert torch.equal(fired, acts[:, k, :].float() != 0), f"fired mask diverges at k={k}"
        r_k = inf.decode_step(sd, a_k, k)
        assert torch.allclose(r_k, recon[:, k, :].float(), atol=1e-5), f"recon diverges at k={k}"


def test_matches_historical_scaled_divide_form():
    """The direct pre*fired form equals the (pre*norm)*fired/norm scripts used."""
    cc, snaps = _tiny_crosscoder()
    sd = {k: v.detach().clone() for k, v in cc.state_dict().items()}
    pre = _helper_pre(sd, snaps)
    thr = inf.jumprelu_threshold(sd)
    for k in range(K):
        # Verbatim historical form (run_step1000_feature_rescue.encode_at_step).
        W_D_k = sd["W_D"][k]
        decoder_norm = torch.linalg.norm(W_D_k.float(), dim=-1)
        scaled = pre * decoder_norm[None, :]
        fired_ref = scaled > thr[None, :]
        acts_ref = (scaled * fired_ref) / decoder_norm.clamp_min(1e-12)[None, :]

        a_k, fired = inf.jumprelu_feature_acts(pre, thr, inf.decoder_norms(sd, k))
        assert torch.equal(fired, fired_ref)
        assert torch.allclose(a_k, acts_ref, atol=1e-6)


def test_rate_convention_gates_without_norms():
    """dec_norm_k=None gates on the raw pre-activation (EV/L0 phase-plane use)."""
    pre = torch.tensor([[0.5, -0.2, 2.0]])
    thr = torch.tensor([1.0, 0.1, 1.0])
    acts, fired = inf.jumprelu_feature_acts(pre, thr)
    assert torch.equal(fired, torch.tensor([[False, False, True]]))
    assert torch.equal(acts, torch.tensor([[0.0, 0.0, 2.0]]))


def test_subset_decode_matches_full_decode_minus_bias():
    cc, snaps = _tiny_crosscoder()
    sd = {k: v.detach().clone() for k, v in cc.state_dict().items()}
    pre = _helper_pre(sd, snaps)
    thr = inf.jumprelu_threshold(sd)
    a_k, _ = inf.jumprelu_feature_acts(pre, thr, inf.decoder_norms(sd, 1))
    full_subset = torch.arange(sd["W_D"].shape[1])
    contrib = inf.subset_decode(sd, a_k, 1, full_subset)
    assert torch.allclose(contrib + sd["b_D"][1].float(), inf.decode_step(sd, a_k, 1), atol=1e-5)
    empty = inf.subset_decode(sd, a_k, 1, torch.empty(0, dtype=torch.long))
    assert empty.shape == (V, D_MODEL)
    assert (empty == 0).all()


def test_preprocess_arrays_roundtrip():
    """preprocess_arrays + invert_preprocess undo preprocess_snapshots."""
    seed_everything(1)
    raw = torch.randn(K, V, D_MODEL) * 3 + 0.5
    proc, stats = preprocess_snapshots(raw, mode="center_scale")
    means, scales = inf.preprocess_arrays(stats)
    assert means.shape == (K, D_MODEL) and scales.shape == (K,)
    for k in range(K):
        back = inf.invert_preprocess(proc[k], means[k], scales[k])
        assert torch.allclose(back, raw[k], atol=1e-5), f"roundtrip fails at k={k}"
