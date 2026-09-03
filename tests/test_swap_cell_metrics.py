"""swap_cell_metrics (readout.probes.readout_swap) against a direct reference computation."""

from __future__ import annotations

import math

import pytest
import torch

from readout.probes.readout_swap import swap_cell_metrics


def _reference(h, W, W_native, ids, h_native=None):
    h_native = h if h_native is None else h_native
    tgt = ids[:, 1:]
    logits = (h.float() @ W.float().T)[:, :-1, :]
    logits_n = (h_native.float() @ W_native.float().T)[:, :-1, :]
    logp = torch.log_softmax(logits, -1)
    logp_n = torch.log_softmax(logits_n, -1)
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).mean().item()
    nll_n = -logp_n.gather(-1, tgt.unsqueeze(-1)).mean().item()
    kl = (logp.exp() * (logp - logp_n)).sum(-1).mean().item()
    am, am_n = logp.argmax(-1), logp_n.argmax(-1)
    cn = logits_n - logits_n.mean(-1, keepdim=True)
    r2 = 1.0 - (logits - logits_n).pow(2).sum().item() / cn.pow(2).sum().item()
    return dict(
        n_tokens=tgt.numel(),
        nll=nll,
        nll_native=nll_n,
        delta_nll=nll - nll_n,
        ppl=math.exp(nll),
        kl_to_native=kl,
        top1=(am == tgt).float().mean().item(),
        top1_native=(am_n == tgt).float().mean().item(),
        top1_agreement=(am == am_n).float().mean().item(),
        centered_logit_r2=r2,
    )


@pytest.fixture
def cell():
    torch.manual_seed(0)
    N, T, d, V = 3, 5, 4, 7
    h = torch.randn(N, T, d)
    h2 = torch.randn(N, T, d)
    W = torch.randn(V, d)
    W_native = torch.randn(V, d)
    ids = torch.randint(0, V, (N, T))
    return h, h2, W, W_native, ids


def test_matches_reference_and_is_batch_invariant(cell):
    h, _, W, W_native, ids = cell
    ref = _reference(h, W, W_native, ids)
    for bs in (1, 2, 3):
        out = swap_cell_metrics(h, W, W_native, ids, device="cpu", batch_seqs=bs)
        assert set(out) == set(ref)
        for k, v in ref.items():
            assert out[k] == pytest.approx(v, rel=1e-5, abs=1e-6), k


def test_h_native_changes_only_the_native_reference(cell):
    h, h2, W, W_native, ids = cell
    ref = _reference(h, W, W_native, ids, h_native=h2)
    out = swap_cell_metrics(h, W, W_native, ids, device="cpu", batch_seqs=2, h_native=h2)
    for k, v in ref.items():
        assert out[k] == pytest.approx(v, rel=1e-5, abs=1e-6), k
    plain = swap_cell_metrics(h, W, W_native, ids, device="cpu", batch_seqs=2)
    assert out["nll"] == pytest.approx(plain["nll"])  # swapped path untouched
    assert out["nll_native"] != pytest.approx(plain["nll_native"])
