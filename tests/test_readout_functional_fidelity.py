"""readout_functional_fidelity{,_strata}.py (crosscoder_main appendix validation) on synthetic tensors.

The scripts live in an experiment ``scripts/`` dir (the strata audit imports the
aggregate script as a sibling), so they are loaded with that dir on ``sys.path``
(pytest's monkeypatch), as in ``test_recipe_control_lifecycle.py``. No data, no GPU.
"""

from __future__ import annotations

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

from readout.core.paths import repo_root

SCRIPTS = repo_root() / "experiments" / "crosscoders" / "crosscoder_main" / "scripts" / "appendix_validation"
K, V, d, D = 3, 11, 4, 6


@pytest.fixture(scope="module")
def mods():
    mp = pytest.MonkeyPatch()
    mp.syspath_prepend(str(SCRIPTS))
    try:
        yield (
            importlib.import_module("readout_functional_fidelity"),
            importlib.import_module("readout_functional_fidelity_strata"),
        )
    finally:
        mp.undo()


def _crosscoder(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "W_E": torch.randn(K, d, D, generator=g),
        "b_E": 0.1 * torch.randn(K, D, generator=g),
        "W_D": torch.randn(K, D, d, generator=g),
        "b_D": 0.1 * torch.randn(K, d, generator=g),
        "activation_function.log_jumprelu_threshold": torch.full((D,), math.log(0.5)),
    }


def _reference_recon(sd, x_norm, k, scale, mean):
    """Direct JumpReLU crosscoder decode at snapshot k in raw units (the historical formula)."""
    pre = sum(x_norm[j] @ sd["W_E"][j] + sd["b_E"][j] for j in range(K))  # (V, D)
    dn = sd["W_D"][k].norm(dim=-1)  # (D,)
    thr = sd["activation_function.log_jumprelu_threshold"].exp()
    scaled = pre * dn
    acts = scaled * (scaled > thr) / dn
    return (acts @ sd["W_D"][k] + sd["b_D"][k]) * scale + mean


def test_reconstruct_one_step_matches_reference(mods):
    rff, _ = mods
    sd = _crosscoder()
    x_norm = torch.randn(K, V, d, generator=torch.Generator().manual_seed(1))  # (K, V, d)
    scale, mean = torch.tensor([[2.0]]), torch.randn(1, d)
    ref = _reference_recon(sd, x_norm, 1, scale, mean)
    for batch_rows in (3, V):  # row batching must not change the result
        recon, matrix = rff.reconstruct_one_step(
            sd, x_norm, ref, {"scale": scale, "mean": mean}, 1, device="cpu", batch_rows=batch_rows
        )
        torch.testing.assert_close(recon, ref, rtol=1e-5, atol=1e-5)
        assert matrix["matrix_ev"] == pytest.approx(1.0, abs=1e-6)
    native = ref + 0.3 * torch.randn(V, d, generator=torch.Generator().manual_seed(2))
    _, matrix = rff.reconstruct_one_step(
        sd, x_norm, native, {"scale": scale, "mean": mean}, 1, device="cpu", batch_rows=4
    )
    sse = (ref - native).pow(2).sum().item()
    var = (native - native.mean(0, keepdim=True)).pow(2).sum().item()
    assert matrix["matrix_ev"] == pytest.approx(1 - sse / var, rel=1e-5)


def _hidden(n_seqs=3, T=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n_seqs + 1, T, d, generator=g)  # cache may hold more sequences than scored
    ids = torch.randint(0, V, (n_seqs, T), generator=g)
    return h, ids


def test_functional_metrics_identity_readout(mods):
    rff, _ = mods
    h, ids = _hidden()
    W = torch.randn(V, d, generator=torch.Generator().manual_seed(3))
    m = rff.functional_metrics(h, ids, W, W.clone(), device="cpu", batch_seqs=2)
    assert m["n_predictions"] == 3 * 5
    assert m["logit_r2"] == pytest.approx(1.0)
    assert m["centered_logit_r2"] == pytest.approx(1.0)
    assert m["kl_native_to_recon"] == pytest.approx(0.0, abs=1e-6)
    assert m["delta_nll"] == pytest.approx(0.0, abs=1e-6)
    assert m["top1_agreement"] == 1.0 and m["top5_overlap"] == pytest.approx(1.0)


def test_functional_metrics_match_direct_formulas(mods):
    rff, _ = mods
    h, ids = _hidden(seed=4)
    g = torch.Generator().manual_seed(5)
    Wn, Wr = torch.randn(V, d, generator=g), torch.randn(V, d, generator=g)
    m = rff.functional_metrics(h, ids, Wn, Wr, device="cpu", batch_seqs=2)
    hh, tgt = h[:3, :-1].double(), ids[:, 1:]
    ln, lr = hh @ Wn.double().T, hh @ Wr.double().T  # (N, T-1, V)
    lpn, lpr = F.log_softmax(ln, -1), F.log_softmax(lr, -1)
    nll_n = -lpn.gather(-1, tgt.unsqueeze(-1)).mean().item()
    nll_r = -lpr.gather(-1, tgt.unsqueeze(-1)).mean().item()
    assert m["delta_nll"] == pytest.approx(nll_r - nll_n, rel=1e-4)
    assert m["kl_native_to_recon"] == pytest.approx((lpn.exp() * (lpn - lpr)).sum(-1).mean().item(), rel=1e-4)
    assert m["logit_r2"] == pytest.approx(
        1 - ((lr - ln) ** 2).sum().item() / ((ln - ln.mean()) ** 2).sum().item(), rel=1e-4
    )
    assert m["top1_agreement"] == pytest.approx((ln.argmax(-1) == lr.argmax(-1)).double().mean().item())


def test_parse_steps(mods):
    rff, _ = mods
    assert rff.parse_steps("all", [0, 1, 2]) == [0, 1, 2]
    assert rff.parse_steps("2,0", [0, 1, 2]) == [2, 0]
    with pytest.raises(ValueError):
        rff.parse_steps("3", [0, 1, 2])


def test_freq_and_text_strata(mods):
    _, strata_mod = mods
    ids = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4])  # distinct counts: no rank ties
    freq = strata_mod.freq_decile_strata(ids, 7)
    assert freq[5].item() == -1 and freq[6].item() == -1  # unseen
    assert freq[4].item() == 9 and freq[2].item() == 0  # most / least frequent seen
    length, coarse, names = strata_mod.text_strata([" the", "a", " ", "12345678", "!!"])
    assert length.tolist() == [1, 0, 0, 3, 1]  # "the"->2-3, "a"->1, whitespace->0, 8 chars->8+, "!!"->2-3
    assert len(names) == len(set(coarse.tolist()))
    assert strata_mod.length_bin_names() == ["1", "2-3", "4-7", "8+"]


def test_strata_partition_the_aggregate(mods):
    rff, strata_mod = mods
    h, ids = _hidden(seed=6)
    g = torch.Generator().manual_seed(7)
    Wn = torch.randn(V, d, generator=g)
    Wr = Wn + 0.5 * torch.randn(V, d, generator=g)
    strata = {"two": torch.tensor([0, 1] * 5 + [0], dtype=torch.int32)}
    mat = strata_mod.stratified_matrix_residuals(Wr, Wn, strata)["two"]
    assert sum(b["n_rows"] for b in mat.values()) == V
    assert sum(b["matrix_sse"] for b in mat.values()) == pytest.approx((Wr - Wn).pow(2).sum().item(), rel=1e-5)
    fun = strata_mod.stratified_functional(h, ids, Wn, Wr, strata, device="cpu", batch_seqs=2)["two"]
    agg = rff.functional_metrics(h, ids, Wn, Wr, device="cpu", batch_seqs=2)
    n = sum(b["n_targets"] for b in fun.values())
    assert n == agg["n_predictions"]
    weighted = sum(b["delta_nll"] * b["n_targets"] for b in fun.values()) / n
    assert weighted == pytest.approx(agg["delta_nll"], rel=1e-5)
    weighted_kl = sum(b["kl_to_native"] * b["n_targets"] for b in fun.values()) / n
    assert weighted_kl == pytest.approx(agg["kl_native_to_recon"], rel=1e-5)


def test_logit_r2_stable_under_large_mean_logit(mods):
    # Every logit ~ 1000 + O(0.1): sum(x^2) - sum(x)^2/n cancels in float32; the merged variance must not.
    rff, _ = mods
    h, ids = _hidden(n_seqs=4, T=64, seed=8)
    h[..., 0] = 1.0
    g = torch.Generator().manual_seed(9)
    Wn = 0.1 * torch.randn(V, d, generator=g)
    Wn[:, 0] = 1000.0
    Wr = Wn + 0.02 * torch.randn(V, d, generator=g)
    m = rff.functional_metrics(h, ids, Wn, Wr, device="cpu", batch_seqs=1)
    hh = h[:4, :-1].double()
    ln, lr = hh @ Wn.double().T, hh @ Wr.double().T
    ref = 1 - ((lr - ln) ** 2).sum().item() / ((ln - ln.mean()) ** 2).sum().item()
    assert m["logit_r2"] == pytest.approx(ref, abs=1e-3)
