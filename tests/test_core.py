"""Smoke tests for src/readout/core/ modifications: data.py, models.py, training.py."""

from __future__ import annotations

import pytest
import torch

# ── data.py ──────────────────────────────────────────────────


def test_center_and_project_normalize_rows():
    from readout.core.data import center_and_project

    torch.manual_seed(0)
    W = torch.randn(100, 32)
    data, centering = center_and_project(W, svd_remove=0, normalize_rows=True)
    # Rows should be unit-norm
    norms = data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert "row_norms" in centering
    assert centering["row_norms"].shape == (100,)


def test_center_and_project_backward_compat():
    from readout.core.data import center_and_project

    torch.manual_seed(0)
    W = torch.randn(100, 32)
    data, centering = center_and_project(W, svd_remove=2)
    assert "row_norms" not in centering
    assert data.shape == (100, 32)


def test_adaptive_center_and_project():
    from readout.core.data import adaptive_center_and_project

    torch.manual_seed(0)
    W = torch.randn(100, 32)
    data, centering = adaptive_center_and_project(W, svd_remove=2)
    assert data.shape == (100, 32)
    assert centering["svd_components"].shape[0] == 2


def test_svd_baseline_module_cache():
    from readout.core.data import svd_baseline

    torch.manual_seed(0)
    data = torch.randn(50, 16)
    mse1 = svd_baseline(data, rank=4)
    mse2 = svd_baseline(data, rank=8)
    assert isinstance(mse1, float)
    assert mse2 <= mse1  # more components = better reconstruction


def test_center_and_project_svd_s_stored():
    from readout.core.data import center_and_project

    torch.manual_seed(0)
    W = torch.randn(100, 32)
    _, centering = center_and_project(W, svd_remove=2, verbose=False)
    assert "svd_S" in centering
    assert centering["svd_S"] is not None
    # S comes from SVD of (100, 32) centered matrix → min(100, 32) = 32
    assert centering["svd_S"].shape == (32,)


def test_center_and_project_svd_s_none_when_no_removal():
    from readout.core.data import center_and_project

    torch.manual_seed(0)
    W = torch.randn(100, 32)
    _, centering = center_and_project(W, svd_remove=0, verbose=False)
    assert "svd_S" in centering
    assert centering["svd_S"] is None


def test_svd_baseline_from_centering_returns_none_without_svd_s():
    from readout.core.data import svd_baseline_from_centering

    centering = {"svd_S": None, "svd_components": None}
    assert svd_baseline_from_centering(centering, rank=4, n_rows=100) is None


def test_svd_baseline_from_centering_returns_zero_large_rank():
    from readout.core.data import center_and_project, svd_baseline_from_centering

    torch.manual_seed(0)
    W = torch.randn(100, 32)
    _, centering = center_and_project(W, svd_remove=2, verbose=False)
    # rank=32 means offset = 2+32 = 34 >= 32 → should return 0.0
    assert svd_baseline_from_centering(centering, rank=32, n_rows=100) == pytest.approx(0.0)


def test_svd_baseline_from_centering_matches_svd_baseline():
    from readout.core.data import _svd_cache, center_and_project, svd_baseline, svd_baseline_from_centering

    torch.manual_seed(42)
    W = torch.randn(200, 64)
    svd_remove = 2
    data, centering = center_and_project(W, svd_remove=svd_remove, verbose=False)
    n_rows = data.shape[0]

    _svd_cache.clear()  # ensure fresh SVD

    for rank in [1, 4, 8, 16, 32]:
        mse_direct = svd_baseline(data, rank=rank)
        mse_analytic = svd_baseline_from_centering(centering, rank=rank, n_rows=n_rows)
        assert mse_analytic is not None
        assert mse_analytic == pytest.approx(mse_direct, rel=1e-4), (
            f"rank={rank}: analytic={mse_analytic:.6f} vs direct={mse_direct:.6f}"
        )


def test_extract_wu_from_model_lm_head():
    from readout.core.data import extract_wu_from_model

    torch.manual_seed(0)

    class FakeHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(50, 32))

    class FakeModel:
        lm_head = FakeHead()

    W = extract_wu_from_model(FakeModel())
    assert W.shape == (50, 32)
    assert W.dtype == torch.float32


def test_extract_wu_from_model_embed_out():
    from readout.core.data import extract_wu_from_model

    torch.manual_seed(0)

    class FakeHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(50, 32))

    class FakeModel:
        embed_out = FakeHead()

    W = extract_wu_from_model(FakeModel())
    assert W.shape == (50, 32)
    assert W.dtype == torch.float32


def test_extract_wu_from_model_raises_on_unknown():
    from readout.core.data import extract_wu_from_model

    class FakeModel:
        pass

    with pytest.raises(ValueError, match="Cannot locate unembedding matrix"):
        extract_wu_from_model(FakeModel())


def test_get_device_priority_cuda_over_mps(monkeypatch):
    """get_device() is the single device picker; README documents cuda > mps > cpu."""
    from readout.core.data import get_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device() == "cpu"


# ── models.py ────────────────────────────────────────────────


def test_topk_sae_exact_l0():
    from readout.core.models import TopKSAE

    torch.manual_seed(0)
    model = TopKSAE(d_input=32, d_dict=64, k=5)
    x = torch.randn(20, 32)
    x_hat, codes, aux = model(x)
    l0 = (codes > 0).float().sum(dim=1)
    assert (l0 == 5).all(), f"TopK L0 should be exactly 5, got {l0}"
    assert x_hat.shape == x.shape


def test_batch_topk_sae_average_l0():
    from readout.core.models import BatchTopKSAE

    torch.manual_seed(0)
    model = BatchTopKSAE(d_input=32, d_dict=64, k=5)
    x = torch.randn(20, 32)
    x_hat, codes, aux = model(x)
    # Exactly min(k * batch_size, #positive pre-activations) entries are active:
    # the batch-average-L0 guarantee, not just an upper bound.
    total_active = (codes > 0).sum().item()
    with torch.no_grad():
        n_positive = (model.encoder(x - model.b_dec) > 0).sum().item()
    assert total_active == min(20 * 5, n_positive), (
        f"BatchTopK selected {total_active} active features, expected {min(20 * 5, n_positive)}"
    )
    assert total_active > 0, "seeded random init must activate at least one feature"
    assert x_hat.shape == x.shape


def test_batch_topk_tie_exact_selection():
    """Documented tie behavior: selection is BY INDEX, so exactly k_total
    entries stay active even when many values tie at the threshold (a
    value-threshold mask would keep every tied entry and break the L0
    guarantee)."""
    from readout.core.models import BatchTopKSAE

    B, D, k = 4, 8, 2
    k_total = B * k  # 8
    # REAL ties: all 32 positive pre-activations share the same value.
    pre = torch.full((B, D), 1.0)
    codes = BatchTopKSAE._batch_topk(pre, k)
    assert (codes > 0).sum().item() == k_total, "tied values must not inflate the active count above k_total"
    # Selected entries keep their value, dropped ties are exactly zero.
    assert set(codes.unique().tolist()) == {0.0, 1.0}

    # Ties at the selection boundary: 4 entries at 2.0, 8 tied at 1.0 competing
    # for the remaining 4 slots -> still exactly k_total active.
    pre = torch.zeros(B, D)
    pre[0, :4] = 2.0
    pre[1, :4] = 1.0
    pre[2, :4] = 1.0
    codes = BatchTopKSAE._batch_topk(pre, k)
    assert (codes > 0).sum().item() == k_total
    assert (codes == 2.0).sum().item() == 4  # the strict top values always survive

    # All-zero pre-activations: ReLU leaves nothing positive to select.
    assert (BatchTopKSAE._batch_topk(torch.zeros(B, D), k) > 0).sum().item() == 0


def test_batch_topk_normalize_and_dict():
    from readout.core.models import BatchTopKSAE

    model = BatchTopKSAE(d_input=32, d_dict=64, k=5)
    model.normalize_decoder()
    norms = model.W_dec.data.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    d = model.dictionary_vectors()
    assert d.shape == (64, 32)


# ── training.py ──────────────────────────────────────────────


def test_train_simple_sae_topk():
    from readout.core.models import TopKSAE
    from readout.crosscoder.training import train_simple_sae

    torch.manual_seed(0)
    data = torch.randn(200, 16)
    model = TopKSAE(d_input=16, d_dict=32, k=4)
    metrics = train_simple_sae(model, data, num_epochs=50, lr=1e-3)
    assert "mse" in metrics
    assert "l0" in metrics
    assert metrics["mse"] < data.pow(2).sum(dim=1).mean().item()  # better than zero reconstruction


def test_train_simple_sae_early_stopping():
    from readout.core.models import TopKSAE
    from readout.crosscoder.training import train_simple_sae

    torch.manual_seed(0)
    data = torch.randn(200, 16)
    model = TopKSAE(d_input=16, d_dict=32, k=4)
    metrics = train_simple_sae(model, data, num_epochs=5000, lr=1e-3, patience=10)
    assert "mse" in metrics
    # Should have stopped well before 5000 epochs
    assert metrics["time"] < 60  # sanity: shouldn't take a minute


def test_cross_validate_sae():
    from readout.core.models import TopKSAE
    from readout.crosscoder.training import cross_validate_sae

    torch.manual_seed(0)
    data = torch.randn(200, 16)
    result = cross_validate_sae(TopKSAE, data, n_folds=3, d_dict=32, k=4, num_epochs=30, lr=1e-3)
    assert len(result["fold_metrics"]) == 3
    assert result["mean_mse"] > 0
    assert result["std_mse"] >= 0


def test_resample_dead_features_device_assert():
    from readout.core.models import TopKSAE
    from readout.crosscoder.training import resample_dead_features

    model = TopKSAE(d_input=16, d_dict=32, k=4)
    torch.manual_seed(0)
    data = torch.randn(50, 16)
    codes = torch.zeros(50, 32)  # all dead
    # Same device — should work
    n = resample_dead_features(model, data, codes)
    assert n == 32  # all features were dead


# ── Matryoshka SAEs: sparsity_aux contract + trainer coupling ─


MATRYOSHKA_NESTED = [8, 16, 32]


def _matryoshka_classes():
    from readout.core.models import MatryoshkaBatchTopKSAE, TiedMatryoshkaBatchTopKSAE

    return (MatryoshkaBatchTopKSAE, TiedMatryoshkaBatchTopKSAE)


def test_matryoshka_sparsity_aux_contract_tied_and_untied():
    """Both variants return the SAME sparsity_aux contract: in training mode
    with train_groups=True it is the stacked per-level reconstruction MSE,
    shape (n_levels,); otherwise zeros of shape (1,)."""
    for cls in _matryoshka_classes():
        torch.manual_seed(0)
        model = cls(d_input=16, d_dict=32, k=4, nested_sizes=MATRYOSHKA_NESTED)
        x = torch.randn(24, 16)

        model.train()
        x_hat, codes, aux = model(x, train_groups=True)
        assert aux.shape == (len(MATRYOSHKA_NESTED),), f"{cls.__name__}: bad sparsity_aux shape {tuple(aux.shape)}"
        assert aux.requires_grad, f"{cls.__name__}: per-level MSEs must carry grad"
        assert (aux >= 0).all() and torch.isfinite(aux).all()
        # The last level is the full dictionary: its MSE equals the returned
        # reconstruction's MSE (pins per-level RECON MSE semantics).
        full_mse = (x - x_hat).pow(2).sum(dim=1).mean()
        assert torch.allclose(aux[-1], full_mse, atol=1e-6), f"{cls.__name__}: level[-1] != full recon MSE"
        assert x_hat.shape == x.shape and codes.shape == (24, 32)

        # Outside group training the aux is inert zeros of shape (1,).
        _, _, aux_no_groups = model(x, train_groups=False)
        assert aux_no_groups.shape == (1,) and (aux_no_groups == 0).all()
        model.eval()
        _, _, aux_eval = model(x, train_groups=True)
        assert aux_eval.shape == (1,) and (aux_eval == 0).all()


def test_trainer_normalizes_matryoshka_levels_by_mean_for_both_variants():
    """train_simple_sae's matryoshka loss is sparsity_aux.MEAN() over levels
    (not .sum()) for tied AND untied: the gradients left on the model after a
    single epoch match a manual mean-loss backward, not the 3x-larger
    sum-loss gradients."""
    import copy

    from readout.crosscoder.training import train_simple_sae

    n_levels = len(MATRYOSHKA_NESTED)
    for cls in _matryoshka_classes():
        torch.manual_seed(0)
        model = cls(d_input=16, d_dict=32, k=4, nested_sizes=MATRYOSHKA_NESTED)
        ref = copy.deepcopy(model)
        torch.manual_seed(1)
        data = torch.randn(64, 16)

        # One epoch: backward runs once, so the grads the trainer leaves on
        # the parameters are exactly the grads of the loss it optimized.
        train_simple_sae(model, data, num_epochs=1, lr=1e-3, seed=0, arch_name=cls.__name__)

        ref.train()
        _, _, aux = ref(data, train_groups=True)
        aux.mean().backward()

        ref_grads = {n: p.grad for n, p in ref.named_parameters()}
        checked = 0
        for name, p in model.named_parameters():
            g, rg = p.grad, ref_grads[name]
            assert g is not None and rg is not None, f"{cls.__name__}.{name}: missing grad"
            assert torch.allclose(g, rg, atol=1e-7), f"{cls.__name__}.{name}: trainer loss != mean over levels"
            if rg.abs().max() > 1e-5:
                # A .sum() consumer would scale this grad by n_levels.
                assert not torch.allclose(g, n_levels * rg, rtol=1e-3), (
                    f"{cls.__name__}.{name}: gradients match the SUM-normalized loss"
                )
                checked += 1
        assert checked > 0, f"{cls.__name__}: no parameter had a discriminating gradient"


def test_matryoshka_noise_sigma_is_not_a_noop():
    """noise_sigma must change matryoshka training (it used to be silently
    ignored on the train_groups path): identical seeds/data with a large
    noise_sigma end in different weights and losses than noise_sigma=0."""
    from readout.crosscoder.training import train_simple_sae

    for cls in _matryoshka_classes():

        def run(noise_sigma, cls=cls):
            torch.manual_seed(0)
            model = cls(d_input=16, d_dict=32, k=4, nested_sizes=MATRYOSHKA_NESTED)
            torch.manual_seed(1)
            data = torch.randn(64, 16)
            metrics = train_simple_sae(
                model, data, num_epochs=5, lr=1e-3, seed=42, noise_sigma=noise_sigma, arch_name=cls.__name__
            )
            return {k: v.detach().clone() for k, v in model.state_dict().items()}, metrics

        sd_clean_a, _ = run(0.0)
        sd_clean_b, _ = run(0.0)
        sd_noisy, _ = run(5.0)

        # Determinism control: same seed, no noise -> bitwise identical.
        assert all(torch.equal(sd_clean_a[k], sd_clean_b[k]) for k in sd_clean_a), (
            f"{cls.__name__}: noise-free training is not deterministic; the noise comparison is meaningless"
        )
        assert any(not torch.equal(sd_clean_a[k], sd_noisy[k]) for k in sd_clean_a), (
            f"{cls.__name__}: noise_sigma=5.0 left training bitwise unchanged (still a no-op)"
        )
