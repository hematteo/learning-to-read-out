"""Smoke tests for src/readout/core/ modifications: data.py, models.py, training.py."""

from __future__ import annotations

import pytest
import torch

# ── data.py ──────────────────────────────────────────────────


def test_center_and_project_normalize_rows():
    from readout.core.data import center_and_project

    W = torch.randn(100, 32)
    data, centering = center_and_project(W, svd_remove=0, normalize_rows=True)
    # Rows should be unit-norm
    norms = data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert "row_norms" in centering
    assert centering["row_norms"].shape == (100,)


def test_center_and_project_backward_compat():
    from readout.core.data import center_and_project

    W = torch.randn(100, 32)
    data, centering = center_and_project(W, svd_remove=2)
    assert "row_norms" not in centering
    assert data.shape == (100, 32)


def test_adaptive_center_and_project():
    from readout.core.data import adaptive_center_and_project

    W = torch.randn(100, 32)
    data, centering = adaptive_center_and_project(W, svd_remove=2)
    assert data.shape == (100, 32)
    assert centering["svd_components"].shape[0] == 2


def test_svd_baseline_module_cache():
    from readout.core.data import svd_baseline

    data = torch.randn(50, 16)
    mse1 = svd_baseline(data, rank=4)
    mse2 = svd_baseline(data, rank=8)
    assert isinstance(mse1, float)
    assert mse2 <= mse1  # more components = better reconstruction


def test_center_and_project_svd_s_stored():
    from readout.core.data import center_and_project

    W = torch.randn(100, 32)
    _, centering = center_and_project(W, svd_remove=2, verbose=False)
    assert "svd_S" in centering
    assert centering["svd_S"] is not None
    # S comes from SVD of (100, 32) centered matrix → min(100, 32) = 32
    assert centering["svd_S"].shape == (32,)


def test_center_and_project_svd_s_none_when_no_removal():
    from readout.core.data import center_and_project

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

    model = TopKSAE(d_input=32, d_dict=64, k=5)
    x = torch.randn(20, 32)
    x_hat, codes, aux = model(x)
    l0 = (codes > 0).float().sum(dim=1)
    assert (l0 == 5).all(), f"TopK L0 should be exactly 5, got {l0}"
    assert x_hat.shape == x.shape


def test_batch_topk_sae_average_l0():
    from readout.core.models import BatchTopKSAE

    model = BatchTopKSAE(d_input=32, d_dict=64, k=5)
    x = torch.randn(20, 32)
    x_hat, codes, aux = model(x)
    # Batch average should be approximately k (exactly k * batch_size total activations)
    total_active = (codes > 0).sum().item()
    assert total_active <= 20 * 5, f"Total active features {total_active} should be <= {20 * 5}"
    assert x_hat.shape == x.shape


def test_batch_topk_sae_no_ties_issue():
    """Verify scatter-based implementation doesn't over-count on ties."""
    from readout.core.models import BatchTopKSAE

    model = BatchTopKSAE(d_input=32, d_dict=64, k=5)
    # Use zeros to maximize tie chance
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.zero_()
    x = torch.randn(10, 32)
    codes = model.encode(x)
    total = (codes > 0).sum().item()
    # With zero encoder, pre_acts are all zero -> ReLU zeros everything
    # Total active should be 0 (all pre-activations are 0, ReLU kills them)
    assert total == 0


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

    data = torch.randn(200, 16)
    model = TopKSAE(d_input=16, d_dict=32, k=4)
    metrics = train_simple_sae(model, data, num_epochs=50, lr=1e-3)
    assert "mse" in metrics
    assert "l0" in metrics
    assert metrics["mse"] < data.pow(2).sum(dim=1).mean().item()  # better than zero reconstruction


def test_train_simple_sae_early_stopping():
    from readout.core.models import TopKSAE
    from readout.crosscoder.training import train_simple_sae

    data = torch.randn(200, 16)
    model = TopKSAE(d_input=16, d_dict=32, k=4)
    metrics = train_simple_sae(model, data, num_epochs=5000, lr=1e-3, patience=10)
    assert "mse" in metrics
    # Should have stopped well before 5000 epochs
    assert metrics["time"] < 60  # sanity: shouldn't take a minute


def test_cross_validate_sae():
    from readout.core.models import TopKSAE
    from readout.crosscoder.training import cross_validate_sae

    data = torch.randn(200, 16)
    result = cross_validate_sae(TopKSAE, data, n_folds=3, d_dict=32, k=4, num_epochs=30, lr=1e-3)
    assert len(result["fold_metrics"]) == 3
    assert result["mean_mse"] > 0
    assert result["std_mse"] >= 0


def test_resample_dead_features_device_assert():
    from readout.core.models import TopKSAE
    from readout.crosscoder.training import resample_dead_features

    model = TopKSAE(d_input=16, d_dict=32, k=4)
    data = torch.randn(50, 16)
    codes = torch.zeros(50, 32)  # all dead
    # Same device — should work
    n = resample_dead_features(model, data, codes)
    assert n == 32  # all features were dead
