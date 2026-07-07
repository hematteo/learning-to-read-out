"""Smoke load test for crosscoder checkpoints (paper plan §6, §10).

Builds a tiny crosscoder on the fly, saves it in the same format as
``src/readout/crosscoder/wu_adapter.py``, reloads it, and exercises the load
path used by ``extract_rates.py`` and the manifest writer. CPU-only,
no ${UM_SSD_ROOT}/ dependency.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from readout.crosscoder.manifest import (
    MANIFEST_HEADER,
    append_manifest_row,
    infer_row_from_checkpoint,
)
from readout.crosscoder.wu_adapter import build_crosscoder, quick_quality

D_MODEL = 16
D_SAE = 32
K = 3
V = 64


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    cc = build_crosscoder(K, D_MODEL, expansion_factor=D_SAE / D_MODEL, device="cpu")
    snaps = torch.randn(K, V, D_MODEL)

    metrics = quick_quality(cc, snaps, batch_size=32, device="cpu")
    # dead_rate is part of paper §6 post-eval; wu_adapter doesn't yet write it,
    # but the manifest writer expects it under quality["dead_rate"].
    with torch.no_grad():
        l0_per = (cc.encode(cc.prepare_input({hp: snaps[i] for i, hp in enumerate(cc.cfg.hook_points)})[0])).reshape(
            -1, D_SAE
        )
        fired_any = (l0_per > 0).any(dim=0)
        dead_rate = float(1.0 - fired_any.float().mean().item())
    metrics["dead_rate"] = dead_rate

    out = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save(
        {
            "state_dict": cc.state_dict(),
            "config": cc.cfg.model_dump(),
            "steps": [0, 1000, 143000],
            "model_name": "EleutherAI/pythia-160m",
            "seed": 0,
            "quality": metrics,
            "training": {
                "lr": 5e-5,
                "l1_coefficient": 0.3,
                "frequency_scale": 0.01,
                "tanh_stretch_coefficient": 4.0,
                "jumprelu_lr_factor": 0.1,
                "init_threshold": 0.1,
                "decoder_transpose_init": 1.0,
                "amp_dtype": "fp32",
                "optimizer": "adam",
                "n_epochs": 1,
                "batch_size": 32,
                "expansion_factor": D_SAE / D_MODEL,
                "d_sae": D_SAE,
                "input_preprocess": "none",
                "l1_warmup_fraction": 0.0,
                "lr_warmup_fraction": 0.0,
                "lr_decay_fraction": 0.0,
                "auxk_coefficient": 0.0,
                "auxk_k": 512,
                "dead_window_steps": 100,
            },
            "preprocess_stats": None,
        },
        out,
    )
    return out


def test_checkpoint_roundtrips(tiny_checkpoint: Path):
    ck = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    assert set(ck.keys()) >= {
        "state_dict",
        "config",
        "steps",
        "model_name",
        "seed",
        "quality",
        "training",
    }
    sd = ck["state_dict"]
    # Round-trip: rebuild from config and load_state_dict.
    from llamascopium.models.crosscoder import Crosscoder, CrosscoderConfig

    cfg = CrosscoderConfig(**ck["config"])
    cc = Crosscoder(cfg)
    missing, unexpected = cc.load_state_dict(sd, strict=False)
    assert not missing, f"missing keys after load: {missing}"
    assert not unexpected, f"unexpected keys after load: {unexpected}"


def test_state_dict_keys_match_adapter_expectations(tiny_checkpoint: Path):
    sd = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    expected = {
        "W_E",
        "b_E",
        "W_D",
        "b_D",
        "activation_function.log_jumprelu_threshold",
    }
    assert expected.issubset(set(sd.keys())), f"expected subset, got {sorted(sd.keys())}"
    # Shapes used by extract_rates.compute_rates.
    assert sd["W_E"].shape == (K, D_MODEL, D_SAE)
    assert sd["b_E"].shape == (K, D_SAE)
    assert sd["activation_function.log_jumprelu_threshold"].shape == (D_SAE,)


def test_extract_rates_direct_path_finite(tiny_checkpoint: Path, tmp_path: Path):
    """Exercises the REAL extract_rates.compute_rates on the tiny checkpoint
    (previously this test re-implemented the 4-line direct-tensor path, so the
    function itself had no coverage)."""
    from readout.crosscoder.extract_rates import compute_rates

    # Tiny synthetic snapshot cache instead of loading from /Volumes.
    torch.manual_seed(1)
    for s in (0, 1000, 143000):
        torch.save(torch.randn(V, D_MODEL), tmp_path / f"EleutherAI_pythia-160m_step{s}_wu.pt")

    rates, steps = compute_rates(tiny_checkpoint, tmp_path)
    assert steps == [0, 1000, 143000]
    assert rates.shape == (K, D_SAE)
    assert torch.isfinite(rates).all()
    assert (rates >= 0).all() and (rates <= 1).all()


def test_forward_pass_finite_via_class(tiny_checkpoint: Path):
    ck = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    from llamascopium.models.crosscoder import Crosscoder, CrosscoderConfig

    cfg = CrosscoderConfig(**ck["config"])
    cc = Crosscoder(cfg)
    cc.load_state_dict(ck["state_dict"])
    cc.eval()
    torch.manual_seed(2)
    snaps = torch.randn(K, 8, D_MODEL)
    batch = {hp: snaps[i] for i, hp in enumerate(cc.cfg.hook_points)}
    with torch.no_grad():
        x, enc_kw, dec_kw = cc.prepare_input(batch)
        feats, _ = cc.encode(x, return_hidden_pre=True, **enc_kw)
        recon = cc.decode(feats, **dec_kw)
    assert torch.isfinite(feats).all()
    assert torch.isfinite(recon).all()
    assert recon.shape == x.shape


def test_quality_fields_present_and_sane(tiny_checkpoint: Path):
    quality = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)["quality"]
    for k in ("explained_variance", "mean_l0", "dead_rate"):
        assert k in quality, f"missing quality field: {k}"
    ev = quality["explained_variance"]
    l0 = quality["mean_l0"]
    dr = quality["dead_rate"]
    assert math.isfinite(ev) and -10.0 <= ev <= 1.0 + 1e-6
    assert math.isfinite(l0) and 0.0 <= l0 <= D_SAE + 1e-6
    assert math.isfinite(dr) and 0.0 <= dr <= 1.0 + 1e-6


# --- quick_quality -----------------------------------------------------------


def test_quick_quality_exact_reconstruction_gives_ev_one():
    """A crosscoder wired as an exact autoencoder (identity encode/decode,
    every feature firing) must score EV ~= 1.0 and ~zero MSE."""
    d = 8
    cc = build_crosscoder(1, d, expansion_factor=1.0, device="cpu")
    c = 10.0
    with torch.no_grad():
        cc.load_state_dict(
            {
                "W_E": torch.eye(d).unsqueeze(0),  # (1, d, d)
                "b_E": torch.full((1, d), c),
                "W_D": torch.eye(d).unsqueeze(0),  # (1, d, d)
                "b_D": torch.full((1, d), -c),
                "activation_function.log_jumprelu_threshold": torch.full((d,), math.log(1e-3)),
            }
        )
    g = torch.Generator().manual_seed(0)
    snaps = torch.rand(1, V, d, generator=g)  # positive: pre = x + c fires everywhere

    q = quick_quality(cc, snaps, batch_size=16, device="cpu")
    assert q["explained_variance"] == pytest.approx(1.0, abs=1e-6)
    assert q["reconstruction_mse"] == pytest.approx(0.0, abs=1e-9)
    assert q["mean_l0"] == pytest.approx(float(d))  # every feature fires on every row
    assert q["dead_rate"] == 0.0


def test_quick_quality_batch_size_invariant():
    """EV/MSE/L0/dead_rate must not depend on eval batch size (the old
    variance-of-batch estimator was batch-size dependent)."""
    torch.manual_seed(3)
    cc = build_crosscoder(K, D_MODEL, expansion_factor=D_SAE / D_MODEL, device="cpu")
    snaps = torch.randn(K, V, D_MODEL)  # V=64: batch 7 leaves a ragged tail batch

    q_small = quick_quality(cc, snaps, batch_size=7, device="cpu")
    q_full = quick_quality(cc, snaps, batch_size=V, device="cpu")
    assert q_small["mean_l0"] == q_full["mean_l0"]
    assert q_small["dead_rate"] == q_full["dead_rate"]
    assert q_small["explained_variance"] == pytest.approx(q_full["explained_variance"], rel=1e-6)
    assert q_small["reconstruction_mse"] == pytest.approx(q_full["reconstruction_mse"], rel=1e-6)


# --- manifest writer ---------------------------------------------------------


def test_infer_row_has_all_header_columns(tiny_checkpoint: Path):
    row = infer_row_from_checkpoint(tiny_checkpoint, notes="smoke")
    assert set(row.keys()) == set(MANIFEST_HEADER)
    # Column rename regression: the step-count column is n_steps, not the old
    # ambiguous 'steps' name (manifests are regenerated artifacts).
    assert "steps" not in MANIFEST_HEADER
    assert row["n_steps"] == "3"
    assert row["d_sae"] == str(D_SAE)
    assert row["d_model"] == str(D_MODEL)
    assert row["seed"] == "0"
    assert row["model"] == "EleutherAI/pythia-160m"
    assert row["notes"] == "smoke"
    assert row["path"].endswith("tiny.pt")


def test_append_manifest_idempotent(tiny_checkpoint: Path, tmp_path: Path):
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(",".join(MANIFEST_HEADER) + "\n")

    r1 = append_manifest_row(tiny_checkpoint, manifest_csv=csv_path, notes="first")
    r2 = append_manifest_row(tiny_checkpoint, manifest_csv=csv_path, notes="second")

    text = csv_path.read_text().splitlines()
    # Header + exactly one data row (idempotent).
    assert len(text) == 2, f"expected header + 1 row, got {text}"
    assert "second" in text[1], "later call should overwrite earlier notes in place"
    assert r1["path"] == r2["path"]


def test_append_manifest_strict_missing_dir(tiny_checkpoint: Path, tmp_path: Path):
    bogus = tmp_path / "no_such_dir" / "manifest.csv"
    with pytest.raises(FileNotFoundError):
        append_manifest_row(tiny_checkpoint, manifest_csv=bogus, strict=True)


def test_manifest_roundtrip_with_cusum_p(tiny_checkpoint: Path, tmp_path: Path):
    """Tier-1 DoD: training writes EV/L0/dead_rate, a downstream tool injects
    CUSUM_p via update_or_insert, and the same row carries both."""
    import csv as _csv

    from readout.crosscoder.manifest import (
        _atomic_write,
        _read_existing,
        update_or_insert,
    )

    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(",".join(MANIFEST_HEADER) + "\n")

    # 1) Training-side append (EV, L0, dead_rate populated from quality dict).
    append_manifest_row(tiny_checkpoint, manifest_csv=csv_path, notes="t1")

    # 2) Downstream CUSUM injection — same key, partial row.
    header, rows = _read_existing(csv_path)
    rows, action = update_or_insert(
        rows,
        {"path": str(tiny_checkpoint.resolve()), "CUSUM_p": "0.0123"},
        key="path",
    )
    assert action == "updated", "second write must update existing row, not insert"
    _atomic_write(csv_path, header, rows)

    with csv_path.open() as f:
        data = list(_csv.DictReader(f))
    assert len(data) == 1, f"expected single row, got {len(data)}: {data}"
    row = data[0]
    assert row["EV"], "EV column should be populated by wu_adapter path"
    assert row["L0"], "L0 column should be populated by wu_adapter path"
    assert row["dead_rate"], "dead_rate should be populated from quick_quality"
    assert row["CUSUM_p"] == "0.0123", "CUSUM_p should be injected by downstream tool"
    assert row["path"] == str(tiny_checkpoint.resolve())
