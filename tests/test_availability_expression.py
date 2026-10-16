"""Availability/expression metrics (readout.probes.availability_expression) and the runner.

CPU only, no data: synthetic hidden states, W_U and LayerNorm snapshots.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys

import numpy as np
import pytest
import torch
from sklearn.model_selection import GroupKFold

from readout.core.paths import repo_root
from readout.probes import availability_expression as AE

SCRIPTS = repo_root() / "experiments/probes/contrastive_readout_swap/scripts"


def test_group_kfold_matches_sklearn_without_ties_and_is_stable_with_ties():
    rng = np.random.default_rng(0)
    sizes = rng.permutation(np.arange(3, 23))  # 20 distinct sizes: no ties
    groups = np.repeat([f"g{i:02d}" for i in range(20)], sizes)
    ours = AE.group_kfold_splits(groups, 5)
    ref = list(GroupKFold(5).split(np.zeros(len(groups)), groups=groups))
    for (tr, te), (rtr, rte) in zip(ours, ref):
        assert np.array_equal(tr, rtr) and np.array_equal(te, rte)

    # 20 equal groups: stable order places them round-robin from the last unique key.
    groups = np.repeat([f"g{i:02d}" for i in range(20)], 4)
    folds = [sorted(set(groups[te])) for _, te in AE.group_kfold_splits(groups, 5)]
    assert folds[0] == ["g04", "g09", "g14", "g19"] and folds[4] == ["g00", "g05", "g10", "g15"]
    for tr, te in AE.group_kfold_splits(groups, 5):
        assert not set(groups[tr]) & set(groups[te])


def test_x86_simd_argsort_tie_order():
    # Group sizes of the published Pythia sva_across_pp dataset (np.unique order of head lemmas);
    # this tie order reproduces the published GroupKFold folds, a stable sort does not.
    sizes = [84, 52, 68, 68, 72, 64, 64, 70, 76, 56, 62, 68, 48, 80, 54, 78, 82, 74, 62, 78]
    sizes += [58, 66, 66, 52, 84, 64, 52, 52, 80, 66]
    expected = [12, 1, 27, 23, 26, 14, 9, 20, 10, 18, 25, 5, 6, 22, 21, 29, 2, 11, 3, 7]
    expected += [4, 17, 8, 15, 19, 13, 28, 16, 0, 24]
    assert AE.x86_simd_argsort(np.array(sizes)).tolist() == expected
    assert not np.array_equal(expected, np.argsort(sizes, kind="stable"))
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.integers(0, 12, size=int(rng.integers(1, 300)))
        order = AE.x86_simd_argsort(a)
        assert sorted(order.tolist()) == list(range(len(a))) and np.all(np.diff(a[order]) >= 0)
    assert AE.x86_simd_argsort(np.full(40, 7)).tolist() == list(range(40))  # sorted input: identity


def test_feature_labels_and_group_keys():
    sva = {"family": "sva", "prompt": "The keys to the cabinet", "meta": {"noun": "keys", "number": "plural"}}
    assert AE.feature_label(sva) == "plural" and AE.derive_group_key(sva) == "key"
    num = {"family": "numeric_gt", "meta": {"big": 17, "small": 3, "answer_position": "second"}}
    assert AE.feature_label(num) == "second" and AE.derive_group_key(num) == "num::17::3"
    ioi = {"family": "ioi_role_balanced", "meta": {"recipient_position": "first", "unordered_pair": "A::B"}}
    assert AE.feature_label(ioi) == "first" and AE.derive_group_key(ioi) == "A::B"
    det = {"family": "blimp_determiner_noun_agreement_1", "meta": {"good_token": " dogs", "bad_token": " dog"}}
    assert AE.feature_label(det) == "plural" and AE.derive_group_key(det) == "detnoun::dog"
    ana = {"family": "blimp_anaphor_gender_agreement", "meta": {"good_token": " themselves", "sentence_good": "x"}}
    assert AE.feature_label(ana) is None
    assert AE.feature_label({"family": "fv_antonym", "meta": {"input": "hot"}}) is None
    assert AE.derive_groups([sva, {"family": "induction", "meta": {}}]) is None


def test_availability_probe_separable_vs_nulls():
    rng = np.random.default_rng(0)
    n, d = 200, 32
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, d))  # (N, d)
    X[:, 0] += 4.0 * (2 * y - 1)
    groups = np.array([f"g{i % 20}" for i in range(n)], dtype=object)
    out = AE.availability_probe(X, y, n_splits=5, seed=0, groups=groups)
    assert out["probe_splitter"] == "GroupKFold(k=5)"
    assert out["probe_acc"] > 0.95 and out["probe_acc_1d"] > 0.95
    assert out["probe_acc_label_shuffle"] < 0.7 and out["probe_acc_random_label"] < 0.7
    assert out["probe_acc_1d_ci_lo"] <= out["probe_acc_1d"] <= out["probe_acc_1d_ci_hi"]
    again = AE.availability_probe(X, y, n_splits=5, seed=0, groups=groups)
    assert {k: v for k, v in out.items() if k != "probe_acc_randinit"} == {
        k: v for k, v in again.items() if k != "probe_acc_randinit"
    }
    skip = AE.availability_probe(X, y, groups=np.array(["one"] * n, dtype=object))
    assert np.isnan(skip["probe_acc"]) and "GroupKFold infeasible" in skip["skip_reason"]


class _Snaps:
    def __init__(self, wus: dict, lnfs: dict):
        self._wu, self._ln = wus, lnfs
        self.steps = sorted(set(wus) & set(lnfs))

    def wu(self, s):
        return self._wu.get(s)

    def lnf(self, s):
        return self._ln.get(s)


def _toy_snapshots(d: int = 8, V: int = 16, steps=(0, 10, 20, 143000)):
    rng = torch.Generator().manual_seed(0)
    wus = {s: torch.randn(V, d, generator=rng) for s in steps}
    lnfs = {
        s: {"weight": torch.rand(d, generator=rng) + 0.5, "bias": torch.randn(d, generator=rng), "eps": 1e-5}
        for s in steps
    }
    return wus, lnfs


def test_native_and_best_swept_against_direct_computation():
    wus, lnfs = _toy_snapshots()
    snaps = _Snaps(wus, lnfs)
    rng = np.random.default_rng(1)
    z = rng.normal(size=(40, 8)).astype(np.float32)  # (N, d)
    yp, ym = rng.integers(0, 16, 40), rng.integers(0, 16, 40)
    ym[ym == yp] = (ym[ym == yp] + 1) % 16

    def acc(s):
        h = torch.nn.functional.layer_norm(torch.from_numpy(z), (8,), lnfs[s]["weight"], lnfs[s]["bias"], 1e-5)
        logits = h @ wus[s].T  # (N, V)
        return float((logits[np.arange(40), yp] > logits[np.arange(40), ym]).float().mean()), logits

    out = AE.native_and_best_swept(z, yp, ym, 10, snaps, seed=0)
    native, logits = acc(10)
    assert out["native_readout_acc"] == pytest.approx(native)
    ranks = (logits > logits[np.arange(40), yp][:, None]).sum(1) + 1
    assert out["native_full_vocab_mean_rank"] == pytest.approx(float(ranks.float().mean()))
    swept = {s: acc(s)[0] for s in (0, 10, 20)}  # terminal step excluded from the sweep
    assert out["best_swept_readout_acc"] == pytest.approx(max(swept.values()))
    assert out["best_swept_s_step"] in swept and swept[out["best_swept_s_step"]] == max(swept.values())
    assert out["best_swept_s_step_heldout"] in swept
    assert out["delta_best_minus_native"] == pytest.approx(max(swept.values()) - native)
    none = AE.native_and_best_swept(None, yp, ym, 10, snaps)
    assert np.isnan(none["native_readout_acc"]) and none["full_vocab_rank_skip_reason"]


def test_runner_end_to_end(tmp_path, monkeypatch):
    d, V, n = 8, 16, 40
    wus, lnfs = _toy_snapshots(d, V, steps=(0, 10))
    wu_dir, hid, ds, out = (tmp_path / x for x in ("wu", "hidden", "ds", "out"))
    for p in (wu_dir, hid, ds):
        p.mkdir()
    slug = "EleutherAI_pythia-160m"
    for s in (0, 10):
        torch.save(wus[s], wu_dir / f"{slug}_step{s}_wu.pt")
        torch.save(lnfs[s], wu_dir / f"{slug}_step{s}_lnf.pt")
    rng = np.random.default_rng(0)
    nouns = [("keys", "key"), ("books", "book"), ("doors", "door"), ("cars", "car"), ("cups", "cup")]
    rows = []
    for i in range(n):
        pl, sg = nouns[i % 5]
        noun, number = (pl, "plural") if i % 2 else (sg, "singular")
        rows.append(
            {
                "family": "sva",
                "prompt": f"The {noun} on the table",
                "prompt_ids": [1, 2],
                "y_plus": 3 if number == "plural" else 4,
                "y_minus": 4 if number == "plural" else 3,
                "meta": {"noun": noun, "number": number},
            }
        )
    (ds / "sva.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for s in (0, 10):
        z = torch.from_numpy(rng.normal(size=(n, d)).astype(np.float32))  # (N, d)
        torch.save(z, hid / f"sva_z{s}.pt")
        torch.save(
            torch.nn.functional.layer_norm(z, (d,), lnfs[s]["weight"], lnfs[s]["bias"], 1e-5), hid / f"sva_h{s}.pt"
        )

    monkeypatch.syspath_prepend(str(SCRIPTS))
    runner = importlib.import_module("run_availability_probes")
    argv = ["x", "--model", "EleutherAI/pythia-160m", "--hidden-dir", str(hid), "--datasets-dir", str(ds)]
    argv += ["--wu-dir", str(wu_dir), "--group-split", "--out-dir", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    got = list(csv.DictReader((out / "probe_summary.csv").open()))
    assert [(r["family"], r["h_step"]) for r in got] == [("sva", "0"), ("sva", "10")]
    assert all(r["probe_splitter"] == "GroupKFold(k=5)" and r["best_swept_s_step"] in ("0", "10") for r in got)
    assert json.loads((out / "probe_summary.provenance.json").read_text())["swept_steps"] == [0, 10]
    runner.main()  # resume: nothing recomputed, rows unchanged
    assert list(csv.DictReader((out / "probe_summary.csv").open())) == got


def test_extractor_streams_tiny_model(monkeypatch):
    transformers = pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    ext = importlib.import_module("extract_hidden_dense")
    torch.manual_seed(0)
    cfg = transformers.GPTNeoXConfig(
        vocab_size=64, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32
    )
    model = transformers.GPTNeoXForCausalLM(cfg).eval()
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11]]
    batched = ext.final_position_streams(model, prompts, device="cpu", batch_size=3, save_dtype=torch.float32)
    single = ext.final_position_streams(model, prompts, device="cpu", batch_size=1, save_dtype=torch.float32)
    assert batched["h"].shape == batched["z"].shape == (4, 16)
    for k in ("h", "z"):
        assert torch.allclose(batched[k], single[k], atol=1e-5)
    ln = ext.lnf_params(model)
    assert torch.allclose(AE.layernorm(single["z"], ln), single["h"], atol=1e-6)
    logits = model(torch.tensor([prompts[1]])).logits[0, -1]  # (V,)
    W = model.embed_out.weight
    assert torch.allclose(single["h"][1] @ W.T, logits, atol=1e-5)
