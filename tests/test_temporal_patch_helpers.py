"""Pins for the shipped replacements of the archived intervention helpers.

encode_snapshot_local and script_of replaced the non-shipped
_archive/legacy_crosscoder_160m helpers; on the real release data the swap was
verified bitwise (helper outputs and the end-to-end smoke pipeline). These
tests pin the same contracts on synthetic data so CI holds them.
"""

from __future__ import annotations

import torch

from readout.core.repro import seed_everything
from readout.dynamics.temporal_patch import encode_snapshot_local
from readout.probes.token_scripts import script_of


def _legacy_encode(W_U, pieces):
    """Verbatim archived 02_intervention.encode."""
    x_proc = (W_U - pieces["mean_i"]) / pieces["scale_i"]
    pre = x_proc @ pieces["W_E"] + pieces["b_E"]
    return (pre > pieces["thr"]).float() * pre


def test_encode_snapshot_local_matches_archived_formula():
    seed_everything(0)
    V, d, D = 64, 8, 32
    pieces = {
        "W_E": torch.randn(d, D) * 0.3,
        "b_E": torch.randn(D) * 0.1,
        "thr": torch.rand(D) * 0.2,
        "mean_i": torch.randn(d) * 0.5,
        "scale_i": torch.tensor(2.0),
    }
    W_U = torch.randn(V, d)
    a_new = encode_snapshot_local(W_U, pieces)
    a_old = _legacy_encode(W_U, pieces)
    assert torch.equal(a_new, a_old)
    assert (a_new != 0).any() and (a_new == 0).any()  # gate actually active


def test_script_of_branches():
    cases = {
        "": "empty",
        "   ": "space",
        " the": "ASCII",
        "!": "ASCII",
        " Привет": "Cyrillic",
        "ไทย": "Thai",
        "مرحبا": "Arabic",
        "नमस्ते": "Devanagari",
        "বাংলা": "Bengali",
        "中文": "CJK",
        "ひらがな": "Japanese",
        "한국어": "Korean",
        "Ωμέγα": "other_unicode",  # Greek falls through to the catch-all
    }
    for text, expected in cases.items():
        assert script_of(text) == expected, f"{text!r} -> {script_of(text)} != {expected}"
