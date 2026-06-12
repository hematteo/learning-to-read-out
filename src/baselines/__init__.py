"""Baseline statistics and null-distribution helpers."""

from __future__ import annotations

import numpy as np


def mad_floor_z(x, axis=0):
    """Robust z-score with 25th-percentile floor on σ_raw.

    Median-Absolute-Deviation z-score with a percentile floor applied to the
    estimated σ to prevent division by near-zero MAD when many entries share
    a value (common for sparse firing-rate matrices).
    """
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    sigma_raw = 1.4826 * mad
    nonzero = sigma_raw[sigma_raw > 0]
    floor = np.quantile(nonzero, 0.25) if nonzero.size else 1e-8
    sigma = np.maximum(sigma_raw, max(1e-8, floor))
    return (x - med) / sigma


__all__ = ["mad_floor_z"]
