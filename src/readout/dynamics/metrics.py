"""Decoder-direction metrics for the lifecycle analyses.

This module is the single source of truth for them; analysis scripts call into
here and never re-derive metrics ad hoc.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def adjacent_rotation(decoder_weights: np.ndarray | torch.Tensor) -> np.ndarray:
    """1 - cos between adjacent decoder columns.

    Input: (K, D, d) decoder tensor (W_D in checkpoint).
    Output: (K-1, D) with values in [0, 2].
    """
    W = _to_np(decoder_weights).astype(np.float64)
    if W.ndim != 3:
        raise ValueError(f"decoder_weights must be (K, D, d), got {W.shape}")
    a = W[:-1]
    b = W[1:]
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return 1.0 - num / den


def terminal_direction_distance(
    decoder_weights: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """1 - cos between each decoder column and its terminal-snapshot direction.

    Input: (K, D, d) decoder tensor (W_D in checkpoint).
    Output: (K, D), where the final row is approximately zero for nonzero
    terminal columns. The reorganization score's movement toward the terminal
    decoder direction is the absolute change of this term between checkpoints.
    """
    W = _to_np(decoder_weights).astype(np.float64)
    if W.ndim != 3:
        raise ValueError(f"decoder_weights must be (K, D, d), got {W.shape}")
    terminal = W[-1:]  # (1, D, d)
    num = (W * terminal).sum(axis=-1)
    den = np.linalg.norm(W, axis=-1) * np.linalg.norm(terminal, axis=-1) + 1e-12
    return 1.0 - num / den
