"""In-place readout / embedding weight swaps on a loaded causal LM.

The recipe-control readout swap grid scores grafted readouts on cached hidden
states (``readout.probes.readout_swap.swap_cell_metrics``); these three helpers
are its correctness oracle: physically copy another checkpoint's ``W_U`` (or
``W_E``) into the model and re-evaluate the model end to end.
"""

from __future__ import annotations

import math

import torch


def get_weight(model, target: str = "wu") -> torch.nn.Parameter:
    """The readout (``wu``: ``embed_out.weight``) or input-embedding (``we``) parameter."""
    if target == "wu":
        return model.embed_out.weight
    if target == "we":
        return model.gpt_neox.embed_in.weight
    raise ValueError(f"target must be 'wu' or 'we', got {target!r}")


@torch.no_grad()
def swap_weight(model, new_weight: torch.Tensor, target: str = "wu") -> None:
    """Copy ``new_weight`` into the model's ``target`` parameter (device/dtype cast)."""
    w = get_weight(model, target)
    if tuple(new_weight.shape) != tuple(w.shape):
        raise ValueError(f"shape mismatch for {target}: {tuple(new_weight.shape)} vs {tuple(w.shape)}")
    w.data.copy_(new_weight.to(w.device, dtype=w.dtype))


@torch.no_grad()
def eval_bpt(model, tokens: torch.Tensor, batch_size: int = 4, device: str = "cpu") -> float:
    """Next-token loss in bits per token over ``tokens`` ``(N, T)`` (token-weighted mean)."""
    model.eval()
    total_nats, n_pred = 0.0, 0
    for start in range(0, tokens.shape[0], batch_size):
        batch = tokens[start : start + batch_size].to(device)
        out = model(batch, labels=batch)
        n = batch.shape[0] * (batch.shape[1] - 1)  # HF shifts labels: T-1 predictions per row
        total_nats += out.loss.item() * n
        n_pred += n
    return (total_nats / max(n_pred, 1)) / math.log(2)
