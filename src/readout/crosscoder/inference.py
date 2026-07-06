"""Raw state-dict inference primitives for trained JumpReLU trajectory crosscoders.

Every experiment that evaluates a trained crosscoder from its saved state dict
(EV/L0 sweeps, per-snapshot fidelity, feature rescue, attribution) shares this
math; it was previously re-implemented per script. The streaming/chunking
strategy stays in each script — these helpers are the per-snapshot core only.

State-dict key layout (llamascopium ``Crosscoder``):

    W_E (K, d_model, D)    b_E (K, D)
    W_D (K, D, d_model)    b_D (K, d_model)
    activation_function.log_jumprelu_threshold (D,)

with K snapshots, D dictionary features. The joint pre-activation for a row
batch x of shape (B, K, d_model) is ``sum_k x[:, k] @ W_E[k] + sum_k b_E[k]``
(shared encoder summing over snapshot heads); scripts accumulate it snapshot
by snapshot via :func:`encoder_preacts` + :func:`encoder_bias_total`.
"""

from __future__ import annotations

import torch

THRESHOLD_KEY = "activation_function.log_jumprelu_threshold"


def jumprelu_threshold(sd: dict[str, torch.Tensor], device=None) -> torch.Tensor:
    """JumpReLU firing threshold ``exp(log_threshold)``, shape (D,)."""
    thr = sd[THRESHOLD_KEY].float().exp()
    return thr.to(device) if device is not None else thr


def decoder_norms(sd: dict[str, torch.Tensor], k: int | None = None, *, eps: float = 0.0, device=None) -> torch.Tensor:
    """Per-feature decoder column norms ``||W_D[k, j]||``.

    Returns (K, D) for ``k=None``, else (D,) for one snapshot. ``eps`` clamps
    from below (pass e.g. 1e-8 when the result will be divided by).
    """
    W_D = sd["W_D"] if k is None else sd["W_D"][k]
    if device is not None:
        W_D = W_D.to(device)
    n = W_D.float().pow(2).sum(dim=-1).sqrt()
    return n.clamp_min(eps) if eps > 0 else n


def encoder_preacts(sd: dict[str, torch.Tensor], x_k: torch.Tensor, k: int, device=None) -> torch.Tensor:
    """One snapshot's encoder contribution ``x_k @ W_E[k]`` -> (B, D).

    ``x_k`` must already be in the crosscoder's preprocessed input space.
    Accumulate over k and add :func:`encoder_bias_total` for the joint
    pre-activation.
    """
    W_E_k = sd["W_E"][k]
    if device is not None:
        W_E_k = W_E_k.to(device)
    return x_k @ W_E_k.float()


def encoder_bias_total(sd: dict[str, torch.Tensor], device=None) -> torch.Tensor:
    """``sum_k b_E[k]`` -> (D,), the bias term of the joint pre-activation."""
    b = sd["b_E"].float().sum(dim=0)
    return b.to(device) if device is not None else b


def jumprelu_feature_acts(
    pre_joint: torch.Tensor,
    threshold: torch.Tensor,
    dec_norm_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """JumpReLU feature activations from the joint pre-activation.

    ``dec_norm_k`` given (the production convention, matching training with
    ``sparsity_include_decoder_norm=True``): a feature fires at snapshot k iff
    ``pre * ||W_D[k]|| > threshold``. ``dec_norm_k=None`` gates on the raw
    pre-activation instead — the snapshot-independent "rate convention" used
    by the EV/L0 phase-plane analyses.

    Returns ``(feature_acts, fired)`` with feature_acts = pre_joint on fired
    entries and 0 elsewhere (equivalent to the historical
    ``(pre * norm) * fired / norm`` form, without the divide).
    """
    gate_input = pre_joint if dec_norm_k is None else pre_joint * dec_norm_k
    fired = gate_input > threshold
    return pre_joint * fired, fired


def decode_step(
    sd: dict[str, torch.Tensor],
    feature_acts: torch.Tensor,
    k: int,
    device=None,
) -> torch.Tensor:
    """Snapshot-k reconstruction ``acts @ W_D[k] + b_D[k]`` in preprocessed space."""
    W_D_k = sd["W_D"][k]
    b_D_k = sd["b_D"][k]
    if device is not None:
        W_D_k = W_D_k.to(device)
        b_D_k = b_D_k.to(device)
    return feature_acts @ W_D_k.float() + b_D_k.float()


def subset_decode(
    sd: dict[str, torch.Tensor],
    feature_acts: torch.Tensor,
    k: int,
    subset: torch.Tensor,
    device=None,
) -> torch.Tensor:
    """Rank-|S| contribution ``sum_{j in S} a_j W_D[k, j]`` -> (B, d_model).

    No decoder bias — this is the *contribution* of a feature subset, used by
    the rescue/attribution experiments. Empty subsets return zeros.
    """
    W_D_k = sd["W_D"][k]
    if device is not None:
        W_D_k = W_D_k.to(device)
    W_D_k = W_D_k.float()
    if subset.numel() == 0:
        B = feature_acts.shape[0]
        return torch.zeros(B, W_D_k.shape[-1], device=feature_acts.device, dtype=torch.float32)
    return feature_acts[:, subset] @ W_D_k[subset]


def encode_at_step(
    sd: dict[str, torch.Tensor],
    pre_joint: torch.Tensor,
    k: int,
    *,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Snapshot-k JumpReLU encode from the joint pre-activation.

    Returns ``(feature_acts, decoder_norm_k, fired)`` — the signature the
    rescue/attribution experiments consume.
    """
    thr = jumprelu_threshold(sd, device=device)
    dn = decoder_norms(sd, k, device=device)
    acts, fired = jumprelu_feature_acts(pre_joint, thr, dn)
    return acts, dn, fired


def reconstruct_step(
    sd: dict[str, torch.Tensor],
    feature_acts: torch.Tensor,
    k: int,
    mean_k: torch.Tensor | None = None,
    scale_k: torch.Tensor | None = None,
    device=None,
) -> torch.Tensor:
    """Snapshot-k reconstruction, optionally mapped back to raw W_U units.

    With ``mean_k``/``scale_k`` (from :func:`preprocess_arrays`) the result is
    in raw model space; without them it stays in preprocessed space.
    """
    recon = decode_step(sd, feature_acts, k, device=device)
    if scale_k is not None and mean_k is not None:
        return invert_preprocess(recon, mean_k, scale_k)
    return recon


def reconstruct_preprocess_stats(snapshots) -> dict:
    """Deterministically rebuild ``center_scale`` preprocess stats, streaming.

    Release safetensors sidecars do not embed the training-time stats (the
    legacy ``.pt`` checkpoints did). This recomputes them one raw snapshot at
    a time, delegating each to ``wu_adapter.preprocess_snapshots`` so the
    formula cannot drift from training. Verified against the training-time
    stats embedded in the legacy Run-3 checkpoint: means bitwise identical,
    scales within 1-2 float32 ulps (reduction-order noise).

    ``snapshots`` is any iterable of raw (V, d_model) tensors in step order.
    Returns ``{"mean": (K, 1, d), "scale": (K, 1, 1), "mode": "center_scale"}``.
    """
    from readout.crosscoder.wu_adapter import preprocess_snapshots  # local: keeps this module light to import

    means, scales = [], []
    for x in snapshots:
        _, stats = preprocess_snapshots(x.float().unsqueeze(0), mode="center_scale")
        means.append(stats["mean"])
        scales.append(stats["scale"])
    if not means:
        raise ValueError("no snapshots provided")
    return {"mean": torch.cat(means), "scale": torch.cat(scales), "mode": "center_scale"}


def preprocess_arrays(preprocess_stats: dict, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize stored ``preprocess_stats`` to ``means (K, d), scales (K,)``.

    Checkpoints store the stats from ``wu_adapter.preprocess_snapshots`` as
    mean (K, 1, d) and scale (K, 1, 1); older sidecars vary in squeeze state.
    """
    means = preprocess_stats["mean"].float().reshape(preprocess_stats["mean"].shape[0], -1)
    scales = preprocess_stats["scale"].float().reshape(-1)
    if device is not None:
        means = means.to(device)
        scales = scales.to(device)
    return means, scales


def invert_preprocess(x_norm: torch.Tensor, mean_k: torch.Tensor, scale_k: torch.Tensor) -> torch.Tensor:
    """Map a preprocessed-space tensor back to raw W_U units: ``x * scale + mean``."""
    return x_norm * scale_k + mean_k
