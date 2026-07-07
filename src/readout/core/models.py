"""
SAE model architectures for proto-token dictionary learning.
=============================================================
All architectures share a common interface:
  - encode(x) -> sparse codes
  - decode(c) -> reconstruction
  - forward(x) -> (x_hat, codes, sparsity_aux)
  - normalize_decoder()
  - dictionary_vectors() -> (d_dict, d_input)

sparsity_aux contract (third element of forward), per architecture:
  - GatedSAE:                        sigmoid gate probabilities, (B, d_dict);
                                     the trainer L1-penalizes sum(dim=1).mean()
  - JumpReLUSAE:                     soft step probabilities, (B, d_dict); same L1 path
  - VanillaL1SAE:                    |codes|, (B, d_dict); same L1 path
  - TopKSAE / TiedTopKSAE:           AuxK dead-feature loss, shape (1,)
  - BatchTopKSAE / TiedBatchTopKSAE: zeros, shape (1,) (no penalty)
  - Matryoshka variants
    (train_groups=True, training):   stacked per-level recon MSE, shape
                                     (n_levels,); the trainer takes .mean()

Architectures:
  - GatedSAE              (Rajamanoharan et al., 2024)
  - JumpReLUSAE           (Rajamanoharan et al., 2024)
  - TopKSAE               (Gao et al., 2024)
  - BatchTopKSAE          (batch-level top-k sparsity)
  - MatryoshkaBatchTopKSAE(nested Matryoshka + BatchTopK)
  - TiedTopKSAE           (TopK with W_enc = W_dec^T)
  - VanillaL1SAE          (standard L1-penalized)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────
# Gated SAE
# ──────────────────────────────────────────────────────────────


class GatedSAE(nn.Module):
    """Gated Sparse Autoencoder.

    Gate path:      gate = H(W_enc . x_c + b_gate)       [binary, STE gradient]
    Magnitude path: mag  = ReLU(exp(r) . W_enc . x_c + b_mag)
    Sparse code:    c    = gate . mag
    Reconstruction: x_hat = W_dec . c + b_dec

    L1 penalty applies only to gate probabilities, eliminating shrinkage bias.

    VARIANT of Rajamanoharan et al. (2024): sparsity_aux is the sigmoid gate
    probabilities (the paper L1-penalizes ReLU(gate_pre) scaled by decoder
    norms), and the paper's auxiliary reconstruction loss through a frozen
    decoder copy (Eq. 5) is omitted entirely.
    """

    def __init__(self, d_input: int, d_dict: int):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict

        self.W_enc = nn.Parameter(torch.empty(d_dict, d_input))
        self.b_gate = nn.Parameter(torch.zeros(d_dict))
        self.r_mag = nn.Parameter(torch.zeros(d_dict))
        self.b_mag = nn.Parameter(torch.zeros(d_dict))
        self.W_dec = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, c, _ = self._forward_impl(x)
        return c

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor):
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor):
        x_c = x - self.b_dec
        z = F.linear(x_c, self.W_enc)

        # Gate pathway
        gate_pre = z + self.b_gate
        gate_probs = torch.sigmoid(gate_pre)
        gate_hard = (gate_pre > 0).float()
        gate = gate_hard - gate_probs.detach() + gate_probs  # STE

        # Magnitude pathway
        magnitude = F.relu(self.r_mag.exp() * z + self.b_mag)

        c = gate * magnitude
        x_hat = self.decode(c)
        return x_hat, c, gate_probs

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def dictionary_vectors(self) -> torch.Tensor:
        return self.W_dec.data.T.clone()


# ──────────────────────────────────────────────────────────────
# JumpReLU SAE
# ──────────────────────────────────────────────────────────────


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder.

    Pre-activations: z = W_enc . x_c + b_enc
    Sparse code:     c = z . H(z - theta)     [H = Heaviside, STE gradient]
    Reconstruction:  x_hat = W_dec . c + b_dec

    Learnable per-feature threshold theta_j eliminates L1 entirely.
    """

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        initial_threshold: float = 0.001,
        bandwidth: float = 0.01,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict
        self.bandwidth = bandwidth

        self.encoder = nn.Linear(d_input, d_dict)
        self.W_dec = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        self.log_threshold = nn.Parameter(torch.full((d_dict,), math.log(initial_threshold)))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, c, _ = self._forward_impl(x)
        return c

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor):
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor):
        x_c = x - self.b_dec
        pre_acts = self.encoder(x_c)
        theta = self.threshold

        step_hard = (pre_acts > theta).float()
        step_soft = torch.sigmoid((pre_acts - theta) / self.bandwidth)
        step = step_hard - step_soft.detach() + step_soft  # STE

        c = pre_acts * step
        x_hat = self.decode(c)
        return x_hat, c, step_soft

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def dictionary_vectors(self) -> torch.Tensor:
        return self.W_dec.data.T.clone()


# ──────────────────────────────────────────────────────────────
# TopK SAE
# ──────────────────────────────────────────────────────────────


def _auxk_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_acts: torch.Tensor,
    W_dec: torch.Tensor,
    dead_mask: torch.Tensor | None,
    k_aux: int,
    training: bool,
) -> torch.Tensor:
    """AuxK loss: reconstruct the residual using dead features only (Gao et al., 2024)."""
    auxk_loss = torch.zeros(1, device=x.device)
    if training and dead_mask is not None and dead_mask.any():
        residual = x - x_hat.detach()
        dead_pre = pre_acts[:, dead_mask]
        k_aux = min(k_aux, dead_pre.shape[1])
        if k_aux > 0:
            aux_topk_vals, aux_topk_idx = dead_pre.topk(k_aux, dim=1)
            aux_acts = torch.zeros_like(dead_pre)
            aux_acts.scatter_(1, aux_topk_idx, F.relu(aux_topk_vals))
            aux_recon = aux_acts @ W_dec[:, dead_mask].T
            auxk_loss = (residual - aux_recon).pow(2).sum(dim=1).mean().unsqueeze(0)
    return auxk_loss


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder (Gao et al., 2024).

    Hard sparsity: keep exactly k largest pre-activations per input.
    No L1 penalty, no learned thresholds.
    """

    def __init__(self, d_input: int, d_dict: int, k: int = 12, k_aux: int = 0):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict
        self.k = k
        self.k_aux = k_aux if k_aux > 0 else k * 2

        self.encoder = nn.Linear(d_input, d_dict)
        self.W_dec = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        self._dead_mask: torch.Tensor | None = None

        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @staticmethod
    def _topk(pre_acts: torch.Tensor, k: int) -> torch.Tensor:
        acts = F.relu(pre_acts)  # relu first — only positive values compete
        topk_vals, topk_idx = acts.topk(k, dim=1)
        result = torch.zeros_like(acts)
        result.scatter_(1, topk_idx, topk_vals)
        return result

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.b_dec
        pre_acts = self.encoder(x_c)
        return self._topk(pre_acts, self.k)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor):
        x_c = x - self.b_dec
        pre_acts = self.encoder(x_c)
        acts = self._topk(pre_acts, self.k)
        x_hat = self.decode(acts)
        auxk_loss = _auxk_loss(x, x_hat, pre_acts, self.W_dec, self._dead_mask, self.k_aux, self.training)
        return x_hat, acts, auxk_loss

    @torch.no_grad()
    def update_dead_mask(self, codes: torch.Tensor):
        """Update the dead feature mask from a batch of codes."""
        active = (codes.abs() > 1e-8).float().mean(dim=0)
        self._dead_mask = active == 0

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def dictionary_vectors(self) -> torch.Tensor:
        return self.W_dec.data.T.clone()


# ──────────────────────────────────────────────────────────────
# Tied TopK SAE (W_enc = W_dec^T)
# ──────────────────────────────────────────────────────────────


class TiedTopKSAE(nn.Module):
    """TopK SAE with tied weights: W_enc = W_dec^T.

    Only a bias vector b_enc is learned for the encoder, halving
    encoder parameters and preventing memorisation of per-row routing.
    Includes AuxK dead-feature loss (Gao et al., 2024).
    """

    def __init__(self, d_input: int, d_dict: int, k: int = 12, k_aux: int = 0):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict
        self.k = k
        self.k_aux = k_aux if k_aux > 0 else k * 2

        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.W_dec = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        self._dead_mask: torch.Tensor | None = None

        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = F.linear(x - self.b_dec, self.W_dec.T, self.b_enc)
        return TopKSAE._topk(pre_acts, self.k)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor):
        x_c = x - self.b_dec
        pre_acts = F.linear(x_c, self.W_dec.T, self.b_enc)
        acts = TopKSAE._topk(pre_acts, self.k)
        x_hat = self.decode(acts)
        auxk_loss = _auxk_loss(x, x_hat, pre_acts, self.W_dec, self._dead_mask, self.k_aux, self.training)
        return x_hat, acts, auxk_loss

    @torch.no_grad()
    def update_dead_mask(self, codes: torch.Tensor):
        active = (codes.abs() > 1e-8).float().mean(dim=0)
        self._dead_mask = active == 0

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def dictionary_vectors(self) -> torch.Tensor:
        return self.W_dec.data.T.clone()


# ──────────────────────────────────────────────────────────────
# BatchTopK SAE
# ──────────────────────────────────────────────────────────────


class BatchTopKSAE(nn.Module):
    """BatchTopK SAE: variable per-token sparsity, batch-average L0 = k.

    Instead of selecting top-k per token, selects top (batch_size * k)
    activations globally across the batch, allowing some tokens to use
    more features and others fewer.

    WARNING: codes are batch-composition-dependent at inference — the
    top-(batch_size * k) pool is shared across the batch, so the same row can
    receive different codes depending on which rows it is batched with. For
    consistent codes, encode the full dataset in one pass (see
    ``readout.crosscoder.training.get_all_codes``).
    """

    def __init__(self, d_input: int, d_dict: int, k: int = 12):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict
        self.k = k

        self.encoder = nn.Linear(d_input, d_dict)
        self.W_dec = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @staticmethod
    def _batch_topk(pre_acts: torch.Tensor, k: int) -> torch.Tensor:
        acts = F.relu(pre_acts)  # relu first — only positive values compete
        flat = acts.reshape(-1)
        k_total = min(pre_acts.shape[0] * k, flat.numel())
        # Index-based selection. A value-threshold mask (`acts >= threshold`) inflates
        # the active-count above k_total when multiple entries tie at the threshold,
        # which silently breaks the L0 guarantee. Selecting by index is exact.
        _, idx = flat.topk(k_total, sorted=False)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[idx] = True
        return (flat * mask).reshape_as(acts)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.b_dec
        pre_acts = self.encoder(x_c)
        return self._batch_topk(pre_acts, self.k)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor):
        x_c = x - self.b_dec
        pre_acts = self.encoder(x_c)
        acts = self._batch_topk(pre_acts, self.k)
        x_hat = self.decode(acts)
        return x_hat, acts, torch.zeros(1, device=x.device)

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def dictionary_vectors(self) -> torch.Tensor:
        return self.W_dec.data.T.clone()


# ──────────────────────────────────────────────────────────────
# Tied BatchTopK SAE
# ──────────────────────────────────────────────────────────────


class TiedBatchTopKSAE(nn.Module):
    """BatchTopK SAE with tied weights: W_enc = W_dec^T.

    Codes inherit BatchTopKSAE's batch-composition dependence at inference
    (see its docstring); use full-dataset encoding for consistent codes.
    """

    def __init__(self, d_input: int, d_dict: int, k: int = 12):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict
        self.k = k

        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.W_dec = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = F.linear(x - self.b_dec, self.W_dec.T, self.b_enc)
        return BatchTopKSAE._batch_topk(pre_acts, self.k)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor):
        c = self.encode(x)
        return self.decode(c), c, torch.zeros(1, device=x.device)

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def dictionary_vectors(self) -> torch.Tensor:
        return self.W_dec.data.T.clone()


# ──────────────────────────────────────────────────────────────
# Tied Matryoshka BatchTopK SAE
# ──────────────────────────────────────────────────────────────


class TiedMatryoshkaBatchTopKSAE(nn.Module):
    """Matryoshka BatchTopK SAE with tied weights (W_enc = W_dec^T).

    Per-prefix independent encoding: each prefix level gets its own top-k
    pass using only the first active_d encoder rows and decoder columns.
    With ``train_groups=True`` in training mode, sparsity_aux is the stacked
    per-level reconstruction MSE, shape (n_levels,); the trainer takes its
    mean. Otherwise sparsity_aux is zeros, shape (1,). Codes inherit
    BatchTopKSAE's batch-composition dependence at inference.
    """

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        k: int = 12,
        nested_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict
        self.k = k

        if nested_sizes is None:
            sizes = sorted(
                set(max(k, d_dict // (2**i)) for i in range(4)),
            )
            if sizes[-1] != d_dict:
                sizes.append(d_dict)
            self.nested_sizes = sizes
        else:
            self.nested_sizes = list(nested_sizes)

        assert self.nested_sizes[-1] == d_dict, (
            f"Last nested size must equal d_dict ({d_dict}), got {self.nested_sizes[-1]}"
        )

        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.W_dec = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def _forward_at_prefix(self, x_c: torch.Tensor, active_d: int):
        pre_acts = F.linear(x_c, self.W_dec[:, :active_d].T, self.b_enc[:active_d])
        acts = BatchTopKSAE._batch_topk(pre_acts, self.k)
        x_hat = acts @ self.W_dec[:, :active_d].T + self.b_dec
        return x_hat, acts

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.b_dec
        pre_acts = F.linear(x_c, self.W_dec.T, self.b_enc)
        return BatchTopKSAE._batch_topk(pre_acts, self.k)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor, train_groups: bool = False):
        x_c = x - self.b_dec

        if train_groups and self.training:
            nested_mses = []
            x_hat_full = None
            acts_full = None
            for m in self.nested_sizes:
                x_hat_m, acts_m = self._forward_at_prefix(x_c, m)
                nested_mses.append((x - x_hat_m).pow(2).sum(dim=1).mean())
                if m == self.d_dict:
                    x_hat_full = x_hat_m
                    acts_full = acts_m
            # (n_levels,) per-level recon MSE — the trainer takes .mean()
            return x_hat_full, acts_full, torch.stack(nested_mses)
        else:
            x_hat, acts = self._forward_at_prefix(x_c, self.d_dict)
            return x_hat, acts, torch.zeros(1, device=x.device)

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def dictionary_vectors(self) -> torch.Tensor:
        return self.W_dec.data.T.clone()


# ──────────────────────────────────────────────────────────────
# Matryoshka BatchTopK SAE (untied)
# ──────────────────────────────────────────────────────────────


class MatryoshkaBatchTopKSAE(nn.Module):
    """Matryoshka BatchTopK SAE (untied encoder).

    Nested-prefix training: with ``train_groups=True`` in training mode, each
    nested size m gets an independent BatchTopK encode/decode over the first
    m features, and sparsity_aux is the stacked per-level reconstruction MSE,
    shape (n_levels,); the trainer takes its mean. In eval mode (or with
    ``train_groups=False``) a single full-dictionary pass is used and
    sparsity_aux is zeros, shape (1,). Codes inherit BatchTopKSAE's
    batch-composition dependence at inference.
    """

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        k: int = 12,
        nested_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict
        self.k = k

        if nested_sizes is None:
            # Generate nesting by halving d_dict, clamped from below by k
            sizes = sorted(
                set(max(k, d_dict // (2**i)) for i in range(4)),
                reverse=False,
            )
            if sizes[-1] != d_dict:
                sizes.append(d_dict)
            self.nested_sizes = sizes
        else:
            self.nested_sizes = list(nested_sizes)

        assert self.nested_sizes[-1] == d_dict, (
            f"Last nested size must equal d_dict ({d_dict}), got {self.nested_sizes[-1]}"
        )

        self.encoder = nn.Linear(d_input, d_dict)
        self.W_dec = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def _forward_at_prefix(self, x_c: torch.Tensor, active_d: int):
        """Encode and decode using only the first active_d features."""
        # Subset of encoder weights for this prefix
        pre_acts = F.linear(x_c, self.encoder.weight[:active_d], self.encoder.bias[:active_d])  # (batch, active_d)
        acts = BatchTopKSAE._batch_topk(pre_acts, self.k)
        x_hat = acts @ self.W_dec[:, :active_d].T + self.b_dec
        return x_hat, acts

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full-dictionary encode (for evaluation)."""
        x_c = x - self.b_dec
        pre_acts = self.encoder(x_c)
        return BatchTopKSAE._batch_topk(pre_acts, self.k)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor, train_groups: bool = False):
        x_c = x - self.b_dec

        if train_groups and self.training:
            # Per-prefix independent encoding: each prefix gets its own top-k
            nested_mses = []
            x_hat_full = None
            acts_full = None
            for m in self.nested_sizes:
                x_hat_m, acts_m = self._forward_at_prefix(x_c, m)
                mse_m = (x - x_hat_m).pow(2).sum(dim=1).mean()
                nested_mses.append(mse_m)
                if m == self.d_dict:
                    x_hat_full = x_hat_m
                    # Pad acts to full d_dict for logging
                    acts_full = torch.zeros(x.shape[0], self.d_dict, device=x.device)
                    acts_full[:, :m] = acts_m

            nested_mse_tensor = torch.stack(nested_mses)
            return x_hat_full, acts_full, nested_mse_tensor
        else:
            # Eval mode: single full-dictionary forward pass
            pre_acts = self.encoder(x_c)
            acts = BatchTopKSAE._batch_topk(pre_acts, self.k)
            x_hat = self.decode(acts)
            return x_hat, acts, torch.zeros(1, device=x.device)

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def dictionary_vectors(self) -> torch.Tensor:
        return self.W_dec.data.T.clone()


# ──────────────────────────────────────────────────────────────
# Vanilla L1 SAE
# ──────────────────────────────────────────────────────────────


class VanillaL1SAE(nn.Module):
    """Standard L1-penalized SAE."""

    def __init__(self, d_input: int, d_dict: int):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict

        self.encoder = nn.Linear(d_input, d_dict)
        self.W_dec = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.b_dec
        return F.relu(self.encoder(x_c))

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor):
        c = self.encode(x)
        x_hat = self.decode(c)
        return x_hat, c, c.abs()

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def dictionary_vectors(self) -> torch.Tensor:
        return self.W_dec.data.T.clone()
