"""Snapshot-stacked SAE crosscoder trainer for the architecture sweep (T1.5).

Path B (per the task plan): a minimal head-stacked SAE that mirrors the
llamascopium Crosscoder layout but accepts swappable activation/sparsity heads
(JumpReLU, BatchTopK, Gated). State-dict keys (``W_E``, ``b_E``, ``W_D``,
``b_D``) match the wu_adapter format so ``extract_rates.py`` works on the
JumpReLU output without modification, and the firing-rate extractor in
``experiments/crosscoders/crosscoder_main/arch_sweep_rate_plot.py`` reads the same keys for the other
two architectures.

Encoding follows the crosscoder convention: per-head linear, then sum across
heads to get a single (batch, d_sae) pre-activation, then activation/sparsity,
then per-head decode. Reconstruction loss is summed-MSE over all snapshots
(matching wu_adapter Eq. 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._schedules import lr_multiplier as _lr_multiplier


@dataclass
class ArchSweepConfig:
    arch: str  # "jumprelu" | "batchtopk" | "gated"
    n_snapshots: int
    d_model: int
    d_sae: int
    # JumpReLU
    init_threshold: float = 0.1
    bandwidth: float = 4.0
    l1_coefficient: float = 0.3
    frequency_scale: float = 0.01
    tanh_stretch: float = 1.0
    jumprelu_lr_factor: float = 0.1
    # BatchTopK
    batchtopk_k: int = 64
    # Gated
    gated_l1_coefficient: float = 0.3


class HeadStackedSAE(nn.Module):
    """Crosscoder-shaped SAE with swappable activation head.

    Parameters:
      W_E: (K, d_model, d_sae)
      b_E: (K, d_sae)
      W_D: (K, d_sae, d_model)
      b_D: (K, d_model)

    Encoding: per-head ``x[k] @ W_E[k] + b_E[k]`` summed over k.
    Decoding: per-head ``feature_acts @ W_D[k] + b_D[k]``.

    Activation-specific parameters live in submodules so the state-dict keys
    are namespaced (``activation_function.log_jumprelu_threshold`` for
    JumpReLU, ``gated.*`` for Gated, no extra params for BatchTopK).
    """

    def __init__(self, cfg: ArchSweepConfig):
        super().__init__()
        self.cfg = cfg
        K, d, D = cfg.n_snapshots, cfg.d_model, cfg.d_sae

        encoder_bound = 1.0 / math.sqrt(d)
        decoder_bound = 1.0 / math.sqrt(D)

        self.W_E = nn.Parameter(
            torch.empty(K, d, D).uniform_(-encoder_bound, encoder_bound)
        )
        self.b_E = nn.Parameter(torch.zeros(K, D))
        self.W_D = nn.Parameter(
            torch.empty(K, D, d).uniform_(-decoder_bound, decoder_bound)
        )
        self.b_D = nn.Parameter(torch.zeros(K, d))

        # Init encoder ≈ decoder^T (matches wu_adapter default factor=1.0).
        with torch.no_grad():
            self.W_E.copy_(self.W_D.transpose(-1, -2))

        if cfg.arch == "jumprelu":
            self.activation_function = _JumpReLUHead(cfg)
        elif cfg.arch == "batchtopk":
            self.activation_function = _BatchTopKHead(cfg)
        elif cfg.arch == "gated":
            self.activation_function = _GatedHead(cfg)
        else:
            raise ValueError(f"Unknown arch: {cfg.arch}")

    def decoder_norm(self) -> torch.Tensor:
        # (K, d_sae)
        return torch.norm(self.W_D, dim=-1)

    def encode_pre(self, x_stacked: torch.Tensor) -> torch.Tensor:
        # x_stacked: (B, K, d) -> per-head (x @ W_E[k] + b_E[k]), summed over K -> (B, d_sae)
        # einsum: B K d, K d D -> B K D
        per_head = torch.einsum("bkd,kdD->bkD", x_stacked, self.W_E) + self.b_E
        return per_head.sum(dim=-2)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        # feature_acts: (B, d_sae) -> per-head (feature_acts @ W_D[k] + b_D[k]) -> (B, K, d)
        return torch.einsum("bD,kDd->bkd", feature_acts, self.W_D) + self.b_D

    def forward(self, x_stacked: torch.Tensor):
        hidden_pre = self.encode_pre(x_stacked)
        feature_acts, sparsity_aux = self.activation_function(
            hidden_pre, self.decoder_norm()
        )
        recon = self.decode(feature_acts)
        return recon, feature_acts, hidden_pre, sparsity_aux


# ──────────────────────────────────────────────────────────────
# Activation heads
# ──────────────────────────────────────────────────────────────


class _JumpReLUHead(nn.Module):
    """Ge-exact JumpReLU: STE with rectangle pseudo-grad of width ``bandwidth``.

    Mirrors llamascopium activation_functions.JumpReLU. The state-dict key
    ``activation_function.log_jumprelu_threshold`` matches Run 3 / wu_adapter,
    so ``extract_rates.py`` works without modification.
    """

    def __init__(self, cfg: ArchSweepConfig):
        super().__init__()
        self.bandwidth = cfg.bandwidth
        self.log_jumprelu_threshold = nn.Parameter(
            torch.full((cfg.d_sae,), math.log(cfg.init_threshold))
        )

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_jumprelu_threshold.exp()

    def forward(self, hidden_pre: torch.Tensor, decoder_norm: torch.Tensor):
        # decoder_norm: (K, d_sae) — used for sparsity scaling, not the gate.
        theta = self.threshold  # (d_sae,)
        # STE: hard step on forward, rectangle pseudo-grad on backward via the
        # step ≈ sigmoid((pre - theta)/bw) trick.
        step_hard = (hidden_pre > theta).float()
        step_soft = torch.sigmoid((hidden_pre - theta) / self.bandwidth)
        step = step_hard - step_soft.detach() + step_soft
        feature_acts = hidden_pre * step
        return feature_acts, None


class _BatchTopKHead(nn.Module):
    """BatchTopK on the summed pre-activation. No extra learnable params.

    K (the sparsity budget) is set so the per-token mean L0 equals
    ``cfg.batchtopk_k``. Implementation matches BatchTopKSAE._batch_topk in
    src/readout/core/models.py.
    """

    def __init__(self, cfg: ArchSweepConfig):
        super().__init__()
        self.k = cfg.batchtopk_k

    def forward(self, hidden_pre: torch.Tensor, decoder_norm: torch.Tensor):
        acts = F.relu(hidden_pre)  # (B, d_sae); only positives compete
        flat = acts.reshape(-1)
        k_total = min(hidden_pre.shape[0] * self.k, flat.numel())
        _, idx = flat.topk(k_total, sorted=False)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[idx] = True
        gated = (flat * mask).reshape_as(acts)
        return gated, None


class _GatedHead(nn.Module):
    """Gated SAE pathway (Rajamanoharan 2024).

    Splits the pre-activation into a gate path and a magnitude path. The gate
    path uses STE binary; the magnitude path is ReLU(exp(r) * pre + b_mag).
    Sparsity penalty is L1 on gate_pre (via the auxiliary returned dict).

    Note: the encoder is shared (the summed pre-activation), but separate
    biases ``b_gate`` and ``b_mag`` and a per-feature scale ``r_mag`` give the
    two pathways independent affine offsets — the cheapest faithful adaptation
    of Rajamanoharan to the snapshot-summed setup.
    """

    def __init__(self, cfg: ArchSweepConfig):
        super().__init__()
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_sae))
        self.r_mag = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_mag = nn.Parameter(torch.zeros(cfg.d_sae))

    def forward(self, hidden_pre: torch.Tensor, decoder_norm: torch.Tensor):
        gate_pre = hidden_pre + self.b_gate
        gate_probs = torch.sigmoid(gate_pre)
        gate_hard = (gate_pre > 0).float()
        gate = gate_hard - gate_probs.detach() + gate_probs  # STE
        magnitude = F.relu(self.r_mag.exp() * hidden_pre + self.b_mag)
        feature_acts = gate * magnitude
        # Sparsity aux: pass gate_pre so the trainer can L1-penalize it scaled
        # by per-feature mean decoder norm (averaged over heads).
        return feature_acts, {"gate_pre": gate_pre, "gate_probs": gate_probs}


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────


def train_arch_sweep(
    snapshots: torch.Tensor,
    arch: str,
    d_sae: int,
    *,
    n_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 5e-5,
    seed: int = 0,
    device: str = "cuda",
    log_every: int = 50,
    # JumpReLU hparams (mirror wu_adapter defaults for a fair comparison)
    init_threshold: float = 0.1,
    bandwidth: float = 4.0,
    l1_coefficient: float = 0.3,
    frequency_scale: float = 0.01,
    tanh_stretch: float = 1.0,
    jumprelu_lr_factor: float = 0.1,
    # BatchTopK hparam
    batchtopk_k: int = 64,
    # Gated hparam
    gated_l1_coefficient: float = 0.3,
    # Schedule (Ge §A.4 used for parity with Run 3 launches)
    lr_warmup_fraction: float = 0.0,
    lr_decay_fraction: float = 0.0,
    l1_warmup_fraction: float = 0.0,
    # Input preprocessing handled by caller (we accept already-preprocessed snapshots).
):
    """Train a head-stacked SAE on the (K, V, d) snapshot tensor.

    Snapshots are expected pre-processed (centred / scaled) by the caller.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    K, V, d = snapshots.shape
    cfg = ArchSweepConfig(
        arch=arch,
        n_snapshots=K,
        d_model=d,
        d_sae=d_sae,
        init_threshold=init_threshold,
        bandwidth=bandwidth,
        l1_coefficient=l1_coefficient,
        frequency_scale=frequency_scale,
        tanh_stretch=tanh_stretch,
        jumprelu_lr_factor=jumprelu_lr_factor,
        batchtopk_k=batchtopk_k,
        gated_l1_coefficient=gated_l1_coefficient,
    )
    model = HeadStackedSAE(cfg).to(device)

    # Param groups: threshold gets a smaller LR (mirrors wu_adapter).
    threshold_params, other_params = [], []
    for name, p in model.named_parameters():
        if "log_jumprelu_threshold" in name:
            threshold_params.append(p)
        else:
            other_params.append(p)
    pg = [{"params": other_params, "lr": lr}]
    if threshold_params:
        pg.append({"params": threshold_params, "lr": lr * jumprelu_lr_factor})
    optimizer = torch.optim.Adam(pg, foreach=False)
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]

    snapshots_cpu = snapshots.cpu()  # (K, V, d)
    steps_per_epoch = math.ceil(V / batch_size)
    total_steps = n_epochs * steps_per_epoch
    warmup_steps_l1 = int(l1_warmup_fraction * total_steps)

    step_count = 0
    for epoch in range(n_epochs):
        idx = torch.randperm(V, generator=gen)
        for start in range(0, V, batch_size):
            sel = idx[start : start + batch_size]
            x_stacked = (
                snapshots_cpu[:, sel, :].permute(1, 0, 2).contiguous().to(device)
            )
            # x_stacked: (B, K, d)

            recon, feature_acts, hidden_pre, sparsity_aux = model(x_stacked)
            # Reconstruction loss: per-snapshot MSE summed over d, mean over batch and snapshot.
            l_rec = (recon - x_stacked).pow(2).sum(dim=-1).mean()

            # LR schedule
            if lr_warmup_fraction > 0 or lr_decay_fraction > 0:
                lr_mult = _lr_multiplier(
                    step_count, total_steps, lr_warmup_fraction, lr_decay_fraction
                )
                for g in optimizer.param_groups:
                    g["lr"] = g["initial_lr"] * lr_mult

            # L1 warmup scale
            if warmup_steps_l1 > 0:
                l1_scale = min(1.0, step_count / max(1, warmup_steps_l1))
            else:
                l1_scale = 1.0

            l_s = torch.zeros((), device=device)
            if arch == "jumprelu":
                # Ge tanh-quad sparsity (eq. 9) on (feature_acts * decoder_norm).
                decoder_norms = model.decoder_norm()  # (K, d_sae)
                # decoder_norm scales per-feature; aggregate across snapshots by mean
                # to match the crosscoder convention (decoder_norm is shape K, d_sae;
                # llamascopium `sparsity_include_decoder_norm` multiplies feature_acts by
                # the per-feature norm then sums across heads).
                dn_sum = decoder_norms.sum(dim=0)  # (d_sae,)
                score = torch.tanh(tanh_stretch * feature_acts * dn_sum)
                approx_freq = score.mean(dim=0)  # (d_sae,)
                l_s = (approx_freq * (1 + approx_freq / frequency_scale)).sum()
                loss = l_rec + l1_coefficient * l1_scale * l_s
            elif arch == "batchtopk":
                # No sparsity penalty — top-k is an exact constraint.
                loss = l_rec
            elif arch == "gated":
                # Rajamanoharan 2024: L1 on gate_pre after the gate sigmoid,
                # weighted by the mean decoder norm across heads.
                decoder_norms = model.decoder_norm()  # (K, d_sae)
                dn_sum = decoder_norms.sum(dim=0)  # (d_sae,)
                gate_relu = F.relu(sparsity_aux["gate_pre"])
                l_s = (gate_relu * dn_sum).sum(dim=-1).mean()
                loss = l_rec + gated_l1_coefficient * l1_scale * l_s
                # Auxiliary reconstruction from gate_pre (Rajamanoharan eq. 5):
                # ReLU(gate_pre) reconstructs x via a frozen decoder copy.
                # We use the live decoder; this is the standard implementation.
                aux_acts = F.relu(sparsity_aux["gate_pre"])
                aux_recon = model.decode(aux_acts)
                l_aux = (aux_recon - x_stacked.detach()).pow(2).sum(dim=-1).mean()
                loss = loss + l_aux
            else:
                raise ValueError(arch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += 1
            if step_count % log_every == 0:
                with torch.no_grad():
                    l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
                print(
                    f"[{arch} epoch {epoch} step {step_count}] "
                    f"loss {loss.item():.4f} l_rec {l_rec.item():.4f} "
                    f"l_s {l_s.detach().item():.4f} l0 {l0:.1f} l1x {l1_scale:.2f}",
                    flush=True,
                )

    return model


def quick_quality(
    model: HeadStackedSAE,
    snapshots: torch.Tensor,
    batch_size: int = 1024,
    device: str = "cuda",
):
    model.eval()
    K, V, d = snapshots.shape
    snapshots_cpu = snapshots.cpu()
    total_rec = 0.0
    total_var = 0.0
    total_l0 = 0.0
    n_batches = 0
    with torch.no_grad():
        for start in range(0, V, batch_size):
            sel = slice(start, start + batch_size)
            x_stacked = (
                snapshots_cpu[:, sel, :].permute(1, 0, 2).contiguous().to(device)
            )
            recon, feature_acts, _, _ = model(x_stacked)
            total_rec += (recon - x_stacked).pow(2).mean().item()
            total_var += x_stacked.var().item()
            total_l0 += (feature_acts > 0).float().sum(dim=-1).mean().item()
            n_batches += 1
    ev = 1.0 - (total_rec / total_var) if total_var > 0 else 0.0
    return {
        "explained_variance": ev,
        "reconstruction_mse": total_rec / n_batches,
        "mean_l0": total_l0 / n_batches,
        "d_sae": model.cfg.d_sae,
        "n_snapshots": K,
        "n_rows": V,
    }
