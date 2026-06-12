"""
Training loops and utilities for proto-token SAEs.
====================================================
Unified training for all architectures (Gated, JumpReLU, TopK, L1)
with adaptive sparsity targeting and dead feature resampling.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from src.core.data import center_and_project, get_device
from src.core.models import (
    BatchTopKSAE,
    GatedSAE,
    JumpReLUSAE,
    MatryoshkaBatchTopKSAE,
    TiedBatchTopKSAE,
    TiedMatryoshkaBatchTopKSAE,
    TiedTopKSAE,
    TopKSAE,
)  # fmt: skip

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    model_name: str = "gpt2"
    d_dict: int = 4096
    target_l0: float = 12.0
    lr: float = 3e-4
    num_epochs: int = 15_000
    batch_size: int = 0  # 0 = full-batch
    sparsity_warmup: int = 2_000
    resample_every: int = 2_000
    seed: int = 42
    save_dir: str = "results"
    log_every: int = 250

    # Centering
    svd_remove: int = 0

    # Regularisation
    weight_decay: float = 0.0

    # JumpReLU specific
    initial_threshold: float = 0.001
    bandwidth: float = 0.01
    threshold_lr_mult: float = 3.0

    @property
    def device(self) -> str:
        return get_device()


# ──────────────────────────────────────────────────────────────
# Encoding helper
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def get_all_codes(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int = 8192,
    keep_device: bool = False,
) -> torch.Tensor:
    """Encode the full dataset and return sparse codes.

    For BatchTopK/MatryoshkaBatchTopK, processes the full dataset in one
    pass to match training-time batch sparsity behavior.

    Args:
        keep_device: If True, keep codes on the model's device (faster for
                     on-GPU metrics). If False, move to CPU (original behavior).
    """
    model.eval()
    device = next(model.parameters()).device

    # BatchTopK sparsity depends on batch size — use full batch to match training
    if isinstance(
        model,
        (
            BatchTopKSAE,
            TiedBatchTopKSAE,
            MatryoshkaBatchTopKSAE,
            TiedMatryoshkaBatchTopKSAE,
        ),
    ):
        data_dev = data.to(device)
        c = model.encode(data_dev)
        model.train()
        return c if keep_device else c.cpu()

    codes = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i : i + batch_size].to(device)
        c = model.encode(batch)
        codes.append(c if keep_device else c.cpu())
    model.train()
    return torch.cat(codes, dim=0)


# ──────────────────────────────────────────────────────────────
# Dead feature resampling
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def resample_dead_features(model: nn.Module, data: torch.Tensor, codes: torch.Tensor) -> int:
    """Reinitialise dead features toward worst-reconstructed tokens.

    Fully vectorized — no Python loops. Works on whatever device
    data/model are on (CPU, CUDA, MPS).
    """
    assert data.device == codes.device, f"data ({data.device}) and codes ({codes.device}) must be on same device"
    feature_freq = (codes > 0).float().mean(dim=0)
    dead_mask = feature_freq == 0
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return 0

    # Batched decode to avoid OOM at large d_dict
    x_hat_parts = []
    for i in range(0, codes.shape[0], 4096):
        x_hat_parts.append(model.decode(codes[i : i + 4096]))
    x_hat = torch.cat(x_hat_parts, dim=0)
    del x_hat_parts
    recon_error = (data - x_hat).pow(2).sum(dim=1)

    error_sum = recon_error.sum()
    if error_sum == 0:
        return 0  # perfect reconstruction — nothing to resample from
    probs = recon_error / error_sum
    replacement = n_dead > data.shape[0]  # avoid duplicate directions when possible
    sample_idx = torch.multinomial(probs, n_dead, replacement=replacement)
    directions = F.normalize(data[sample_idx] - x_hat[sample_idx], dim=1)

    dead_indices = dead_mask.nonzero().squeeze(-1)

    # Batched decoder reset: W_dec is (d_input, d_dict)
    model.W_dec.data[:, dead_indices] = directions.T

    # Batched encoder reset (architecture-specific)
    if hasattr(model, "W_enc"):  # GatedSAE
        model.W_enc.data[dead_indices] = directions
        model.b_gate.data[dead_indices] = 0.0
        model.r_mag.data[dead_indices] = 0.0
        model.b_mag.data[dead_indices] = 0.0
    elif hasattr(model, "encoder"):  # JumpReLU / TopK / L1
        model.encoder.weight.data[dead_indices] = directions
        model.encoder.bias.data[dead_indices] = 0.0
        if hasattr(model, "log_threshold"):  # JumpReLU
            model.log_threshold.data[dead_indices] = math.log(0.001)
    elif hasattr(model, "b_enc"):  # TiedTopKSAE and tied variants
        # W_dec[:, dead_indices] already reset above; W_enc = W_dec.T picks it up
        model.b_enc.data[dead_indices] = 0.0

    return n_dead


# ──────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(model: nn.Module, data: torch.Tensor) -> dict:
    """Compute MSE, L0, dead feature count on full dataset."""
    model.eval()
    device = next(model.parameters()).device
    codes = get_all_codes(model, data, keep_device=True)
    x_hat = model.decode(codes)
    data_dev = data.to(device)
    mse = (data_dev - x_hat).pow(2).sum(dim=1).mean().item()
    active = codes > 0  # (V, d_dict) bool — compute once, reuse
    per_token_l0 = active.sum(dim=1).float()  # (V,)
    l0 = per_token_l0.mean().item()
    l0_std = per_token_l0.std().item()
    dead = (active.float().mean(dim=0) == 0).sum().item()
    # Total variance: data is already centered, so E[||x||^2] = total variance
    data_var = data_dev.pow(2).sum(dim=1).mean().item()
    fve = 1.0 - mse / data_var if data_var > 0 else 0.0
    d_dict = codes.shape[1]
    dead_pct = dead / d_dict
    return {
        "mse": mse,
        "l0": l0,
        "dead_features": int(dead),
        "fve": fve,
        "l0_std": l0_std,
        "dead_pct": dead_pct,
    }


# ──────────────────────────────────────────────────────────────
# Main training loop (Gated / JumpReLU)
# ──────────────────────────────────────────────────────────────


def train_model(
    model: nn.Module,
    W_U: torch.Tensor,
    cfg: TrainConfig,
    patience: int = 500,
    normalize_rows: bool = False,
    *,
    preprocessed_data: torch.Tensor | None = None,
    train_data_override: torch.Tensor | None = None,
    val_data_override: torch.Tensor | None = None,
) -> tuple:
    """Train a Gated or JumpReLU SAE with adaptive sparsity targeting.

    Args:
        preprocessed_data: If provided, skip center_and_project and use this
            directly. centering is returned as None.
        train_data_override / val_data_override: If both provided, use these
            pre-computed splits instead of creating a new random split. This
            ensures cross-architecture comparability.

    Returns (trained_model, logs, centering_dict).
    """
    # NOTE: this seeds training-time stochasticity only (the held-out split,
    # batch shuffling, resampling). It does NOT control model init — `model` is
    # constructed by the caller before being passed in, so its weights are
    # already fixed by the time we get here. Seed at the construction site if
    # init reproducibility is required.
    torch.manual_seed(cfg.seed)
    device = cfg.device

    if preprocessed_data is not None:
        data = preprocessed_data.to(device)
        centering = None
    else:
        data, centering = center_and_project(
            W_U, svd_remove=cfg.svd_remove, device=device, normalize_rows=normalize_rows
        )
    V, d = data.shape

    # Early stopping: hold out 10% when patience > 0
    if (train_data_override is None) != (val_data_override is None):
        raise ValueError("Provide both train_data_override and val_data_override, or neither.")
    if train_data_override is not None and val_data_override is not None:
        train_data = train_data_override.to(device)
        val_data = val_data_override.to(device)
        V_train = train_data.shape[0]
    elif patience > 0:
        n_val = max(1, V // 10)
        perm_split = torch.randperm(V, device=device)
        val_data = data[perm_split[:n_val]]
        train_data = data[perm_split[n_val:]]
        V_train = train_data.shape[0]
    else:
        train_data = data
        val_data = None
        V_train = V

    model = model.to(device)
    bs = V_train if cfg.batch_size <= 0 else min(cfg.batch_size, V_train)

    # Optimiser with separate lr for JumpReLU thresholds
    if isinstance(model, JumpReLUSAE):
        main_params = [p for n, p in model.named_parameters() if "log_threshold" not in n]
        param_groups = [
            {"params": main_params, "lr": cfg.lr},
            {"params": [model.log_threshold], "lr": cfg.lr * cfg.threshold_lr_mult},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": cfg.lr}]

    optimizer = torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)

    # Adaptive sparsity coefficient (Lagrange multiplier)
    log_lambda = math.log(1e-2)
    sparsity_coeff = math.exp(log_lambda)

    logs = {
        "epoch": [],
        "loss": [],
        "recon": [],
        "sparsity": [],
        "l0": [],
        "sparsity_coeff": [],
        "dead_features": [],
    }

    arch_name = "Gated" if isinstance(model, GatedSAE) else "JumpReLU"
    use_compile = (
        str(device).startswith("cuda") if not isinstance(device, torch.device) else device.type == "cuda"
    ) and hasattr(torch, "compile")
    print(f"\n{'=' * 60}")
    print(f"Training {arch_name} SAE")
    print(f"  D={cfg.d_dict}  target_L0={cfg.target_l0}  lr={cfg.lr}")
    print(f"  epochs={cfg.num_epochs}  warmup={cfg.sparsity_warmup}  device={device}")
    print(
        f"  batch_size={'full' if bs == V_train else bs}  patience={patience}  "
        f"resample_every={cfg.resample_every}  compile={use_compile}"
    )
    if isinstance(model, JumpReLUSAE):
        print(
            f"  initial_threshold={cfg.initial_threshold}  "
            f"bandwidth={cfg.bandwidth}  "
            f"threshold_lr={cfg.lr * cfg.threshold_lr_mult:.1e}"
        )
    print(f"{'=' * 60}")

    # Compile only the forward method (not the whole module) so that
    # in-place ops like normalize_decoder() don't invalidate the graph.
    if use_compile:
        model._orig_forward = model.forward
        model.forward = torch.compile(model.forward)
    forward_fn = model
    is_mps = device.type == "mps" if isinstance(device, torch.device) else device == "mps"

    best_val_mse = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in trange(cfg.num_epochs, desc=f"{arch_name} SAE"):
        warmup_factor = min(1.0, epoch / max(cfg.sparsity_warmup, 1))
        effective_lambda = sparsity_coeff * warmup_factor

        perm = torch.randperm(V_train, device=device)
        epoch_recon = 0.0
        epoch_sparsity = 0.0
        epoch_l0 = 0.0
        n_batches = 0
        nan_detected = False

        for i in range(0, V_train, bs):
            batch = train_data[perm[i : i + bs]]
            x_hat, codes, sparsity_aux = forward_fn(batch)

            recon_loss = (batch - x_hat).pow(2).sum(dim=1).mean()
            sparsity_loss = sparsity_aux.sum(dim=1).mean()
            loss = recon_loss + effective_lambda * sparsity_loss

            if loss.isnan() or loss.isinf():
                tqdm.write(f"  [{arch_name}] NaN/Inf loss at epoch {epoch + 1}, stopping")
                nan_detected = True
                break

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

            with torch.no_grad():
                epoch_recon += recon_loss.item()
                epoch_sparsity += sparsity_loss.item()
                epoch_l0 += (codes > 0).float().sum(dim=1).mean().item()
                n_batches += 1

        if nan_detected:
            break

        avg_recon = epoch_recon / max(n_batches, 1)
        avg_l0 = epoch_l0 / n_batches

        # Adaptive lambda update (arch-specific gain, k-normalised)
        if warmup_factor >= 1.0:
            error = avg_l0 - cfg.target_l0
            if isinstance(model, GatedSAE):
                gain = 0.02
            elif isinstance(model, JumpReLUSAE):
                gain = 0.05
            else:
                gain = 0.03  # VanillaL1SAE
            step = gain * error / max(cfg.target_l0, 1)
            log_lambda += step
            # Arch-specific bounds
            from .models import VanillaL1SAE

            if isinstance(model, VanillaL1SAE):
                log_lambda = max(math.log(1e-8), min(math.log(10.0), log_lambda))
            else:
                log_lambda = max(math.log(1e-6), min(math.log(100.0), log_lambda))
            sparsity_coeff = math.exp(log_lambda)

        # Early stopping (only after warmup, eval every 10 epochs to save compute)
        if patience > 0 and warmup_factor >= 1.0 and val_data is not None and epoch % 10 == 0:
            with torch.no_grad():
                val_hat, _, _ = model(val_data)
                val_mse = (val_data - val_hat).pow(2).sum(dim=1).mean().item()
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 10  # count in epochs, not val checks
                if patience_counter >= patience:
                    tqdm.write(f"  [{arch_name}] Early stopping at epoch {epoch + 1} (val_mse={val_mse:.6f})")
                    break

        # Dead feature resampling (use train_data only to avoid val leakage)
        cached_dead = None
        if (epoch + 1) % cfg.resample_every == 0 and epoch < cfg.num_epochs - 1000:
            with torch.no_grad():
                if is_mps:
                    # MPS: move to CPU to avoid OOM
                    model.cpu()
                    data_cpu = train_data.cpu()
                    all_codes = get_all_codes(model, data_cpu)
                    n_resampled = resample_dead_features(model, data_cpu, all_codes)
                    cached_dead = ((all_codes > 0).float().mean(dim=0) == 0).sum().item()
                    model.to(device)
                    del data_cpu, all_codes
                    torch.mps.empty_cache()
                else:
                    # CUDA/CPU: resample in-place on device
                    all_codes = get_all_codes(model, train_data, keep_device=True)
                    cached_dead = ((all_codes > 0).float().mean(dim=0) == 0).sum().item()
                    n_resampled = resample_dead_features(model, train_data, all_codes)
                    del all_codes
                if n_resampled > 0:
                    tqdm.write(f"  [epoch {epoch + 1}] Resampled {n_resampled} dead features")

        # Logging
        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            if cached_dead is not None:
                dead = cached_dead
            else:
                with torch.no_grad():
                    all_codes = get_all_codes(model, train_data, keep_device=True)
                    dead = ((all_codes > 0).float().mean(dim=0) == 0).sum().item()
                    del all_codes

            logs["epoch"].append(epoch + 1)
            logs["loss"].append(avg_recon + effective_lambda * (epoch_sparsity / n_batches))
            logs["recon"].append(avg_recon)
            logs["sparsity"].append(epoch_sparsity / n_batches)
            logs["l0"].append(avg_l0)
            logs["sparsity_coeff"].append(sparsity_coeff)
            logs["dead_features"].append(dead)

            if (epoch + 1) % (cfg.log_every * 4) == 0 or epoch == 0:
                msg = (
                    f"  [epoch {epoch + 1:>6}]  recon={avg_recon:.4f}  "
                    f"L0={avg_l0:.1f}  \u03bb={sparsity_coeff:.5f}  dead={dead}"
                )
                if isinstance(model, JumpReLUSAE):
                    theta = model.threshold
                    msg += (
                        f"\n    thresholds: min={theta.min().item():.4f}  "
                        f"median={theta.median().item():.4f}  "
                        f"max={theta.max().item():.4f}"
                    )
                tqdm.write(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Early stopping / epoch metadata
    actual_epochs = epoch + 1 if logs["epoch"] else 0
    logs["final_epoch"] = actual_epochs
    logs["early_stopped"] = patience > 0 and actual_epochs < cfg.num_epochs

    # Train/val metrics for overfitting monitoring and consistent cross-arch evaluation
    if val_data is not None:
        train_m = evaluate_model(model, train_data)
        val_m = evaluate_model(model, val_data)
        logs["train_fve"] = train_m["fve"]
        logs["val_fve"] = val_m["fve"]
        logs["val_mse"] = val_m["mse"]
        logs["train_mse"] = train_m["mse"]
        logs["val_metrics"] = val_m

    return model, logs, centering


# ──────────────────────────────────────────────────────────────
# Simple training loop (TopK / Vanilla L1)
# ──────────────────────────────────────────────────────────────


def train_simple_sae(
    model: nn.Module,
    data: torch.Tensor,
    num_epochs: int = 8000,
    lr: float = 3e-4,
    sparsity_coeff: float = 0.01,
    target_l0: float = 12.0,
    resample_every: int = 0,
    arch_name: str = "SAE",
    patience: int = 0,
    weight_decay: float = 0.0,
    seed: int | None = None,
    noise_sigma: float = 0.0,
    lr_warmup: int = 0,
    grad_clip: float = 0.0,
    *,
    train_data_override: torch.Tensor | None = None,
    val_data_override: torch.Tensor | None = None,
) -> dict:
    """Train TopK/BatchTopK or Vanilla L1 SAE with optional resampling.

    Args:
        seed: If provided, seeds torch RNG at the start for reproducibility.
        noise_sigma: If > 0, add Gaussian noise to training data each epoch.
        lr_warmup: Number of epochs for linear LR warmup from 0 to lr.
        grad_clip: If > 0, clip gradient max_norm to this value.
        train_data_override / val_data_override: If both provided, use these
            pre-computed splits instead of creating a new random split. This
            ensures cross-architecture comparability.

    Returns metrics dict with mse, l0, dead_features, time.
    """
    if seed is not None:
        torch.manual_seed(seed)
    import time

    device = data.device
    model = model.to(device)
    V = data.shape[0]

    # Early stopping: hold out 10% when patience > 0
    if (train_data_override is None) != (val_data_override is None):
        raise ValueError("Provide both train_data_override and val_data_override, or neither.")
    if train_data_override is not None and val_data_override is not None:
        train_data = train_data_override.to(device)
        val_data = val_data_override.to(device)
    elif patience > 0:
        n_val = max(1, V // 10)
        perm_split = torch.randperm(V, device=device)
        val_data = data[perm_split[:n_val]]
        train_data = data[perm_split[n_val:]]
    else:
        train_data = data
        val_data = None

    if weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_lambda = math.log(sparsity_coeff)
    is_topk = isinstance(
        model,
        (
            TopKSAE,
            TiedTopKSAE,
            BatchTopKSAE,
            TiedBatchTopKSAE,
            MatryoshkaBatchTopKSAE,
            TiedMatryoshkaBatchTopKSAE,
        ),
    )
    warmup = 0 if is_topk else 2000  # TopK archs have no adaptive lambda

    use_compile = str(device).startswith("cuda") and hasattr(torch, "compile")
    if use_compile:
        model._orig_forward = model.forward
        model.forward = torch.compile(model.forward)
    forward_fn = model
    is_mps = device.type == "mps" if isinstance(device, torch.device) else device == "mps"

    best_val_mse = float("inf")
    best_state = None
    patience_counter = 0
    nan_stopped = False

    # LR warmup scheduler
    if lr_warmup > 0:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=lr_warmup
        )
    else:
        lr_scheduler = None

    t0 = time.time()
    for epoch in trange(num_epochs, desc=arch_name, leave=False):
        warmup_factor = min(1.0, epoch / max(warmup, 1))

        # Noise augmentation: encode noisy input, reconstruct clean target
        if noise_sigma > 0:
            train_input = train_data + noise_sigma * torch.randn_like(train_data)
        else:
            train_input = train_data

        is_matryoshka = isinstance(model, (MatryoshkaBatchTopKSAE, TiedMatryoshkaBatchTopKSAE))
        if is_matryoshka:
            x_hat, codes, sparsity_aux = forward_fn(train_input, train_groups=True)
        else:
            x_hat, codes, sparsity_aux = forward_fn(train_input)
        recon_loss = (train_data - x_hat).pow(2).sum(dim=1).mean()

        if is_topk:
            if is_matryoshka:
                if noise_sigma > 0 and hasattr(model, "_forward_at_prefix"):
                    # Recompute multi-scale loss against CLEAN target
                    x_c_clean = train_data - model.b_dec
                    clean_loss = torch.zeros(1, device=train_data.device)
                    for m in model.nested_sizes:
                        x_hat_m, _ = model._forward_at_prefix(x_c_clean, m)
                        clean_loss = clean_loss + ((train_data - x_hat_m).pow(2).sum(dim=1).mean())
                    loss = clean_loss / len(model.nested_sizes)
                else:
                    # sparsity_aux is averaged multi-scale loss (scalar) for tied,
                    # stacked tensor for untied — handle both
                    loss = sparsity_aux.sum() if sparsity_aux.dim() > 0 else sparsity_aux
            else:
                loss = recon_loss
                # AuxK loss for TopK/TiedTopK (Gao et al., alpha=1/32)
                if isinstance(model, (TopKSAE, TiedTopKSAE)) and sparsity_aux.requires_grad:
                    loss = loss + (1.0 / 32.0) * sparsity_aux
        else:
            effective_lambda = math.exp(log_lambda) * warmup_factor
            sparsity_loss = sparsity_aux.sum(dim=1).mean()
            loss = recon_loss + effective_lambda * sparsity_loss

        if loss.isnan() or loss.isinf():
            tqdm.write(f"    [{arch_name}] NaN/Inf loss at epoch {epoch + 1}, stopping")
            nan_stopped = True
            if best_state is None:
                # Save current params (not yet corrupted — break is before optimizer.step)
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        model.normalize_decoder()

        # Adaptive lambda for L1 (k-normalised gain)
        if not is_topk and warmup_factor >= 1.0:
            with torch.no_grad():
                cur_l0 = (codes > 0).float().sum(dim=1).mean().item()
            error = cur_l0 - target_l0
            step = 0.03 * error / max(target_l0, 1)
            log_lambda += step
            log_lambda = max(math.log(1e-8), min(math.log(10.0), log_lambda))

        # Early stopping (only after warmup, eval every 10 epochs to save compute)
        if patience > 0 and warmup_factor >= 1.0 and epoch % 10 == 0:
            with torch.no_grad():
                val_hat, _, _ = model(val_data)
                val_mse = (val_data - val_hat).pow(2).sum(dim=1).mean().item()
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 10  # count in epochs, not val checks
                if patience_counter >= patience:
                    tqdm.write(f"    [{arch_name}] Early stopping at epoch {epoch + 1} (val_mse={val_mse:.6f})")
                    break

        # Update AuxK dead mask every 100 epochs (TopK and TiedTopK)
        # Reuse codes from the main forward pass — no extra forward needed
        if isinstance(model, (TopKSAE, TiedTopKSAE)) and hasattr(model, "update_dead_mask") and (epoch + 1) % 100 == 0:
            with torch.no_grad():
                model.update_dead_mask(codes.detach())

        # Dead feature resampling (use train_data only to avoid val leakage)
        if resample_every > 0 and (epoch + 1) % resample_every == 0 and epoch < num_epochs - 1000:
            with torch.no_grad():
                if is_mps:
                    model.cpu()
                    td_cpu = train_data.cpu()
                    codes_cpu = model.encode(td_cpu)
                    n = resample_dead_features(model, td_cpu, codes_cpu)
                    model.to(device)
                    torch.mps.empty_cache()
                    del td_cpu, codes_cpu
                else:
                    codes_gpu = get_all_codes(model, train_data, keep_device=True)
                    n = resample_dead_features(model, train_data, codes_gpu)
                    del codes_gpu
                if n > 0:
                    tqdm.write(f"    [{arch_name} epoch {epoch + 1}] Resampled {n} dead features")

    actual_epochs = epoch + 1
    elapsed = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation: use val split as primary (unbiased), train split for overfitting check
    if val_data is not None:
        val_m = evaluate_model(model, val_data)
        train_m = evaluate_model(model, train_data)
        metrics = val_m  # primary metrics from held-out data
        metrics["train_fve"] = train_m["fve"]
        metrics["val_fve"] = val_m["fve"]
        metrics["train_mse"] = train_m["mse"]
    else:
        metrics = evaluate_model(model, data)
        metrics["train_fve"] = metrics["fve"]
        metrics["val_fve"] = metrics["fve"]
        metrics["train_mse"] = metrics["mse"]

    metrics["time"] = elapsed
    metrics["final_epoch"] = actual_epochs
    metrics["early_stopped"] = (patience > 0 and patience_counter >= patience) or nan_stopped

    return metrics


# ──────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────


def cross_validate_sae(
    model_class: type,
    data: torch.Tensor,
    n_folds: int = 5,
    **train_kwargs,
) -> dict:
    d_model = data.shape[1]
    device = data.device
    d_dict = train_kwargs.pop("d_dict")
    k = train_kwargs.pop("k")
    fold_size = data.shape[0] // n_folds
    usable = fold_size * n_folds
    perm = torch.randperm(data.shape[0], device=device)[:usable]
    data_shuffled = data[perm]
    fold_metrics: list[dict] = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_split = data_shuffled[val_start:val_end]
        train_split = torch.cat([data_shuffled[:val_start], data_shuffled[val_end:]], dim=0)

        model = model_class(d_input=d_model, d_dict=d_dict, k=k).to(device)
        train_simple_sae(model, train_split, **train_kwargs)
        metrics = evaluate_model(model, val_split)
        fold_metrics.append(metrics)
        print(f"  Fold {fold + 1}/{n_folds}: mse={metrics['mse']:.4f}  l0={metrics['l0']:.1f}")

    mses = [m["mse"] for m in fold_metrics]
    return {
        "fold_metrics": fold_metrics,
        "mean_mse": sum(mses) / len(mses),
        "std_mse": (sum((m - sum(mses) / len(mses)) ** 2 for m in mses) / len(mses)) ** 0.5,
    }
