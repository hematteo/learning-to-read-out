"""Simple SAE training utilities.

`train_simple_sae` is the fixed-coefficient training loop used by the
persnap-SAE baselines, the test suite, and the minimal example;
`cross_validate_sae`, `resample_dead_features`, and the encode/eval helpers
support it. The production trajectory-crosscoder training loop lives in
`readout.crosscoder.wu_adapter.train`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from readout.core.models import (
    BatchTopKSAE,
    MatryoshkaBatchTopKSAE,
    TiedBatchTopKSAE,
    TiedMatryoshkaBatchTopKSAE,
    TiedTopKSAE,
    TopKSAE,
)  # fmt: skip

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
    was_training = model.training
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
        model.train(was_training)
        return c if keep_device else c.cpu()

    codes = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i : i + batch_size].to(device)
        c = model.encode(batch)
        codes.append(c if keep_device else c.cpu())
    model.train(was_training)
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
    was_training = model.training
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
    model.train(was_training)
    return {
        "mse": mse,
        "l0": l0,
        "dead_features": int(dead),
        "fve": fve,
        "l0_std": l0_std,
        "dead_pct": dead_pct,
    }


# ──────────────────────────────────────────────────────────────
# Simple training loop (persnap-SAE baselines, tests, examples)
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
    if num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")
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
    try:
        for epoch in trange(num_epochs, desc=arch_name, leave=False):
            warmup_factor = min(1.0, epoch / max(warmup, 1))

            # Noise augmentation: encode noisy input, reconstruct clean target
            if noise_sigma > 0:
                train_input = train_data + noise_sigma * torch.randn_like(train_data)
            else:
                train_input = train_data

            is_matryoshka = isinstance(model, (MatryoshkaBatchTopKSAE, TiedMatryoshkaBatchTopKSAE))
            if is_matryoshka and noise_sigma > 0:
                # Denoising for matryoshka: forward() scores each level against
                # its own (noisy) input, so build the per-level losses here —
                # encode the NOISY input, reconstruct the CLEAN target.
                x_c_noisy = train_input - model.b_dec
                level_mses = []
                x_hat = codes = None
                for m in model.nested_sizes:
                    x_hat_m, acts_m = model._forward_at_prefix(x_c_noisy, m)
                    level_mses.append((train_data - x_hat_m).pow(2).sum(dim=1).mean())
                    if m == model.d_dict:
                        x_hat, codes = x_hat_m, acts_m
                sparsity_aux = torch.stack(level_mses)  # (n_levels,)
            elif is_matryoshka:
                x_hat, codes, sparsity_aux = forward_fn(train_input, train_groups=True)
            else:
                x_hat, codes, sparsity_aux = forward_fn(train_input)
            recon_loss = (train_data - x_hat).pow(2).sum(dim=1).mean()

            if is_topk:
                if is_matryoshka:
                    # sparsity_aux: (n_levels,) stacked per-level recon MSE
                    loss = sparsity_aux.mean()
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
            if (
                isinstance(model, (TopKSAE, TiedTopKSAE))
                and hasattr(model, "update_dead_mask")
                and (epoch + 1) % 100 == 0
            ):
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
    finally:
        if use_compile:
            # Restore the eager forward so the compiled wrapper doesn't leak
            # out of this function with the trained model.
            model.forward = model._orig_forward
            del model._orig_forward

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
    seed: int = 0,
    **train_kwargs,
) -> dict:
    """K-fold cross-validation for an SAE architecture.

    Args:
        seed: Seeds a dedicated torch.Generator for the fold shuffle so folds
            are reproducible regardless of ambient RNG state.
        **train_kwargs: `d_dict` is required; `k` is optional (only forwarded
            to the constructor when given, so non-topk archs work too); the
            rest is passed to `train_simple_sae`.
    """
    d_model = data.shape[1]
    device = data.device
    d_dict = train_kwargs.pop("d_dict")
    k = train_kwargs.pop("k", None)
    fold_size = data.shape[0] // n_folds
    usable = fold_size * n_folds
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(data.shape[0], generator=gen)[:usable].to(device)
    data_shuffled = data[perm]
    fold_metrics: list[dict] = []

    model_kwargs = {"d_input": d_model, "d_dict": d_dict}
    if k is not None:
        model_kwargs["k"] = k

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_split = data_shuffled[val_start:val_end]
        train_split = torch.cat([data_shuffled[:val_start], data_shuffled[val_end:]], dim=0)

        model = model_class(**model_kwargs).to(device)
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
