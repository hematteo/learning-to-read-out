"""Train a llamascopium Crosscoder on W_U rows across Pythia training snapshots.

Row v of W_U^(t) is a (d_model,) vector per snapshot t, per token v. We stack
snapshots along the `hook_points` dimension (K = n_snapshots) and treat the V
vocabulary rows as the "batch" axis the crosscoder iterates over.
"""

import argparse
import math
import os
from pathlib import Path

import einops
import torch
from llamascopium.models.crosscoder import Crosscoder, CrosscoderConfig
from transformers import AutoModelForCausalLM

from readout.core.data import get_device
from readout.core.model_specs import DEFAULT_STEPS_32, DEFAULT_STEPS_BY_MODEL, SPECS
from readout.core.paths import snapshot_dir as _snapshot_dir
from readout.core.repro import git_commit, log_run_provenance, seed_everything
from readout.core.resume import atomic_write_torch

from ._schedules import lr_multiplier as _lr_multiplier

# Ge et al. 2026 (arxiv 2509.17196) §4 "Experimental setup": 32 snapshots out of 154.
# Canonical schedule lives in readout.core.model_specs (single source of truth).
DEFAULT_STEPS = DEFAULT_STEPS_32

DEFAULT_MODEL = "EleutherAI/pythia-160m"
# Default to the canonical SSD layout (per readout.core.paths.snapshot_dir).
# Override via WU_CACHE_DIR for non-canonical caches (e.g. W_E extraction).
DEFAULT_CACHE = Path(os.environ.get("WU_CACHE_DIR", str(_snapshot_dir(DEFAULT_MODEL))))


def _terminal_step_for(model_name):
    """Last step of the model's canonical snapshot schedule (Pythia: 143000)."""
    for label, spec in SPECS.items():
        if spec.name == model_name:
            return DEFAULT_STEPS_BY_MODEL[label][-1]
    return DEFAULT_STEPS[-1]


def load_wu_snapshot(model_name, step, cache_dir, dtype=torch.float32):
    """Return the (V, d_model) W_U snapshot for ``step``, from cache or extracted on miss."""
    slug = model_name.replace("/", "_")
    cache_path = cache_dir / f"{slug}_step{step}_wu.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu").to(dtype)
    # Cache miss: only Pythia's `step{N}` revision format and `embed_out` lm-head
    # path are supported here. Other families (OLMo: `stage1-step{N}-tokens*B` +
    # `lm_head.weight`) must be pre-extracted via a family-specific extractor.
    if not model_name.lower().startswith("eleutherai/pythia"):
        raise FileNotFoundError(
            f"Snapshot {cache_path} missing and model {model_name!r} is not Pythia. "
            f"wu_adapter.load_wu_snapshot only knows the Pythia 'step{{N}}' revision format "
            f"and the `embed_out.weight` unembedding path. Pre-extract via the "
            f"family-specific extractor (e.g. experiments/crosscoders/crosscoder_olmo/scripts/extract_wu_olmo.py "
            f"for OLMo) and re-run."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=f"step{step}",
        dtype=torch.float32,
    )
    W_U = model.embed_out.weight.detach().clone().cpu()
    # Atomic write: a kill mid-save must not leave a corrupt .pt in the cache.
    atomic_write_torch(cache_path, W_U)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return W_U.to(dtype)


def load_snapshots(model_name=DEFAULT_MODEL, steps=None, cache_dir=DEFAULT_CACHE, dtype=torch.float32):
    steps = list(steps) if steps is not None else list(DEFAULT_STEPS)
    snapshots = [load_wu_snapshot(model_name, s, cache_dir, dtype=dtype) for s in steps]
    # (K, V, d)
    return torch.stack(snapshots, dim=0), steps


def preprocess_snapshots(snapshots, mode="none"):
    """Per-snapshot input normalization (Rajamanoharan 2024, Gemma Scope 2024).

    mode="center"       : subtract per-snapshot row-mean.
    mode="center_scale" : subtract mean then scale so E[||x||^2] = d_model per snapshot.
                          NOTE: changes input magnitude ~50x for raw Pythia W_U;
                          l1_coefficient should be retuned (try ~0.3 a la Ge et al.).
    mode="none"         : no-op (default, preserves Run-2 behavior).

    Returns (processed, stats) where stats is a dict with per-snapshot 'mean' (K,1,d) and
    'scale' (K,1,1), or None for mode="none". Stats are invertible for downstream
    weight-swap / inference on raw W_U.
    """
    if mode == "none":
        return snapshots, None
    mean = snapshots.mean(dim=1, keepdim=True)  # (K, 1, d)
    centered = snapshots - mean
    K = snapshots.shape[0]
    if mode == "center":
        scale = torch.ones(K, 1, 1, dtype=snapshots.dtype)
        return centered, {"mean": mean, "scale": scale, "mode": mode}
    if mode == "center_scale":
        d = snapshots.shape[-1]
        mean_sq = centered.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)  # (K, 1, 1)
        scale = (mean_sq / d).clamp_min(1e-8).sqrt()
        return centered / scale, {"mean": mean, "scale": scale, "mode": mode}
    raise ValueError(f"Unknown preprocess mode: {mode}")


def build_crosscoder(
    n_snapshots,
    d_model,
    expansion_factor,
    device=None,
    dtype=torch.float32,
    jumprelu_threshold_window=4.0,
    init_threshold=0.1,
    init_encoder_with_decoder_transpose_factor=1.0,
    device_mesh=None,
):
    """Build a llamascopium Crosscoder with optional head-parallel sharding.

    When ``device_mesh`` is None (default), the crosscoder is constructed as
    a regular non-distributed model — same path Run 3 / Run 5 used. When a
    DeviceMesh is provided, weights are constructed as DTensors sharded
    along the n_heads axis and forward/backward are head-parallel via
    All-Reduce on the shared pre-activation (Ge §A.3).

    ``device=None`` auto-selects via ``readout.core.data.get_device()``
    (cuda > mps > cpu).
    """
    if device is None:
        device = get_device()
    cfg = CrosscoderConfig(
        sae_type="crosscoder",
        hook_points=[f"wu_step_{i}" for i in range(n_snapshots)],
        d_model=d_model,
        expansion_factor=expansion_factor,
        act_fn="jumprelu",
        jumprelu_threshold_window=jumprelu_threshold_window,
        sparsity_include_decoder_norm=True,
        use_decoder_bias=True,
        use_triton_kernel=False,
        norm_activation="inference",
        device=device,
        dtype=dtype,
    )
    model = Crosscoder(cfg, device_mesh=device_mesh)
    encoder_bound = 1.0 / math.sqrt(d_model)
    decoder_bound = 1.0 / math.sqrt(cfg.d_sae)
    model.init_parameters(
        encoder_uniform_bound=encoder_bound,
        decoder_uniform_bound=decoder_bound,
        init_log_jumprelu_threshold_value=math.log(init_threshold),
    )
    if init_encoder_with_decoder_transpose_factor > 0:
        # NOTE: untested under DTensor (device_mesh != None). If this fails
        # at production launch, set factor=0 and accept ~0.4 EV cost rather
        # than blocking the run. See train_distributed.py risk note.
        model.init_encoder_with_decoder_transpose(factor=init_encoder_with_decoder_transpose_factor)
    return model


def batch_iter(snapshots, hook_points, batch_size, shuffle=True, generator=None, device=None):
    # snapshots: (K, V, d) on CPU; yields dicts keyed by hook_points, optionally moved to device.
    K, V, _ = snapshots.shape
    assert len(hook_points) == K
    indices = torch.randperm(V, generator=generator) if shuffle else torch.arange(V)
    for start in range(0, V, batch_size):
        idx = indices[start : start + batch_size]
        batch = {hp: snapshots[i, idx, :] for i, hp in enumerate(hook_points)}
        if device is not None:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        yield batch


def _gather_state_dict_to_cpu(model):
    """Gather DTensor shards of `model.state_dict()` to full CPU tensors.

    Collective: every rank must call this. Returns a dict of plain CPU tensors
    suitable for torch.save. Streams each tensor through CPU one at a time to
    avoid piling up GPU memory during the gather (holding multiple full tensors
    on GPU can OOM at K=32, d_sae=32k).
    """
    import gc

    from torch.distributed.tensor import DTensor as _DTensor

    out = {}
    for k, v in model.state_dict().items():
        if isinstance(v, _DTensor):
            full = v.full_tensor().detach().cpu()
            out[k] = full
            del full
            torch.cuda.empty_cache()
        else:
            out[k] = v.detach().cpu() if hasattr(v, "detach") else v
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _load_full_state_into_distributed_model(model, full_state_dict):
    """Load a full-tensor state dict into a possibly-DTensor model.

    For each parameter that's a DTensor in the model, redistribute the loaded
    full tensor to match the DTensor's device_mesh + placements. For plain
    tensors, copy directly.
    """
    from torch.distributed.tensor import DTensor as _DTensor
    from torch.distributed.tensor import distribute_tensor

    sd = model.state_dict()
    new_sd = {}
    for k, full in full_state_dict.items():
        if k not in sd:
            continue
        target = sd[k]
        if isinstance(target, _DTensor):
            full = full.to(target.device, non_blocking=True)
            new_sd[k] = distribute_tensor(full, target.device_mesh, list(target.placements))
        else:
            tgt_dev = target.device if hasattr(target, "device") else "cuda"
            new_sd[k] = full.to(tgt_dev, non_blocking=True)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if unexpected:
        print(f"[resume] WARN: unexpected keys in checkpoint: {unexpected}", flush=True)
    if missing:
        print(f"[resume] WARN: missing keys (using init values): {missing}", flush=True)


def _optimizer_state_to_cpu(optimizer):
    """CPU copy of ``optimizer.state_dict()`` with DTensor states gathered full.

    Collective under device_mesh: every rank must call this (same contract as
    ``_gather_state_dict_to_cpu``).
    """
    from torch.distributed.tensor import DTensor as _DTensor

    sd = optimizer.state_dict()
    state_cpu = {}
    for idx, st in sd["state"].items():
        state_cpu[idx] = {
            k: (
                v.full_tensor().detach().cpu()
                if isinstance(v, _DTensor)
                else (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            )
            for k, v in st.items()
        }
    return {"state": state_cpu, "param_groups": sd["param_groups"]}


def _load_optimizer_state_dict(optimizer, opt_sd):
    """Restore optimizer state saved by ``_optimizer_state_to_cpu``.

    ``Optimizer.load_state_dict`` moves state tensors to each param's device but
    never re-wraps them as DTensors, which would later mix plain moments with
    DTensor grads in step(); redistribute to each param's placements first.
    """
    from torch.distributed.tensor import DTensor as _DTensor
    from torch.distributed.tensor import distribute_tensor

    # Saved state keys index the concatenation of param_groups' params, the
    # same order torch.optim.Optimizer.state_dict uses.
    live_params = [p for g in optimizer.param_groups for p in g["params"]]
    for idx, st in opt_sd.get("state", {}).items():
        p = live_params[idx]
        if not isinstance(p, _DTensor):
            continue
        for k, v in st.items():
            if isinstance(v, torch.Tensor) and v.shape == p.shape:
                st[k] = distribute_tensor(v.to(p.dtype), p.device_mesh, list(p.placements))
    optimizer.load_state_dict(opt_sd)


def train(
    snapshots,
    expansion_factor=8.0,
    lr=5e-5,
    l1_coefficient=0.3,
    frequency_scale=0.01,
    tanh_stretch_coefficient=4.0,
    n_epochs=20,
    batch_size=2048,
    device=None,  # None = auto-select via get_device() (cuda > mps > cpu)
    seed=0,
    log_every=50,
    jumprelu_lr_factor=0.1,
    init_threshold=0.1,
    use_sparse_adam=True,
    amp_dtype=None,  # None = fp32; pass torch.bfloat16 for amp
    noise_std=0.0,
    init_encoder_with_decoder_transpose_factor=1.0,
    l1_warmup_fraction=0.0,  # fraction of total steps for linear L1 coefficient warmup
    lr_warmup_fraction=0.0,  # Ge §A.4 specifies 0.10; 0 = constant LR (Run 3 default)
    lr_decay_fraction=0.0,  # Ge §A.4 specifies 0.20; 0 = no decay (Run 3 default)
    auxk_coefficient=0.0,  # Gao et al. 2024 auxk loss weight (0 = disabled, 1/32 recommended)
    auxk_k=512,  # number of dead latents to use for residual reconstruction
    dead_window_steps=100,  # a latent is "dead" if it hasn't fired in this many steps
    device_mesh=None,  # pass a torch DeviceMesh for head-parallel training
    ckpt_every_epochs=0,  # 0 = disabled. Otherwise: save rolling ckpt every N epochs.
    ckpt_path=None,  # rolling-checkpoint path (Path or str). Atomic rename via .tmp.
    resume_from=None,  # explicit checkpoint path to resume from (overrides auto-detect)
    is_rank_0=True,  # only this rank writes checkpoints (collective is on all ranks)
):
    if device is None:
        device = get_device()
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    K, V, d = snapshots.shape
    snapshots = snapshots.cpu()

    crosscoder = build_crosscoder(
        K,
        d,
        expansion_factor,
        device=device,
        device_mesh=device_mesh,
        init_threshold=init_threshold,
        init_encoder_with_decoder_transpose_factor=init_encoder_with_decoder_transpose_factor,
    )
    hook_points = crosscoder.cfg.hook_points

    threshold_params, other_params = [], []
    for name, p in crosscoder.named_parameters():
        if "jumprelu" in name.lower() or "threshold" in name.lower():
            threshold_params.append(p)
        else:
            other_params.append(p)

    if use_sparse_adam:
        from llamascopium.optim import SparseAdam as _Opt
    else:
        _Opt = torch.optim.Adam
    _opt_kwargs = {} if use_sparse_adam else {"foreach": False}
    optimizer = _Opt(
        [
            {"params": other_params, "lr": lr},
            {"params": threshold_params, "lr": lr * jumprelu_lr_factor},
        ],
        **_opt_kwargs,
    )
    # Save initial LR per param-group; the scheduler multiplies these each step.
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    d_sae = crosscoder.cfg.d_sae
    steps_per_epoch = math.ceil(V / batch_size)
    total_steps = n_epochs * steps_per_epoch
    warmup_steps = int(l1_warmup_fraction * total_steps) if l1_warmup_fraction > 0 else 0
    # Last step at which each latent fired; -inf means never. Used for auxk dead detection.
    last_fired_step = torch.full((d_sae,), -float("inf"), device=device)

    step_count = 0
    start_epoch = 0

    # Resume logic. If `resume_from` is given, use it. Otherwise auto-detect:
    # if ckpt_every_epochs > 0 and the rolling ckpt exists, resume from it.
    _resume_path = None
    if resume_from is not None:
        _resume_path = Path(resume_from)
    elif ckpt_every_epochs > 0 and ckpt_path is not None and Path(ckpt_path).exists():
        _resume_path = Path(ckpt_path)

    if _resume_path is not None and _resume_path.exists():
        print(f"[resume] Loading checkpoint from {_resume_path}", flush=True)
        _ckpt = torch.load(_resume_path, map_location="cpu", weights_only=False)
        _load_full_state_into_distributed_model(crosscoder, _ckpt["state_dict"])
        step_count = int(_ckpt.get("step_count", 0))
        start_epoch = int(_ckpt.get("epoch", -1)) + 1  # resume on the NEXT epoch
        if _ckpt.get("optimizer") is not None:
            _load_optimizer_state_dict(optimizer, _ckpt["optimizer"])
        else:
            print(
                "[resume] WARNING: checkpoint has no optimizer state (old format); "
                "Adam moments restart from zero — resume is NOT exact.",
                flush=True,
            )
        if "gen_state" in _ckpt:
            try:
                gen.set_state(_ckpt["gen_state"])
            except Exception as exc:
                print(
                    f"[resume] WARN: gen.set_state failed ({exc}); seeding fresh",
                    flush=True,
                )
        if "last_fired_step" in _ckpt and auxk_coefficient > 0:
            last_fired_step = _ckpt["last_fired_step"].to(device)
        del _ckpt
        import gc as _gc

        _gc.collect()
        torch.cuda.empty_cache()
        print(
            f"[resume] step_count={step_count} start_epoch={start_epoch}/{n_epochs}",
            flush=True,
        )

    for epoch in range(start_epoch, n_epochs):
        for batch in batch_iter(
            snapshots,
            hook_points,
            batch_size,
            shuffle=True,
            generator=gen,
            device=device,
        ):
            if noise_std > 0:
                batch = {k: v + torch.randn_like(v) * noise_std for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                x, enc_kw, dec_kw = crosscoder.prepare_input(batch)
                feature_acts, hidden_pre = crosscoder.encode(x, return_hidden_pre=True, **enc_kw)
                recon = crosscoder.decode(feature_acts, **dec_kw)

            l_rec = (recon.float() - x.float()).pow(2).sum(dim=-1).mean()

            # tanh-quad sparsity (Ge et al. Eq. 9): mirrors sparse_dictionary.py:872-894,
            # but we aggregate to a scalar here instead of going through the broken
            # apply_token_mask path in compute_loss.
            decoder_norms = crosscoder.decoder_norm().float()
            score = torch.tanh(tanh_stretch_coefficient * feature_acts.float() * decoder_norms)
            approx_frequency = einops.reduce(score, "... d_sae -> d_sae", "mean")
            l_s = (approx_frequency * (1 + approx_frequency / frequency_scale)).sum()

            # In DTensor head-parallel mode, l_rec lands as a Replicate scalar
            # (already globally reduced via .mean()) while l_s is Partial(SUM)
            # (sum over a head-sharded axis). PyTorch DTensor doesn't support
            # auto-redistribute Replicate -> Partial(SUM), so adding them in
            # `loss = l_rec + ... * l_s` errors with
            # "AssertionError: only support replicate to PartialSUM for now!".
            # Materialize both to full tensors (collective; runs on all ranks)
            # so the addition is on plain local scalars.
            if hasattr(l_rec, "full_tensor"):
                l_rec = l_rec.full_tensor()
            if hasattr(l_s, "full_tensor"):
                l_s = l_s.full_tensor()

            # Linear L1 warmup (Gemma Scope default: ~5% of total steps).
            if warmup_steps > 0:
                l1_scale = min(1.0, step_count / warmup_steps)
            else:
                l1_scale = 1.0
            loss = l_rec + (l1_coefficient * l1_scale) * l_s

            # Auxk: reconstruct residual using top-k dead latent pre-activations
            # (Gao et al. 2024, "Scaling and evaluating SAEs", arxiv 2406.04093).
            l_aux = torch.tensor(0.0, device=device)
            if auxk_coefficient > 0 and step_count > dead_window_steps:
                dead_mask = (step_count - last_fired_step) > dead_window_steps  # (d_sae,)
                n_dead = int(dead_mask.sum().item())
                if n_dead > 0:
                    k_aux = min(auxk_k, n_dead)
                    # Rank dead latents by |hidden_pre|; keep original sign, clamp to non-negative
                    # for the JumpReLU-style activation passed to decode.
                    pre = hidden_pre.float()
                    # Head-parallel DTensor mode: dead_mask is a plain tensor, so
                    # `pre_flat * dead_mask` would hit the mixed Tensor/DTensor
                    # failure the firing tracker below documents. Materialize
                    # hidden_pre first (collective; runs on all ranks) —
                    # full_tensor is differentiable so l_aux still trains the
                    # encoder rows of dead latents.
                    _pre_is_dtensor = hasattr(pre, "full_tensor")
                    if _pre_is_dtensor:
                        pre = pre.full_tensor()
                    pre_flat = pre.reshape(-1, d_sae)
                    rank = pre_flat.abs() * dead_mask.float()
                    _, topk_idx = rank.topk(k_aux, dim=-1)
                    orig = pre_flat.gather(-1, topk_idx).clamp_min(0)
                    aux_acts_flat = torch.zeros_like(pre_flat)
                    aux_acts_flat.scatter_(-1, topk_idx, orig)
                    aux_acts = aux_acts_flat.reshape(feature_acts.shape).to(feature_acts.dtype)
                    if _pre_is_dtensor:
                        # decode() under device_mesh requires a DTensor input.
                        # aux_acts is identical on every rank (built from the
                        # materialized pre + replicated dead_mask), so wrap as
                        # Replicate then slice down to feature_acts' head-sharded
                        # placements (both steps communication-free).
                        from torch.distributed.tensor import DTensor as _DTensor
                        from torch.distributed.tensor import Replicate as _Replicate

                        _mesh = feature_acts.device_mesh
                        aux_acts = _DTensor.from_local(aux_acts, _mesh, [_Replicate()] * _mesh.ndim).redistribute(
                            _mesh, list(feature_acts.placements)
                        )
                    aux_recon = crosscoder.decode(aux_acts, **dec_kw)
                    residual = (x - recon).detach().float()
                    l_aux = (aux_recon.float() - residual).pow(2).sum(dim=-1).mean()
                    # Same Replicate/Partial mismatch as l_rec/l_s above.
                    if hasattr(l_aux, "full_tensor"):
                        l_aux = l_aux.full_tensor()
                    loss = loss + auxk_coefficient * l_aux

            # Apply LR warmup + decay schedule (Ge §A.4) before stepping.
            if lr_warmup_fraction > 0 or lr_decay_fraction > 0:
                lr_mult = _lr_multiplier(step_count, total_steps, lr_warmup_fraction, lr_decay_fraction)
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * lr_mult

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update firing tracker (after step, for next iteration's dead-mask).
            # Skip entirely when auxk is disabled — we don't read last_fired_step
            # in that case, and computing `fired` from a DTensor `feature_acts`
            # produces a DTensor that mixes with `last_fired_step` (plain tensor)
            # in `torch.where`, which PyTorch DTensor doesn't support
            # ("aten.where.self: got mixed torch.Tensor and DTensor").
            if auxk_coefficient > 0:
                with torch.no_grad():
                    # In DTensor head-parallel mode, materialize BEFORE the
                    # reshape: flattening (B, K) crosses the head-sharded dim,
                    # which DTensor reshape rejects.
                    fa = feature_acts
                    if hasattr(fa, "full_tensor"):
                        fa = fa.full_tensor()
                    fired = (fa.reshape(-1, d_sae) > 0).any(dim=0)
                    last_fired_step = torch.where(
                        fired,
                        torch.full_like(last_fired_step, float(step_count)),
                        last_fired_step,
                    )

            step_count += 1
            if step_count % log_every == 0:
                # `last_fired_step` is only maintained when auxk is on (above),
                # so with auxk off it stays frozen and the step-window dead count
                # is meaningless (it saturates to d_sae). In that case estimate
                # dead features from the current batch instead — display only; the
                # authoritative dead_rate is the full-data quick_quality value
                # printed once at the end of training.
                if auxk_coefficient > 0:
                    n_dead_log = (
                        int(((step_count - last_fired_step) > dead_window_steps).sum().item())
                        if step_count > dead_window_steps
                        else 0
                    )
                else:
                    with torch.no_grad():
                        # Materialize before reshape (see firing tracker above).
                        fa = feature_acts
                        if hasattr(fa, "full_tensor"):
                            fa = fa.full_tensor()
                        n_dead_log = int((~(fa.reshape(-1, d_sae) > 0).any(dim=0)).sum().item())
                print(
                    f"[epoch {epoch} step {step_count}] "
                    f"loss {loss.item():.4f} l_rec {l_rec.item():.4f} l_s {l_s.item():.4f} "
                    f"l_aux {l_aux.detach().item():.4f} l1x {l1_scale:.2f} dead {n_dead_log}/{d_sae}"
                )

        # End of epoch: rolling checkpoint. Collective (all ranks gather);
        # only rank 0 writes. Atomic via .tmp + rename so a crash mid-write
        # does not corrupt the previous good checkpoint.
        if ckpt_every_epochs > 0 and ckpt_path is not None and (epoch + 1) % ckpt_every_epochs == 0:
            sd_cpu = _gather_state_dict_to_cpu(crosscoder)
            opt_cpu = _optimizer_state_to_cpu(optimizer)  # collective: all ranks
            if is_rank_0:
                _path = Path(ckpt_path)
                _path.parent.mkdir(parents=True, exist_ok=True)
                _tmp = _path.with_suffix(_path.suffix + ".tmp")
                _payload = {
                    "state_dict": sd_cpu,
                    "optimizer": opt_cpu,
                    "step_count": step_count,
                    "epoch": epoch,
                    "gen_state": gen.get_state(),
                    "last_fired_step": (last_fired_step.detach().cpu() if auxk_coefficient > 0 else None),
                }
                torch.save(_payload, _tmp)
                _tmp.replace(_path)
                print(
                    f"[ckpt] epoch={epoch} step={step_count} -> {_path}",
                    flush=True,
                )
            del sd_cpu, opt_cpu
            import gc as _gc

            _gc.collect()
            torch.cuda.empty_cache()
    return crosscoder


def quick_quality(crosscoder, snapshots, batch_size=2048, device=None, dead_eps=1e-6):
    if device is None:
        device = get_device()
    crosscoder.eval()
    hook_points = crosscoder.cfg.hook_points
    snapshots = snapshots.cpu()
    K, V, _ = snapshots.shape
    d_sae = crosscoder.cfg.d_sae
    # Exact (batch-size invariant) accumulation: EV = 1 - SSE/SStot with SSE
    # summed over every element and SStot around the global mean of the full
    # eval tensor; L0 and dead_rate are row-weighted over all V rows.
    sse = 0.0
    l0_active = 0.0
    feature_fires = torch.zeros(d_sae, dtype=torch.float64)
    n_seen = 0
    with torch.no_grad():
        for batch in batch_iter(snapshots, hook_points, batch_size, shuffle=False, device=device):
            x, enc_kwargs, dec_kwargs = crosscoder.prepare_input(batch)
            feature_acts, _ = crosscoder.encode(x, return_hidden_pre=True, **enc_kwargs)
            recon = crosscoder.decode(feature_acts, **dec_kwargs)
            # In the head-parallel DTensor path, x and recon are head-sharded;
            # materialize both (collective; runs on all ranks) before the
            # residual math, same pattern as feature_acts below.
            if hasattr(x, "full_tensor"):
                x = x.full_tensor()
            if hasattr(recon, "full_tensor"):
                recon = recon.full_tensor()
            sse += (recon.float() - x.float()).pow(2).sum().item()
            # feature_acts is (B, d_sae) or (B, K, d_sae); collapse all but d_sae.
            # In the head-parallel DTensor path, feature_acts is sharded on the
            # head axis; .reshape(-1, d_sae) on a DTensor returns a local view
            # of only this rank's heads, which would understate per-feature
            # firing by a factor of world_size. Materialize the full tensor first.
            fa_raw = feature_acts.detach()
            if hasattr(fa_raw, "full_tensor"):
                fa_raw = fa_raw.full_tensor()
            fa = fa_raw.reshape(-1, d_sae).cpu().to(torch.float64)
            fired = fa > 0
            feature_fires += fired.sum(dim=0).to(torch.float64)
            l0_active += float(fired.sum().item())
            n_seen += fa.shape[0]
    # SStot around the global elementwise mean of the full eval tensor (already
    # in memory); fp64 accumulation, sliced over K to bound temp memory at one
    # (V, d) fp32 tensor.
    x_bar = snapshots.mean(dtype=torch.float64).item()
    sstot = 0.0
    for k_i in range(K):
        sstot += (snapshots[k_i] - x_bar).pow(2).sum(dtype=torch.float64).item()
    n_elem = float(snapshots.numel())
    ev = 1.0 - (sse / sstot) if sstot > 0 else 0.0
    if n_seen > 0:
        per_feature_rate = (feature_fires / n_seen).numpy()
        dead_rate = float((per_feature_rate <= dead_eps).mean())
        mean_l0 = l0_active / n_seen
    else:
        dead_rate = float("nan")
        mean_l0 = float("nan")
    return {
        "explained_variance": ev,
        "reconstruction_mse": sse / n_elem,
        "mean_l0": mean_l0,
        "dead_rate": dead_rate,
        "d_sae": d_sae,
        "n_snapshots": K,
        "n_rows": V,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--expansion-factor", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--l1-coefficient", type=float, default=0.3)
    parser.add_argument("--frequency-scale", type=float, default=0.01)
    parser.add_argument("--tanh-stretch", type=float, default=4.0)
    parser.add_argument("--jumprelu-lr-factor", type=float, default=0.1)
    parser.add_argument("--init-threshold", type=float, default=0.1)
    parser.add_argument("--amp-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sparse_adam"],
        default=None,
        help="Explicit optimizer selection. Overrides --use-sparse-adam when set.",
    )
    parser.add_argument("--use-sparse-adam", action="store_true", default=False)
    parser.add_argument(
        "--decoder-transpose-init",
        type=float,
        default=1.0,
        help="Factor for init_encoder_with_decoder_transpose; 0 disables",
    )
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run for sanity-checking (2 snapshots, 1 epoch)",
    )
    parser.add_argument(
        "--input-preprocess",
        choices=["none", "center", "center_scale"],
        default="none",
        help="Per-snapshot input normalization. 'center_scale' changes input magnitude; retune l1.",
    )
    parser.add_argument(
        "--l1-warmup-fraction",
        type=float,
        default=0.0,
        help="Linear L1 coefficient warmup over this fraction of total steps. Gemma Scope uses 0.05.",
    )
    parser.add_argument(
        "--lr-warmup-fraction",
        type=float,
        default=0.0,
        help="Linear LR warmup fraction. Ge §A.4 specifies 0.10. Default 0 = constant LR (Run 3).",
    )
    parser.add_argument(
        "--lr-decay-fraction",
        type=float,
        default=0.0,
        help="Linear LR decay fraction at end of training. Ge §A.4 specifies 0.20. Default 0 = no decay.",
    )
    parser.add_argument(
        "--auxk-coefficient",
        type=float,
        default=0.0,
        help="Gao et al. 2024 auxk loss weight for dead-feature revival. 0 disables; 1/32 = 0.03125 recommended.",
    )
    parser.add_argument(
        "--auxk-k",
        type=int,
        default=512,
        help="Number of dead latents used for auxk reconstruction",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--dead-window-steps",
        type=int,
        default=100,
        help="Latents idle for this many steps count as dead",
    )
    parser.add_argument(
        "--append-manifest",
        action="store_true",
        default=False,
        help="After post-eval, append/update one row in results/crosscoder_sweep_manifest.csv.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("results/crosscoder_sweep_manifest.csv"),
        help="Manifest CSV path (only used when --append-manifest is set).",
    )
    parser.add_argument(
        "--manifest-notes",
        type=str,
        default="",
        help="Free-text notes column for the manifest row.",
    )
    args = parser.parse_args()

    # Seed all RNG (Python/NumPy/torch + PYTHONHASHSEED) before any data or
    # model construction so the run is reproducible from --seed alone.
    seed_everything(args.seed)

    if args.optimizer is not None:
        args.use_sparse_adam = args.optimizer == "sparse_adam"
    optimizer_name = "sparse_adam" if args.use_sparse_adam else "adam"

    if args.smoke:
        steps = [args.steps[0] if args.steps else 0, _terminal_step_for(args.model)]
        n_epochs = 1
        print(f"Smoke test: steps={steps}, 1 epoch")
    else:
        steps = args.steps if args.steps else None
        n_epochs = args.n_epochs

    snapshots, resolved_steps = load_snapshots(
        model_name=args.model,
        steps=steps,
        cache_dir=args.cache_dir,
    )
    print(f"Snapshots: {tuple(snapshots.shape)} (K, V, d)")

    snapshots, preprocess_stats = preprocess_snapshots(snapshots, mode=args.input_preprocess)
    if preprocess_stats is not None:
        mean_norms = preprocess_stats["mean"].norm(dim=-1).squeeze().tolist()
        scales = preprocess_stats["scale"].squeeze().tolist()
        if not isinstance(mean_norms, list):
            mean_norms = [mean_norms]
        if not isinstance(scales, list):
            scales = [scales]
        print(f"Preprocess '{args.input_preprocess}': per-snap mean-shift {mean_norms[:3]}... scale {scales[:3]}...")

    amp_dtype_map = {"fp32": None, "bf16": torch.bfloat16}
    crosscoder = train(
        snapshots,
        expansion_factor=args.expansion_factor,
        lr=args.lr,
        l1_coefficient=args.l1_coefficient,
        frequency_scale=args.frequency_scale,
        tanh_stretch_coefficient=args.tanh_stretch,
        jumprelu_lr_factor=args.jumprelu_lr_factor,
        n_epochs=n_epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        init_threshold=args.init_threshold,
        amp_dtype=amp_dtype_map[args.amp_dtype],
        noise_std=args.noise_std,
        use_sparse_adam=args.use_sparse_adam,
        init_encoder_with_decoder_transpose_factor=args.decoder_transpose_init,
        l1_warmup_fraction=args.l1_warmup_fraction,
        lr_warmup_fraction=args.lr_warmup_fraction,
        lr_decay_fraction=args.lr_decay_fraction,
        auxk_coefficient=args.auxk_coefficient,
        auxk_k=args.auxk_k,
        dead_window_steps=args.dead_window_steps,
        log_every=args.log_every,
    )

    metrics = quick_quality(crosscoder, snapshots, batch_size=args.batch_size, device=args.device)
    print(f"Quality: {metrics}")

    # weights_only=True loaders reject str subclasses (e.g. TorchVersion from
    # torch.__version__); normalize provenance values to plain builtins.
    provenance = {k: str(v) if isinstance(v, str) else v for k, v in log_run_provenance(seed=args.seed).items()}

    # Atomic write: hours of training must not be lost to a kill mid-save.
    atomic_write_torch(
        args.output,
        {
            "state_dict": crosscoder.state_dict(),
            "config": crosscoder.cfg.model_dump(),
            "steps": resolved_steps,
            "model_name": args.model,
            "seed": args.seed,
            "quality": metrics,
            "training": {
                "lr": args.lr,
                "l1_coefficient": args.l1_coefficient,
                "frequency_scale": args.frequency_scale,
                "tanh_stretch_coefficient": args.tanh_stretch,
                "jumprelu_lr_factor": args.jumprelu_lr_factor,
                "init_threshold": args.init_threshold,
                "decoder_transpose_init": args.decoder_transpose_init,
                "amp_dtype": args.amp_dtype,
                "optimizer": optimizer_name,
                "n_epochs": n_epochs,
                "batch_size": args.batch_size,
                "expansion_factor": args.expansion_factor,
                "d_sae": crosscoder.cfg.d_sae,
                "input_preprocess": args.input_preprocess,
                "l1_warmup_fraction": args.l1_warmup_fraction,
                "lr_warmup_fraction": args.lr_warmup_fraction,
                "lr_decay_fraction": args.lr_decay_fraction,
                "auxk_coefficient": args.auxk_coefficient,
                "auxk_k": args.auxk_k,
                "dead_window_steps": args.dead_window_steps,
            },
            "preprocess_stats": preprocess_stats,
            "provenance": provenance,
            "git_commit": git_commit(),
        },
    )
    print(f"Saved to {args.output}")

    if args.append_manifest:
        from readout.crosscoder.manifest import append_manifest_row

        try:
            row = append_manifest_row(
                args.output,
                manifest_csv=args.manifest_path,
                notes=args.manifest_notes,
                strict=True,
            )
            print(f"Manifest row written to {args.manifest_path}: kind={row['kind']} d_sae={row['d_sae']}")
        except FileNotFoundError as e:
            print(f"WARN: --append-manifest skipped ({e}). Mount the SSD or pass --manifest-path.")
