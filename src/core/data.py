"""
Data extraction and preprocessing for proto-token dictionary learning.
=====================================================================
Single source of truth for W_U extraction, centering, SVD removal,
and SVD baseline computation.
"""

from __future__ import annotations

import csv
import hashlib
import struct
from collections.abc import Generator
from pathlib import Path

import torch


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` to ``path`` as CSV (header from the first row's keys).

    No-op when ``rows`` is empty. Parent directories are created if missing.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def extract_wu_from_model(model) -> torch.Tensor:
    """Extract W_U from an already-loaded HuggingFace model (no reload)."""
    if hasattr(model, "embed_out"):
        W_U = model.embed_out.weight.detach().clone()
    elif hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().clone()
    else:
        raise ValueError(f"Cannot locate unembedding matrix in {type(model).__name__}")
    return W_U.float()


def extract_wu(model_name: str, device: str = "cpu", **load_kwargs):
    """Extract the unembedding matrix W_U and return (W_U, tokenizer).

    Supports GPT-2 (lm_head), Pythia/GPT-NeoX (embed_out), and
    Llama-style models including Qwen and Gemma (lm_head).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )

    W_U = extract_wu_from_model(model)
    print(f"Extracted W_U: {W_U.shape} from {model_name}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return W_U.to(device), tokenizer


def extract_we(model_name: str, device: str = "cpu", **load_kwargs):
    """Extract the input embedding matrix W_E and return (W_E, tokenizer).

    Mirrors ``extract_wu``'s model coverage: GPT-2 (transformer.wte),
    Pythia/GPT-NeoX (gpt_neox.embed_in), and Llama-style models including
    Qwen, Gemma, and OLMo (model.embed_tokens).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )

    if hasattr(model, "gpt_neox"):
        W_E = model.gpt_neox.embed_in.weight.detach().clone()
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        W_E = model.transformer.wte.weight.detach().clone()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        # Llama/Qwen/Gemma/OLMo-style decoder stacks.
        W_E = model.model.embed_tokens.weight.detach().clone()
    else:
        raise ValueError(f"Cannot locate embedding matrix in {model_name}")

    print(f"Extracted W_E: {W_E.shape} from {model_name}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return W_E.float().to(device), tokenizer


def iter_checkpoints(
    model_name: str,
    steps: list[int],
    device: str = "cpu",
    extract: str = "wu",
) -> Generator[tuple[int, torch.Tensor], None, None]:
    extract_fn = {"wu": extract_wu, "we": extract_we}[extract]
    for step in steps:
        with torch.no_grad():
            W, _tok = extract_fn(model_name, device=device, revision=f"step{step}")
        yield step, W


def center_and_project(
    W_U: torch.Tensor,
    svd_remove: int = 0,
    device=None,
    verbose: bool = True,
    normalize_rows: bool = False,
):
    if device is None:
        device = W_U.device
    data = W_U.to(device)
    data_mean = data.mean(dim=0)
    data = data - data_mean

    svd_components = None
    if svd_remove > 0:
        U, S, Vt = torch.linalg.svd(data, full_matrices=False)
        svd_components = Vt[:svd_remove].clone()
        proj = data @ svd_components.T
        data = data - proj @ svd_components
        if verbose:
            removed_var = S[:svd_remove].pow(2).sum() / S.pow(2).sum()
            print(
                f"  Removed top {svd_remove} SVD components "
                f"(variance: {removed_var:.1%}, "
                f"singular values: {', '.join(f'{s:.2f}' for s in S[:svd_remove])})"
            )

    centering: dict = {
        "mean": data_mean,
        "svd_components": svd_components,
        "svd_S": S.clone() if svd_remove > 0 else None,
    }

    if normalize_rows:
        row_norms = data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        centering["row_norms"] = row_norms.squeeze(1)
        data = data / row_norms

    return data, centering


def adaptive_center_and_project(
    W: torch.Tensor, svd_remove: int = 2, device: str = "cpu"
) -> tuple[torch.Tensor, dict]:
    return center_and_project(W, svd_remove=svd_remove, device=device)


def apply_centering(W_U: torch.Tensor, centering: dict) -> torch.Tensor:
    """Apply stored centering transform to raw W_U (for inference)."""
    data = W_U - centering["mean"].to(W_U.device)
    if centering.get("svd_components") is not None:
        comps = centering["svd_components"].to(W_U.device)
        data = data - (data @ comps.T) @ comps
    if centering.get("row_norms") is not None:
        row_norms = centering["row_norms"].to(data.device).unsqueeze(1).clamp(min=1e-8)
        data = data / row_norms
    return data


def center_scale_stats(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-snapshot center_scale stats (mean over rows, scale = sqrt(E[||x-μ||²]/d)).

    Matches wu_adapter.preprocess_snapshots(..., mode="center_scale") for one (V, d).
    Returns (mean (1, d), scale scalar tensor).
    """
    mean = W.mean(dim=0, keepdim=True)
    centered = W - mean
    d = W.shape[-1]
    mean_sq = centered.pow(2).sum(dim=-1).mean()
    scale = (mean_sq / d).clamp_min(1e-8).sqrt()
    return mean, scale


def explained_variance(W_true: torch.Tensor, W_hat: torch.Tensor) -> float:
    var_true = W_true.var().item()
    var_res = (W_true - W_hat).var().item()
    return 1.0 - (var_res / max(var_true, 1e-12))


_svd_cache: dict = {}


def _svd_fingerprint(data: torch.Tensor) -> tuple:
    """Content-based cache key for SVD.

    Keyed on shape + dtype + a hash of the full tensor bytes, so distinct
    matrices never collide (the previous 3x3-corner key let different matrices
    sharing a corner alias to a stale cached SVD).
    """
    contig = data.detach().cpu().contiguous()
    digest = hashlib.sha1(contig.numpy().tobytes()).hexdigest()
    return (
        tuple(contig.shape),
        str(contig.dtype),
        digest,
    )


def svd_baseline(data: torch.Tensor, rank: int) -> float:
    """Reconstruction MSE of rank-k SVD (for comparison with SAE).

    Caches the SVD decomposition so repeated calls with different ranks
    don't recompute the full SVD each time.
    """
    fp = _svd_fingerprint(data)
    if _svd_cache.get("fingerprint") != fp:
        data_cpu = data.detach().cpu()
        U, S, Vt = torch.linalg.svd(data_cpu, full_matrices=False)
        _svd_cache.clear()
        _svd_cache.update(
            {"fingerprint": fp, "U": U, "S": S, "Vt": Vt, "data_cpu": data_cpu}
        )
    U, S, Vt, data_cpu = (
        _svd_cache["U"],
        _svd_cache["S"],
        _svd_cache["Vt"],
        _svd_cache["data_cpu"],
    )
    recon = (U[:, :rank] * S[:rank]) @ Vt[:rank]
    return (data_cpu - recon).pow(2).sum(dim=1).mean().item()


def svd_baseline_from_centering(
    centering: dict, rank: int, n_rows: int
) -> float | None:
    """Estimate rank-k SVD baseline MSE from stored singular values.

    WARNING: This is an approximation. The stored singular values are from
    the pre-removal SVD. After projecting out the top components, the true
    singular values of the residual are NOT simply S[svd_remove:].
    For exact baselines, use svd_baseline() on the actual data tensor.

    Returns None if centering lacks stored SVD info (e.g. svd_remove=0).
    """
    S = centering.get("svd_S")
    if S is None:
        return None
    svd_remove = (
        centering["svd_components"].shape[0]
        if centering.get("svd_components") is not None
        else 0
    )
    offset = svd_remove + rank
    if offset >= S.shape[0]:
        return 0.0
    return S[offset:].pow(2).sum().item() / n_rows


def get_device() -> str:
    """Return best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Checkpoint cache ─────────────────────────────────────────────


def _cache_path(cache_dir: Path, matrix: str, step: int) -> Path:
    return cache_dir / matrix / f"step{step}.pt"


def _cleanup_hf_revision(model_name: str, step: int) -> None:
    """Remove cached HF model files for this model to free disk space.

    Each Pythia checkpoint has unique weights, so blobs can't be reused
    between revisions. Deleting the entire model cache is safe — the next
    step will re-download.
    """
    import os
    import shutil

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    # HF caches models under hub/models--{org}--{name}/
    repo_dir_name = "models--" + model_name.replace("/", "--")
    model_cache = Path(hf_home) / "hub" / repo_dir_name
    if model_cache.exists():
        shutil.rmtree(model_cache, ignore_errors=True)


def cache_checkpoints(
    model_name: str,
    steps: list[int],
    cache_dir: str | Path,
    matrices: list[str] = ("wu",),
    dtype: torch.dtype = torch.float32,
) -> None:
    """Extract and cache weight matrices to disk, skipping already-cached steps."""
    cache_dir = Path(cache_dir)
    extract_fns = {"wu": extract_wu, "we": extract_we}

    for matrix in matrices:
        (cache_dir / matrix).mkdir(parents=True, exist_ok=True)

    tokenizer = None
    for step in steps:
        needed = [m for m in matrices if not _cache_path(cache_dir, m, step).exists()]
        if not needed:
            continue

        for matrix in needed:
            W, tok = extract_fns[matrix](
                model_name, device="cpu", revision=f"step{step}"
            )
            torch.save(W.to(dtype), _cache_path(cache_dir, matrix, step))
            if tokenizer is None:
                tokenizer = tok
            del W
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"  Cached step {step}: {', '.join(needed)}")

        # Free HF cache for this revision to avoid accumulating full model weights
        _cleanup_hf_revision(model_name, step)

    # Save tokenizer alongside cache
    tok_path = cache_dir / "tokenizer"
    if tokenizer is not None and not tok_path.exists():
        tokenizer.save_pretrained(str(tok_path))


def load_cached(
    cache_dir: str | Path,
    step: int,
    matrix: str = "wu",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load a cached weight matrix, upcast to dtype (default float32)."""
    path = _cache_path(Path(cache_dir), matrix, step)
    return torch.load(path, map_location=device, weights_only=True).to(dtype)


def load_cached_tokenizer(cache_dir: str | Path):
    from transformers import AutoTokenizer

    cache_dir = Path(cache_dir)
    tok_subdir = cache_dir / "tokenizer"
    # Try canonical subdirectory first, then the cache root directly
    load_path = tok_subdir if tok_subdir.exists() else cache_dir
    return AutoTokenizer.from_pretrained(str(load_path))


# ── Precision diagnostics ───────────────────────────────────────


def _classify_precision(value_f32: float) -> str:
    """Classify whether a float32 value sits on the bf16 or fp16 grid.

    bf16 has 7 mantissa bits -> trailing 16 zeros in the 23-bit mantissa.
    fp16 has 10 mantissa bits -> trailing 13 zeros in the 23-bit mantissa.
    """
    # Pack as IEEE 754 float32, interpret as uint32
    bits = struct.unpack(">I", struct.pack(">f", value_f32))[0]
    mantissa = bits & 0x007FFFFF  # lower 23 bits

    # bf16: mantissa bits [0:16] must all be zero
    if (mantissa & 0x0000FFFF) == 0:
        return "bf16"
    # fp16: mantissa bits [0:13] must all be zero
    if (mantissa & 0x00001FFF) == 0:
        return "fp16"
    return "fp32"


def check_checkpoint_precision(
    cache_dir: str | Path,
    steps: list[int],
    matrix: str = "wu",
    expected: str = "bf16",
) -> list[tuple[int, str]]:
    """Check cached checkpoints for dtype precision anomalies.

    Samples 1000 non-zero values per checkpoint and classifies each as
    bf16, fp16, or fp32 based on mantissa bit patterns. Returns a list of
    (step, detected_precision) for steps where the majority precision
    differs from *expected*.
    """
    cache_dir = Path(cache_dir)
    anomalies: list[tuple[int, str]] = []

    for step in steps:
        path = _cache_path(cache_dir, matrix, step)
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping precision check")
            continue

        W = torch.load(path, map_location="cpu", weights_only=True).float()
        flat = W.flatten()

        # Sample up to 1000 finite non-zero values
        valid_mask = flat.isfinite() & (flat != 0.0)
        valid_vals = flat[valid_mask]
        n_sample = min(1000, valid_vals.numel())
        if n_sample == 0:
            print(f"  WARNING: step {step} has no finite non-zero values, skipping")
            continue
        gen = torch.Generator().manual_seed(step)
        indices = torch.randperm(valid_vals.numel(), generator=gen)[:n_sample]
        sample = valid_vals[indices]

        # Classify each sampled value
        counts: dict[str, int] = {"bf16": 0, "fp16": 0, "fp32": 0}
        for val in sample.tolist():
            counts[_classify_precision(val)] += 1

        # Majority vote
        detected = max(counts, key=counts.get)  # type: ignore[arg-type]
        if detected != expected:
            pct = counts[detected] / n_sample * 100
            print(
                f"  WARNING: step {step} detected as {detected} "
                f"({pct:.0f}% of {n_sample} samples), expected {expected}. "
                f"Counts: {counts}"
            )
            anomalies.append((step, detected))

    if not anomalies:
        print(f"  All {len(steps)} checkpoints match expected precision ({expected}).")

    return anomalies
