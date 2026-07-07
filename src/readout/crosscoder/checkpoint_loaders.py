"""Load (rates, decoder_norms, decoder_weights, steps) for a crosscoder run.

Cache priority:
  1. Run-level pre-cached arrays (e.g. firing_rates_all_seeds.npy from Run 3).
  2. Per-seed extracted rates (.pt from src/readout/crosscoder/extract_rates.py).
  3. Re-derive from checkpoint via compute_rates_canonical (slow, needs W_U snaps).

Phase scripts use `load_run(row)` once per (run_id, seed) row from the analysis
table. Reloads are cheap if the cached arrays exist, expensive otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class RunArrays:
    """Per-run arrays in canonical form."""

    rates: np.ndarray  # (K, D) firing rate
    decoder_norms: np.ndarray  # (K, D) ||W_D[k, j, :]||_2
    decoder_weights: np.ndarray | None  # (K, D, d) raw decoder, optional (memory-heavy)
    steps: list[int]  # length K
    run_id: str
    seed: int
    d_sae: int


def _load_npy_or_pt(path: Path) -> np.ndarray | dict | None:
    if not path.exists():
        return None
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(f"unsupported cache file extension: {path}")


def _decoder_norms_from_ckpt(
    ckpt_path: Path,
    *,
    steps_fallback: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Read W_D and steps from a crosscoder checkpoint.

    Returns:
      decoder_norms: (K, D) float32
      decoder_weights: (K, D, d) float32 (kept on CPU; caller decides whether to drop it)
      steps: list[int]

    Some stripped-down checkpoints (e.g. Pythia-1B d_sae>=16384 from the
    capacity sweep) drop the `steps` key. Pass `steps_fallback` from the
    discovery row to recover; lengths must match the W_D leading axis.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ck["state_dict"]
    W_D = sd["W_D"].float()  # (K, D, d)
    if "steps" in ck:
        steps = list(ck["steps"])
    elif steps_fallback is not None and len(steps_fallback) == W_D.shape[0]:
        steps = list(steps_fallback)
    else:
        raise KeyError(
            f"checkpoint {ckpt_path} missing 'steps' and no compatible steps_fallback (W_D K={W_D.shape[0]})"
        )
    norms = torch.linalg.norm(W_D, dim=-1).numpy().astype(np.float32)
    return norms, W_D.numpy().astype(np.float32), steps


def _decoder_norms_from_per_snap_dir(
    ckpt_dir: Path,
    *,
    steps: list[int],
    d_sae: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Stack W_D from a directory of per-snapshot SAE checkpoints (T1.3).

    Each file `wu_sae_dsae{D}_step{S}.pt` is a single-snapshot SAE; we load
    them in `steps` order and stack to (K, D, d_model). Files for steps not
    present are skipped (and the run drops to the available subset). The
    return shape is consistent with `_decoder_norms_from_ckpt` so downstream
    code is identical.
    """
    stacked: list[torch.Tensor] = []
    actual_steps: list[int] = []
    for s in steps:
        candidates = [
            ckpt_dir / f"wu_sae_dsae{d_sae}_step{s}.pt",
            ckpt_dir / f"sae_dsae{d_sae}_step{s}.pt",
        ]
        ck_path = next((c for c in candidates if c.exists()), None)
        if ck_path is None:
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        # Per-snap SAE stores a single snapshot's decoder; promote to (1, D, d).
        wd = sd["W_D"].float()
        if wd.ndim == 2:
            wd = wd.unsqueeze(0)
        stacked.append(wd)
        actual_steps.append(s)
    if not stacked:
        raise FileNotFoundError(
            f"no per-snap SAE checkpoints found under {ckpt_dir} matching wu_sae_dsae{d_sae}_step*.pt"
        )
    W_D = torch.cat(stacked, dim=0)  # (K, D, d)
    norms = torch.linalg.norm(W_D, dim=-1).numpy().astype(np.float32)
    return norms, W_D.numpy().astype(np.float32), actual_steps


def load_run(
    *,
    run_id: str,
    seed: int,
    d_sae: int,
    ckpt_path: Path | str,
    rates_cache: Path | str | None = None,
    norms_cache: Path | str | None = None,
    seed_index_in_cache: int | None = None,
    keep_decoder_weights: bool = False,
    steps_fallback: list[int] | None = None,
    allow_missing_rates: bool = False,
) -> RunArrays:
    """Load arrays for a single (run, seed).

    rates_cache / norms_cache may be:
      - a per-seed .pt of shape (K, D)
      - a multi-seed .npy of shape (S, K, D); seed_index_in_cache selects the row
      - None: derived from ckpt_path

    `steps_fallback` (typically the row's `steps` field) recovers from
    checkpoints that drop the `steps` key (e.g. Pythia-1B d_sae>=16384).

    `allow_missing_rates=True` returns rates=None instead of raising when no
    cache is provided and ckpt-derivation fails (snapshots missing, or
    Gated/BatchTopK/PerSnapSAE architecture not supported by
    compute_rates_canonical). Norms are still loaded so downstream callers can
    compute decoder-only artifacts (rotation, direction-to-terminal, cusum on
    norms).

    keep_decoder_weights=True returns the full (K, D, d) tensor (memory: ~4 GB
    for d_sae=8192, ~12 GB for d_sae=24576, ~50 GB for d_sae=98304). Default
    False; callers that need rotation should compute it themselves and drop W_D.
    """
    ckpt_path = Path(ckpt_path)
    is_dir_ckpt = ckpt_path.is_dir()

    rates = None
    if rates_cache is not None:
        rates_obj = _load_npy_or_pt(Path(rates_cache))
        if isinstance(rates_obj, np.ndarray):
            if rates_obj.ndim == 3 and seed_index_in_cache is not None:
                rates = rates_obj[seed_index_in_cache].astype(np.float32)
            elif rates_obj.ndim == 2:
                rates = rates_obj.astype(np.float32)
            else:
                raise ValueError(
                    f"rates cache {rates_cache} has shape {rates_obj.shape}; need seed_index_in_cache for 3D"
                )
        elif isinstance(rates_obj, dict):
            # extract_rates.py output: {"rates_per_seed": {stem: tensor}, "steps": [...]}.
            # Keys are the ckpt stems at extraction time; require a match so a
            # single-entry cache from another seed is never silently served.
            rps = rates_obj.get("rates_per_seed")
            stem = ckpt_path.stem
            if rps is None or stem not in rps:
                raise KeyError(f"rates cache {rates_cache} missing key {stem}; have {list(rps or [])}")
            rates = rps[stem].numpy().astype(np.float32)

    norms = None
    if norms_cache is not None:
        norms_obj = _load_npy_or_pt(Path(norms_cache))
        if isinstance(norms_obj, np.ndarray):
            if norms_obj.ndim == 3 and seed_index_in_cache is not None:
                norms = norms_obj[seed_index_in_cache].astype(np.float32)
            elif norms_obj.ndim == 2:
                norms = norms_obj.astype(np.float32)
            else:
                raise ValueError(
                    f"norms cache {norms_cache} has shape {norms_obj.shape}; need seed_index_in_cache for 3D"
                )

    decoder_weights = None
    steps: list[int] = []
    need_ckpt = (rates is None) or (norms is None) or keep_decoder_weights
    if need_ckpt:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint {ckpt_path} not found; cached arrays do not cover the request")
        if is_dir_ckpt:
            assert steps_fallback is not None, (
                f"directory ckpt {ckpt_path} requires steps_fallback (the row's steps list)"
            )
            norms_from_ckpt, decoder_weights_from_ckpt, steps = _decoder_norms_from_per_snap_dir(
                ckpt_path, steps=steps_fallback, d_sae=d_sae
            )
        else:
            norms_from_ckpt, decoder_weights_from_ckpt, steps = _decoder_norms_from_ckpt(
                ckpt_path, steps_fallback=steps_fallback
            )
        if norms is None:
            norms = norms_from_ckpt
        if keep_decoder_weights:
            decoder_weights = decoder_weights_from_ckpt
        if rates is None and not is_dir_ckpt:
            try:
                from readout.crosscoder.extract_rates import compute_rates_canonical  # noqa: E402

                cache_dir = _resolve_wu_cache_dir(ckpt_path)
                rates_t, _ = compute_rates_canonical(ckpt_path, cache_dir, device="cpu")
                rates = rates_t.numpy().astype(np.float32)
            except (FileNotFoundError, KeyError):
                if not allow_missing_rates:
                    raise
                rates = None
        elif rates is None and is_dir_ckpt:
            if not allow_missing_rates:
                raise FileNotFoundError(
                    f"per-snap SAE dir {ckpt_path} has no cached rates and "
                    f"compute_rates_canonical does not support directory ckpts"
                )
    else:
        if is_dir_ckpt:
            # Per-snap SAE dirs carry no aggregate metadata file to read steps from.
            if steps_fallback is None:
                raise KeyError(
                    f"per-snap SAE dir {ckpt_path} has no aggregate metadata; "
                    f"pass steps_fallback (the row's steps list)"
                )
            steps = list(steps_fallback)
        else:
            # Steps only; torch.load still deserializes the full payload, so this
            # is cheap only relative to re-deriving rates/norms.
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "steps" in ck:
                steps = list(ck["steps"])
            elif steps_fallback is not None:
                steps = list(steps_fallback)
            else:
                raise KeyError(f"checkpoint {ckpt_path} missing 'steps' and no fallback")

    K = norms.shape[0]
    if rates is not None:
        if rates.shape != norms.shape:
            raise ValueError(f"shape mismatch: rates {rates.shape} vs norms {norms.shape} for {run_id} seed {seed}")
    D = norms.shape[1]
    if D != d_sae:
        raise ValueError(f"d_sae mismatch: cache D={D} vs row d_sae={d_sae}")
    if len(steps) != K:
        raise ValueError(f"steps length {len(steps)} != K={K} from norms shape")

    return RunArrays(
        rates=rates,
        decoder_norms=norms,
        decoder_weights=decoder_weights,
        steps=steps,
        run_id=run_id,
        seed=seed,
        d_sae=d_sae,
    )


def _resolve_wu_cache_dir(ckpt_path: Path) -> Path:
    """Best-effort guess at the W_U snapshot cache for derive-from-checkpoint."""
    import os

    env = os.environ.get("WU_CACHE_DIR")
    if env:
        return Path(env)
    # Default for local SSD layout.
    from readout.core.paths import ssd_path

    candidates = [
        ssd_path("wu_crosscoder", "snapshots"),
        Path(os.environ.get("CLUSTER_SCRATCH", str(Path.home() / "scratch"))) / "wu_crosscoder/snapshots",
        Path.cwd() / "cluster_data/wu_snapshots",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "WU_CACHE_DIR not set and no default snapshot cache found; "
        "set WU_CACHE_DIR or pass rates_cache/norms_cache to skip re-derivation"
    )
