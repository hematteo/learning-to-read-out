"""Build the next three W_E / W_U follow-up analyses.

The three outputs correspond to the highest-priority follow-ups from the
read/write appendix:

1. use the archived d24576 W_E activation-rate sidecar;
2. add BPE-rank / token-length matched controls for family lead-lag;
3. measure dense W_E -> W_U orthogonal Procrustes alignment over training.

Each output is saved under figures/crosscoder_we/ with CSV plus .pt caches
for auditability.
"""

from __future__ import annotations

import argparse
import csv
import json

# Same-experiment sibling module; the insert makes this robust to import
# from outside the scripts/ dir (direct file execution already has it first).
import sys as _sys  # noqa: E402
from pathlib import Path
from pathlib import Path as _P  # noqa: E402

import numpy as np
import torch

_sys.path.insert(0, str(_P(__file__).resolve().parent))
from we_common import (  # noqa: E402
    MODEL_NAME,
    MODEL_SHORT,
    STEPS_32,
    TOKENIZER_VOCAB,
    _decoded_token_texts,
    _family_selected_features,
    _feature_family_fraction,
    _figure_bases,
    _load_we_rates_and_norms,
    _load_wu_rates_and_norms,
    _terminal_decoder,
    _token_family_masks,
    _top_tokens_per_feature,
    _write_csv_flexible,
    _write_rows_and_cache,
)

from readout.core.paths import release_path, repo_root, ssd_root
from readout.core.repro import git_commit, log_run_provenance
from readout.crosscoder.snapshots import load_snapshot
from readout.dynamics.metrics import lifecycle

REPO = repo_root()


def _load_we_d24576_rates_and_norms(
    ssd_root: Path,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    rate_path = ssd_root / "derived" / "rates" / "we-d24576" / "we_rates_dsae24576_seed0.pt"
    norm_path = ssd_root / "derived" / "rates" / "we-d24576" / "we_cc_dsae24576_seed0_norms.npy"
    payload = torch.load(rate_path, map_location="cpu", weights_only=False)
    if payload.get("steps") != STEPS_32:
        raise ValueError(f"unexpected d24576 W_E steps in {rate_path}")
    rates_dict = payload["rates_per_seed"]
    if "we_cc_dsae24576_seed0" not in rates_dict:
        raise KeyError(f"missing we_cc_dsae24576_seed0 in {rate_path}")
    rates = rates_dict["we_cc_dsae24576_seed0"].float().numpy()
    norms = np.load(norm_path).astype(np.float32)
    if rates.shape != (32, 24576):
        raise ValueError(f"unexpected d24576 rate shape {rates.shape}")
    if norms.shape != (32, 24576):
        raise ValueError(f"unexpected d24576 norm shape {norms.shape}")
    return rates, norms, rate_path, norm_path


def _peak_steps_for_rates(rates: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """Return peak-rate step per seed/feature.

    Accepts (S,K,D) or (K,D), returns (S,D).
    """
    if rates.ndim == 2:
        rates = rates[None, ...]
        norms = norms[None, ...]
    rows = []
    step_arr = np.asarray(STEPS_32)
    for seed in range(rates.shape[0]):
        lc = lifecycle(rates[seed], decoder_norms=norms[seed])
        idx = lc.peak_step.copy()
        out = np.full(idx.shape, np.nan, dtype=np.float64)
        ok = idx < len(STEPS_32)
        out[ok] = step_arr[idx[ok]]
        rows.append(out)
    return np.stack(rows, axis=0)


def build_d24576_rate_timing(
    wu_rates: np.ndarray,
    wu_norms: np.ndarray,
    we_rates: np.ndarray,
    we_norms: np.ndarray,
    we_d24576_rates: np.ndarray,
    we_d24576_norms: np.ndarray,
    out_bases: list[Path],
    *,
    rate_path: Path,
    norm_path: Path,
) -> None:
    rows: list[dict] = []
    series = [
        ("W_E", 8192, "mean5", we_rates, we_norms),
        ("W_E", 24576, 0, we_d24576_rates[None, ...], we_d24576_norms[None, ...]),
        ("W_U", 8192, "mean5", wu_rates, wu_norms),
    ]
    for matrix, dim, seed_label, rates, norms in series:
        l0 = rates.sum(axis=2)
        mean_rate = rates.mean(axis=2)
        mean_norm = norms.mean(axis=2)
        for step_idx, step in enumerate(STEPS_32):
            rows.append(
                {
                    "matrix": matrix,
                    "d_sae": dim,
                    "seed": seed_label,
                    "step": step,
                    "mean_l0": float(l0[:, step_idx].mean()),
                    "sd_l0": float(l0[:, step_idx].std()),
                    "mean_feature_rate": float(mean_rate[:, step_idx].mean()),
                    "mean_decoder_norm": float(mean_norm[:, step_idx].mean()),
                }
            )

    peak_rows = []
    for matrix, dim, seed_label, rates, norms in series:
        peak = _peak_steps_for_rates(rates, norms)
        for seed in range(peak.shape[0]):
            label = seed if seed_label == "mean5" else seed_label
            vals = peak[seed]
            peak_rows.append(
                {
                    "matrix": matrix,
                    "d_sae": dim,
                    "seed": label,
                    "median_peak_step": float(np.nanmedian(vals)),
                    "mean_peak_step": float(np.nanmean(vals)),
                    "q25_peak_step": float(np.nanquantile(vals, 0.25)),
                    "q75_peak_step": float(np.nanquantile(vals, 0.75)),
                }
            )

    _write_rows_and_cache(
        out_bases,
        rows + [{"kind": "peak_summary", **r} for r in peak_rows],
        {
            "steps": STEPS_32,
            "rate_path": str(rate_path),
            "norm_path": str(norm_path),
            "rows": rows,
            "peak_rows": peak_rows,
            "wu_rates": torch.from_numpy(wu_rates),
            "we_rates": torch.from_numpy(we_rates),
            "we_d24576_rates": torch.from_numpy(we_d24576_rates),
        },
    )


def _token_metadata(vocab_rows: int) -> tuple[np.ndarray, np.ndarray]:
    texts = _decoded_token_texts()
    token_ids = np.arange(vocab_rows, dtype=np.float32)
    lengths = np.zeros(vocab_rows, dtype=np.float32)
    for token_id, text in enumerate(texts):
        stripped = text.strip()
        lengths[token_id] = max(1, len(stripped))
    lengths[len(texts) :] = 1
    return token_ids, lengths


def _feature_token_stats(top_tokens: np.ndarray, token_ids: np.ndarray, token_lengths: np.ndarray) -> np.ndarray:
    clipped = np.minimum(top_tokens, token_ids.shape[0] - 1)
    med_id = np.median(token_ids[clipped], axis=1) / max(1.0, float(TOKENIZER_VOCAB))
    med_len = np.median(token_lengths[clipped], axis=1)
    len_scale = max(1.0, float(np.quantile(token_lengths[:TOKENIZER_VOCAB], 0.95)))
    med_len = np.minimum(med_len / len_scale, 3.0)
    return np.stack([med_id, med_len], axis=1).astype(np.float32)


def _nearest_controls(
    selected_idx: np.ndarray,
    pool_idx: np.ndarray,
    stats: np.ndarray,
    *,
    chunk: int = 1024,
) -> np.ndarray:
    if selected_idx.size == 0:
        return selected_idx
    if pool_idx.size == 0:
        return selected_idx
    selected_stats = stats[selected_idx]
    pool_stats = stats[pool_idx]
    out = []
    for start in range(0, selected_stats.shape[0], chunk):
        end = min(selected_stats.shape[0], start + chunk)
        diff = selected_stats[start:end, None, :] - pool_stats[None, :, :]
        dist = np.sum(diff * diff, axis=2)
        out.append(pool_idx[np.argmin(dist, axis=1)])
    return np.concatenate(out).astype(np.int64)


def _summary_delta_rows(rows: list[dict], masks: dict[str, np.ndarray]) -> list[dict]:
    summary = []
    for family in masks:
        obs_wu = [r["observed_median_peak_step"] for r in rows if r["matrix"] == "W_U" and r["family"] == family]
        obs_we = [r["observed_median_peak_step"] for r in rows if r["matrix"] == "W_E" and r["family"] == family]
        ctrl_wu = [r["control_median_peak_step"] for r in rows if r["matrix"] == "W_U" and r["family"] == family]
        ctrl_we = [r["control_median_peak_step"] for r in rows if r["matrix"] == "W_E" and r["family"] == family]
        obs_delta = float(np.nanmedian(obs_wu) - np.nanmedian(obs_we))
        ctrl_delta = float(np.nanmedian(ctrl_wu) - np.nanmedian(ctrl_we))
        summary.append(
            {
                "matrix": "summary",
                "seed": "median5",
                "family": family,
                "observed_delta_wu_minus_we": obs_delta,
                "control_delta_wu_minus_we": ctrl_delta,
                "adjusted_delta_wu_minus_we": obs_delta - ctrl_delta,
            }
        )
    return summary


def build_matched_control_leadlag(
    wu_rates: np.ndarray,
    wu_norms: np.ndarray,
    we_rates: np.ndarray,
    we_norms: np.ndarray,
    masks: dict[str, np.ndarray],
    out_bases: list[Path],
    *,
    top_k: int,
    family_threshold: float,
) -> None:
    terminal_wu = load_snapshot(MODEL_NAME, STEPS_32[-1], kind="wu")
    token_ids, token_lengths = _token_metadata(terminal_wu.shape[0])
    peak_steps = {
        "W_U": _peak_steps_for_rates(wu_rates, wu_norms),
        "W_E": _peak_steps_for_rates(we_rates, we_norms),
    }
    rows: list[dict] = []
    for matrix in ["W_U", "W_E"]:
        for seed in range(5):
            decoder = _terminal_decoder(release_path(MODEL_SHORT, matrix, dim=8192, seed=seed))
            top_tokens = _top_tokens_per_feature(decoder, terminal_wu, top_k=top_k)
            stats = _feature_token_stats(top_tokens, token_ids, token_lengths)
            selected_by_family = _family_selected_features(top_tokens, masks, threshold=family_threshold)
            for family, selected_idx in selected_by_family.items():
                frac = _feature_family_fraction(top_tokens, masks[family])
                pool_idx = np.flatnonzero(frac < min(0.1, family_threshold / 2.0))
                if pool_idx.size < 24:
                    pool_idx = np.setdiff1d(np.arange(frac.size), selected_idx)
                controls = _nearest_controls(selected_idx, pool_idx, stats)
                obs_vals = peak_steps[matrix][seed, selected_idx]
                ctrl_vals = peak_steps[matrix][seed, controls]
                rows.append(
                    {
                        "matrix": matrix,
                        "seed": seed,
                        "family": family,
                        "n_observed_features": int(selected_idx.size),
                        "n_control_pool": int(pool_idx.size),
                        "n_control_features": int(controls.size),
                        "mean_selected_family_fraction": float(frac[selected_idx].mean()),
                        "mean_control_family_fraction": float(frac[controls].mean()),
                        "observed_median_peak_step": float(np.nanmedian(obs_vals)),
                        "control_median_peak_step": float(np.nanmedian(ctrl_vals)),
                        "observed_minus_control_peak_step": float(np.nanmedian(obs_vals) - np.nanmedian(ctrl_vals)),
                    }
                )
            del decoder, top_tokens
    summary = _summary_delta_rows(rows, masks)

    _write_rows_and_cache(
        out_bases,
        rows + summary,
        {
            "steps": STEPS_32,
            "top_k": top_k,
            "family_threshold": family_threshold,
            "control": "nearest feature controls matched on median top-token BPE id and decoded length",
            "rows": rows,
            "summary": summary,
        },
    )


def _procrustes_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Metrics for min_R ||x R - y||_F with R orthogonal."""
    cross = x.T @ y
    u, s, vh = torch.linalg.svd(cross, full_matrices=False)
    r = u @ vh
    rotated = x @ r
    x_norm = torch.linalg.vector_norm(x)
    y_norm = torch.linalg.vector_norm(y)
    residual = torch.linalg.vector_norm(rotated - y) / y_norm.clamp_min(1e-12)
    scale = s.sum() / x_norm.square().clamp_min(1e-12)
    scaled_residual = torch.linalg.vector_norm(scale * rotated - y) / y_norm.clamp_min(1e-12)
    alignment_cosine = s.sum() / (x_norm * y_norm).clamp_min(1e-12)
    return {
        "alignment_cosine": float(alignment_cosine.item()),
        "orthogonal_residual_rel_y": float(residual.item()),
        "scaled_residual_rel_y": float(scaled_residual.item()),
        "isotropic_scale": float(scale.item()),
        "top_singular_value": float(s[0].item()),
        "median_singular_value": float(s[len(s) // 2].item()),
    }


def build_procrustes_alignment(out_bases: list[Path]) -> None:
    rows: list[dict] = []
    for step in STEPS_32:
        we = load_snapshot(MODEL_NAME, step, kind="we")
        wu = load_snapshot(MODEL_NAME, step, kind="wu")
        for centered in [False, True]:
            x = we - we.mean(dim=0, keepdim=True) if centered else we
            y = wu - wu.mean(dim=0, keepdim=True) if centered else wu
            metrics = _procrustes_metrics(x, y)
            row_cos = torch.nn.functional.cosine_similarity(x, y, dim=1)
            rows.append(
                {
                    "step": step,
                    "centered": centered,
                    "same_token_row_cosine_mean": float(row_cos.mean().item()),
                    "same_token_row_cosine_median": float(row_cos.median().item()),
                    **metrics,
                }
            )
        del we, wu

    _write_rows_and_cache(
        out_bases,
        rows,
        {
            "steps": STEPS_32,
            "rows": rows,
            "definition": "orthogonal Procrustes solves min_R ||W_E R - W_U||_F",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssd-root", type=Path, default=ssd_root())
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--family-threshold", type=float, default=0.35)
    args = parser.parse_args()

    root = repo_root()
    fig_dir = root / "figures" / "crosscoder_we"
    result_dir = root / "results" / "experiments" / "crosscoder_we"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "build_we_wu_top3_experiments.provenance.json").write_text(
        json.dumps({**log_run_provenance(), "git_commit": git_commit()}, indent=2)
    )

    we_rates, we_norms = _load_we_rates_and_norms(args.ssd_root)
    wu_rates, wu_norms = _load_wu_rates_and_norms(args.ssd_root)
    we_d24576_rates, we_d24576_norms, rate_path, norm_path = _load_we_d24576_rates_and_norms(args.ssd_root)

    terminal_wu = load_snapshot(MODEL_NAME, STEPS_32[-1], kind="wu")
    masks, family_rows = _token_family_masks(terminal_wu.shape[0])
    del terminal_wu

    build_d24576_rate_timing(
        wu_rates,
        wu_norms,
        we_rates,
        we_norms,
        we_d24576_rates,
        we_d24576_norms,
        _figure_bases("d24576_rate_timing", fig_dir),
        rate_path=rate_path,
        norm_path=norm_path,
    )
    build_matched_control_leadlag(
        wu_rates,
        wu_norms,
        we_rates,
        we_norms,
        masks,
        _figure_bases("matched_control_lead_lag", fig_dir),
        top_k=args.top_k,
        family_threshold=args.family_threshold,
    )
    build_procrustes_alignment(_figure_bases("procrustes_alignment", fig_dir))

    _write_csv_flexible(result_dir / "token_family_masks_top3.csv", family_rows)
    torch.save(
        {"families": masks, "rows": family_rows},
        result_dir / "token_family_masks_top3.pt",
    )
    for name in [
        "d24576_rate_timing",
        "matched_control_lead_lag",
        "procrustes_alignment",
    ]:
        src = fig_dir / f"{name}.csv"
        if src.exists():
            with src.open() as f:
                _write_csv_flexible(result_dir / f"{name}.csv", list(csv.DictReader(f)))

    print(f"wrote top-three W_E/W_U metric CSV/.pt caches to {fig_dir}")
    print(f"wrote top-three audit CSVs to {result_dir}")


if __name__ == "__main__":
    main()
