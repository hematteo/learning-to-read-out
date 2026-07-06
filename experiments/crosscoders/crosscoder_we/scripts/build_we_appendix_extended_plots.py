"""Build extended W_E appendix audits for the read/write section.

These figures complete the companion appendix around the available
Pythia-160M W_E artifacts:

  - token-family peak timing from cached d8192 firing rates;
  - top-token overlap for nearest W_E/W_U terminal decoder matches, with
    random and within-matrix controls;
  - dense W_E/W_U row-geometry trajectories across Pythia checkpoints;
  - W_E cross-seed Hungarian correspondence on high-activity features;
  - a d24576 capacity note using quality and decoder-norm trajectories.

Audit payloads are saved as CSV plus .pt alongside each output base.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment

from experiments.crosscoders.crosscoder_we.scripts.we_common import (  # noqa: E402
    MODEL_NAME,
    MODEL_SHORT,
    STEPS_32,
    _family_selected_features,
    _feature_family_fraction,
    _figure_bases,
    _load_we_rates_and_norms,
    _load_wu_rates_and_norms,
    _quality_rows,
    _terminal_decoder,
    _token_family_masks,
    _top_tokens_per_feature,
    _write_csv,
    _write_rows_and_cache,
)
from readout.core.paths import release_path, repo_root, ssd_root
from readout.crosscoder.snapshots import load_snapshot
from readout.dynamics.metrics import lifecycle

REPO = repo_root()


def _norm_rows_from_safetensors(path: Path) -> np.ndarray:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        w_d = f.get_slice("W_D")
        shape = w_d.get_shape()
        norms = []
        for step_idx in range(shape[0]):
            norms.append(torch.linalg.vector_norm(w_d[step_idx].float(), dim=1).numpy())
    return np.stack(norms, axis=0)


def _terminal_decoder_norms(path: Path) -> np.ndarray:
    return torch.linalg.vector_norm(_terminal_decoder(path), dim=1).numpy()


def _decoder_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a.float(), dim=1)
    b = torch.nn.functional.normalize(b.float(), dim=1)
    return a @ b.T


def _lifecycle_step_tables(rates: np.ndarray, norms: np.ndarray) -> dict[str, np.ndarray]:
    first_rows = []
    peak_rows = []
    for seed in range(rates.shape[0]):
        lc = lifecycle(rates[seed], decoder_norms=norms[seed])
        first_idx = lc.first_active.copy()
        first_step = np.full(first_idx.shape, np.nan, dtype=np.float64)
        active = first_idx < len(STEPS_32)
        first_step[active] = np.asarray(STEPS_32, dtype=np.float64)[first_idx[active]]
        peak_idx = lc.peak_step.copy()
        peak_step = np.full(peak_idx.shape, np.nan, dtype=np.float64)
        peaked = peak_idx < len(STEPS_32)
        peak_step[peaked] = np.asarray(STEPS_32, dtype=np.float64)[peak_idx[peaked]]
        first_rows.append(first_step)
        peak_rows.append(peak_step)
    return {
        "first_active_step": np.stack(first_rows, axis=0),
        "peak_step": np.stack(peak_rows, axis=0),
    }


def build_lead_lag_plot(
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
    wu_lifecycle = _lifecycle_step_tables(wu_rates, wu_norms)
    we_lifecycle = _lifecycle_step_tables(we_rates, we_norms)
    rows: list[dict] = []
    plot_rows: list[dict] = []

    for matrix, life in [("W_U", wu_lifecycle), ("W_E", we_lifecycle)]:
        for seed in range(5):
            ckpt = release_path(MODEL_SHORT, matrix, dim=8192, seed=seed)
            top_tokens = _top_tokens_per_feature(_terminal_decoder(ckpt), terminal_wu, top_k=top_k)
            selected = _family_selected_features(top_tokens, masks, threshold=family_threshold)
            for family, idx in selected.items():
                first_vals = life["first_active_step"][seed, idx]
                first_vals = first_vals[np.isfinite(first_vals)]
                peak_vals = life["peak_step"][seed, idx]
                peak_vals = peak_vals[np.isfinite(peak_vals)]
                frac = _feature_family_fraction(top_tokens, masks[family])
                rows.append(
                    {
                        "matrix": matrix,
                        "seed": seed,
                        "family": family,
                        "n_features": int(idx.size),
                        "mean_top_token_family_fraction": float(frac[idx].mean()),
                        "median_first_active_step": float(np.median(first_vals)) if first_vals.size else math.nan,
                        "mean_first_active_step": float(np.mean(first_vals)) if first_vals.size else math.nan,
                        "median_peak_step": float(np.median(peak_vals)) if peak_vals.size else math.nan,
                        "mean_peak_step": float(np.mean(peak_vals)) if peak_vals.size else math.nan,
                    }
                )

    for family in masks:
        wu_by_seed = {int(r["seed"]): r for r in rows if r["matrix"] == "W_U" and r["family"] == family}
        we_by_seed = {int(r["seed"]): r for r in rows if r["matrix"] == "W_E" and r["family"] == family}
        wu_vals = [r["median_first_active_step"] for r in wu_by_seed.values()]
        we_vals = [r["median_first_active_step"] for r in we_by_seed.values()]
        wu_peak_vals = [r["median_peak_step"] for r in wu_by_seed.values()]
        we_peak_vals = [r["median_peak_step"] for r in we_by_seed.values()]
        paired_peak_deltas = np.asarray(
            [
                wu_by_seed[seed]["median_peak_step"] - we_by_seed[seed]["median_peak_step"]
                for seed in sorted(wu_by_seed)
                if seed in we_by_seed
            ],
            dtype=np.float64,
        )
        wu_first_med = float(np.nanmedian(wu_vals))
        we_first_med = float(np.nanmedian(we_vals))
        wu_peak_med = float(np.nanmedian(wu_peak_vals))
        we_peak_med = float(np.nanmedian(we_peak_vals))
        delta_med = float(np.nanmedian(paired_peak_deltas))
        delta_q25 = float(np.nanquantile(paired_peak_deltas, 0.25))
        delta_q75 = float(np.nanquantile(paired_peak_deltas, 0.75))
        plot_rows.append(
            {
                "family": family,
                "we_median_first_active_step": we_first_med,
                "wu_median_first_active_step": wu_first_med,
                "delta_first_active_step_wu_minus_we": wu_first_med - we_first_med,
                "we_median_peak_step": we_peak_med,
                "wu_median_peak_step": wu_peak_med,
                "delta_peak_step_wu_minus_we": delta_med,
                "delta_peak_step_q25": delta_q25,
                "delta_peak_step_q75": delta_q75,
                "n_seed_pairs": int(paired_peak_deltas.size),
            }
        )

    _write_rows_and_cache(
        out_bases,
        rows + [{"matrix": "summary", **r} for r in plot_rows],
        {
            "steps": STEPS_32,
            "top_k": top_k,
            "family_threshold": family_threshold,
            "timing_metric": "median_peak_step",
            "first_active_note": "relative first-active is saturated at step 0 for these rate measurements",
            "rows": rows,
            "summary": plot_rows,
        },
    )


def _jaccard_rows(
    source_top: np.ndarray,
    target_top: np.ndarray,
    target_match: np.ndarray,
    match_cos: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    comparison: str,
    source_matrix: str,
    target_matrix: str,
) -> list[dict]:
    rows: list[dict] = []
    families = list(masks)
    family_fracs = np.stack([_feature_family_fraction(source_top, masks[f]) for f in families], axis=1)
    primary_idx = np.argmax(family_fracs, axis=1)
    for feature in range(source_top.shape[0]):
        matched_feature = int(target_match[feature])
        a = set(int(x) for x in source_top[feature])
        b = set(int(x) for x in target_top[matched_feature])
        inter = len(a & b)
        union = len(a | b)
        family = families[int(primary_idx[feature])]
        rows.append(
            {
                "comparison": comparison,
                "source_matrix": source_matrix,
                "target_matrix": target_matrix,
                "source_feature": feature,
                "matched_feature": matched_feature,
                "decoder_cosine": float(match_cos[feature]),
                "top_token_jaccard": float(inter / union),
                "top_token_intersection": int(inter),
                "primary_family": family,
                "primary_family_fraction": float(family_fracs[feature, primary_idx[feature]]),
            }
        )
    return rows


def _nearest_matches(source_decoder: torch.Tensor, target_decoder: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    cos = _decoder_cosine(source_decoder, target_decoder)
    match_cos, match = torch.max(cos, dim=1)
    return match.numpy(), match_cos.numpy()


def build_token_overlap_plot(
    masks: dict[str, np.ndarray],
    out_bases: list[Path],
    *,
    top_k: int,
) -> None:
    terminal_wu = load_snapshot(MODEL_NAME, STEPS_32[-1], kind="wu")
    wu0_decoder = _terminal_decoder(release_path(MODEL_SHORT, "W_U", dim=8192, seed=0))
    wu1_decoder = _terminal_decoder(release_path(MODEL_SHORT, "W_U", dim=8192, seed=1))
    we0_decoder = _terminal_decoder(release_path(MODEL_SHORT, "W_E", dim=8192, seed=0))
    we1_decoder = _terminal_decoder(release_path(MODEL_SHORT, "W_E", dim=8192, seed=1))
    wu0_top = _top_tokens_per_feature(wu0_decoder, terminal_wu, top_k=top_k)
    wu1_top = _top_tokens_per_feature(wu1_decoder, terminal_wu, top_k=top_k)
    we0_top = _top_tokens_per_feature(we0_decoder, terminal_wu, top_k=top_k)
    we1_top = _top_tokens_per_feature(we1_decoder, terminal_wu, top_k=top_k)

    rng = np.random.default_rng(0)
    wu0_we0_match, wu0_we0_cos = _nearest_matches(wu0_decoder, we0_decoder)
    wu0_wu1_match, wu0_wu1_cos = _nearest_matches(wu0_decoder, wu1_decoder)
    we0_we1_match, we0_we1_cos = _nearest_matches(we0_decoder, we1_decoder)
    random_we_match = rng.permutation(we0_top.shape[0])[: wu0_top.shape[0]]
    random_cos = np.full(wu0_top.shape[0], np.nan, dtype=np.float32)

    rows: list[dict] = []
    rows.extend(
        _jaccard_rows(
            wu0_top,
            we0_top,
            wu0_we0_match,
            wu0_we0_cos,
            masks,
            comparison="W_U0-W_E0 nearest",
            source_matrix="W_U seed0",
            target_matrix="W_E seed0",
        )
    )
    rows.extend(
        _jaccard_rows(
            wu0_top,
            we0_top,
            random_we_match,
            random_cos,
            masks,
            comparison="W_U0-random W_E0",
            source_matrix="W_U seed0",
            target_matrix="W_E seed0 random",
        )
    )
    rows.extend(
        _jaccard_rows(
            wu0_top,
            wu1_top,
            wu0_wu1_match,
            wu0_wu1_cos,
            masks,
            comparison="W_U0-W_U1 nearest",
            source_matrix="W_U seed0",
            target_matrix="W_U seed1",
        )
    )
    rows.extend(
        _jaccard_rows(
            we0_top,
            we1_top,
            we0_we1_match,
            we0_we1_cos,
            masks,
            comparison="W_E0-W_E1 nearest",
            source_matrix="W_E seed0",
            target_matrix="W_E seed1",
        )
    )

    summary_rows = []
    family_rows = []
    for comparison in sorted({r["comparison"] for r in rows}):
        vals = np.asarray(
            [r["top_token_jaccard"] for r in rows if r["comparison"] == comparison],
            dtype=np.float64,
        )
        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(comparison)) % (2**32)
        boot_rng = np.random.default_rng(seed)
        boot_means = np.asarray([boot_rng.choice(vals, size=vals.size, replace=True).mean() for _ in range(500)])
        summary_rows.append(
            {
                "comparison": comparison,
                "n_features": int(vals.size),
                "mean_jaccard": float(vals.mean()),
                "median_jaccard": float(np.median(vals)),
                "q25_jaccard": float(np.quantile(vals, 0.25)),
                "q75_jaccard": float(np.quantile(vals, 0.75)),
                "q95_jaccard": float(np.quantile(vals, 0.95)),
                "mean_jaccard_ci_low": float(np.quantile(boot_means, 0.025)),
                "mean_jaccard_ci_high": float(np.quantile(boot_means, 0.975)),
            }
        )
        for family in masks:
            fvals = [
                r["top_token_jaccard"]
                for r in rows
                if r["comparison"] == comparison
                and r["primary_family"] == family
                and r["primary_family_fraction"] >= 0.2
            ]
            family_rows.append(
                {
                    "comparison": comparison,
                    "family": family,
                    "n_features": len(fvals),
                    "mean_jaccard": float(np.mean(fvals)) if fvals else math.nan,
                    "median_jaccard": float(np.median(fvals)) if fvals else math.nan,
                }
            )

    _write_rows_and_cache(
        out_bases,
        rows
        + [{"source_feature": "comparison_summary", **r} for r in summary_rows]
        + [{"source_feature": "family_summary", **r} for r in family_rows],
        {
            "top_k": top_k,
            "rows": rows,
            "comparison_summary": summary_rows,
            "family_summary": family_rows,
            "wu0_to_we0_match": wu0_we0_match,
            "wu0_to_we0_cosine": wu0_we0_cos,
            "wu0_to_wu1_match": wu0_wu1_match,
            "wu0_to_wu1_cosine": wu0_wu1_cos,
            "we0_to_we1_match": we0_we1_match,
            "we0_to_we1_cosine": we0_we1_cos,
            "random_we_match": random_we_match,
        },
    )


def build_geometry_plot(out_bases: list[Path]) -> None:
    rows: list[dict] = []
    hist_payload: dict[int, dict[str, np.ndarray]] = {}
    for step in STEPS_32:
        wu = load_snapshot(MODEL_NAME, step, kind="wu")
        we = load_snapshot(MODEL_NAME, step, kind="we")
        wu_norm = torch.linalg.vector_norm(wu, dim=1).numpy()
        we_norm = torch.linalg.vector_norm(we, dim=1).numpy()
        cos = torch.nn.functional.cosine_similarity(wu, we, dim=1).numpy()
        rows.append(
            {
                "step": step,
                "wu_row_norm_mean": float(wu_norm.mean()),
                "wu_row_norm_median": float(np.median(wu_norm)),
                "we_row_norm_mean": float(we_norm.mean()),
                "we_row_norm_median": float(np.median(we_norm)),
                "we_over_wu_mean_norm_ratio": float(we_norm.mean() / wu_norm.mean()),
                "per_token_cosine_mean": float(cos.mean()),
                "per_token_cosine_median": float(np.median(cos)),
                "per_token_cosine_q05": float(np.quantile(cos, 0.05)),
                "per_token_cosine_q95": float(np.quantile(cos, 0.95)),
            }
        )
        if step in {0, 1000, STEPS_32[-1]}:
            hist_payload[step] = {
                "wu_norm": wu_norm,
                "we_norm": we_norm,
                "per_token_cosine": cos,
            }
        del wu, we

    _write_rows_and_cache(out_bases, rows, {"rows": rows, "histograms": hist_payload})


def build_hungarian_plot(
    we_rates: np.ndarray,
    out_bases: list[Path],
    *,
    top_n: int,
) -> None:
    decoders = []
    for seed in range(5):
        decoders.append(_terminal_decoder(release_path(MODEL_SHORT, "W_E", dim=8192, seed=seed)))

    rows: list[dict] = []
    distributions: dict[str, list[np.ndarray]] = {"Hungarian": [], "random": []}
    rng = np.random.default_rng(0)
    top_ranks = [np.argsort(we_rates[seed, -1])[-top_n:] for seed in range(5)]
    normed = [torch.nn.functional.normalize(decoders[seed][top_ranks[seed]].float(), dim=1) for seed in range(5)]
    for seed_a in range(5):
        for seed_b in range(seed_a + 1, 5):
            cos = (normed[seed_a] @ normed[seed_b].T).numpy()
            row_ind, col_ind = linear_sum_assignment(-cos)
            matched = cos[row_ind, col_ind]
            random_col = rng.permutation(cos.shape[1])[: cos.shape[0]]
            random_matched = cos[np.arange(cos.shape[0]), random_col]
            distributions["Hungarian"].append(matched)
            distributions["random"].append(random_matched)
            seed_pair = f"{seed_a}-{seed_b}"
            for local_i, local_j, score in zip(row_ind, col_ind, matched):
                rows.append(
                    {
                        "comparison": "Hungarian",
                        "seed_pair": seed_pair,
                        "seed_a_feature": int(top_ranks[seed_a][local_i]),
                        "seed_b_feature": int(top_ranks[seed_b][local_j]),
                        "decoder_cosine": float(score),
                        "rank_subset_n": top_n,
                    }
                )
            for local_i, local_j, score in zip(np.arange(cos.shape[0]), random_col, random_matched):
                rows.append(
                    {
                        "comparison": "random",
                        "seed_pair": seed_pair,
                        "seed_a_feature": int(top_ranks[seed_a][local_i]),
                        "seed_b_feature": int(top_ranks[seed_b][local_j]),
                        "decoder_cosine": float(score),
                        "rank_subset_n": top_n,
                    }
                )

    quality = [r for r in _quality_rows() if r["matrix"] == "W_E" and r["d_sae"] == 8192]
    for r in quality:
        rows.append(
            {
                "comparison": "quality",
                "seed_pair": "quality",
                "seed_a_feature": r["seed"],
                "seed_b_feature": r["seed"],
                "decoder_cosine": r["explained_variance"],
                "rank_subset_n": top_n,
                "mean_l0": r["mean_l0"],
            }
        )

    _write_rows_and_cache(
        out_bases,
        rows,
        {
            "top_n": top_n,
            "rows": rows,
            "distributions": distributions,
            "quality_rows": quality,
        },
    )


def _rate_sidecar_candidates(ssd_root: Path) -> list[str]:
    names = [
        ssd_root / "derived" / "rates" / "we-d24576" / "we_rates_dsae24576_seed0.pt",
        ssd_root / "wu_crosscoder" / "cluster_results" / "we_multiseed" / "we_rates_dsae24576_seed0.pt",
        ssd_root / "wu_crosscoder" / "cluster_results" / "we_multiseed" / "we_rates_d24576_seed0.pt",
        ssd_root / "archive" / "cluster_results" / "t3_2_we_dsae24576" / "rates.pt",
        ssd_root / "archive" / "cluster_results" / "t3_2_we_dsae24576" / "we_rates_dsae24576_seed0.pt",
    ]
    return [str(p) for p in names if p.exists()]


def _load_we_d24576_rates(ssd_root: Path) -> tuple[np.ndarray, str]:
    candidates = _rate_sidecar_candidates(ssd_root)
    if not candidates:
        raise FileNotFoundError("no W_E d24576 rate sidecar found")
    path = Path(candidates[0])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("steps") != STEPS_32:
        raise ValueError(f"unexpected W_E d24576 steps in {path}")
    per_seed = payload["rates_per_seed"]
    if len(per_seed) != 1:
        raise ValueError(f"expected one d24576 rate tensor in {path}, got {per_seed.keys()}")
    return next(iter(per_seed.values())).float().numpy(), str(path)


def build_capacity_note_plot(
    wu_rates: np.ndarray,
    wu_norms: np.ndarray,
    we_rates: np.ndarray,
    we_norms: np.ndarray,
    out_bases: list[Path],
    *,
    ssd_root: Path,
) -> None:
    we_d24576_path = release_path(MODEL_SHORT, "W_E", dim=24576, seed=0)
    we_d24576_norms = _norm_rows_from_safetensors(we_d24576_path)
    we_d24576_rates, we_d24576_rate_path = _load_we_d24576_rates(ssd_root)
    quality_rows = _quality_rows()
    rows: list[dict] = []
    for step_idx, step in enumerate(STEPS_32):
        rows.append(
            {
                "kind": "active_trajectory",
                "matrix": "W_E",
                "d_sae": 8192,
                "seed": "mean5",
                "step": step,
                "mean_l0": float(we_rates[:, step_idx, :].sum(axis=1).mean()),
                "mean_feature_rate": float(we_rates[:, step_idx, :].mean()),
            }
        )
        rows.append(
            {
                "kind": "active_trajectory",
                "matrix": "W_U",
                "d_sae": 8192,
                "seed": "mean5",
                "step": step,
                "mean_l0": float(wu_rates[:, step_idx, :].sum(axis=1).mean()),
                "mean_feature_rate": float(wu_rates[:, step_idx, :].mean()),
            }
        )
        rows.append(
            {
                "kind": "active_trajectory",
                "matrix": "W_E",
                "d_sae": 24576,
                "seed": 0,
                "step": step,
                "mean_l0": float(we_d24576_rates[step_idx].sum()),
                "mean_feature_rate": float(we_d24576_rates[step_idx].mean()),
            }
        )
        rows.append(
            {
                "kind": "norm_trajectory",
                "matrix": "W_E",
                "d_sae": 8192,
                "seed": "mean5",
                "step": step,
                "mean_decoder_norm": float(we_norms[:, step_idx, :].mean()),
            }
        )
        rows.append(
            {
                "kind": "norm_trajectory",
                "matrix": "W_U",
                "d_sae": 8192,
                "seed": "mean5",
                "step": step,
                "mean_decoder_norm": float(wu_norms[:, step_idx, :].mean()),
            }
        )
        rows.append(
            {
                "kind": "norm_trajectory",
                "matrix": "W_E",
                "d_sae": 24576,
                "seed": 0,
                "step": step,
                "mean_decoder_norm": float(we_d24576_norms[step_idx].mean()),
            }
        )
    for r in quality_rows:
        rows.append(
            {
                "kind": "quality",
                "matrix": r["matrix"],
                "d_sae": r["d_sae"],
                "seed": r["seed"],
                "step": "",
                "mean_decoder_norm": "",
                "explained_variance": r["explained_variance"],
                "mean_l0": r["mean_l0"],
                "rate_sidecar": we_d24576_rate_path,
            }
        )

    _write_rows_and_cache(
        out_bases,
        rows,
        {
            "steps": STEPS_32,
            "rows": rows,
            "we_d24576_decoder_norms": we_d24576_norms,
            "we_d24576_rates": we_d24576_rates,
            "we_d24576_rate_path": we_d24576_rate_path,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssd-root", type=Path, default=ssd_root())
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--family-threshold", type=float, default=0.35)
    parser.add_argument("--hungarian-top-n", type=int, default=1024)
    args = parser.parse_args()

    root = repo_root()
    fig_dir = root / "figures" / "crosscoder_we"
    result_dir = root / "results" / "experiments" / "crosscoder_we"
    result_dir.mkdir(parents=True, exist_ok=True)

    we_rates, we_norms = _load_we_rates_and_norms(args.ssd_root)
    wu_rates, wu_norms = _load_wu_rates_and_norms(args.ssd_root)
    terminal_wu = load_snapshot(MODEL_NAME, STEPS_32[-1], kind="wu")
    masks, family_rows = _token_family_masks(terminal_wu.shape[0])
    _write_csv(result_dir / "token_family_masks.csv", family_rows)
    torch.save({"families": masks, "rows": family_rows}, result_dir / "token_family_masks.pt")
    del terminal_wu

    build_lead_lag_plot(
        wu_rates,
        wu_norms,
        we_rates,
        we_norms,
        masks,
        _figure_bases("lead_lag_family_heatmap", fig_dir),
        top_k=args.top_k,
        family_threshold=args.family_threshold,
    )
    build_token_overlap_plot(
        masks,
        _figure_bases("token_overlap_jaccard", fig_dir),
        top_k=args.top_k,
    )
    build_geometry_plot([fig_dir / "wu_we_geometry_160m"])
    build_hungarian_plot(
        we_rates,
        _figure_bases("multiseed_hungarian", fig_dir),
        top_n=args.hungarian_top_n,
    )
    build_capacity_note_plot(
        wu_rates,
        wu_norms,
        we_rates,
        we_norms,
        _figure_bases("capacity_note", fig_dir),
        ssd_root=args.ssd_root,
    )

    for name in [
        "lead_lag_family_heatmap",
        "token_overlap_jaccard",
        "wu_we_geometry_160m",
        "we_wu_geometry_160m",
        "multiseed_hungarian",
        "capacity_note",
    ]:
        src = fig_dir / f"{name}.csv"
        if src.exists():
            rows = list(csv.DictReader(src.open()))
            _write_csv(result_dir / f"{name}.csv", rows)

    print(f"wrote extended W_E appendix audit payloads to {fig_dir}")
    print(f"wrote extended audit CSVs to {result_dir}")


if __name__ == "__main__":
    main()
