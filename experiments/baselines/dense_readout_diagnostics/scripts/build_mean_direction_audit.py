"""Build mean-direction and preprocessing audit metrics.

The average W_U row can dominate raw geometry.
This script runs the relevant audit for the paper's current Pythia model set:
160M, 1B, and 6.9B. It avoids full SVD by estimating the top two right singular
directions with block power iteration, which is enough for the mean-alignment,
spectral-gap, and stable-rank preprocessing checks.

Outputs:
  results/experiments/dense_readout_diagnostics/mean_direction_metrics.csv
  results/experiments/dense_readout_diagnostics/mean_direction_metrics.pt
  results/experiments/dense_readout_diagnostics/mean_direction_provenance.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.core.model_specs import (  # noqa: E402
    DEFAULT_STEPS_BY_MODEL,
)
from src.core.model_specs import MODEL_HF_NAMES as MODEL_NAMES  # noqa: E402
from src.core.paths import snapshot_path  # noqa: E402
from src.crosscoder.snapshots import load_snapshot  # noqa: E402

OUT_DIR = REPO / "results/experiments/dense_readout_diagnostics"


@dataclass(frozen=True)
class Top2Audit:
    s1: float
    s2: float
    v1: torch.Tensor


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return None


def _x_for_centered_q(
    w: torch.Tensor, q: torch.Tensor, mu: torch.Tensor
) -> torch.Tensor:
    y = w @ q
    return y - (mu @ q).unsqueeze(0)


def _xt_for_centered_z(
    w: torch.Tensor, z: torch.Tensor, mu: torch.Tensor
) -> torch.Tensor:
    return w.T @ z - mu.unsqueeze(1) @ z.sum(dim=0, keepdim=True)


def estimate_top2(
    w: torch.Tensor,
    *,
    mu: torch.Tensor | None,
    q_dim: int,
    n_iter: int,
    seed: int,
) -> Top2Audit:
    """Estimate the top two right singular values/vectors of W or W - mean(W).

    Shape convention: `w` is (vocab, d_model), right singular vectors live in
    d_model space. `q_dim` should be >2 for a small oversampling buffer.
    """

    d_model = w.shape[1]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q = torch.randn(d_model, q_dim, generator=gen, dtype=w.dtype)
    q, _ = torch.linalg.qr(q, mode="reduced")

    for _ in range(n_iter):
        if mu is None:
            z = w @ q
            m = w.T @ z
        else:
            z = _x_for_centered_q(w, q, mu)
            m = _xt_for_centered_z(w, z, mu)
        q, _ = torch.linalg.qr(m, mode="reduced")

    y = w @ q if mu is None else _x_for_centered_q(w, q, mu)
    small_cov = y.T @ y
    evals, evecs = torch.linalg.eigh(small_cov)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    singular_values = torch.sqrt(torch.clamp(evals, min=0))
    v = q @ evecs
    return Top2Audit(
        s1=float(singular_values[0].item()),
        s2=float(singular_values[1].item()),
        v1=v[:, 0].detach().cpu(),
    )


def _mean_pairwise_cosine_from_unit_sum(unit_sum: torch.Tensor, n_rows: int) -> float:
    if n_rows <= 1:
        return float("nan")
    numerator = float(unit_sum.dot(unit_sum).item()) - float(n_rows)
    return numerator / float(n_rows * (n_rows - 1))


def _unit_sum_chunked(
    w: torch.Tensor,
    *,
    mu: torch.Tensor | None,
    row_norm_sq: torch.Tensor | None = None,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.zeros(w.shape[1], dtype=torch.float64)
    eps = torch.finfo(w.dtype).eps
    for start in range(0, w.shape[0], chunk_size):
        chunk = w[start : start + chunk_size]
        if mu is None:
            if row_norm_sq is None:
                norms = chunk.norm(dim=1)
            else:
                norms = torch.sqrt(
                    torch.clamp(row_norm_sq[start : start + chunk_size], min=eps)
                )
            unit = chunk / norms.unsqueeze(1).clamp_min(eps)
        else:
            centered = chunk - mu
            norms = centered.norm(dim=1)
            unit = centered / norms.unsqueeze(1).clamp_min(eps)
        out += unit.double().sum(dim=0)
    return out


def compute_one(
    model_label: str,
    model_name: str,
    step: int,
    *,
    q_dim: int,
    n_iter: int,
    chunk_size: int,
    seed: int,
) -> dict:
    t0 = time.time()
    w = load_snapshot(model_name, step, kind="wu", dtype=torch.float32).contiguous()
    n_rows, d_model = w.shape
    mu = w.mean(dim=0)
    mu_norm = float(mu.norm().item())
    row_norm_sq = (w * w).sum(dim=1)
    row_norm = torch.sqrt(torch.clamp(row_norm_sq, min=0))
    raw_fro_sq = float(row_norm_sq.sum().item())
    raw_median_row_norm = float(row_norm.median().item())

    # ||w_i - mu||^2 = ||w_i||^2 - 2 w_i dot mu + ||mu||^2.
    w_dot_mu = w @ mu
    centered_row_norm_sq = row_norm_sq - 2.0 * w_dot_mu + mu_norm * mu_norm
    centered_row_norm_sq = torch.clamp(centered_row_norm_sq, min=0)
    centered_row_norm = torch.sqrt(centered_row_norm_sq)
    centered_fro_sq = float(centered_row_norm_sq.sum().item())
    centered_median_row_norm = float(centered_row_norm.median().item())
    center_scale = (centered_row_norm_sq.mean() / float(d_model)).sqrt().item()

    raw_unit_sum = _unit_sum_chunked(
        w, mu=None, row_norm_sq=row_norm_sq, chunk_size=chunk_size
    )
    centered_unit_sum = _unit_sum_chunked(
        w, mu=mu, row_norm_sq=None, chunk_size=chunk_size
    )

    raw_top = estimate_top2(
        w, mu=None, q_dim=q_dim, n_iter=n_iter, seed=seed + step + 17
    )
    centered_top = estimate_top2(
        w, mu=mu, q_dim=q_dim, n_iter=n_iter, seed=seed + step + 31
    )

    if mu_norm > 0:
        mu_unit = mu / mu.norm()
        raw_v1_mean_alignment = abs(float(torch.dot(mu_unit, raw_top.v1).item()))
        centered_v1_mean_alignment = abs(
            float(torch.dot(mu_unit, centered_top.v1).item())
        )
    else:
        raw_v1_mean_alignment = float("nan")
        centered_v1_mean_alignment = float("nan")

    row = {
        "model": model_label,
        "model_name": model_name,
        "step": step,
        "vocab": n_rows,
        "d_model": d_model,
        "snapshot_path": str(snapshot_path(model_name, step, kind="wu")),
        "mu_norm": mu_norm,
        "mu_norm_over_raw_median_row_norm": mu_norm / raw_median_row_norm,
        "mu_norm_over_centered_median_row_norm": mu_norm / centered_median_row_norm,
        "raw_median_row_norm": raw_median_row_norm,
        "centered_median_row_norm": centered_median_row_norm,
        "center_scale": float(center_scale),
        "raw_fro_sq": raw_fro_sq,
        "centered_fro_sq": centered_fro_sq,
        "raw_s1": raw_top.s1,
        "raw_s2": raw_top.s2,
        "centered_s1": centered_top.s1,
        "centered_s2": centered_top.s2,
        "raw_spectral_gap": raw_top.s1 / raw_top.s2,
        "centered_spectral_gap": centered_top.s1 / centered_top.s2,
        "raw_stable_rank_top2": raw_fro_sq / (raw_top.s1 * raw_top.s1),
        "centered_stable_rank_top2": centered_fro_sq
        / (centered_top.s1 * centered_top.s1),
        "raw_v1_mean_alignment": raw_v1_mean_alignment,
        "centered_v1_mean_alignment": centered_v1_mean_alignment,
        "raw_mean_pairwise_cos": _mean_pairwise_cosine_from_unit_sum(
            raw_unit_sum, n_rows
        ),
        "centered_mean_pairwise_cos": _mean_pairwise_cosine_from_unit_sum(
            centered_unit_sum, n_rows
        ),
        "top2_method": "block_power_iteration",
        "top2_q_dim": q_dim,
        "top2_n_iter": n_iter,
        "elapsed_sec": time.time() - t0,
    }
    del w, mu, row_norm_sq, row_norm, w_dot_mu, centered_row_norm_sq
    return row


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_models(values: Iterable[str]) -> list[tuple[str, str]]:
    out = []
    for value in values:
        if value not in MODEL_NAMES:
            raise ValueError(
                f"unknown model {value}; expected one of {sorted(MODEL_NAMES)}"
            )
        out.append((value, MODEL_NAMES[value]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pythia-160m", "pythia-1b", "pythia-6.9b", "olmo-2-7b"],
        help="Model labels to audit.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Training steps to audit for all selected models. "
            "Default uses each model's paper cross-snapshot-32 schedule."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--q-dim", type=int, default=6)
    parser.add_argument("--n-iter", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    model_pairs = _parse_models(args.models)
    for model_label, model_name in model_pairs:
        model_steps = (
            list(args.steps)
            if args.steps is not None
            else DEFAULT_STEPS_BY_MODEL[model_label]
        )
        for step in model_steps:
            path = snapshot_path(model_name, step, kind="wu")
            if not path.exists():
                print(f"[skip] missing {model_label} step {step}: {path}", flush=True)
                continue
            print(f"[audit] {model_label} step {step}", flush=True)
            row = compute_one(
                model_label,
                model_name,
                step,
                q_dim=args.q_dim,
                n_iter=args.n_iter,
                chunk_size=args.chunk_size,
                seed=args.seed,
            )
            rows.append(row)
            print(
                "  "
                f"mu/median={row['mu_norm_over_raw_median_row_norm']:.3g} "
                f"align={row['raw_v1_mean_alignment']:.3f} "
                f"gap raw/ctr={row['raw_spectral_gap']:.2f}/{row['centered_spectral_gap']:.2f} "
                f"({row['elapsed_sec']:.1f}s)",
                flush=True,
            )

    csv_path = args.out_dir / "mean_direction_metrics.csv"
    pt_path = args.out_dir / "mean_direction_metrics.pt"
    provenance_path = args.out_dir / "mean_direction_provenance.json"
    _write_csv(rows, csv_path)
    torch.save({"rows": rows}, pt_path)
    provenance = {
        "script": str(Path(__file__).resolve()),
        "git_commit": _git_commit(),
        "models": args.models,
        "steps": args.steps,
        "steps_by_model": {
            model_label: list(args.steps)
            if args.steps is not None
            else DEFAULT_STEPS_BY_MODEL[model_label]
            for model_label, _ in model_pairs
        },
        "top2_method": "block_power_iteration",
        "top2_q_dim": args.q_dim,
        "top2_n_iter": args.n_iter,
        "chunk_size": args.chunk_size,
        "outputs": {
            "csv": str(csv_path),
            "pt": str(pt_path),
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {pt_path}")
    print(f"[done] wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
