"""Lifecycle statistics across dictionary fits (tab:app-lifecycle-stability).

Recomputes the refined profile fractions (tab:lifecycle-profile-rules) and the
reorganization window (sec:app-temporal-localization) for all 21 Pythia
dictionaries fitted on the 32-checkpoint schedule, with the functions that
produce the selected-dictionary numbers: readout.dynamics.lifecycle.profile_fractions
and pair_metrics / window_metrics / summarize_windows from the sibling
find_reorganization_steps.py, under the same null seed convention
(RNG_SEED + sum(ord(c) for c in key)).

Stages (``--stage``, default ``all``), each writing one CSV under ``--out-dir``:
  validate    the four selected dictionaries from the lifecycle caches (the window
              script's inputs), beside the regenerated selected-dictionary CSVs in
              ``--reference-dir``; run plot_lifecycle_profile_composition.py and
              find_reorganization_steps.py first              -> validation.csv
  pythia1b    widths 8192 / 16384 / 24576, seed 0              -> pythia1b_widths.csv
  pythia160m  cross-snapshot-32 widths and seeds               -> pythia160m_widths_seeds.csv
  lambda      the d8192 lambda sweep                           -> pythia160m_lambda_sweep.csv
  pythia69b   d32768 at the default lambda 0.3                 -> pythia69b_sparsity.csv
The selected Pythia-6.9B row (lambda 0.6, seed0-sparse) is the validate stage's.

Per fit, decoder norms and decoder geometry come from the released checkpoint's
W_D, and activation rates are the canonical rates (compute_rates_canonical) from
the checkpoint and the W_U snapshots. Where an SSD aggregate holds the fit's norms
and rates (Pythia-1B, Pythia-6.9B, whose encoders are too large to rerun on a
laptop), those are read from it instead; the release norms are then only checked
against them.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import find_reorganization_steps as find
import numpy as np
import torch
from lifecycle_common import PYTHIA_STEPS_32, Run, _geometry_from_decoder
from safetensors import safe_open

from readout.core.data import write_csv
from readout.core.paths import aggregate_path, release_path, release_root, snapshot_dir, snapshot_path
from readout.core.repro import log_run_provenance, seed_everything
from readout.crosscoder.extract_rates import compute_rates_canonical
from readout.crosscoder.inference import reconstruct_preprocess_stats
from readout.crosscoder.snapshots import load_snapshot_at
from readout.dynamics.lifecycle import profile_fractions

OUT = find.OUT / "lifecycle_stability"
GEOMETRY_CHUNK = 1024  # features per W_D slice; bounds memory for the 17 GB 6.9B decoder
MODEL_NAMES = {"pythia-160m": "EleutherAI/pythia-160m", "pythia-1b": "EleutherAI/pythia-1b"}
STAGE_FILES = {
    "validate": "validation.csv",
    "pythia1b": "pythia1b_widths.csv",
    "pythia160m": "pythia160m_widths_seeds.csv",
    "lambda": "pythia160m_lambda_sweep.csv",
    "pythia69b": "pythia69b_sparsity.csv",
}


@dataclass(frozen=True)
class Fit:
    stage: str
    key: str  # null-seed key; the selected dictionaries keep their window-script key
    model: str  # release model dir
    d_sae: int
    seed: int
    checkpoint: Path
    aggregate: Path | None = None  # norms + rates source; None -> derived from checkpoint + snapshots
    lam: float | None = None  # reported only for the sparsity-sweep stages


def _lambda_tag(lam: float) -> str:
    return f"lam{lam:.2f}".replace(".", "p")  # 0.4 -> lam0p40


def build_fits() -> tuple[Fit, ...]:
    """The 20 non-selected-sparse Pythia fits; the 21st (6.9B seed0-sparse) is in validate."""
    fits = [
        Fit(
            "pythia1b",
            f"pythia1b_d{d}",
            "pythia-1b",
            d,
            0,
            release_path("pythia-1b", dim=d, seed=0),
            aggregate=aggregate_path(f"pythia-1b_d{d}_seed0"),
        )
        for d in (8192, 16384, 24576)
    ]
    for d, seed in ((24576, 0), (24576, 1), (24576, 2), (16384, 0), *((8192, s) for s in range(5))):
        key = "pythia160m_d24576" if (d, seed) == (24576, 0) else f"pythia160m_d{d}_seed{seed}"
        fits.append(Fit("pythia160m", key, "pythia-160m", d, seed, release_path("pythia-160m", dim=d, seed=seed)))
    sweep = release_root() / "pythia-160m" / "W_U" / "lambda-sweep" / "d8192"
    for lam, seed in ((0.4, 0), (1.0, 0), (1.2, 0), (1.35, 0), (1.35, 1), (1.35, 2), (1.8, 0)):
        tag = _lambda_tag(lam)
        fits.append(
            Fit(
                "lambda",
                f"pythia160m_d8192_{tag}_seed{seed}",
                "pythia-160m",
                8192,
                seed,
                sweep / f"{tag}_seed{seed}.safetensors",
                lam=lam,
            )
        )
    fits.append(
        Fit(
            "pythia69b",
            "pythia69b_d32768_default",
            "pythia-6.9b",
            32768,
            0,
            release_path("pythia-6.9b", dim=32768, seed=0),
            aggregate=aggregate_path("pythia-6.9b_d32768_seed0"),
            lam=0.3,
        )
    )
    return tuple(fits)


def window_summary(key: str, label: str, norms, rotation, rates, cos_terminal, steps) -> dict[str, object]:
    """Selected reorganization window with the paper's null seed for ``key``."""
    metrics = find.pair_metrics(norms, rotation, rates, cos_terminal)
    windows = find.window_metrics(steps, metrics, null_seed=find.RNG_SEED + sum(ord(c) for c in key))
    run = Run(key=key, label=label, rates_path=Path(), steps=steps, xticks=())
    summary = find.summarize_windows(run, windows)
    return {
        "window_start": summary["primary_start_step"],
        "window_end": summary["primary_end_step"],
        "null_p": summary["primary_null_p"],
    }


def release_decoder_arrays(path: Path, chunk: int = GEOMETRY_CHUNK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decoder norms (K, D), adjacent rotation in radians (K-1, D), cos-to-terminal (K, D).

    Reads W_D (K, D, d) in feature slices; the geometry uses the lifecycle cache's
    conventions (lifecycle_common._geometry_from_decoder).
    """
    with safe_open(str(path), framework="pt") as f:
        w_d = f.get_slice("W_D")
        K, D, _ = w_d.get_shape()
        norms = np.empty((K, D), dtype=np.float32)
        rotation = np.empty((K - 1, D), dtype=np.float32)
        cos_terminal = np.empty((K, D), dtype=np.float32)
        for start in range(0, D, chunk):
            end = min(start + chunk, D)
            block = w_d[:, start:end, :].float()  # (K, C, d)
            norms[:, start:end] = torch.linalg.norm(block, dim=-1).numpy()
            rotation[:, start:end], cos_terminal[:, start:end] = _geometry_from_decoder(block, chunk=chunk)
    return norms, rotation, cos_terminal


_STATS_CACHE: dict[str, dict] = {}


def model_preprocess_stats(model: str, steps: np.ndarray) -> dict:
    """center_scale stats rebuilt once per model from its snapshots (shared by all its fits)."""
    if model not in _STATS_CACHE:
        name = MODEL_NAMES[model]
        snaps = (load_snapshot_at(snapshot_path(name, int(step))) for step in steps)
        _STATS_CACHE[model] = reconstruct_preprocess_stats(snaps)
    return _STATS_CACHE[model]


def fit_arrays(fit: Fit, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norms, rotation, cos_terminal = release_decoder_arrays(fit.checkpoint)
    if fit.aggregate is not None:
        blob = torch.load(fit.aggregate, map_location="cpu", weights_only=False)
        agg_norms = blob["decoder_norms"].float().numpy()  # (K, D)
        print(f"[{fit.key}] |release - aggregate| decoder norms: {np.abs(norms - agg_norms).max():.2e}", flush=True)
        return agg_norms, rotation, blob["rates"].float().numpy(), cos_terminal
    stats = model_preprocess_stats(fit.model, PYTHIA_STEPS_32)
    rates, steps = compute_rates_canonical(
        fit.checkpoint, snapshot_dir(MODEL_NAMES[fit.model]), device=device, preprocess_stats=stats
    )
    if list(steps) != PYTHIA_STEPS_32.tolist():
        raise ValueError(f"{fit.checkpoint}: steps {steps} are not the 32-checkpoint schedule")
    return norms, rotation, rates.numpy(), cos_terminal


def fit_row(fit: Fit, device: str) -> dict[str, object]:
    norms, rotation, rates, cos_terminal = fit_arrays(fit, device)
    prof = profile_fractions(norms, PYTHIA_STEPS_32)
    win = window_summary(fit.key, fit.key, norms, rotation, rates, cos_terminal, PYTHIA_STEPS_32)
    row: dict[str, object] = {"model": fit.model, "d_sae": fit.d_sae}
    if fit.lam is not None:
        row["lambda"] = fit.lam
    row["seed"] = fit.seed
    row.update({k: round(v, 4) for k, v in prof.items()})
    row.update(win)
    print(row, flush=True)
    return row


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def validation_rows(reference_dir: Path) -> list[dict[str, object]]:
    """Selected dictionaries through the window script's own loaders, beside its regenerated CSVs."""
    ref_prof = {
        (r["model"], r["profile"]): float(r["fraction"])
        for r in read_csv(reference_dir / "selected_lifecycle_profile_composition_refined_summary.csv")
    }
    ref_win = {r["model"]: r for r in read_csv(reference_dir / "reorganization_window_summary_selected.csv")}
    rows = []
    for run in find.RUNS:
        norms, rotation, rates, cos_terminal = find.load_arrays(run)
        prof = profile_fractions(norms, run.steps)
        win = window_summary(run.key, run.label, norms, rotation, rates, cos_terminal, run.steps)
        ref = ref_win[run.key]
        row: dict[str, object] = {"model": run.key}
        for name in ("early_decay", "persistent", "late_emerge"):  # Table E.3 column order
            row[name] = round(prof[name], 4)
            row[f"reference_{name}"] = round(ref_prof[(run.key, name)], 4)
        row["window"] = f"{win['window_start']}-{win['window_end']}"
        row["reference_window"] = f"{ref['primary_start_step']}-{ref['primary_end_step']}"
        row["null_p"] = round(win["null_p"], 4)
        row["reference_null_p"] = round(float(ref["primary_null_p"]), 4)
        print(row, flush=True)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=(*STAGE_FILES, "all"), default="all")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--reference-dir", type=Path, default=find.OUT)
    ap.add_argument("--device", default="cpu", help="device for the canonical-rate encoder pass")
    args = ap.parse_args()

    seed_everything(find.RNG_SEED)
    log_run_provenance(seed=find.RNG_SEED)
    torch.set_grad_enabled(False)
    stages = tuple(STAGE_FILES) if args.stage == "all" else (args.stage,)
    fits = build_fits()
    for stage in stages:
        if stage == "validate":
            rows = validation_rows(args.reference_dir)
        else:
            rows = [fit_row(fit, args.device) for fit in fits if fit.stage == stage]
        write_csv(args.out_dir / STAGE_FILES[stage], rows, require_rows=True)
        print(f"wrote {args.out_dir / STAGE_FILES[stage]}", flush=True)


if __name__ == "__main__":
    main()
