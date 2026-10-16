"""Language-stratified fixed-hidden-state readout swaps.

Repeats the raw (alignment ``none``) readout-swap grid of
``temporal_localization_patching`` while retaining per-language and
consecutive-block sufficient statistics. The statistics support paired
bootstrap intervals and row-wise argmin probabilities without saving
token-by-vocabulary logits.

The aggregate ``all`` stratum includes every scored target and is regression
checked against the raw swap grid when ``--reference-summary`` exists (the
``summary_global.csv`` written by ``temporal_patch_grid.py``). Per-language
strata exclude targets whose context token lies in the preceding language
slice of the concatenated corpus.

Usage:
    uv run python experiments/causal/language_stratified_readout_swap/scripts/run_language_stratified_swap.py \\
        --model pythia-1b --seed 0 \\
        --h-eval-steps 256 512 1000 2000 3000 14000 47000 143000 \\
        --out-dir results/experiments/causal/language_stratified_readout_swap/pythia-1b_seed0
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import readout.dynamics.temporal_patch as TPM
from readout.core.hf_revisions import resolve_revision
from readout.core.repro import git_commit, log_run_provenance, seed_everything
from readout.core.resume import atomic_write_json, atomic_write_torch

DEFAULT_H_STEPS = [256, 512, 1000, 2000, 3000, 14000, 47000, 143000]
LANGUAGE_ORDER = ["en", "ru", "zh", "ja", "th", "ar", "hi", "ko", "bn"]
MODEL_CFGS = {
    "pythia-160m": TPM.CFG_PYTHIA_160M,
    "pythia-1b": TPM.CFG_PYTHIA_1B,
    "pythia-6.9b": TPM.CFG_PYTHIA_6_9B,
}
# Max |NLL difference| for the `all` stratum to count as matching the reference grid.
PARITY_TOLERANCE = 1e-5


@dataclass
class BlockLayout:
    ids_seqs: torch.Tensor  # (n_seqs, seq_len)
    all_block_ids: torch.Tensor  # (n_seqs, seq_len - 1) block id of each target in `all`
    language_block_ids: torch.Tensor  # (n_seqs, seq_len - 1) language block id, -1 = excluded
    counts: torch.Tensor  # (n_blocks_total,) targets per block
    block_language: list[str]
    block_index: list[int]
    language_slices: dict[str, tuple[int, int]]  # stratum -> [start, stop) block range
    n_cross_language_targets: int


def load_readout(ctx: TPM.PatchContext, step: int, *, cache_dir: Path | None) -> torch.Tensor:
    """Load an exact W_U snapshot, extracting it from Hugging Face if absent.

    Canonical snapshots under ``${UM_SSD_ROOT}/snapshots`` are preferred. When
    they are missing (e.g. on a cluster without the snapshot mirror),
    ``cache_dir`` enables a bounded fallback: download one revision, keep only
    its float32 readout, and remove the temporary full-model download.
    """
    try:
        return TPM.load_snapshot(ctx, step).float()
    except FileNotFoundError as canonical_error:
        if cache_dir is None:
            raise FileNotFoundError(
                f"canonical readout snapshot for step {step} is missing; "
                "pass --readout-cache-dir to enable exact Hugging Face extraction"
            ) from canonical_error

    cache_dir.mkdir(parents=True, exist_ok=True)
    readout_path = cache_dir / f"W_U_step{step}.pt"
    if readout_path.exists():
        payload = torch.load(readout_path, map_location="cpu", weights_only=False)
        readout = payload["W_U"] if isinstance(payload, dict) else payload
        return readout.float()

    revision = resolve_revision(ctx.model_name, step)
    download_dir = cache_dir / f".hf-step{step}"
    print(
        f"  [readout cache] extracting {ctx.model_name} {revision} -> {readout_path.name}",
        flush=True,
    )
    from transformers import AutoModelForCausalLM

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            ctx.model_name,
            revision=revision,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            cache_dir=download_dir,
        )
        readout = TPM._resolve_readout_module(model).weight.detach().cpu().to(torch.float32).clone()
        atomic_write_torch(
            readout_path,
            {
                "W_U": readout,
                "model": ctx.model_name,
                "revision": revision,
                "snapshot_step": step,
            },
        )
    finally:
        del model
        shutil.rmtree(download_dir, ignore_errors=True)
    return readout


def build_block_layout(
    ids: torch.Tensor,
    languages: list[str],
    *,
    seq_len: int,
    block_tokens: int,
) -> BlockLayout:
    """Align target-token language labels and build bootstrap blocks.

    The raw swap grid truncates the flat corpus to complete ``seq_len``
    windows and scores positions 1..L-1 in each window; the ``all`` blocks
    preserve that convention exactly. Per-language blocks omit a target when
    its preceding token belongs to another language slice.
    """
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")
    if len(ids) != len(languages):
        raise ValueError(f"ids/languages length mismatch: {len(ids)} != {len(languages)}")

    n_full = (len(ids) // seq_len) * seq_len
    ids_seqs = ids[:n_full].view(-1, seq_len)  # (n_seqs, seq_len)
    lang_arr = np.asarray(languages[:n_full], dtype=object).reshape(-1, seq_len)
    target_lang = lang_arr[:, 1:].reshape(-1)  # (n_seqs * (seq_len - 1),)
    context_lang = lang_arr[:, :-1].reshape(-1)
    same_language = target_lang == context_lang

    n_predictions = int(target_lang.size)
    all_ids_flat = torch.arange(n_predictions, dtype=torch.long) // block_tokens
    n_all_blocks = int(all_ids_flat.max().item()) + 1
    all_block_ids = all_ids_flat.view(ids_seqs.shape[0], seq_len - 1)

    language_ids_flat = torch.full((n_predictions,), -1, dtype=torch.long)
    block_language = ["all"] * n_all_blocks
    block_index = list(range(n_all_blocks))
    language_slices: dict[str, tuple[int, int]] = {"all": (0, n_all_blocks)}
    offset = n_all_blocks

    observed = set(target_lang.tolist())
    ordered_languages = [x for x in LANGUAGE_ORDER if x in observed]
    ordered_languages.extend(sorted(observed - set(ordered_languages)))
    for language in ordered_languages:
        positions = np.flatnonzero(same_language & (target_lang == language))
        n_blocks = math.ceil(len(positions) / block_tokens)
        start, stop = offset, offset + n_blocks
        language_slices[language] = (start, stop)
        if len(positions):
            local_blocks = torch.arange(len(positions), dtype=torch.long) // block_tokens
            language_ids_flat[torch.from_numpy(positions)] = local_blocks + start
        block_language.extend([language] * n_blocks)
        block_index.extend(range(n_blocks))
        offset = stop

    language_block_ids = language_ids_flat.view(ids_seqs.shape[0], seq_len - 1)
    counts = torch.bincount(all_block_ids.reshape(-1), minlength=offset)
    valid_language = language_block_ids.reshape(-1)
    valid_language = valid_language[valid_language >= 0]
    counts += torch.bincount(valid_language, minlength=offset)

    return BlockLayout(
        ids_seqs=ids_seqs,
        all_block_ids=all_block_ids,
        language_block_ids=language_block_ids,
        counts=counts,
        block_language=block_language,
        block_index=block_index,
        language_slices=language_slices,
        n_cross_language_targets=int((~same_language).sum()),
    )


def evaluate_readout_by_block(
    h_ln: torch.Tensor,
    readout: torch.Tensor,
    layout: BlockLayout,
    *,
    device: str,
    batch_seqs: int,
) -> torch.Tensor:
    """Return float64 NLL sums for every bootstrap block, shape (n_blocks_total,)."""
    if h_ln.shape[:2] != layout.ids_seqs.shape:
        raise ValueError(f"hidden/corpus shape mismatch: {tuple(h_ln.shape[:2])} != {tuple(layout.ids_seqs.shape)}")
    n_blocks = len(layout.block_language)
    block_sums = torch.zeros(n_blocks, dtype=torch.float64)
    readout_device = readout.to(device=device, dtype=torch.float32)  # (V, d_model)

    for start in range(0, h_ln.shape[0], batch_seqs):
        stop = min(start + batch_seqs, h_ln.shape[0])
        hidden = h_ln[start:stop].to(device=device, dtype=torch.float32)  # (b, seq_len, d_model)
        targets = layout.ids_seqs[start:stop, 1:].to(device)  # (b, seq_len - 1)

        logits = hidden[:, :-1] @ readout_device.T  # (b, seq_len - 1, V)
        target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        nll = (torch.logsumexp(logits, dim=-1) - target_logits).detach().cpu()
        weights = nll.to(torch.float64).reshape(-1)

        all_ids = layout.all_block_ids[start:stop].reshape(-1)
        block_sums += torch.bincount(all_ids, weights=weights, minlength=n_blocks)

        language_ids = layout.language_block_ids[start:stop].reshape(-1)
        valid = language_ids >= 0
        block_sums += torch.bincount(language_ids[valid], weights=weights[valid], minlength=n_blocks)

        del hidden, targets, logits, target_logits, nll, weights
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return block_sums


def bootstrap_mean(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    """95% percentile interval for a block-resampled ratio of sums."""
    if numerator.ndim != 1 or denominator.ndim != 1:
        raise ValueError("bootstrap arrays must be one-dimensional")
    if len(numerator) != len(denominator) or not len(numerator):
        raise ValueError("bootstrap arrays must have equal non-zero length")
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 256):
        stop = min(start + 256, reps)
        draw = rng.integers(0, len(numerator), size=(stop - start, len(numerator)))  # (chunk, n_blocks)
        values[start:stop] = numerator[draw].sum(axis=1) / denominator[draw].sum(axis=1)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def stable_seed(base: int, h_step: int, s_step: int, language: str) -> int:
    """Process-stable 32-bit bootstrap seed per (hidden step, readout step, stratum)."""
    language_code = sum((i + 1) * ord(c) for i, c in enumerate(language))
    return int((base * 1_000_003 + h_step * 1009 + s_step * 9176 + language_code) % (2**32))


def summarize_cell(
    *,
    model: str,
    seed: int,
    h_step: int,
    s_step: int,
    cell_sums: torch.Tensor,
    native_sums: torch.Tensor,
    layout: BlockLayout,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> list[dict]:
    """One row per stratum: NLL, native NLL, and paired ΔNLL with bootstrap interval."""
    rows: list[dict] = []
    counts = layout.counts.numpy().astype(np.float64)
    cell = cell_sums.numpy()
    native = native_sums.numpy()
    for language, (start, stop) in layout.language_slices.items():
        den = counts[start:stop]
        nll_num = cell[start:stop]
        native_num = native[start:stop]
        delta_num = nll_num - native_num
        delta_lo, delta_hi = bootstrap_mean(
            delta_num,
            den,
            reps=bootstrap_reps,
            seed=stable_seed(bootstrap_seed, h_step, s_step, language),
        )
        n_tokens = int(den.sum())
        nll = float(nll_num.sum() / n_tokens)
        native_nll = float(native_num.sum() / n_tokens)
        rows.append(
            {
                "model": model,
                "seed": seed,
                "h_eval_step": h_step,
                "snapshot_step": s_step,
                "language": language,
                "n_tokens": n_tokens,
                "n_blocks": stop - start,
                "nll": nll,
                "nll_native": native_nll,
                "delta_nll": nll - native_nll,
                "delta_nll_ci_low": delta_lo,
                "delta_nll_ci_high": delta_hi,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_cell_json(shard_dir: Path, h_steps: list[int], snap_steps: list[int]) -> list[dict]:
    rows: list[dict] = []
    for h_step in h_steps:
        for s_step in snap_steps:
            payload = json.loads((shard_dir / f"h{h_step}_s{s_step}.json").read_text())
            if not isinstance(payload, list):
                raise TypeError(f"expected list payload in h{h_step}_s{s_step}.json")
            rows.extend(payload)
    rows.sort(key=lambda r: (r["language"], r["h_eval_step"], r["snapshot_step"]))
    return rows


def argmin_probabilities(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap probability that each readout attains the minimum mean NLL.

    ``sums`` is (n_readouts, n_blocks) per-block NLL sums and ``counts`` is
    (n_blocks,) targets per block; every readout is resampled with the same
    block draw (paired). Ties go to the first readout, as in ``np.argmin``.
    """
    win_counts = np.zeros(sums.shape[0], dtype=np.int64)
    n_blocks = sums.shape[1]
    for rep_start in range(0, reps, 128):
        rep_stop = min(rep_start + 128, reps)
        draw = rng.integers(0, n_blocks, size=(rep_stop - rep_start, n_blocks))  # (chunk, n_blocks)
        sampled_sums = sums[:, draw].sum(axis=2)  # (n_readouts, chunk)
        sampled_counts = counts[draw].sum(axis=1)  # (chunk,)
        sampled_means = sampled_sums / sampled_counts[None, :]
        winners = np.argmin(sampled_means, axis=0)
        win_counts += np.bincount(winners, minlength=sums.shape[0])
    return win_counts / reps


def argmin_bootstrap(
    *,
    out_dir: Path,
    model: str,
    seed: int,
    h_steps: list[int],
    snap_steps: list[int],
    layout: BlockLayout,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    probability_rows: list[dict] = []
    counts_all = layout.counts.numpy().astype(np.float64)

    for h_step in h_steps:
        cell_arrays = [
            torch.load(
                out_dir / "cells" / f"h{h_step}_s{s_step}.pt",
                map_location="cpu",
                weights_only=False,
            )["block_nll_sums"].numpy()
            for s_step in snap_steps
        ]
        stacked = np.stack(cell_arrays, axis=0)  # (n_readouts, n_blocks_total)
        for language, (start, stop) in layout.language_slices.items():
            sums = stacked[:, start:stop]
            counts = counts_all[start:stop]
            observed = sums.sum(axis=1) / counts.sum()
            observed_best_index = int(np.argmin(observed))
            observed_best_step = snap_steps[observed_best_index]
            native_index = snap_steps.index(h_step)

            rng = np.random.default_rng(stable_seed(bootstrap_seed + 1, h_step, 0, language))
            probabilities = argmin_probabilities(sums, counts, reps=bootstrap_reps, rng=rng)
            summary_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "h_eval_step": h_step,
                    "language": language,
                    "best_snapshot_step": observed_best_step,
                    "best_nll": float(observed[observed_best_index]),
                    "native_nll": float(observed[native_index]),
                    "native_minus_best_nll": float(observed[native_index] - observed[observed_best_index]),
                    "best_argmin_probability": float(probabilities[observed_best_index]),
                    "native_argmin_probability": float(probabilities[native_index]),
                    "n_tokens": int(counts.sum()),
                    "n_blocks": stop - start,
                }
            )
            for s_step, nll, probability in zip(snap_steps, observed, probabilities, strict=True):
                probability_rows.append(
                    {
                        "h_eval_step": h_step,
                        "language": language,
                        "snapshot_step": s_step,
                        "nll": float(nll),
                        "argmin_probability": float(probability),
                    }
                )
    return summary_rows, probability_rows


def reference_parity(reference_path: Path, summary_rows: list[dict], *, model: str) -> dict:
    """Compare the `all` stratum with a raw swap-grid summary on overlapping cells.

    Accepts ``temporal_patch_grid.py``'s ``summary_global.csv`` (no alignment
    column) or ``run_aligned_swap_grid.py``'s ``summary.csv`` (``alignment ==
    none`` rows only). The two grids must be built on the same corpus: the
    aligned grid's reference runs cover only a prefix of it and will not match.
    """
    if not reference_path.exists():
        return {"status": "not_checked", "reason": f"missing {reference_path}"}
    ours = {
        (int(r["h_eval_step"]), int(r["snapshot_step"])): float(r["nll"])
        for r in summary_rows
        if r["language"] == "all"
    }
    diffs: list[dict] = []
    with reference_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("alignment", "none") != "none":
                continue
            if row.get("model", model) != model:
                continue
            key = (int(row["h_eval_step"]), int(row["snapshot_step"]))
            if key not in ours:
                continue
            diffs.append(
                {
                    "h_eval_step": key[0],
                    "snapshot_step": key[1],
                    "difference": ours[key] - float(row["nll"]),
                }
            )
    if not diffs:
        return {"status": "not_checked", "reason": "no overlapping raw-swap cells"}
    max_abs = max(abs(x["difference"]) for x in diffs)
    return {
        "status": "pass" if max_abs < PARITY_TOLERANCE else "fail",
        "reference_summary": str(reference_path),
        "n_overlapping_cells": len(diffs),
        "max_abs_nll_difference": max_abs,
        "tolerance": PARITY_TOLERANCE,
        "largest_differences": sorted(diffs, key=lambda x: abs(x["difference"]), reverse=True)[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(MODEL_CFGS), default="pythia-1b")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Run seed (seed_everything) and the seed label written to every row.",
    )
    parser.add_argument("--h-eval-steps", type=int, nargs="+", default=DEFAULT_H_STEPS)
    parser.add_argument(
        "--snap-steps",
        type=int,
        nargs="+",
        default=None,
        help="Readout checkpoints; default = the model's 32 canonical steps.",
    )
    parser.add_argument("--batch-seqs", type=int, default=4)
    parser.add_argument("--device", default=TPM.DEVICE)
    parser.add_argument("--block-tokens", type=int, default=256)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--eval-tokens",
        type=Path,
        default=TPM.CORPUS_TOKENS_PYTHIA,
        help="Eval corpus with per-token `languages` (default: the released eval_tokens.pt).",
    )
    parser.add_argument(
        "--hln-cache-dir",
        type=Path,
        default=None,
        help="Hidden-state cache directory (default: the model config's hLN_cache_dir).",
    )
    parser.add_argument(
        "--readout-cache-dir",
        type=Path,
        default=None,
        help=("Cache for exact W_U matrices extracted from Hugging Face when canonical snapshots are unavailable."),
    )
    parser.add_argument(
        "--delete-hln-cache-after-step",
        action="store_true",
        help="Delete each h_LN cache file after all readout cells for that step finish.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/experiments/causal/language_stratified_readout_swap/pythia-1b_seed0"),
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=Path(
            "results/experiments/causal/temporal_localization_patching/"
            "temporal_patch_grid/pythia-1b_seed0/summary_global.csv"
        ),
        help="temporal_patch_grid.py summary_global.csv for the `all`-stratum parity check.",
    )
    args = parser.parse_args()
    if args.bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive")
    seed_everything(args.seed)
    provenance = log_run_provenance(args.seed)

    model_cfg = MODEL_CFGS[args.model]
    if args.hln_cache_dir is not None:
        model_cfg = dataclasses.replace(model_cfg, hLN_cache_dir=args.hln_cache_dir)
    # d_sae is unused here (no crosscoder is loaded); PatchContext requires one.
    ctx = TPM.PatchContext(cfg=model_cfg, d_sae=0, corpus_tokens=args.eval_tokens)
    cfg = ctx.cfg
    snap_steps = args.snap_steps or list(cfg.steps_canonical)
    missing_native = sorted(set(args.h_eval_steps) - set(snap_steps))
    if missing_native:
        raise ValueError(
            f"all hidden-state steps must be present in snap-steps for the native baseline; missing={missing_native}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    cell_dir = args.out_dir / "cells"
    native_dir = args.out_dir / "native"
    for directory in (shard_dir, cell_dir, native_dir):
        directory.mkdir(parents=True, exist_ok=True)

    eval_data = torch.load(ctx.corpus_tokens, map_location="cpu", weights_only=False)
    if "languages" not in eval_data:
        raise KeyError(f"{ctx.corpus_tokens} has no per-token `languages` list")
    layout = build_block_layout(
        eval_data["ids"],
        eval_data["languages"],
        seq_len=TPM.SEQ_LEN,
        block_tokens=args.block_tokens,
    )
    atomic_write_torch(
        args.out_dir / "layout.pt",
        {
            "all_block_ids": layout.all_block_ids,
            "language_block_ids": layout.language_block_ids,
            "counts": layout.counts,
            "block_language": layout.block_language,
            "block_index": layout.block_index,
            "language_slices": layout.language_slices,
            "n_cross_language_targets": layout.n_cross_language_targets,
        },
    )

    manifest = {
        "experiment_id": "language_stratified_readout_swap",
        "model": cfg.model_name,
        "seed": args.seed,
        "alignment": "none",
        "h_eval_steps": args.h_eval_steps,
        "snapshot_steps": snap_steps,
        "corpus": str(ctx.corpus_tokens),
        "corpus_sha1": TPM._corpus_fingerprint(layout.ids_seqs),
        "n_eval_sequences": int(layout.ids_seqs.shape[0]),
        "seq_len": TPM.SEQ_LEN,
        "n_all_targets": int(layout.counts[slice(*layout.language_slices["all"])].sum()),
        "n_cross_language_targets_excluded_from_language_strata": layout.n_cross_language_targets,
        "language_token_counts": {
            language: int(layout.counts[start:stop].sum())
            for language, (start, stop) in layout.language_slices.items()
            if language != "all"
        },
        "block_tokens": args.block_tokens,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "batch_seqs": args.batch_seqs,
        "device": args.device,
        "hln_cache_dir": str(cfg.hLN_cache_dir),
        "readout_cache_dir": str(args.readout_cache_dir) if args.readout_cache_dir else None,
        "delete_hln_cache_after_step": args.delete_hln_cache_after_step,
        "reference_summary": str(args.reference_summary),
        "git_commit": git_commit(),
        "provenance": provenance,
        "command": sys.argv,
    }
    atomic_write_json(args.out_dir / "manifest.json", manifest)

    print(
        f"[setup] {len(args.h_eval_steps)} hidden steps x {len(snap_steps)} "
        f"readouts = {len(args.h_eval_steps) * len(snap_steps)} cells; "
        f"{manifest['n_all_targets']:,} targets; "
        f"languages={list(manifest['language_token_counts'])}",
        flush=True,
    )

    started = time.time()
    for h_step in args.h_eval_steps:
        undone = [s_step for s_step in snap_steps if not (shard_dir / f"h{h_step}_s{s_step}.json").exists()]
        if not undone:
            print(f"[resume] h={h_step}: all {len(snap_steps)} cells complete", flush=True)
            continue

        print(f"[hidden] h={h_step}: loading/building cache; {len(undone)} cells remain", flush=True)
        hidden_payload = TPM.cache_or_build_hLN(ctx, h_step, layout.ids_seqs)
        h_ln = hidden_payload["h_LN"]  # (n_seqs, seq_len, d_model)
        native_readout = hidden_payload.get("W_U_orig")
        if native_readout is None:
            native_readout = load_readout(ctx, h_step, cache_dir=args.readout_cache_dir)

        native_path = native_dir / f"h{h_step}.pt"
        if native_path.exists():
            native_sums = torch.load(native_path, map_location="cpu", weights_only=False)["block_nll_sums"]
        else:
            print(f"[native] h={h_step}", flush=True)
            native_sums = evaluate_readout_by_block(
                h_ln,
                native_readout,
                layout,
                device=args.device,
                batch_seqs=args.batch_seqs,
            )
            atomic_write_torch(native_path, {"h_eval_step": h_step, "block_nll_sums": native_sums})

        for s_step in undone:
            cell_json = shard_dir / f"h{h_step}_s{s_step}.json"
            cell_pt = cell_dir / f"h{h_step}_s{s_step}.pt"
            print(f"[cell] h={h_step:>6} readout={s_step:>6}", flush=True)
            readout = load_readout(ctx, s_step, cache_dir=args.readout_cache_dir)
            cell_sums = evaluate_readout_by_block(
                h_ln,
                readout,
                layout,
                device=args.device,
                batch_seqs=args.batch_seqs,
            )
            atomic_write_torch(
                cell_pt,
                {"h_eval_step": h_step, "snapshot_step": s_step, "block_nll_sums": cell_sums},
            )
            rows = summarize_cell(
                model=cfg.model_name,
                seed=args.seed,
                h_step=h_step,
                s_step=s_step,
                cell_sums=cell_sums,
                native_sums=native_sums,
                layout=layout,
                bootstrap_reps=args.bootstrap_reps,
                bootstrap_seed=args.bootstrap_seed,
            )
            atomic_write_json(cell_json, rows)
            all_row = rows[0]
            print(f"       all NLL={all_row['nll']:.5f} (delta={all_row['delta_nll']:+.5f})", flush=True)
            del readout, cell_sums
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        del h_ln, native_readout, native_sums, hidden_payload
        if args.delete_hln_cache_after_step:
            hln_path = cfg.hLN_cache_dir / f"hLN_step{h_step}.pt"
            hln_path.unlink(missing_ok=True)
            print(f"  [scratch] deleted completed h_LN cache {hln_path}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    expected = len(args.h_eval_steps) * len(snap_steps)
    completed = sum((shard_dir / f"h{h}_s{s}.json").exists() for h in args.h_eval_steps for s in snap_steps)
    if completed != expected:
        print(f"[partial] {completed}/{expected} cells complete; rerun to resume")
        return

    summary_rows = aggregate_cell_json(shard_dir, args.h_eval_steps, snap_steps)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(
        args.out_dir / "english_only_summary.csv",
        [row for row in summary_rows if row["language"] == "en"],
    )
    argmin_rows, probability_rows = argmin_bootstrap(
        out_dir=args.out_dir,
        model=cfg.model_name,
        seed=args.seed,
        h_steps=args.h_eval_steps,
        snap_steps=snap_steps,
        layout=layout,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_csv(args.out_dir / "argmin_summary.csv", argmin_rows)
    write_csv(args.out_dir / "argmin_probabilities.csv", probability_rows)
    atomic_write_json(
        args.out_dir / "reference_parity.json",
        reference_parity(args.reference_summary, summary_rows, model=cfg.model_name),
    )
    atomic_write_json(
        args.out_dir / "manifest.json",
        {
            **manifest,
            "status": "complete",
            "elapsed_s": round(time.time() - started, 2),
            "n_summary_rows": len(summary_rows),
        },
    )
    print(
        f"[done] {completed} cells; outputs in {args.out_dir}; elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
