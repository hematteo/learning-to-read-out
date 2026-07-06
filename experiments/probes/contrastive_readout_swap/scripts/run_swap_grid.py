"""Contrastive readout-swap heatmap: (h_step, s_step) per task family.

For each (h_step, s_step, alignment) we compute the contrastive margin
``h_t · W_s[y+] - h_t · W_s[y-]`` over a per-family example set, and report
mean_margin / accuracy / delta_margin / delta_accuracy relative to the native
(h_step, h_step) cell.

Hidden states at the FINAL prompt token are extracted once per h_step (cached
to disk, since prompts are deterministic) by forwarding the model at revision
``step{h_step}`` through a forward-pre-hook on the readout module.

Resume-safe per (alignment, h_step, s_step, family). Per-cell shards under
``shards/`` aggregate to ``summary.csv`` on completion.

Usage:
    uv run python experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py \\
        --model pythia-1b \\
        --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \\
        --h-steps 256 512 1000 2000 8000 143000 \\
        --alignments none scale row_norm procrustes
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

from readout.core.hf_revisions import resolve_revision
from readout.core.model_specs import (  # noqa: E402
    DEFAULT_STEPS_32 as PYTHIA_STEPS_32,
)
from readout.core.model_specs import (
    OLMO_STEPS_32 as OLMO_STEPS,
)
from readout.core.paths import repo_root  # noqa: E402
from readout.core.repro import git_commit  # noqa: E402
from readout.core.resume import (  # noqa: E402
    aggregate_json_shards,
    atomic_write_json,
    atomic_write_torch,  # noqa: F401 — used in main() under --save-per-example
    iter_undone,
)
from readout.crosscoder.snapshots import load_snapshot  # noqa: E402
from readout.probes import contrastive_tasks as CT  # noqa: E402
from readout.probes import readout_swap as RS  # noqa: E402

REPO = repo_root()

MODEL_HF = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "olmo-2-7b": "allenai/OLMo-2-1124-7B",
}

# OLMo HF revisions use the format "stage1-step{N}-tokens{T}B"; the {T}B suffix
# is looked up at runtime. The step schedule is the canonical OLMO_STEPS_32
# (src/readout/core/model_specs.py), matching extract_wu_olmo.py and thesis Appendix A.


def _model_steps(model_short: str) -> list[int]:
    if model_short.startswith("pythia"):
        return PYTHIA_STEPS_32
    if model_short.startswith("olmo"):
        return OLMO_STEPS
    raise KeyError(f"unknown model {model_short!r}")


_OLMO_REV_CACHE: dict[str, list[str]] = {}


_resolve_revision = resolve_revision  # canonical resolver: readout.core.hf_revisions


def _load_W_U(model_name: str, step: int) -> torch.Tensor:
    """Load the snapshot W_U (V, d) at ``step`` from the SSD."""
    return load_snapshot(model_name, step, kind="wu")


def _hidden_cache_path(out_dir: Path, family: str, h_step: int) -> Path:
    return out_dir / "hidden" / f"{family}_h{h_step}.pt"


def _build_or_load_hidden(
    model_name: str,
    h_step: int,
    examples: list[CT.Example],
    *,
    out_dir: Path,
    family: str,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Forward the HF model at revision step{h_step}; cache final-pos hidden state."""
    p = _hidden_cache_path(out_dir, family, h_step)
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM

    revision = _resolve_revision(model_name, h_step)
    print(
        f"  [hidden] forwarding {model_name} {revision} on {family} ({len(examples)} ex)",
        flush=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    h = RS.extract_final_hidden(
        model,
        [ex.prompt_ids for ex in examples],
        device=device,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    torch.save(h, p)
    # Wipe the revision-specific HF blobs to bound on-disk cache growth.
    # Each Pythia-6.9B revision is ~28 GB; without this, /tmp fills after
    # ~10 revisions and the next download fails with ENOSPC.
    if os.environ.get("HF_HOME"):
        import shutil

        slug = model_name.replace("/", "--")
        # Delete the entire model cache; the tokenizer/config gets re-fetched on
        # the next iteration (cheap; the dominant cost is the weight blobs).
        for d in (Path(os.environ["HF_HOME"]) / "hub").glob(f"models--{slug}"):
            shutil.rmtree(d, ignore_errors=True)
    print(f"    -> {tuple(h.shape)} in {time.time() - t0:.1f}s", flush=True)
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=list(MODEL_HF.keys()),
        default="pythia-1b",
    )
    ap.add_argument(
        "--datasets-dir",
        type=Path,
        required=True,
        help="Directory containing <family>.jsonl files (built by build_task_datasets.py).",
    )
    ap.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Subset of families to evaluate (default = every <family>.jsonl found).",
    )
    ap.add_argument(
        "--h-steps",
        type=int,
        nargs="+",
        default=[256, 512, 1000, 2000, 8000, 143000],
    )
    ap.add_argument(
        "--s-steps",
        type=int,
        nargs="+",
        default=None,
        help="Snapshot readout steps; default = canonical 32-step schedule.",
    )
    ap.add_argument(
        "--alignments",
        nargs="+",
        default=["none"],
        help=f"Subset of {RS.ALIGN_MODES}",
    )
    ap.add_argument(
        "--use-corrupt",
        action="store_true",
        help="Also evaluate <family>_corrupt.jsonl (clean/corrupted control).",
    )
    ap.add_argument(
        "--use-random-distractors",
        action="store_true",
        help="Replace y_minus with random_distractors sidecar (matched-distractor control).",
    )
    ap.add_argument(
        "--use-label-permutation",
        action="store_true",
        help="MCQ-letter families only: replace y_minus with a uniformly random "
        "non-gold letter (sidecar <family>_label_permutation.json). Stronger "
        "than --use-random-distractors for piqa / arc / sciq.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    ap.add_argument(
        "--save-per-example",
        action="store_true",
        help="Write per-example margins/y_plus/y_minus to a sidecar shards/<cell>.pt; "
        "the JSON shard stays aggregate-only. Required for Step 4 (availability probes) "
        "and Step 6 (held-out best-s + paired bootstrap).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/experiments/probes/contrastive_readout_swap/run0"),
    )
    args = ap.parse_args()

    s_steps = args.s_steps or _model_steps(args.model)
    bad = sorted(set(args.alignments) - set(RS.ALIGN_MODES))
    if bad:
        raise ValueError(f"unknown alignments: {bad}; valid={RS.ALIGN_MODES}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    model_hf = MODEL_HF[args.model]
    dtype_t = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.dtype
    ]

    # Discover families.
    if args.families is None:
        family_paths = sorted(args.datasets_dir.glob("*.jsonl"))
        family_paths = [
            p for p in family_paths if not p.name.endswith("_corrupt.jsonl")
        ]
        families = [p.stem for p in family_paths]
    else:
        families = args.families
    if not families:
        raise FileNotFoundError(
            f"no <family>.jsonl files found under {args.datasets_dir}"
        )
    print(f"[setup] model={model_hf} families={families}", flush=True)
    print(
        f"[setup] h_steps={args.h_steps} s_steps={s_steps} alignments={args.alignments}",
        flush=True,
    )

    manifest = {
        "model": model_hf,
        "datasets_dir": str(args.datasets_dir),
        "families": families,
        "h_steps": args.h_steps,
        "s_steps": s_steps,
        "alignments": args.alignments,
        "use_corrupt": args.use_corrupt,
        "use_random_distractors": args.use_random_distractors,
        "save_per_example": args.save_per_example,
        "git_commit": git_commit(),
    }
    atomic_write_json(args.out_dir / "manifest.json", manifest)

    # Cells: (family, alignment, h_step, s_step). One shard per cell.
    items: list[tuple[str, str, int, int]] = [
        (f, al, h, s)
        for f in families
        for al in args.alignments
        for h in args.h_steps
        for s in s_steps
    ]

    def shard_path(item) -> Path:
        f, al, h, s = item
        return shard_dir / f"{f}__a-{al}__h{h}__s{s}.json"

    def per_example_path(item) -> Path:
        return shard_path(item).with_suffix(".pt")

    # Precompute per-family example tensors (y_plus, y_minus).
    per_family: dict[str, dict] = {}
    for f in families:
        ex = CT.load_examples(args.datasets_dir / f"{f}.jsonl")
        if not ex:
            print(f"[warn] family {f} has 0 examples, skipping", flush=True)
            continue
        y_plus = torch.tensor([e.y_plus for e in ex], dtype=torch.long)
        y_minus = torch.tensor([e.y_minus for e in ex], dtype=torch.long)
        if args.use_random_distractors and args.use_label_permutation:
            raise ValueError(
                "--use-random-distractors and --use-label-permutation are "
                "mutually exclusive; pick one control source per run."
            )
        if args.use_random_distractors:
            ctrl_path = args.datasets_dir / f"{f}_random_distractors.json"
            if ctrl_path.exists():
                rand = json.loads(ctrl_path.read_text())
                y_minus = torch.tensor(rand[: len(ex)], dtype=torch.long)
        elif args.use_label_permutation:
            ctrl_path = args.datasets_dir / f"{f}_label_permutation.json"
            if ctrl_path.exists():
                rand = json.loads(ctrl_path.read_text())
                y_minus = torch.tensor(rand[: len(ex)], dtype=torch.long)
            else:
                print(
                    f"[warn] --use-label-permutation requested but no "
                    f"{f}_label_permutation.json in {args.datasets_dir} — "
                    f"family {f} will use its native y_minus.",
                    flush=True,
                )
        per_family[f] = {"examples": ex, "y_plus": y_plus, "y_minus": y_minus}

    # Cache W_U per s_step (and h_step, since native readout is the model's).
    W_U_cache: dict[int, torch.Tensor] = {}

    def get_W_U(step: int) -> torch.Tensor:
        if step not in W_U_cache:
            W_U_cache[step] = _load_W_U(model_hf, step)
            if len(W_U_cache) > 6:  # bound cache
                old = next(iter(W_U_cache))
                if old != step:
                    del W_U_cache[old]
        return W_U_cache[step]

    # Build hidden states lazily, one h_step x family at a time.
    h_cache: dict[tuple[str, int], torch.Tensor] = {}

    def get_h(family: str, h_step: int) -> torch.Tensor:
        key = (family, h_step)
        if key not in h_cache:
            h_cache[key] = _build_or_load_hidden(
                model_hf,
                h_step,
                per_family[family]["examples"],
                out_dir=args.out_dir,
                family=family,
                device=args.device,
                dtype=dtype_t,
            )
            if len(h_cache) > 12:
                old = next(iter(h_cache))
                if old != key:
                    del h_cache[old]
        return h_cache[key]

    def done_path(item) -> Path:
        """Sentinel path for resume. When --save-per-example is on, the .pt must
        also exist; we encode that by checking both here. iter_undone only
        inspects the returned path, so we use a thin wrapper file when needed."""
        return shard_path(item)

    def cell_complete(item) -> bool:
        if not shard_path(item).exists():
            return False
        if args.save_per_example and not per_example_path(item).exists():
            return False
        return True

    # Pre-filter: items whose JSON shard exists but .pt is missing under
    # --save-per-example need a rerun — delete the stale .json so iter_undone
    # picks them up. Atomic-replace makes this safe.
    if args.save_per_example:
        for item in items:
            if shard_path(item).exists() and not per_example_path(item).exists():
                shard_path(item).unlink()

    t0 = time.time()
    for item in iter_undone(items, done_path, label="cell"):
        f, al, h_step, s_step = item
        if f not in per_family:
            continue
        h = get_h(f, h_step)
        W_t = get_W_U(h_step)
        W_s = get_W_U(s_step)
        y_plus = per_family[f]["y_plus"]
        y_minus = per_family[f]["y_minus"]
        result = RS.evaluate_swap_cell(
            h,
            W_s,
            W_t,
            y_plus,
            y_minus,
            alignment=al,
            device=args.device,
            return_per_example=args.save_per_example,
        )
        if args.save_per_example:
            m, per_ex = result
            # Write the .pt sidecar first; JSON is the resume marker, so it
            # must only land after the per-example tensors are durable.
            atomic_write_torch(
                per_example_path(item),
                {
                    "model": model_hf,
                    "family": f,
                    "alignment": al,
                    "h_step": h_step,
                    "s_step": s_step,
                    **per_ex,
                },
            )
        else:
            m = result
        row = {
            "model": model_hf,
            "family": f,
            "alignment": al,
            "h_step": h_step,
            "s_step": s_step,
            **m,
        }
        atomic_write_json(shard_path(item), row)
        print(
            f"  [{f:<18} {al:<10} h={h_step:>6} s={s_step:>6}] "
            f"acc={m['accuracy']:.3f} (Δ={m['delta_accuracy']:+.3f}) "
            f"mg={m['mean_margin']:+.3f} (Δ={m['delta_margin']:+.3f})  n={m['n']}",
            flush=True,
        )
        if args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    n_done = sum(1 for it in items if cell_complete(it))
    if n_done == len(items):
        n = aggregate_json_shards(shard_dir, args.out_dir / "summary.csv", key="s_step")
        print(f"[done] aggregated {n} rows -> summary.csv", flush=True)
    else:
        print(f"[partial] {n_done}/{len(items)} cells; rerun to complete", flush=True)
    print(f"[elapsed] {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
