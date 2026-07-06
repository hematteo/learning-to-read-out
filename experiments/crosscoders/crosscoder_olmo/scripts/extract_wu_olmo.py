"""Extract W_U (lm_head) tensors from OLMo-2-7B-1124 checkpoints.

OLMo-2 uses a LLaMA-style architecture where the unembedding matrix is at
``model.lm_head.weight`` (vs Pythia's GPT-NeoX ``model.embed_out.weight``).
Revisions follow ``stage1-step{N}-tokens{T}B`` rather than Pythia's ``step{N}``.

Output cache files use the same naming convention as ``wu_adapter.extract_wu``
so the existing trainer finds them as cached and skips re-extraction:

    {cache_dir}/{slug}_step{N}_wu.pt        where slug = model.replace("/", "_")

Usage:
    python experiments/crosscoders/crosscoder_olmo/scripts/extract_wu_olmo.py \
        --model allenai/OLMo-2-1124-7B \
        --cache-dir /workspace/wu_cache_olmo \
        --steps 150,600,700,850,900,1000,2000,...

Disk: cleans HF cache between checkpoints. Peak disk per checkpoint is the
size of one OLMo-2-7B fp16 download (~14 GB). W_U cache per snapshot is
~1.6 GB at fp32 (V=100352 × d=4096 × 4 bytes). Plan for ~50 GB total cache
at the 32-snapshot schedule.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM

# Pre-registered schedule from SCHEDULE.md (committed 2026-04-27).
OLMO2_7B_STEPS = [
    150,
    600,
    700,
    850,
    900,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    14000,
    21000,
    27000,
    34000,
    47000,
    110000,
    173000,
    236000,
    299000,
    362000,
    425000,
    488000,
    614000,
    677000,
    740000,
    803000,
    866000,
    928000,
]
assert len(OLMO2_7B_STEPS) == 32


def log(msg: str, log_file: Path | None = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def list_revisions(model_name: str) -> list[str]:
    """Hit HF API once to enumerate all branch names."""
    r = requests.get(f"https://huggingface.co/api/models/{model_name}/refs", timeout=30)
    r.raise_for_status()
    return [b["name"] for b in r.json().get("branches", [])]


def resolve_revision(
    revisions: list[str], step: int, stage: str = "stage1"
) -> str | None:
    """Find the revision matching ``{stage}-step{step}-tokens*B``.

    Returns the full branch name, e.g. 'stage1-step150-tokens1B'. If the
    requested step has no exact match, returns None.
    """
    pattern = re.compile(rf"^{re.escape(stage)}-step{step}(-tokens.*)?$")
    for rev in revisions:
        if pattern.match(rev):
            return rev
    return None


def clean_hf_cache(model_name: str) -> None:
    """Wipe the HF blobs+snapshots for this repo to bound disk.

    Honors HF_HOME if set; transformers caches under HF_HOME/hub when the env
    var is exported, not the default $HOME/.cache. Without this guard, a
    redirected cache (common on HPC) grows unbounded across snapshots.
    """
    import os

    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    hf_cache = Path(hf_home) / "hub"
    if not hf_cache.exists():
        return
    repo_glob = "models--" + model_name.replace("/", "--") + "*"
    for repo_dir in hf_cache.glob(repo_glob):
        snaps = repo_dir / "snapshots"
        if snaps.exists():
            for snap in snaps.iterdir():
                shutil.rmtree(snap, ignore_errors=True)
        blobs = repo_dir / "blobs"
        if blobs.exists():
            shutil.rmtree(blobs, ignore_errors=True)
            blobs.mkdir()


def extract_one(
    model_name: str,
    revision: str,
    step: int,
    cache_dir: Path,
    download_dtype: torch.dtype = torch.bfloat16,
    cache_dtype: torch.dtype = torch.float32,
    log_file: Path | None = None,
) -> bool:
    """Download checkpoint, save lm_head.weight as fp32 to cache. Returns True on success."""
    slug = model_name.replace("/", "_")
    out_path = cache_dir / f"{slug}_step{step}_wu.pt"
    if out_path.exists():
        log(f"  step{step}: cached at {out_path}", log_file)
        return True

    log(
        f"  step{step}: downloading revision={revision} (dtype={download_dtype})",
        log_file,
    )
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=download_dtype,
            low_cpu_mem_usage=True,
        )
        # OLMo-2 / LLaMA-style: unembedding at model.lm_head.weight, shape (V, d_model).
        wu = model.lm_head.weight.detach().cpu().to(cache_dtype).contiguous().clone()
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(wu, out_path)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clean_hf_cache(model_name)
        elapsed = time.time() - t0
        log(
            f"  step{step}: saved W_U {tuple(wu.shape)} ({wu.dtype}) "
            f"in {elapsed:.0f}s -> {out_path}",
            log_file,
        )
        return True
    except Exception as e:
        log(f"  step{step}: FAILED ({type(e).__name__}: {e})", log_file)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B")
    ap.add_argument(
        "--stage", default="stage1", help="Pretraining stage (stage1=pretrain)"
    )
    ap.add_argument(
        "--steps",
        default=",".join(str(s) for s in OLMO2_7B_STEPS),
        help="Comma-separated list of training steps to extract.",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Where to write {slug}_step{N}_wu.pt files. Use an SSD path with >=50 GB free.",
    )
    ap.add_argument(
        "--download-dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Halves download bandwidth at fp16/bf16 (model native is bf16). "
        "Cache always saved as fp32 for crosscoder training compatibility.",
    )
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve revisions and report what would be extracted, without downloading.",
    )
    args = ap.parse_args()

    from readout.core.repro import log_run_provenance

    log_run_provenance()

    log_file = args.log_file
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("")

    steps = [int(s) for s in args.steps.split(",")]
    log(f"Extracting W_U for {args.model} at {len(steps)} steps", log_file)
    log(f"Cache dir: {args.cache_dir}", log_file)

    log("Querying HF API for revision list...", log_file)
    revisions = list_revisions(args.model)
    log(f"  found {len(revisions)} branches", log_file)

    resolved: list[tuple[int, str]] = []
    missing: list[int] = []
    for step in steps:
        rev = resolve_revision(revisions, step, stage=args.stage)
        if rev is None:
            missing.append(step)
        else:
            resolved.append((step, rev))

    log(f"Resolved {len(resolved)}/{len(steps)} steps to revisions.", log_file)
    if missing:
        log(f"MISSING (no revision matched): {missing}", log_file)
        log(
            "  -> halting. Pre-registered schedule requires every listed step. "
            "Either fix the schedule or report the deviation explicitly.",
            log_file,
        )
        return 2

    if args.dry_run:
        log("Dry run; not downloading. Resolved revisions:", log_file)
        for step, rev in resolved:
            log(f"  step={step:>7d}  rev={rev}", log_file)
        return 0

    download_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[
        args.download_dtype
    ]

    t0 = time.time()
    n_ok = 0
    for i, (step, rev) in enumerate(resolved):
        log(f"\n[{i + 1}/{len(resolved)}] step {step} (rev {rev})", log_file)
        ok = extract_one(
            args.model,
            rev,
            step,
            args.cache_dir,
            download_dtype=download_dtype,
            log_file=log_file,
        )
        if ok:
            n_ok += 1
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(resolved) - i - 1)
        log(
            f"  cumulative: {n_ok}/{i + 1}, elapsed={elapsed:.0f}s, eta={eta:.0f}s",
            log_file,
        )

    log(
        f"\nDone: {n_ok}/{len(resolved)} steps extracted in {time.time() - t0:.0f}s",
        log_file,
    )
    return 0 if n_ok == len(resolved) else 1


if __name__ == "__main__":
    sys.exit(main())
