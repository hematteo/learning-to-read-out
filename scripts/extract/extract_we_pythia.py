"""Extract W_E (input embeddings) from Pythia checkpoints into the
wu_adapter cache layout.

For the Pythia-160M W_E multi-seed crosscoder. The wu_adapter trainer reads snapshots from a single
cache directory whose files match `{model_slug}_step{step}_wu.pt`. To train
on W_E without a code fork we extract `model.gpt_neox.embed_in.weight` per
snapshot and save it under the same naming convention into a *separate*
cache directory (so W_U and W_E caches do not collide).

Saves Ge-exact 32-step snapshots by default
(matches src/crosscoder/wu_adapter.py:DEFAULT_STEPS), so the resulting
crosscoder is directly comparable to the production 160M W_U run
(experiments.yaml: crosscoder_main).

Usage:
  uv run python scripts/extract/extract_we_pythia.py \
      --model EleutherAI/pythia-160m \
      --cache-dir ${UM_SSD_ROOT}/wu_crosscoder/we_snapshots

Then point the trainer at the same cache (scripts/train/train_crosscoder.py
wraps wu_adapter.main; the library module has no __main__ block):
  uv run python scripts/train/train_crosscoder.py \
      --cache-dir ${UM_SSD_ROOT}/wu_crosscoder/we_snapshots \
      --output /workspace/results/we_multiseed/we_cc_dsae8192_seed${SEED}.pt \
      ...
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import argparse
import shutil
from pathlib import Path

import torch

from src.core.paths import ssd_path, ssd_root
from src.crosscoder.wu_adapter import DEFAULT_STEPS  # noqa: E402


def extract_one(
    model_name: str,
    step: int,
    cache_dir: Path,
    *,
    overwrite: bool = False,
    cleanup_hf_cache: bool = True,
) -> Path:
    from transformers import AutoModelForCausalLM

    slug = model_name.replace("/", "_")
    out_path = cache_dir / f"{slug}_step{step}_wu.pt"
    if out_path.exists() and not overwrite:
        return out_path
    print(f"  step {step}: downloading {model_name}@step{step}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=f"step{step}",
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # Pythia / GPT-NeoX input embeddings. The W_U extractor pulls
    # `model.embed_out.weight` (the unembedding) from the same model object.
    W_E = model.gpt_neox.embed_in.weight.detach().contiguous().cpu().clone()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(W_E, out_path)
    print(f"  step {step}: saved W_E {tuple(W_E.shape)} -> {out_path.name}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cleanup_hf_cache:
        import os  # local: survives top-level autoformat/import-prune passes

        # Honor HF_HOME if set; transformers caches under HF_HOME/hub when
        # the env var is exported, not under the default $HOME/.cache path.
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        hf_cache = Path(hf_home) / "hub"
        if hf_cache.exists():
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
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ssd_path("wu_crosscoder", "we_snapshots"),
        help="W_E snapshot cache; must NOT overlap the W_U cache (different W tensor).",
    )
    ap.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_STEPS),
        help="Default = Ge-exact 32-step schedule from wu_adapter.DEFAULT_STEPS.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip HF cache cleanup between checkpoints (faster but uses more disk).",
    )
    args = ap.parse_args()

    if str(args.cache_dir).startswith(str(ssd_root())):
        if not ssd_root().is_dir():
            raise SystemExit(
                "ERROR: external SSD ${UM_SSD_ROOT} is not mounted. "
                "Attach the drive (the W_E cache must persist) or pass --cache-dir."
            )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[extract-we] model={args.model} cache={args.cache_dir} steps={len(args.steps)}",
        flush=True,
    )

    n_ok = 0
    for i, step in enumerate(args.steps):
        try:
            extract_one(
                args.model,
                step,
                args.cache_dir,
                overwrite=args.overwrite,
                cleanup_hf_cache=not args.no_cleanup,
            )
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  step {step}: FAILED ({type(e).__name__}: {e})", flush=True)
        print(f"  [{i + 1}/{len(args.steps)}] cumulative {n_ok} ok", flush=True)

    print(f"[extract-we] done: {n_ok}/{len(args.steps)} extracted", flush=True)
    return 0 if n_ok == len(args.steps) else 1


if __name__ == "__main__":
    _sys.exit(main())
