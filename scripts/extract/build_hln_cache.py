"""Pre-readout hidden-state (h_LN) and W_U snapshot builder for Pythia checkpoints.

Each requested step:
1. Forwards a fixed eval-token shard through the model and captures the input
   to the readout via a forward-pre hook -> ``--out-dir/hLN_step{step}.pt``.
2. Saves the readout weight matrix to the canonical snapshot layout
   ``--snapshots-dir/<slug>_step{step}_wu.pt``.
3. Cleans the HuggingFace cache for that revision before the next step (each
   Pythia revision is a unique blob set, so the cache grows unbounded otherwise).

h_LN is stored in --save-dtype (default bf16); W_U is stored in fp32 to match
the layout consumed by ``src.crosscoder.snapshots.load_snapshot``.

Resume-safe: per-step exists-check + atomic write. A revision is skipped if
both its h_LN and W_U snapshot are already present.

Usage:
    uv run python scripts/extract/build_hln_cache.py \
        --model pythia-1b \
        --steps all \
        --eval-tokens path/to/eval_tokens.pt \
        --out-dir /path/to/derived/readout_edit_timing_pythia1b \
        --snapshots-dir /path/to/data/snapshots/pythia-1b \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.resume import atomic_write_torch

PYTHIA_DEFAULT_STEPS = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
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
    61000,
    75000,
    89000,
    102000,
    116000,
    130000,
    143000,
]

HF_NAME = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
}

SEQ_LEN = 512


def parse_steps(raw: str) -> list[int]:
    if raw == "all":
        return list(PYTHIA_DEFAULT_STEPS)
    return [int(x) for x in raw.split(",") if x.strip()]


def resolve_revision(model_name: str, step: int) -> str:
    if "olmo" in model_name.lower():
        raise NotImplementedError("OLMo revision lookup not ported yet.")
    return f"step{step}"


def resolve_readout(model):
    if hasattr(model, "embed_out"):
        return model.embed_out
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError(f"{type(model).__name__} has no .embed_out or .lm_head")


def load_ids_seqs(eval_tokens: Path, n_sequences: int) -> torch.Tensor:
    data = torch.load(eval_tokens, map_location="cpu", weights_only=False)
    ids = data["ids"].detach().cpu().to(torch.long)
    n_full = (len(ids) // SEQ_LEN) * SEQ_LEN
    return ids[:n_full].view(-1, SEQ_LEN)[:n_sequences]


def clean_hf_cache(hf_name: str) -> None:
    """Delete all blobs and snapshots for ``hf_name`` to bound disk usage.

    HF caches each revision as separate blobs for Pythia (weights differ per
    step), so without this the cache grows ~14 GB per 6.9B revision.
    """
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    hub = Path(hf_home) / "hub"
    if not hub.exists():
        return
    repo_glob = "models--" + hf_name.replace("/", "--") + "*"
    for repo_dir in hub.glob(repo_glob):
        snaps = repo_dir / "snapshots"
        if snaps.exists():
            for snap in snaps.iterdir():
                shutil.rmtree(snap, ignore_errors=True)
        blobs = repo_dir / "blobs"
        if blobs.exists():
            shutil.rmtree(blobs, ignore_errors=True)
            blobs.mkdir()


def cache_one_step(
    hf_name: str,
    step: int,
    ids_seqs: torch.Tensor,
    *,
    device: str,
    model_dtype: torch.dtype,
    save_dtype: torch.dtype,
    batch_seqs: int,
    need_hln: bool,
    need_wu: bool,
) -> tuple[dict | None, torch.Tensor | None, str]:
    from transformers import AutoModelForCausalLM

    revision = resolve_revision(hf_name, step)
    print(
        f"[step {step}] loading {hf_name}@{revision} dtype={model_dtype} "
        f"need_hln={need_hln} need_wu={need_wu}",
        flush=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        revision=revision,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    payload: dict | None = None
    wu: torch.Tensor | None = None

    if need_hln:
        captured: list[torch.Tensor] = []

        def hook(_mod, inputs):
            captured.append(inputs[0].detach().to(save_dtype).cpu())

        readout = resolve_readout(model)
        handle = readout.register_forward_pre_hook(hook)
        with torch.no_grad():
            for start in range(0, ids_seqs.shape[0], batch_seqs):
                batch = ids_seqs[start : start + batch_seqs].to(device)
                _ = model(batch)
        handle.remove()
        h_LN = torch.cat(captured, dim=0)
        payload = {"h_LN": h_LN, "revision": revision, "model": hf_name}

    if need_wu:
        readout = resolve_readout(model)
        wu = readout.weight.detach().to(torch.float32).cpu().clone()

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    elapsed = time.time() - t0
    print(f"[step {step}] elapsed={elapsed:.1f}s", flush=True)
    return payload, wu, revision


def parse_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(HF_NAME.keys()), required=True)
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps or 'all' (= 32-step Ge schedule).",
    )
    parser.add_argument("--eval-tokens", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="h_LN cache dir.")
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=None,
        help="W_U snapshots dir (canonical layout). Skip W_U if not provided.",
    )
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--batch-seqs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "bf16", "fp16"],
        default=None,
        help="Model inference dtype. Default: fp32 for 160m, bf16 for 1b/6.9b.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["fp32", "bf16", "fp16"],
        default="bf16",
        help="On-disk h_LN dtype (default bf16). W_U snapshots are always fp32.",
    )
    parser.add_argument(
        "--no-clean-hf-cache",
        action="store_true",
        help="Disable per-revision HF cache cleanup (default: clean).",
    )
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.snapshots_dir is not None:
        args.snapshots_dir.mkdir(parents=True, exist_ok=True)

    if args.dtype is None:
        args.dtype = "fp32" if args.model == "pythia-160m" else "bf16"
    model_dtype = parse_dtype(args.dtype)
    save_dtype = parse_dtype(args.save_dtype)
    hf_name = HF_NAME[args.model]
    slug = hf_name.replace("/", "_")

    ids_seqs = load_ids_seqs(args.eval_tokens, args.max_seqs)
    print(
        f"[setup] model={args.model} ({hf_name}) steps={steps} "
        f"eval={ids_seqs.shape[0]}x{SEQ_LEN} "
        f"model_dtype={args.dtype} save_dtype={args.save_dtype} device={args.device} "
        f"snapshots_dir={args.snapshots_dir}",
        flush=True,
    )

    def hln_path(step: int) -> Path:
        return args.out_dir / f"hLN_step{step}.pt"

    def wu_path(step: int) -> Path:
        if args.snapshots_dir is None:
            return Path("/dev/null")
        return args.snapshots_dir / f"{slug}_step{step}_wu.pt"

    n_total = len(steps)
    done_full = sum(
        1
        for s in steps
        if hln_path(s).exists() and (args.snapshots_dir is None or wu_path(s).exists())
    )
    if done_full:
        print(f"[resume] {done_full}/{n_total} steps already complete", flush=True)
    else:
        print(f"[resume] starting fresh, {n_total} steps todo", flush=True)

    for step in steps:
        need_hln = not hln_path(step).exists()
        need_wu = args.snapshots_dir is not None and not wu_path(step).exists()
        if not need_hln and not need_wu:
            continue

        payload, wu, revision = cache_one_step(
            hf_name,
            step,
            ids_seqs,
            device=args.device,
            model_dtype=model_dtype,
            save_dtype=save_dtype,
            batch_seqs=args.batch_seqs,
            need_hln=need_hln,
            need_wu=need_wu,
        )
        if need_hln and payload is not None:
            atomic_write_torch(hln_path(step), payload)
            print(
                f"[step {step}] saved h_LN shape={tuple(payload['h_LN'].shape)} "
                f"dtype={payload['h_LN'].dtype}",
                flush=True,
            )
        if need_wu and wu is not None:
            atomic_write_torch(wu_path(step), wu)
            print(
                f"[step {step}] saved W_U shape={tuple(wu.shape)} -> {wu_path(step)}",
                flush=True,
            )

        if not args.no_clean_hf_cache:
            clean_hf_cache(hf_name)

    n_done_hln = sum(1 for s in steps if hln_path(s).exists())
    n_done_wu = (
        sum(1 for s in steps if wu_path(s).exists())
        if args.snapshots_dir is not None
        else None
    )
    print(
        f"[done] h_LN={n_done_hln}/{n_total}, "
        f"W_U={n_done_wu}/{n_total if n_done_wu is not None else '-'} in {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
