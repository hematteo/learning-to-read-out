"""Standalone Pythia-1B W_U + hidden-state extractor (single GPU; no src deps).

Per checkpoint step:
  - load model at HF revision step{N}
  - save embed_out.weight to wu-out-dir/{slug}_step{N}_wu.pt          (always)
  - if step in --hidden-steps, forward over each family's prompt_ids
    and save hidden-out-dir/{family}_h{N}.pt                          (conditional)
  - evict the entire HF cache for that revision

Designed to run on one GPU; bash launcher runs N copies in parallel
with disjoint --steps subsets and CUDA_VISIBLE_DEVICES per worker.

No deps on src.* — only torch + transformers (+ optional hf_transfer).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch


def _purge_revision_cache(hf_home: Path, model_repo: str) -> None:
    repo_dir = hf_home / "hub" / f"models--{model_repo.replace('/', '--')}"
    if not repo_dir.exists():
        return
    for sub in ("snapshots", "blobs", "refs"):
        p = repo_dir / sub
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


@torch.no_grad()
def _final_hidden(model, prompt_ids_list, device):
    captured = []

    def hook(_mod, inputs):
        captured.append(inputs[0].detach())

    if hasattr(model, "embed_out"):
        readout = model.embed_out
    elif hasattr(model, "lm_head"):
        readout = model.lm_head
    else:
        raise AttributeError("model has no embed_out or lm_head")
    handle = readout.register_forward_pre_hook(hook)
    rows = []
    try:
        for ids in prompt_ids_list:
            captured.clear()
            t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            _ = model(t)
            rows.append(captured[0][:, -1, :].detach().to("cpu", torch.float32))
    finally:
        handle.remove()
    return torch.cat(rows, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="EleutherAI/pythia-1b")
    ap.add_argument(
        "--steps",
        type=int,
        nargs="+",
        required=True,
        help="Checkpoint steps this worker handles (W_U always extracted).",
    )
    ap.add_argument(
        "--hidden-steps",
        type=int,
        nargs="*",
        default=[],
        help="Subset of --steps that ALSO need hidden-state extraction.",
    )
    ap.add_argument("--datasets-dir", type=Path, required=True)
    ap.add_argument(
        "--families",
        nargs="+",
        default=[
            "sva",
            "ioi",
            "relational_facts",
            "hypernym",
            "induction",
            "numeric_gt",
        ],
    )
    ap.add_argument("--wu-out-dir", type=Path, required=True)
    ap.add_argument("--hidden-out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--hf-home", type=Path, default=None)
    args = ap.parse_args()

    args.wu_out_dir.mkdir(parents=True, exist_ok=True)
    args.hidden_out_dir.mkdir(parents=True, exist_ok=True)
    if args.hf_home is None:
        args.hf_home = Path(f"/tmp/hfh_{os.getpid()}")
    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    slug = args.hf_model.replace("/", "_")

    families_examples = {}
    if args.hidden_steps:
        for fam in args.families:
            p = args.datasets_dir / f"{fam}.jsonl"
            if not p.exists():
                print(f"[warn] missing dataset {p}", flush=True)
                continue
            ex = []
            with p.open() as f:
                for line in f:
                    d = json.loads(line)
                    ex.append((d["prompt_ids"], int(d["y_plus"]), int(d["y_minus"])))
            families_examples[fam] = ex
            print(f"[setup] {fam}: n={len(ex)}", flush=True)

    dtype_t = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.dtype
    ]
    from transformers import AutoModelForCausalLM

    pid_tag = os.environ.get("WORKER_TAG", str(os.getpid()))
    t0 = time.time()
    hidden_set = set(args.hidden_steps)
    for i, step in enumerate(args.steps):
        wu_path = args.wu_out_dir / f"{slug}_step{step}_wu.pt"
        need_hidden_steps = step in hidden_set
        need_hidden_files = []
        if need_hidden_steps:
            for fam in families_examples:
                hp = args.hidden_out_dir / f"{fam}_h{step}.pt"
                if not hp.exists():
                    need_hidden_files.append((fam, hp))
        if wu_path.exists() and not need_hidden_files:
            print(
                f"[{pid_tag}][{i + 1}/{len(args.steps)}] step={step} skip (W_U + hidden present)",
                flush=True,
            )
            continue
        revision = f"step{step}"
        print(
            f"\n[{pid_tag}][{i + 1}/{len(args.steps)}] step={step} rev={revision} "
            f"@ elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        t_step = time.time()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.hf_model,
                revision=revision,
                dtype=dtype_t,
                low_cpu_mem_usage=True,
            ).to(args.device)
            model.eval()
        except Exception as e:
            print(f"  [error] load {revision}: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"  loaded in {time.time() - t_step:.1f}s", flush=True)

        if not wu_path.exists():
            if hasattr(model, "embed_out"):
                W_U = (
                    model.embed_out.weight.detach()
                    .to("cpu", torch.float32)
                    .contiguous()
                )
            elif hasattr(model, "lm_head"):
                W_U = (
                    model.lm_head.weight.detach().to("cpu", torch.float32).contiguous()
                )
            else:
                raise AttributeError("model has neither embed_out nor lm_head")
            tmp = wu_path.with_suffix(".pt.tmp")
            torch.save(W_U, tmp)
            os.replace(tmp, wu_path)
            print(f"  W_U {tuple(W_U.shape)} -> {wu_path.name}", flush=True)
            del W_U

        if need_hidden_files:
            for fam, hp in need_hidden_files:
                t_fwd = time.time()
                ex = families_examples[fam]
                h = _final_hidden(model, [e[0] for e in ex], args.device)
                tmp = hp.with_suffix(".pt.tmp")
                torch.save(h, tmp)
                os.replace(tmp, hp)
                print(
                    f"  hidden {fam:<18} n={len(ex):>5} {tuple(h.shape)} "
                    f"{time.time() - t_fwd:.1f}s -> {hp.name}",
                    flush=True,
                )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _purge_revision_cache(args.hf_home, args.hf_model)
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(args.steps) - (i + 1))
        print(f"  cum={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    shutil.rmtree(args.hf_home, ignore_errors=True)
    print(
        f"\n[{pid_tag}][done] {len(args.steps)} steps in {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
