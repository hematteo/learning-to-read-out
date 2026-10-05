"""Final-position hidden states (pre- and post-final-LN) and readout snapshots per checkpoint.

For each HF revision ``step<S>``: loads the model once, forwards every family's
``prompt_ids`` (right-padded batches) and saves, at the final prompt token,

  <out-dir>/<family>_h<S>.pt   post-LN residual h = LN_S(z), the readout input  (N, d)
  <out-dir>/<family>_z<S>.pt   pre-LN residual z, the final LayerNorm input     (N, d)
  <snapshots-dir>/<slug>_step<S>_lnf.pt   final-LN {weight, bias, eps} (fp32)
  <snapshots-dir>/<slug>_step<S>_wu.pt    readout matrix (V, d) fp32, if missing

``run_availability_probes.py`` reads all four. Resumable: a step whose files all
exist is skipped. With ``--hf-home`` each revision is deleted from that cache
after use (only one checkpoint on disk at a time); without it the ambient HF
cache is used and left untouched.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import torch

from readout.core.paths import model_slug, snapshot_dir
from readout.core.repro import log_run_provenance, seed_everything
from readout.core.resume import atomic_write_json, atomic_write_torch
from readout.probes.readout_swap import _resolve_readout_module

# Checkpoints of the published availability panel (Pythia-1B and -6.9B).
PANEL_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
PANEL_STEPS += [16000, 32000, 64000, 128000, 143000]


def resolve_final_ln(model):
    for path in (("gpt_neox", "final_layer_norm"), ("model", "norm"), ("transformer", "ln_f")):
        mod = model
        for attr in path:
            mod = getattr(mod, attr, None)
            if mod is None:
                break
        if mod is not None:
            return mod
    raise AttributeError(f"{type(model).__name__}: could not resolve the final LayerNorm")


def lnf_params(model) -> dict:
    ln = resolve_final_ln(model)
    weight = ln.weight.detach().to("cpu", torch.float32).contiguous()
    bias = getattr(ln, "bias", None)
    bias = bias.detach().to("cpu", torch.float32).contiguous() if bias is not None else torch.zeros_like(weight)
    eps = float(getattr(ln, "eps", getattr(ln, "variance_epsilon", 1e-5)))
    return {"weight": weight, "bias": bias, "eps": eps}


@torch.no_grad()
def final_position_streams(
    model, prompt_ids_list: list[list[int]], *, device: str, batch_size: int, save_dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """{"h": (N, d) readout input, "z": (N, d) final-LN input} at each prompt's last token."""
    cap: dict[str, list[torch.Tensor]] = {"h": [], "z": []}
    handles = [
        _resolve_readout_module(model).register_forward_pre_hook(lambda _m, inp: cap["h"].append(inp[0].detach())),
        resolve_final_ln(model).register_forward_pre_hook(lambda _m, inp: cap["z"].append(inp[0].detach())),
    ]
    rows: dict[str, list[torch.Tensor]] = {"h": [], "z": []}
    try:
        for start in range(0, len(prompt_ids_list), batch_size):
            batch = prompt_ids_list[start : start + batch_size]
            lengths = torch.tensor([len(ids) for ids in batch], dtype=torch.long, device=device)
            toks = torch.zeros((len(batch), int(lengths.max())), dtype=torch.long, device=device)  # (B, T)
            mask = torch.zeros_like(toks)
            for i, ids in enumerate(batch):
                toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                mask[i, : len(ids)] = 1
            for v in cap.values():
                v.clear()
            model(toks, attention_mask=mask)
            last = torch.arange(len(batch), device=device), lengths - 1
            for key in ("h", "z"):
                rows[key].append(cap[key][0][last].to("cpu", save_dtype))  # (B, d)
    finally:
        for handle in handles:
            handle.remove()
    return {k: torch.cat(v, dim=0) for k, v in rows.items()}


def purge_revision_cache(hf_home: Path, model_repo: str) -> None:
    repo_dir = hf_home / "hub" / f"models--{model_repo.replace('/', '--')}"
    for sub in ("snapshots", "blobs", "refs"):
        shutil.rmtree(repo_dir / sub, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hf-model", required=True, help="e.g. EleutherAI/pythia-6.9b")
    ap.add_argument("--datasets-dir", type=Path, required=True)
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--h-steps", type=int, nargs="+", default=PANEL_STEPS)
    ap.add_argument("--out-dir", type=Path, required=True, help="hidden-state caches")
    ap.add_argument("--snapshots-dir", type=Path, default=None, help="W_U / final-LN snapshots (default snapshot_dir)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32", help="model + cache dtype")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hf-home", type=Path, default=None, help="dedicated HF cache; each revision is purged after use")
    ap.add_argument("--keep-cache", action="store_true", help="with --hf-home: do not purge revisions")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    seed_everything(0)
    provenance = log_run_provenance(0)
    if args.hf_home is not None:
        os.environ["HF_HOME"] = str(args.hf_home)
        args.hf_home.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM  # after HF_HOME is set

    snaps = args.snapshots_dir or snapshot_dir(args.hf_model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    snaps.mkdir(parents=True, exist_ok=True)
    slug = model_slug(args.hf_model)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    families = {}
    for fam in args.families:
        families[fam] = [json.loads(line)["prompt_ids"] for line in (args.datasets_dir / f"{fam}.jsonl").open()]
        print(f"[setup] {fam}: n={len(families[fam])}", flush=True)
    atomic_write_json(
        args.out_dir / "extract_provenance.json",
        {**provenance, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
    )

    def paths(step: int) -> list[Path]:
        out = [snaps / f"{slug}_step{step}_lnf.pt", snaps / f"{slug}_step{step}_wu.pt"]
        return out + [args.out_dir / f"{f}_{k}{step}.pt" for f in families for k in ("h", "z")]

    todo = [s for s in args.h_steps if not all(p.exists() for p in paths(s))]
    print(f"[setup] {len(todo)} steps to process: {todo}", flush=True)
    t0 = time.time()
    for i, step in enumerate(todo):
        print(f"\n[{i + 1}/{len(todo)}] step{step} @ {time.time() - t0:.0f}s", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            revision=f"step{step}",
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=args.local_files_only,
        ).to(args.device)
        model.eval()
        lnf_p, wu_p = snaps / f"{slug}_step{step}_lnf.pt", snaps / f"{slug}_step{step}_wu.pt"
        if not lnf_p.exists():
            atomic_write_torch(lnf_p, lnf_params(model))
        if not wu_p.exists():
            W = _resolve_readout_module(model).weight.detach().to("cpu", torch.float32).contiguous()  # (V, d)
            atomic_write_torch(wu_p, W)
        for fam, prompts in families.items():
            hp, zp = args.out_dir / f"{fam}_h{step}.pt", args.out_dir / f"{fam}_z{step}.pt"
            if hp.exists() and zp.exists():
                continue
            streams = final_position_streams(
                model, prompts, device=args.device, batch_size=args.batch_size, save_dtype=dtype
            )
            atomic_write_torch(hp, streams["h"])
            atomic_write_torch(zp, streams["z"])
            print(f"  {fam:<26} n={len(prompts):>5} -> h/z {tuple(streams['h'].shape)}", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.hf_home is not None and not args.keep_cache:
            purge_revision_cache(args.hf_home, args.hf_model)
    print(f"[done] {len(todo)} steps in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
