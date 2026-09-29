"""Shared helpers for the recipe-control analysis scripts in this directory.

Sibling module (same ``scripts/`` dir): default paths, argument groups, the
held-out token loader, the one-forward hidden state cache, and CSV /
provenance writers. Anything another experiment would reuse belongs in
``readout`` instead (checkpoint loading is ``readout.probes.recipe_control_models``).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from readout.core.paths import repo_root, ssd_path
from readout.core.repro import git_commit, log_run_provenance
from readout.probes.recipe_control_models import PAPER_CONDITIONS, default_runs_root

EXPERIMENT_RESULTS = Path("results") / "experiments" / "ablations" / "pretraining_recipe_control"
DEFAULT_CONDITIONS = list(PAPER_CONDITIONS)


def default_results_dir() -> Path:
    """Gitignored metrics root for this experiment (``<repo>/results/experiments/ablations/...``)."""
    return repo_root() / EXPERIMENT_RESULTS


def default_eval_bin() -> Path:
    """Where ``fetch_heldout_slice.py`` writes the held-out slice by default."""
    return ssd_path("pile_slice", "heldout_after10B_2M.bin")


def pick_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def add_common_args(ap: argparse.ArgumentParser, *, eval_bin: bool = False) -> None:
    ap.add_argument(
        "--ckpt-root",
        type=Path,
        default=default_runs_root(),
        help="dir holding <condition>/ckpts/step<N>/model_fp16.pt (default ${UM_SSD_ROOT}/runs)",
    )
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="arms to analyse (default: the four the paper reports; add warmup_long for the fifth)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cuda / cpu / auto")
    if eval_bin:
        ap.add_argument(
            "--eval-bin",
            type=Path,
            default=default_eval_bin(),
            help="flat uint16 token file from fetch_heldout_slice.py",
        )
        ap.add_argument("--seq-len", type=int, default=2048)


def load_eval_blocks(eval_bin: Path, eval_tokens: int, seq_len: int) -> torch.Tensor:
    """First ``eval_tokens`` of a flat uint16 token file as ``(N, seq_len)`` int64 blocks."""
    data = np.memmap(eval_bin, dtype=np.uint16, mode="r")
    n_tok = min(int(eval_tokens), data.shape[0])
    n_blocks = n_tok // seq_len
    if n_blocks < 1:
        raise SystemExit(f"eval slice too small: {n_tok} tokens < seq_len {seq_len}")
    arr = np.asarray(data[: n_blocks * seq_len], dtype=np.int64).reshape(n_blocks, seq_len)
    return torch.from_numpy(arr)


@torch.no_grad()
def cache_hidden(
    model,
    ids_seqs: torch.Tensor,
    device: str,
    batch_seqs: int,
    *,
    return_pre_ln: bool = False,
):
    """Forward the body once; return the post-final-LN hidden ``h`` ``(N, T, d)`` on CPU fp32.

    With ``return_pre_ln`` also return the pre-LN residual ``g`` captured by a forward
    pre-hook on ``final_layer_norm`` (``h == LN(g)`` with the body's own gain/bias), so a
    caller can re-normalize with another checkpoint's LayerNorm.
    """
    captured: dict[str, torch.Tensor] = {}
    hook = model.gpt_neox.final_layer_norm.register_forward_pre_hook(lambda _m, inp: captured.__setitem__("g", inp[0]))
    hs, gs = [], []
    try:
        for s in range(0, ids_seqs.shape[0], batch_seqs):
            x = ids_seqs[s : s + batch_seqs].to(device)
            h = model.gpt_neox(input_ids=x).last_hidden_state.float()  # (b, T, d)
            hs.append(h.cpu())
            if return_pre_ln:
                gs.append(captured["g"].float().cpu())
    finally:
        hook.remove()
    if return_pre_ln:
        return torch.cat(hs), torch.cat(gs)
    return torch.cat(hs)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    return path


def write_provenance(out_dir: Path, name: str, args: argparse.Namespace, seed: int, extra: dict | None = None) -> Path:
    """Stamp ``<out_dir>/provenance_<name>.json`` with git hash, env, seed, and the CLI args."""
    info = log_run_provenance(seed)
    info["git_commit_full"] = git_commit()
    info["script"] = name
    info["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    if extra:
        info.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"provenance_{name}.json"
    p.write_text(json.dumps(info, indent=2, default=str) + "\n")
    return p
