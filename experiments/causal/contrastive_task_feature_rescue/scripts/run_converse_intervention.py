"""Converse hidden-state intervention ("hidden projection" columns of the localization ledger).

Projects the top-K crosscoder decoder directions selected by
``run_feature_attribution.py`` out of the hidden states, h <- (I - Q Q^T) h
with Q an orthonormal (QR) basis of the K decoder rows, and re-scores the
contrastive margin under the model's NATIVE readout at ``--h-step`` (and at
``--snapshot-step`` for comparison). This separates "the K features identify
a subspace of h that carries the task" (accuracy collapses) from "the K
features are readout-side token directions with no projection in h"
(accuracy unchanged).

Evaluation uses exactly the held-out half (split B) of the attribution run:
the split parameters are read from the attribution shard JSON and the
permutation is re-derived with the attribution script's per-family
``stable_rng(split_seed, "split", family)``, so the projected directions were
never selected on the scored items.

Inputs (per family):
  - <attribution-dir>/shards/<family>__h<h>__s<s>.{json,pt} from run_feature_attribution.py
    (the .pt carries ``decoder_rows_top_Kmax`` (K_max, d) in selection order)
  - <hidden-dir>/<family>_h<h>.pt and <datasets-dir>/<family>.jsonl (Exp 1)
  - native W_U snapshots at h_step and snapshot_step (${UM_SSD_ROOT}/snapshots/...)

Outputs (resume-safe per family):
  <out-dir>/<family>__h<h>__s<s>.json   baseline + per-K projected margins / logit shifts
  <out-dir>/manifest.json               inputs, split provenance, git commit

Usage (the ledger cell: step-1000 states, terminal-readout attribution):
  uv run python experiments/causal/contrastive_task_feature_rescue/scripts/run_converse_intervention.py \\
      --model pythia-1b \\
      --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \\
      --hidden-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden \\
      --attribution-dir results/experiments/causal/contrastive_task_feature_rescue/run0_pythia1b_s143000_h1000_pos \\
      --h-step 1000 --snapshot-step 143000 \\
      --out-dir results/experiments/causal/contrastive_task_feature_rescue/converse_pythia1b_s143000_h1000_pos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from run_feature_attribution import stable_rng

from readout.core.model_specs import MODEL_HF_NAMES
from readout.core.repro import git_commit, log_run_provenance, seed_everything
from readout.core.resume import atomic_write_json
from readout.crosscoder.snapshots import load_snapshot
from readout.probes import contrastive_tasks as CT
from readout.probes import readout_swap as RS


def project_out(h: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Project h (N, d) onto the orthogonal complement of span(directions (K, d)) -> (N, d)."""
    Q, _ = torch.linalg.qr(directions.T.float())  # (d, K), orthonormal columns
    return h.float() - (h.float() @ Q) @ Q.T


def logit_pair(h: torch.Tensor, W: torch.Tensor, yp: torch.Tensor, ym: torch.Tensor) -> tuple[float, float]:
    """Mean logit of y+ and of y- over examples: (mean_n h_n . W[y+_n], mean_n h_n . W[y-_n])."""
    Wp = W[yp.long()].float()
    Wm = W[ym.long()].float()
    return (h.float() * Wp).sum(-1).mean().item(), (h.float() * Wm).sum(-1).mean().item()


def held_out_indices(fam: str, split: dict) -> np.ndarray:
    """Split-B indices of the attribution run, from its shard's ``split`` record."""
    n_total = int(split["n_total"])
    if not split["use_split"]:
        return np.arange(n_total)
    perm = stable_rng(int(split["split_seed"]), "split", fam).permutation(n_total)
    idx_b = np.sort(perm[int(split["n_attr"]) :])
    if idx_b.size != int(split["n_eval"]):
        raise ValueError(f"{fam}: re-derived split B has {idx_b.size} items, shard records n_eval={split['n_eval']}")
    return idx_b


def logit_shift(proj: tuple[float, float], base: tuple[float, float]) -> dict[str, float]:
    return {
        "yp": proj[0],
        "ym": proj[1],
        "dy_plus": proj[0] - base[0],
        "dy_minus": proj[1] - base[1],
        "dmargin": (proj[0] - proj[1]) - (base[0] - base[1]),
    }


def converse_family(
    h: torch.Tensor,
    yp: torch.Tensor,
    ym: torch.Tensor,
    D_top: torch.Tensor,
    Ks: list[int],
    *,
    W_native_t: torch.Tensor,
    W_native_s: torch.Tensor,
) -> dict:
    """Baseline and per-K projected margins on already-selected eval items.

    h: (N, d) hidden states; yp, ym: (N,) answer ids; D_top: (K_max, d) decoder
    rows in selection order; W_native_t / W_native_s: (V, d) native readouts.
    """
    base_t = logit_pair(h, W_native_t, yp, ym)
    base_s = logit_pair(h, W_native_s, yp, ym)
    out: dict = {
        "baseline": {
            "native_t": RS.margin_summary(RS.compute_margin(h, W_native_t, yp, ym)),
            "native_s": RS.margin_summary(RS.compute_margin(h, W_native_s, yp, ym)),
            "logit_native_t": {"yp": base_t[0], "ym": base_t[1]},
            "logit_native_s": {"yp": base_s[0], "ym": base_s[1]},
        },
        "per_K": {},
    }
    h_norm = h.float().norm(dim=-1)  # (N,)
    for K in Ks:
        K = min(K, D_top.shape[0])
        h_proj = project_out(h, D_top[:K])  # (N, d)
        out["per_K"][K] = {
            "K": int(K),
            "h_residual_frac": (h_proj.norm(dim=-1) / h_norm.clamp_min(1e-12)).mean().item(),  # |h_proj| / |h|
            "summary_proj_native_t": RS.margin_summary(RS.compute_margin(h_proj, W_native_t, yp, ym)),
            "summary_proj_native_s": RS.margin_summary(RS.compute_margin(h_proj, W_native_s, yp, ym)),
            "logit_proj_native_t": logit_shift(logit_pair(h_proj, W_native_t, yp, ym), base_t),
            "logit_proj_native_s": logit_shift(logit_pair(h_proj, W_native_s, yp, ym), base_s),
        }
    return out


def run_family(
    fam: str, args: argparse.Namespace, *, W_native_t: torch.Tensor, W_native_s: torch.Tensor
) -> dict | None:
    ex = CT.load_examples(args.datasets_dir / f"{fam}.jsonl")
    if not ex:
        return None
    h_path = args.hidden_dir / f"{fam}_h{args.h_step}.pt"
    shard = args.attribution_dir / "shards" / f"{fam}__h{args.h_step}__s{args.snapshot_step}.json"
    for p in (h_path, shard, shard.with_suffix(".pt")):
        if not p.exists():
            print(f"[skip] {fam}: missing {p}", flush=True)
            return None
    h = torch.load(h_path, map_location="cpu", weights_only=False).float()  # (N, d)
    if h.shape[0] != len(ex):
        n = min(h.shape[0], len(ex))
        h, ex = h[:n], ex[:n]
    split = json.loads(shard.read_text())["split"]
    if int(split["n_total"]) != h.shape[0]:
        raise ValueError(f"{fam}: {h.shape[0]} examples here vs n_total={split['n_total']} in {shard}")
    idx_b = torch.from_numpy(held_out_indices(fam, split))
    yp = torch.tensor([e.y_plus for e in ex], dtype=torch.long)[idx_b]
    ym = torch.tensor([e.y_minus for e in ex], dtype=torch.long)[idx_b]
    payload = torch.load(shard.with_suffix(".pt"), map_location="cpu", weights_only=False)
    D_top = payload["decoder_rows_top_Kmax"].float()  # (K_max, d)

    result = converse_family(h[idx_b], yp, ym, D_top, args.K, W_native_t=W_native_t, W_native_s=W_native_s)
    return {
        "model": args.model,
        "family": fam,
        "h_step": args.h_step,
        "snapshot_step": args.snapshot_step,
        "n_eval": int(idx_b.numel()),
        "use_split": bool(split["use_split"]),
        "split": split,
        "top_Kmax_indices": [int(i) for i in payload["decoder_rows_top_Kmax_indices"]],
        **result,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODEL_HF_NAMES), required=True)
    ap.add_argument("--datasets-dir", type=Path, required=True)
    ap.add_argument("--hidden-dir", type=Path, required=True)
    ap.add_argument(
        "--attribution-dir",
        type=Path,
        required=True,
        help="run_feature_attribution.py --out-dir for the same (h, s) cell; its shards/ supply the "
        "decoder rows and the split.",
    )
    ap.add_argument("--h-step", type=int, required=True)
    ap.add_argument("--snapshot-step", type=int, required=True)
    ap.add_argument("--K", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--families", nargs="+", default=None, help="Default: every <family>.jsonl (not *_corrupt).")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    seed_everything(0)  # deterministic pipeline; seeded for convention
    provenance = log_run_provenance(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_hf = MODEL_HF_NAMES[args.model]
    W_native_t = load_snapshot(model_hf, args.h_step)  # (V, d) fp32
    W_native_s = load_snapshot(model_hf, args.snapshot_step)  # (V, d) fp32

    if args.families is None:
        families = [p.stem for p in sorted(args.datasets_dir.glob("*.jsonl")) if not p.name.endswith("_corrupt.jsonl")]
    else:
        families = args.families
    print(f"[setup] model={model_hf} families={families} h={args.h_step} s={args.snapshot_step} K={args.K}", flush=True)

    done: dict[str, dict] = {}
    for fam in families:
        out_p = args.out_dir / f"{fam}__h{args.h_step}__s{args.snapshot_step}.json"
        if out_p.exists():
            print(f"  [skip] {out_p} exists", flush=True)
            done[fam] = json.loads(out_p.read_text())["split"]
            continue
        result = run_family(fam, args, W_native_t=W_native_t, W_native_s=W_native_s)
        if result is None:
            continue
        atomic_write_json(out_p, result)
        done[fam] = result["split"]
        base = result["baseline"]["native_t"]["accuracy"]
        for K, d in result["per_K"].items():
            print(
                f"  [{fam} K={K:>4}] native_t acc={base:.3f} -> proj {d['summary_proj_native_t']['accuracy']:.3f} "
                f"(dmargin={d['logit_proj_native_t']['dmargin']:+.3f}, dy+={d['logit_proj_native_t']['dy_plus']:+.3f}, "
                f"dy-={d['logit_proj_native_t']['dy_minus']:+.3f}) |h_proj|/|h|={d['h_residual_frac']:.3f}",
                flush=True,
            )

    atomic_write_json(
        args.out_dir / "manifest.json",
        {
            "model": model_hf,
            "h_step": args.h_step,
            "snapshot_step": args.snapshot_step,
            "K": args.K,
            "datasets_dir": str(args.datasets_dir),
            "hidden_dir": str(args.hidden_dir),
            "attribution_dir": str(args.attribution_dir),
            "per_family_split": done,
            "git_commit": git_commit(),
            "provenance": provenance,
        },
    )
    print(f"[done] -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
