"""Availability vs expression per (family, checkpoint): the probe summary behind Figure 4 (right) and Appendix G.

For every family with a dataset and cached hidden states, and every cached
checkpoint, writes one row of ``readout.probes.availability_expression`` metrics:
probe availability (A1 shrinkage-LDA, A_full logistic, label nulls) on post-LN
``h``, and native / best-swept readout expression plus full-vocab rank on pre-LN
``z`` with each checkpoint's own final LayerNorm and W_U.

Inputs:
  <hidden-dir>/<family>_h<step>.pt, <family>_z<step>.pt   (extract_hidden_dense.py)
  <datasets-dir>/<family>.jsonl                          (build_task_datasets.py --task-set availability)
  <wu-dir>/<slug>_step<S>_wu.pt, <slug>_step<S>_lnf.pt   (swept readouts: --swept-steps, or every S with both)

Outputs (resumable per cell):
  <out-dir>/probe_summary.csv, probe_summary.pt (rows incl. fold accuracies),
  probe_summary.provenance.json

Usage:
    uv run python experiments/probes/contrastive_readout_swap/scripts/run_availability_probes.py \\
        --model EleutherAI/pythia-6.9b --hidden-dir <run>/hidden --datasets-dir <datasets> \\
        --group-split --out-dir <run>/probes
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from readout.core.paths import model_slug, snapshot_dir
from readout.core.repro import log_run_provenance, seed_everything
from readout.core.resume import atomic_write_json, atomic_write_torch
from readout.probes.availability_expression import (
    availability_probe,
    derive_groups,
    empty_probe,
    feature_label,
    native_and_best_swept,
)


class SnapshotDir:
    """Per-checkpoint W_U (mmap-loaded, cached) and final-LN params under one directory."""

    def __init__(self, wu_dir: Path, slug: str, only: list[int] | None = None):
        self.wu_dir, self.slug = wu_dir, slug
        self._wu: dict[int, torch.Tensor | None] = {}
        steps = []
        for p in wu_dir.glob(f"{slug}_step*_wu.pt"):
            try:
                s = int(p.name[len(slug) + len("_step") : -len("_wu.pt")])
            except ValueError:
                continue
            if (only is None or s in only) and self.lnf(s) is not None:
                steps.append(s)
        self.steps = sorted(steps)

    def wu(self, step: int) -> torch.Tensor | None:
        if step not in self._wu:
            p = self.wu_dir / f"{self.slug}_step{step}_wu.pt"
            obj = None
            if p.exists():
                # mmap keeps ~0.8 GB per 6.9B snapshot out of RSS; values are unchanged.
                obj = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
                if isinstance(obj, dict):
                    obj = obj["W_U"] if "W_U" in obj else obj["weight"] if "weight" in obj else next(iter(obj.values()))
                obj = obj.float()  # (V, d)
            self._wu[step] = obj
        return self._wu[step]

    def lnf(self, step: int) -> dict | None:
        p = self.wu_dir / f"{self.slug}_step{step}_lnf.pt"
        if not p.exists():
            return None
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "weight" not in obj:
            return None
        bias = obj.get("bias")
        return {
            "weight": obj["weight"].float(),
            "bias": bias.float() if bias is not None else None,
            "eps": float(obj.get("eps", 1e-5)),
        }


def load_hidden(hidden_dir: Path, family: str, step: int) -> tuple[np.ndarray, np.ndarray | None] | None:
    """(post-LN h, pre-LN z or None), each (N, d) fp32; None if the h cache is missing."""
    hp = hidden_dir / f"{family}_h{step}.pt"
    if not hp.exists():
        return None
    obj = torch.load(hp, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):  # {"h": ..., "z": ...} layout
        h, z = obj.get("h"), obj.get("z")
        h = h if h is not None else z
        return h.float().numpy(), (z.float().numpy() if z is not None else None)
    zp = hidden_dir / f"{family}_z{step}.pt"
    z = torch.load(zp, map_location="cpu", weights_only=False).float().numpy() if zp.exists() else None
    return obj.float().numpy(), z


def run_cell(examples: list[dict], h: np.ndarray, z: np.ndarray | None, family: str, h_step: int, args, snaps) -> dict:
    unprobed = any(family.startswith(p) for p in args.unprobed_prefixes)
    feats = [None if unprobed else feature_label(ex) for ex in examples]
    ungrouped = any(family.startswith(p) for p in args.stratified_prefixes)
    groups = derive_groups(examples) if args.group_split and not ungrouped else None
    if args.group_split and groups is None:
        print(f"  [{family:<14} h={h_step:>6}] no group key; StratifiedKFold", flush=True)

    h_randinit = None
    if args.random_init_hidden_dir is not None:
        loaded = load_hidden(args.random_init_hidden_dir, family, h_step)
        if loaded is None:
            alt = sorted(args.random_init_hidden_dir.glob(f"{family}_h*.pt"))
            loaded = (
                load_hidden(args.random_init_hidden_dir, family, int(alt[0].stem.rpartition("_h")[2])) if alt else None
            )
        h_randinit = loaded[0] if loaded is not None else None

    # Probe the labeled subset (e.g. BLiMP anaphor gender is defined for herself/himself only).
    labeled_idx = np.array([i for i, f in enumerate(feats) if f is not None])
    labeled = [feats[i] for i in labeled_idx]
    if len(labeled_idx) >= 4 and len(set(labeled)) >= 2:
        code = {lab: i for i, lab in enumerate(sorted(set(labeled)))}
        probe = availability_probe(
            h[labeled_idx],
            np.array([code[f] for f in labeled]),
            n_splits=args.n_splits,
            seed=args.seed,
            groups=groups[labeled_idx] if groups is not None else None,
            h_randinit=h_randinit[labeled_idx] if h_randinit is not None else None,
        )
        if len(labeled_idx) < len(feats):
            probe["n_labeled"], probe["n_total"] = int(len(labeled_idx)), int(len(feats))
    else:
        probe = empty_probe(len(examples), "no feature label (open-vocab breadth)")
    print(
        f"  [{family:<14} h={h_step:>6}] A1={probe['probe_acc_1d']:.3f} Afull={probe['probe_acc']:.3f} "
        f"shuffle={probe['probe_acc_label_shuffle']:.3f} n={probe['n']} K={probe['n_classes']} "
        f"[{probe['probe_splitter']}] {probe['skip_reason']}",
        flush=True,
    )
    readout = native_and_best_swept(
        z,
        np.array([int(ex["y_plus"]) for ex in examples]),
        np.array([int(ex["y_minus"]) for ex in examples]),
        h_step,
        snaps,
        seed=args.seed,
        device=args.device,
    )
    return {
        "family": family,
        "h_step": h_step,
        "alignment": "none",
        **{k: v for k, v in probe.items() if not isinstance(v, list)},
        **readout,
        "_probe_folds": probe.get("fold_acc"),
        "_probe_folds_1d": probe.get("fold_acc_1d"),
        "_probe_folds_label_shuffle": probe.get("fold_acc_label_shuffle"),
        "_probe_folds_random_label": probe.get("fold_acc_random_label"),
        "_probe_folds_randinit": probe.get("fold_acc_randinit"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--model", required=True, help="HF id, e.g. EleutherAI/pythia-6.9b (snapshot slug + default --wu-dir)"
    )
    ap.add_argument("--hidden-dir", type=Path, required=True, help="<family>_h<step>.pt / _z<step>.pt caches")
    ap.add_argument("--datasets-dir", type=Path, required=True, help="<family>.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--wu-dir", type=Path, default=None, help="W_U + final-LN snapshots (default: snapshot_dir(model))")
    ap.add_argument("--families", nargs="+", default=None, help="default: every family with a cache and a dataset")
    ap.add_argument("--h-steps", type=int, nargs="+", default=None, help="default: every cached step per family")
    ap.add_argument(
        "--swept-steps",
        type=int,
        nargs="+",
        default=None,
        help="readout checkpoints to sweep (default: every step with _wu.pt and _lnf.pt in --wu-dir)",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-split", action="store_true", help="GroupKFold on derive_group_key (paper setting)")
    ap.add_argument(
        "--stratified-prefixes",
        nargs="*",
        default=[],
        help="family prefixes probed with stratified folds even under --group-split "
        "(`numeric` reproduces the published Pythia-1B breadth run)",
    )
    ap.add_argument(
        "--unprobed-prefixes",
        nargs="*",
        default=[],
        help="family prefixes given no probe label (`blimp` reproduces the published Pythia-1B breadth run)",
    )
    ap.add_argument("--random-init-hidden-dir", type=Path, default=None, help="optional untrained-model floor")
    ap.add_argument("--device", default="cpu", help="device for the full-vocab rank GEMMs (cpu / cuda)")
    args = ap.parse_args()

    seed_everything(args.seed)
    provenance = log_run_provenance(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wu_dir = args.wu_dir or snapshot_dir(args.model)
    snaps = SnapshotDir(wu_dir, model_slug(args.model), args.swept_steps)
    print(f"[setup] {len(snaps.steps)} swept readouts with W_U+LN under {wu_dir}: {snaps.steps}", flush=True)

    available: dict[str, set[int]] = {}
    for p in sorted(args.hidden_dir.glob("*_h*.pt")):
        fam, _, step = p.stem.rpartition("_h")
        if step.isdigit():
            available.setdefault(fam, set()).add(int(step))
    families = [f for f in (args.families or sorted(available)) if (args.datasets_dir / f"{f}.jsonl").exists()]
    if not families:
        raise FileNotFoundError(f"no family has both a cache in {args.hidden_dir} and a jsonl in {args.datasets_dir}")

    out_csv, out_pt = args.out_dir / "probe_summary.csv", args.out_dir / "probe_summary.pt"
    rows: list[dict] = torch.load(out_pt, weights_only=False) if out_pt.exists() else []
    done = {(str(r["family"]), int(r["h_step"])) for r in rows}
    if rows:
        print(f"[resume] {len(rows)} cells already in {out_pt}", flush=True)
    atomic_write_json(
        args.out_dir / "probe_summary.provenance.json",
        {
            **provenance,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "wu_dir": str(wu_dir),
            "swept_steps": snaps.steps,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "machine": platform.machine(),
        },
    )

    def flush() -> None:
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        tmp = out_csv.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(out_csv)
        atomic_write_torch(out_pt, rows)

    for fam in families:
        examples = [json.loads(line) for line in (args.datasets_dir / f"{fam}.jsonl").open()]
        for step in args.h_steps or sorted(available.get(fam, set())):
            if (fam, int(step)) in done:
                continue
            loaded = load_hidden(args.hidden_dir, fam, step)
            if loaded is None:
                print(f"[skip] no hidden cache for {fam} h={step}", flush=True)
                continue
            h, z = loaded  # (N, d) each
            if h.shape[0] != len(examples):
                print(f"[skip] {fam} h={step}: hidden N={h.shape[0]} != examples N={len(examples)}", flush=True)
                continue
            rows.append(run_cell(examples, h, z, fam, int(step), args, snaps))
            done.add((fam, int(step)))
            flush()
    print(f"[done] {len(rows)} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
