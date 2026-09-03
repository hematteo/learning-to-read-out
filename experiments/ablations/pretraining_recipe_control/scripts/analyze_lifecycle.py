#!/usr/bin/env python3
"""Feature-lifecycle statistics of the per-condition recipe-control trajectory crosscoders.

For each condition's crosscoder (``train_control_crosscoder.py``) this computes the
same per-feature lifecycle signals the paper uses for the published models and
persists them so the conditions can be compared:

  * decoder norm trajectory  rho_f(t) = ||D_f^(t)||_2 / max_t ||D_f^(t)||_2   (K, D)
  * alive threshold rho >= 0.5 (``readout.dynamics.lifecycle.ALIVE_THR``): first alive
    step, lifespan in snapshots; peak step / peak tau (log-step time in [0, 1])
  * rule-based profiles (``readout.dynamics.lifecycle.classify_profiles_refined``,
    ``tab:lifecycle-profile-rules``), counted over active features
  * reorganization window: mean |delta rho| between adjacent checkpoints (norm
    turnover) and mean 1 - cos between adjacent decoder directions (rotation,
    ``readout.dynamics.metrics.adjacent_rotation``); the reorganization step is the
    later checkpoint of the pair with the largest norm turnover
  * firing rate per checkpoint (optional; needs the W_U checkpoints)

Outputs (under ``--out-dir``; no figures are rendered here):
  lifecycle_stats.json           per-condition summary (tab:app-recipe-control-summary,
                                 fig:main-recipe-dose-response inputs)
  lifecycle_summary.csv          the same summary, one row per condition
  median_trajectories.csv        median / IQR of rho over active features per checkpoint
                                 (fig:app-recipe-control-median)
  peak_step_population.csv       fraction of active features peaking at each checkpoint and
                                 total decoder norm mass (fig:app-recipe-control-peakstep)
  reorg_window.csv               norm turnover and decoder rotation per adjacent pair
                                 (fig:app-recipe-control-reorg)
  feature_lifecycles_<cond>.csv  per-feature profile / peak / birth / lifespan
  feature_trajectories.pt        per-condition rho (K, D), decoder norms, peak steps,
                                 profiles, the seeded 500-feature display sample, and
                                 firing rates (fig:app-recipe-control-trajectories)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rc_common import add_common_args, default_results_dir, write_csv, write_provenance

from readout.core.repro import seed_everything
from readout.dynamics.lifecycle import (
    ALIVE_THR,
    PROFILE_LABELS,
    PROFILE_ORDER,
    classify_profiles_refined,
    tau_from_steps,
)
from readout.dynamics.metrics import adjacent_rotation
from readout.probes.recipe_control_models import condition_config, load_readout, val_loss_at

# Paper profile names in the order the summary table reports them.
PROFILES = [PROFILE_LABELS[k] for k in ("persistent", "transitional", "early_decay", "late_emerge", "mixed")]
assert set(PROFILES) == {PROFILE_LABELS[k] for k in PROFILE_ORDER}

COND_LABEL = {
    "baseline": "baseline (1x, warmup 1430)",
    "wu_lr_0p25": "wu_lr 0.25x",
    "wu_lr_4x": "wu_lr 4x",
    "warmup_short": "warmup_short (1x, warmup 715)",
    "warmup_long": "warmup_long (1x, warmup 5720)",
}
QUALITY_KEYS = ("explained_variance", "reconstruction_mse", "mean_l0", "dead_rate")


def load_crosscoder(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def decoder_norms(state_dict: dict) -> np.ndarray:
    W_D = state_dict["W_D"].float()  # (K, D, d)
    return torch.linalg.norm(W_D, dim=-1).cpu().numpy()  # (K, D)


@torch.no_grad()
def firing_rates(ck: dict, ckpt_root: Path, cond: str) -> np.ndarray:
    """Per-checkpoint fraction of W_U rows on which each feature fires, (K, D).

    Mirrors the canonical rate extraction for the published models: the shared
    pre-activation is the sum of the per-snapshot encoder heads on the
    ``center_scale``-preprocessed rows, and a feature fires at snapshot k when it
    exceeds the JumpReLU threshold rescaled by that snapshot's decoder norm.
    """
    sd = ck["state_dict"]
    steps = ck["steps"]
    W_E = sd["W_E"].float()  # (K, d, D)
    b_E = sd["b_E"].float()  # (K, D)
    W_D = sd["W_D"].float()  # (K, D, d)
    thr = sd["activation_function.log_jumprelu_threshold"].exp().float()  # (D,)
    stats = ck.get("preprocess_stats")
    K, _, D = W_E.shape
    dec_norm = torch.linalg.norm(W_D, dim=-1)  # (K, D)
    joint_pre = None
    for i, st in enumerate(steps):
        wu = load_readout(ckpt_root, cond, st)["wu"]  # (V, d)
        if stats is not None:
            wu = (wu - stats["mean"][i].reshape(1, -1)) / stats["scale"][i].reshape(-1)
        head_pre = wu @ W_E[i] + b_E[i]  # (V, D)
        joint_pre = head_pre if joint_pre is None else joint_pre + head_pre
    rates = torch.zeros(K, D)
    for k in range(K):
        rates[k] = (joint_pre > thr / dec_norm[k].clamp_min(1e-12)).float().mean(dim=0)
    return rates.numpy()


def reorg_curves(rho: np.ndarray, W_D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Norm turnover mean|delta rho| and decoder rotation mean(1 - cos), each (K-1,)."""
    dnorm = np.abs(np.diff(rho, axis=0)).mean(axis=1)
    rot = adjacent_rotation(W_D).mean(axis=1)
    return dnorm, rot


def analyse_condition(ck: dict, cond: str, *, sample_size: int, seed: int) -> dict:
    sd = ck["state_dict"]
    steps = np.asarray(ck["steps"], dtype=np.int64)
    norms = decoder_norms(sd)  # (K, D)
    W_D = sd["W_D"].float().cpu().numpy()  # (K, D, d)
    tau = tau_from_steps(steps)
    peak = norms.max(axis=0, keepdims=True) + 1e-12
    rho = norms / peak
    alive = rho >= ALIVE_THR
    active = np.where(alive.any(axis=0))[0]  # features with a nonzero decoder somewhere

    prof = classify_profiles_refined(rho[:, active], steps)
    labels = np.full(norms.shape[1], "inactive", dtype=object)
    labels[active] = [PROFILE_LABELS[p] for p in prof["profile"]]
    peak_idx = norms.argmax(axis=0)
    peak_step = steps[peak_idx]
    peak_tau = tau[peak_idx]
    first_alive = alive.argmax(axis=0)
    birth_step = np.where(alive.any(axis=0), steps[first_alive], -1)
    lifespan = alive.sum(axis=0)

    dnorm_turn, rot_turn = reorg_curves(rho[:, active], W_D[:, active])
    reorg_arg = int(np.argmax(dnorm_turn))
    rot_arg = int(np.argmax(rot_turn))
    total_mass = norms.sum(axis=1)
    total_mass = total_mass / (total_mass.max() + 1e-12)

    rng = np.random.default_rng(seed)
    sample = active if active.size <= sample_size else rng.choice(active, sample_size, replace=False)
    sample = sample[np.argsort(peak_step[sample], kind="stable")]

    prof_counts = {p: int((labels[active] == p).sum()) for p in PROFILES}
    return dict(
        cond=cond,
        steps=steps,
        tau=tau,
        norms=norms,
        rho=rho,
        labels=labels,
        active=active,
        birth_step=birth_step,
        lifespan=lifespan,
        peak_idx=peak_idx,
        peak_step=peak_step,
        peak_tau=peak_tau,
        dnorm_turn=dnorm_turn,
        rot_turn=rot_turn,
        reorg_peak_step=int(steps[reorg_arg + 1]),
        reorg_peak_tau=float(0.5 * (tau[reorg_arg] + tau[reorg_arg + 1])),
        rotation_peak_step=int(steps[rot_arg + 1]),
        total_mass=total_mass,
        prof_counts=prof_counts,
        n_active=int(active.size),
        sample=sample,
        quality=ck.get("quality", {}),
    )


def condition_summary(r: dict, ckpt_root: Path) -> dict:
    cond, act, steps = r["cond"], r["active"], r["steps"]
    cfg = condition_config(ckpt_root, cond)
    final_step = int(steps[-1])
    return dict(
        label=COND_LABEL.get(cond, cond),
        readout_lr_mult=cfg.get("readout_lr_mult"),
        warmup_steps=cfg.get("warmup_steps"),
        n_snapshots=int(len(steps)),
        steps=[int(s) for s in steps],
        d_sae=int(r["norms"].shape[1]),
        n_active=r["n_active"],
        profile_counts=r["prof_counts"],
        profile_fractions={p: r["prof_counts"][p] / max(1, r["n_active"]) for p in PROFILES},
        median_peak_step=float(np.median(r["peak_step"][act])),
        median_peak_tau=float(np.median(r["peak_tau"][act])),
        median_lifespan_snaps=float(np.median(r["lifespan"][act])),
        reorg_peak_step=r["reorg_peak_step"],
        reorg_peak_tau=r["reorg_peak_tau"],
        rotation_peak_step=r["rotation_peak_step"],
        max_norm_turnover=float(r["dnorm_turn"].max()),
        max_decoder_rotation=float(r["rot_turn"].max()),
        val_loss_final=val_loss_at(ckpt_root, cond, final_step),
        val_loss_at_reorg=val_loss_at(ckpt_root, cond, r["reorg_peak_step"]),
        quality={k: float(v) for k, v in r["quality"].items() if isinstance(v, (int, float))},
    )


def summary_row(cond: str, s: dict) -> dict:
    return {
        "condition": cond,
        "readout_lr_mult": s["readout_lr_mult"],
        "warmup_steps": s["warmup_steps"],
        "n_snapshots": s["n_snapshots"],
        "d_sae": s["d_sae"],
        "n_active": s["n_active"],
        **{f"quality_{k}": s["quality"].get(k) for k in QUALITY_KEYS},
        "val_loss_final": s["val_loss_final"],
        "val_loss_at_reorg": s["val_loss_at_reorg"],
        **{f"profile_{p}": s["profile_fractions"][p] for p in PROFILES},
        "median_peak_step": s["median_peak_step"],
        "median_peak_tau": s["median_peak_tau"],
        "median_lifespan_snaps": s["median_lifespan_snaps"],
        "reorg_peak_step": s["reorg_peak_step"],
        "reorg_peak_tau": s["reorg_peak_tau"],
        "rotation_peak_step": s["rotation_peak_step"],
        "max_norm_turnover": s["max_norm_turnover"],
        "max_decoder_rotation": s["max_decoder_rotation"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument(
        "--cc-dir",
        type=Path,
        default=default_results_dir() / "crosscoders",
        help="dir holding cc_<cond>_d<d_sae>_seed<seed>.pt from train_control_crosscoder.py",
    )
    ap.add_argument("--d-sae", type=int, default=8192, help="d_sae in the crosscoder filenames")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <results>/lifecycle_seed<seed>")
    ap.add_argument("--no-rates", action="store_true", help="skip the firing-rate pass (no W_U reload)")
    ap.add_argument("--sample-size", type=int, default=500, help="seeded per-condition display sample")
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = args.out_dir or (default_results_dir() / f"lifecycle_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(out_dir, "analyze_lifecycle", args, args.seed)

    runs: dict[str, dict] = {}
    for cond in args.conditions:
        path = args.cc_dir / f"cc_{cond}_d{args.d_sae}_seed{args.seed}.pt"
        print(f"[{cond}] loading {path}", flush=True)
        ck = load_crosscoder(path)
        r = analyse_condition(ck, cond, sample_size=args.sample_size, seed=args.seed)
        r["rates"] = None if args.no_rates else firing_rates(ck, args.ckpt_root, cond)
        runs[cond] = r
        print(
            f"  N_active={r['n_active']} profiles={r['prof_counts']} "
            f"median_peak_step={np.median(r['peak_step'][r['active']]):.0f} "
            f"reorg_peak_step={r['reorg_peak_step']}",
            flush=True,
        )

    stats = {cond: condition_summary(runs[cond], args.ckpt_root) for cond in args.conditions}
    (out_dir / "lifecycle_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    write_csv(out_dir / "lifecycle_summary.csv", [summary_row(c, stats[c]) for c in args.conditions])

    median_rows, peak_rows, reorg_rows = [], [], []
    for cond in args.conditions:
        r = runs[cond]
        act, steps, tau = r["active"], r["steps"], r["tau"]
        rho_act = r["rho"][:, act]
        q25, q75 = np.quantile(rho_act, [0.25, 0.75], axis=1)
        med = np.median(rho_act, axis=1)
        mean = rho_act.mean(axis=1)
        pk = np.bincount(r["peak_idx"][act], minlength=len(steps)) / max(1, r["n_active"])
        for k in range(len(steps)):
            base = dict(condition=cond, snapshot_idx=k, step=int(steps[k]), tau=float(tau[k]))
            median_rows.append(
                {
                    **base,
                    "median_rho": float(med[k]),
                    "q25_rho": float(q25[k]),
                    "q75_rho": float(q75[k]),
                    "mean_rho": float(mean[k]),
                    "n_active": r["n_active"],
                }
            )
            peak_rows.append({**base, "peak_fraction": float(pk[k]), "total_norm_mass": float(r["total_mass"][k])})
        for k in range(len(steps) - 1):
            reorg_rows.append(
                dict(
                    condition=cond,
                    pair_idx=k,
                    step_lo=int(steps[k]),
                    step_hi=int(steps[k + 1]),
                    step_mid=float(np.sqrt(max(steps[k], 1) * max(steps[k + 1], 1))),
                    tau_mid=float(0.5 * (tau[k] + tau[k + 1])),
                    norm_turnover=float(r["dnorm_turn"][k]),
                    decoder_rotation=float(r["rot_turn"][k]),
                )
            )
        write_csv(
            out_dir / f"feature_lifecycles_{cond}.csv",
            [
                dict(
                    feature=int(f),
                    active=bool(f in set(act.tolist())),
                    profile=str(r["labels"][f]),
                    peak_idx=int(r["peak_idx"][f]),
                    peak_step=int(r["peak_step"][f]),
                    peak_tau=float(r["peak_tau"][f]),
                    birth_step=int(r["birth_step"][f]),
                    lifespan_snaps=int(r["lifespan"][f]),
                    max_decoder_norm=float(r["norms"][:, f].max()),
                )
                for f in range(r["norms"].shape[1])
            ],
        )
    write_csv(out_dir / "median_trajectories.csv", median_rows)
    write_csv(out_dir / "peak_step_population.csv", peak_rows)
    write_csv(out_dir / "reorg_window.csv", reorg_rows)

    torch.save(
        {
            cond: dict(
                steps=torch.as_tensor(r["steps"]),
                tau=torch.as_tensor(r["tau"], dtype=torch.float32),
                rho=torch.as_tensor(r["rho"], dtype=torch.float32),  # (K, D)
                decoder_norms=torch.as_tensor(r["norms"], dtype=torch.float32),  # (K, D)
                active=torch.as_tensor(r["active"]),
                peak_step=torch.as_tensor(r["peak_step"]),
                profile=[str(x) for x in r["labels"]],
                sample_idx=torch.as_tensor(r["sample"]),
                rates=None if r["rates"] is None else torch.as_tensor(r["rates"], dtype=torch.float32),
            )
            for cond, r in runs.items()
        },
        out_dir / "feature_trajectories.pt",
    )
    print(f"wrote lifecycle metrics -> {out_dir}", flush=True)
    print(json.dumps({c: {k: stats[c][k] for k in ("median_peak_step", "reorg_peak_step")} for c in stats}, indent=2))


if __name__ == "__main__":
    main()
