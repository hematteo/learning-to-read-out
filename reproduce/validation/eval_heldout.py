"""Held-out checkpoint evaluation for the W_U crosscoder (memory-frugal).

Uses the released 32-snapshot crosscoder as-is and evaluates reconstruction at
8 Pythia checkpoints that were *never* in the 32-step training schedule.

Key identity: the crosscoder's encoder is a sum over heads of
  (x_norm_k @ W_E_k) + b_E_k.
We precompute hp_trained_sum (V, d_sae) once over the 32 trained heads, then
each held-out evaluation only adds a single (V, d_sae) synthetic-head term —
no (B, K+1, d_sae) intermediate, no streaming of all 40 raw snapshots
together.

Reconstruction protocol (matches llamascopium Crosscoder forward pass):
    accumulated   = hp_trained_sum + (W_norm(t*) @ syn_W_E + syn_b_E)
    syn_dn        = ||syn_W_D||_d                                # (D,)
    feature_acts  = JumpReLU(accumulated * syn_dn, threshold) / syn_dn
    recon_norm    = feature_acts @ syn_W_D + syn_b_D
    recon_raw     = recon_norm * scale(t*) + mean(t*)
    EV            = 1 - var(W_U(t*) - recon_raw) / var(W_U(t*))

Baselines on the same held-out steps:
    nn_head             nearer trained neighbor's head (no interpolation)
    direct_wu_interp    (1-α)·W_U(t-) + α·W_U(t+), no crosscoder
    endpoint            log-α interp between W_U(0) and W_U(143000)
    pca_low_rank        project onto top-r right-singular subspace of stacked
                        normalized trained snaps; unnormalize with t*'s stats
    trained_neighbors   in-sample EV at t- and t+  (ceiling)
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path

import torch

from readout.core.data import center_scale_stats, explained_variance
from readout.core.model_specs import DEFAULT_STEPS_32
from readout.core.repro import git_commit, log_run_provenance
from readout.crosscoder import inference  # noqa: E402
from readout.crosscoder.checkpoints import load_checkpoint
from readout.crosscoder.snapshots import load_snapshot_at

# 32-snapshot anchor schedule — matches the released seed0 config.
TRAINED_STEPS = DEFAULT_STEPS_32
# Pythia checkpoints absent from TRAINED_STEPS — never seen by the crosscoder.
HELD_OUT = [10000, 23000, 33000, 43000, 73000, 93000, 113000, 123000]


# ──────────────────────────────────────────────────────────────────────
# Snapshot I/O + preprocessing  (mirrors wu_adapter.preprocess_snapshots)
# ──────────────────────────────────────────────────────────────────────


def load_snapshot(cache_dir: Path, step: int, model_slug: str = "EleutherAI_pythia-160m"):
    return load_snapshot_at(cache_dir / f"{model_slug}_step{step}_wu.pt")


# ──────────────────────────────────────────────────────────────────────
# Crosscoder I/O
# ──────────────────────────────────────────────────────────────────────


def load_crosscoder_state(safetensors_path: Path) -> dict[str, torch.Tensor]:
    return load_checkpoint(safetensors_path).state_dict


def load_pt_checkpoint_state(pt_path: Path):
    cp = load_checkpoint(pt_path)
    sd = cp.state_dict
    out = {k: sd[k] for k in ("W_E", "b_E", "W_D", "b_D")}
    for k in sd:
        if "log_jumprelu_threshold" in k:
            out["activation_function.log_jumprelu_threshold"] = sd[k]
            break
    return out, cp.raw


# ──────────────────────────────────────────────────────────────────────
# Interpolation helpers
# ──────────────────────────────────────────────────────────────────────


def find_neighbors(t_star: int, trained_steps: list[int]) -> tuple[int, int]:
    below = [t for t in trained_steps if t < t_star]
    above = [t for t in trained_steps if t > t_star]
    if not below or not above:
        raise ValueError(f"step {t_star} has no two-sided neighbors in {trained_steps}")
    return max(below), min(above)


def alpha_log(t: int, t_minus: int, t_plus: int) -> float:
    a = math.log1p(t) - math.log1p(t_minus)
    b = math.log1p(t_plus) - math.log1p(t_minus)
    return a / b


# ──────────────────────────────────────────────────────────────────────
# Per-head decode given the precomputed encoder sum
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def decode_with_head(
    accumulated: torch.Tensor,  # (V, D) — total summed pre-activation
    head_W_D: torch.Tensor,  # (D, d)
    head_b_D: torch.Tensor,  # (d,)
    threshold: torch.Tensor,  # (D,)
    batch_size: int = 4096,
) -> torch.Tensor:
    """Apply per-head JumpReLU gating + decode for ONE head.

    Implements llamascopium's
        feature_acts = JumpReLU(accumulated * decoder_norm, θ) / decoder_norm
    for a single head, then decode through (head_W_D, head_b_D).
    """
    V, D = accumulated.shape
    d = head_W_D.shape[-1]
    dn = head_W_D.norm(dim=-1)  # (D,)
    out = torch.empty(V, d, dtype=accumulated.dtype, device=accumulated.device)
    for start in range(0, V, batch_size):
        end = min(start + batch_size, V)
        feature_acts, _ = inference.jumprelu_feature_acts(accumulated[start:end], threshold, dn)
        out[start:end] = feature_acts @ head_W_D + head_b_D
    return out


# ──────────────────────────────────────────────────────────────────────
# Main eval
# ──────────────────────────────────────────────────────────────────────


def run_eval(
    crosscoder_path: Path,
    snapshot_dir: Path,
    out_dir: Path,
    held_out: list[int] = HELD_OUT,
    trained_steps: list[int] = TRAINED_STEPS,
    pca_rank: int = 50,
    batch_size: int = 4096,
    device: str = "cpu",
    model_slug: str = "EleutherAI_pythia-160m",
    self_check: bool = True,
    seed: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    overlap = set(trained_steps) & set(held_out)
    if overlap:
        raise ValueError(f"held_out steps overlap with trained_steps: {sorted(overlap)}")
    print(f"[eval] trained_steps (K={len(trained_steps)}): {trained_steps}")
    print(f"[eval] held_out: {held_out}")

    # 1. Crosscoder state
    if str(crosscoder_path).endswith(".safetensors"):
        state = load_crosscoder_state(crosscoder_path)
        raw_meta = None
    else:
        state, raw_meta = load_pt_checkpoint_state(crosscoder_path)
    state = {k: v.to(device) for k, v in state.items()}
    K = state["W_E"].shape[0]
    if K != len(trained_steps):
        raise ValueError(f"Crosscoder has K={K} heads but trained_steps has {len(trained_steps)}.")
    if raw_meta is not None and "steps" in raw_meta and list(raw_meta["steps"]) != trained_steps:
        raise ValueError("Saved training steps do not match TRAINED_STEPS.")
    W_E = state["W_E"]  # (K, d, D)
    b_E = state["b_E"]  # (K, D)
    W_D = state["W_D"]  # (K, D, d)
    b_D = state["b_D"]  # (K, d)
    threshold = state["activation_function.log_jumprelu_threshold"].exp()  # (D,)

    # 2. Stream pass 1: per-snap stats + accumulate hp_trained_sum +
    #    PCA-fit row subsample.  Never holds more than one normalized snap at
    #    a time, so peak memory is dominated by hp_trained_sum (V*D*4 B) and
    #    the crosscoder state, not by K*V*d.
    print(f"[eval] streaming pass over {len(trained_steps)} trained snaps...")
    trained_stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    # .to(device) here keeps the first iteration's matmul (and its
    # trained_stats entry) on the same device as every later snapshot.
    raw0 = load_snapshot(snapshot_dir, trained_steps[0], model_slug).to(device)
    V, d = raw0.shape
    D = W_E.shape[-1]
    print(f"[eval] V={V}, d={d}, D={D}")
    hp_trained_sum = torch.zeros(V, D, dtype=torch.float32, device=device)
    # PCA fit subsample: ~3k rows per snap × 32 snaps = ~100k rows.  This is
    # plenty for a rank-50 fit on d_model in [768, 4096], and 100k×4096×4 B
    # = 1.6 GB — far cheaper than the full (K*V, d) matrix at 1B (13 GB).
    rows_per_snap = max(2048, min(5000, V // 16))
    pca_buf = torch.empty(rows_per_snap * K, d, dtype=torch.float32, device=device)
    rng = torch.Generator(device="cpu").manual_seed(0)
    for k, s in enumerate(trained_steps):
        raw = raw0 if s == trained_steps[0] else load_snapshot(snapshot_dir, s, model_slug).to(device)
        m, sc = center_scale_stats(raw)
        trained_stats[s] = (m, sc)
        normed = (raw - m) / sc
        hp_trained_sum += normed @ W_E[k]
        idx = torch.randperm(V, generator=rng)[:rows_per_snap]
        pca_buf[k * rows_per_snap : (k + 1) * rows_per_snap] = normed[idx]
        del raw, normed
    del raw0
    hp_trained_sum += b_E.sum(dim=0)
    gc.collect()

    # 3. PCA fit on the row subsample.
    print(f"[eval] PCA fit (r={pca_rank}) on {pca_buf.shape[0]} subsampled rows...")
    rows_mean = pca_buf.mean(dim=0, keepdim=True)
    centered = pca_buf - rows_mean
    torch.manual_seed(seed)  # svd_lowrank draws a random test matrix
    _, _, V_r = torch.svd_lowrank(centered, q=pca_rank, niter=5)  # V_r: (d, r)
    del centered, pca_buf
    gc.collect()

    # 4. In-sample EV per trained head: load each raw transiently, compare
    #    against the unnormalized recon for raw EV; compute normalized EV via
    #    the (raw - m) / sc identity (saves a redundant load).
    ceiling_ev_norm: dict[int, float] = {}
    ceiling_ev_raw: dict[int, float] = {}
    print("[eval] computing trained-head ceiling EV (norm + raw)...")
    for k, s in enumerate(trained_steps):
        recon_norm = decode_with_head(hp_trained_sum, W_D[k], b_D[k], threshold, batch_size=batch_size)
        m_k, sc_k = trained_stats[s]
        raw_k = load_snapshot(snapshot_dir, s, model_slug).to(device)
        recon_raw = recon_norm * sc_k + m_k
        ceiling_ev_raw[s] = explained_variance(raw_k, recon_raw)
        normed_k = (raw_k - m_k) / sc_k
        ceiling_ev_norm[s] = explained_variance(normed_k, recon_norm)
        del recon_norm, recon_raw, raw_k, normed_k
    if self_check:
        mean_ev_norm = sum(ceiling_ev_norm.values()) / len(ceiling_ev_norm)
        print(f"  mean trained EV (normalized) across {K} heads = {mean_ev_norm:.4f}")
    print(f"[eval] trained-snap raw-EV range: {min(ceiling_ev_raw.values()):.4f} – {max(ceiling_ev_raw.values()):.4f}")

    # 7. Endpoint raw snaps (kept until end of held-out loop).
    first_step, last_step = trained_steps[0], trained_steps[-1]
    raw_first = load_snapshot(snapshot_dir, first_step, model_slug).to(device)
    raw_last = load_snapshot(snapshot_dir, last_step, model_slug).to(device)

    # 8. Held-out loop.
    rows: list[dict] = []
    recon_pt: dict[int, dict] = {}
    for t_star in held_out:
        t_minus, t_plus = find_neighbors(t_star, trained_steps)
        a = alpha_log(t_star, t_minus, t_plus)
        i_minus, i_plus = trained_steps.index(t_minus), trained_steps.index(t_plus)

        # Synthetic interpolated head.
        syn_W_E = (1 - a) * W_E[i_minus] + a * W_E[i_plus]  # (d, D)
        syn_b_E = (1 - a) * b_E[i_minus] + a * b_E[i_plus]  # (D,)
        syn_W_D = (1 - a) * W_D[i_minus] + a * W_D[i_plus]  # (D, d)
        syn_b_D = (1 - a) * b_D[i_minus] + a * b_D[i_plus]  # (d,)

        # Held-out raw + normalize with own stats.
        held_raw = load_snapshot(snapshot_dir, t_star, model_slug).to(device)
        held_mean, held_scale = center_scale_stats(held_raw)
        held_norm = (held_raw - held_mean) / held_scale

        # Add the synthetic head's encoder contribution.
        hp_synth = held_norm @ syn_W_E + syn_b_E  # (V, D)
        accumulated = hp_trained_sum + hp_synth  # (V, D)

        recon_synth_norm = decode_with_head(accumulated, syn_W_D, syn_b_D, threshold, batch_size=batch_size)
        recon_synth_raw = recon_synth_norm * held_scale + held_mean
        ev_synth = explained_variance(held_raw, recon_synth_raw)
        del hp_synth, recon_synth_norm

        # NN-head: replace synthetic with the closer neighbor's head verbatim.
        nn_step = t_minus if a < 0.5 else t_plus
        nn_idx = trained_steps.index(nn_step)
        nn_hp = held_norm @ W_E[nn_idx] + b_E[nn_idx]
        nn_accum = hp_trained_sum + nn_hp
        recon_nn_norm = decode_with_head(nn_accum, W_D[nn_idx], b_D[nn_idx], threshold, batch_size=batch_size)
        recon_nn_raw = recon_nn_norm * held_scale + held_mean
        ev_nn = explained_variance(held_raw, recon_nn_raw)
        del nn_hp, nn_accum, recon_nn_norm

        # Direct W_U linear interpolation (load neighbor raws, free immediately).
        raw_minus = load_snapshot(snapshot_dir, t_minus, model_slug).to(device)
        raw_plus = load_snapshot(snapshot_dir, t_plus, model_slug).to(device)
        direct = (1 - a) * raw_minus + a * raw_plus
        ev_direct = explained_variance(held_raw, direct)
        del raw_minus, raw_plus

        # Endpoint baseline.
        beta = alpha_log(t_star, first_step, last_step)
        endpoint = (1 - beta) * raw_first + beta * raw_last
        ev_endpoint = explained_variance(held_raw, endpoint)

        # PCA r=pca_rank: project normalized held onto V_r, unnormalize.
        held_norm_centered = held_norm - rows_mean
        proj_norm = held_norm_centered @ V_r @ V_r.T + rows_mean
        proj_raw = proj_norm * held_scale + held_mean
        ev_pca = explained_variance(held_raw, proj_raw)
        del held_norm_centered, proj_norm

        rows.append(
            {
                "held_out_step": t_star,
                "t_minus": t_minus,
                "t_plus": t_plus,
                "alpha_log": round(a, 4),
                "ev_synth_interp": ev_synth,
                "ev_nn_head": ev_nn,
                "ev_direct_wu_interp": ev_direct,
                "ev_endpoint": ev_endpoint,
                f"ev_pca_r{pca_rank}": ev_pca,
                "ev_ceiling_t_minus_raw": ceiling_ev_raw[t_minus],
                "ev_ceiling_t_plus_raw": ceiling_ev_raw[t_plus],
                "ev_ceiling_t_minus_norm": ceiling_ev_norm[t_minus],
                "ev_ceiling_t_plus_norm": ceiling_ev_norm[t_plus],
            }
        )
        recon_pt[t_star] = {
            "synth_residual_norm": (held_raw - recon_synth_raw).norm(dim=-1).cpu(),
            "nn_residual_norm": (held_raw - recon_nn_raw).norm(dim=-1).cpu(),
            "direct_residual_norm": (held_raw - direct).norm(dim=-1).cpu(),
            "endpoint_residual_norm": (held_raw - endpoint).norm(dim=-1).cpu(),
            "pca_residual_norm": (held_raw - proj_raw).norm(dim=-1).cpu(),
            "true_row_norm": held_raw.norm(dim=-1).cpu(),
        }
        print(
            f"  t*={t_star:>6d}  α={a:.3f}  synth={ev_synth:.4f}  nn={ev_nn:.4f}  "
            f"direct={ev_direct:.4f}  endpoint={ev_endpoint:.4f}  pca{pca_rank}={ev_pca:.4f}  "
            f"ceiling_raw=[{ceiling_ev_raw[t_minus]:.4f}, {ceiling_ev_raw[t_plus]:.4f}]"
        )
        del held_raw, held_norm, accumulated
        gc.collect()

    # 9. Persist.
    csv_path = out_dir / "heldout_ev.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[eval] wrote {csv_path}")

    pt_path = out_dir / "heldout_ev.pt"
    torch.save(
        {
            "rows": rows,
            "trained_steps": trained_steps,
            "held_out": held_out,
            "ceiling_ev_raw": ceiling_ev_raw,
            "ceiling_ev_norm": ceiling_ev_norm,
            "residuals": recon_pt,
            "pca_rank": pca_rank,
            "seed": seed,
            "crosscoder_path": str(crosscoder_path),
            "git_commit": git_commit(),
        },
        pt_path,
    )
    print(f"[eval] wrote {pt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crosscoder", type=Path, required=True)
    p.add_argument("--snapshot-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pca-rank", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-slug", default="EleutherAI_pythia-160m")
    p.add_argument("--no-self-check", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="seeds the svd_lowrank PCA fit")
    args = p.parse_args()
    log_run_provenance(seed=args.seed)
    run_eval(
        crosscoder_path=args.crosscoder,
        snapshot_dir=args.snapshot_dir,
        out_dir=args.out_dir,
        pca_rank=args.pca_rank,
        batch_size=args.batch_size,
        device=args.device,
        model_slug=args.model_slug,
        self_check=not args.no_self_check,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
