"""Next-token fidelity of the reconstructed W_U readout (fig:app-fidelity-curves).

Matrix EV says whether a crosscoder reconstructs W_U rows. This script asks the
functional question: when held-out hidden states are decoded through the
reconstructed readout, how close are the logits and next-token distributions
to those of the native readout?

Per checkpoint step it reconstructs W_U from the crosscoder (streaming over
vocabulary rows, so at most one (V, d) reconstruction is held), then decodes
the cached pre-readout hidden states h_LN of a fixed eval shard (the first
``--max-seqs`` 512-token sequences of the eval corpus) through the native and
reconstructed readouts. The paper figure plots matrix_ev, logit_r2,
kl_native_to_recon and top1_agreement per step; the text reports delta_nll.

Inputs:
  - released crosscoder  ${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/<model>/W_U/...
  - W_U snapshots        ${UM_SSD_ROOT}/snapshots/<model>/...
  - h_LN caches          --hln-dir/hLN_step<N>.pt (built by scripts/extract/build_hln_cache.py
                         from the same --eval-tokens corpus); ``W_U_orig`` in the cache is the
                         native readout for the functional metrics
  - eval corpus          --eval-tokens (default: the released eval_tokens.pt)

Outputs (resume-safe; one JSON shard per step, aggregated when all are present):
  <out-dir>/shards/step<N>.json
  <out-dir>/readout_functional_fidelity.csv
  <out-dir>/readout_functional_fidelity_metadata.json

Usage (the paper's three curves):
  uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/readout_functional_fidelity.py \\
      --model pythia-160m --d-sae 8192 --seed 0 --steps all \\
      --hln-dir "$UM_SSD_ROOT/derived/readout_edit_timing"
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from readout.core.data import get_device
from readout.core.paths import release_path, repo_root, ssd_path
from readout.core.repro import git_commit, log_run_provenance, seed_everything
from readout.core.resume import aggregate_json_shards, atomic_write_json, iter_undone
from readout.crosscoder import inference
from readout.crosscoder.checkpoints import load_checkpoint
from readout.crosscoder.snapshots import load_snapshots
from readout.crosscoder.wu_adapter import preprocess_snapshots

REPO = repo_root()
SEQ_LEN = 512
DEFAULT_EVAL_TOKENS = ssd_path("hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt")
OUT_ROOT = REPO / "experiments/crosscoders/crosscoder_main/derived/appendix_validation/readout_functional_fidelity"
SNAPSHOT_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def parse_steps(raw: str | None, available: list[int]) -> list[int]:
    """``all``/None -> every checkpoint step; else a comma list that must be in ``available``."""
    if raw is None or raw == "all":
        return list(available)
    requested = [int(x) for x in raw.split(",") if x.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"requested steps not in checkpoint schedule: {missing}")
    return requested


def cast_snapshots(snapshots: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Storage dtype for the K-stack of W_U snapshots (bf16 halves RAM for 1B / 6.9B)."""
    return snapshots.to(SNAPSHOT_DTYPES[dtype_name])


def _empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def reconstruct_one_step(
    sd: dict[str, torch.Tensor],
    x_norm: torch.Tensor,
    native_step: torch.Tensor,
    stats_step: dict[str, torch.Tensor] | None,
    step_idx: int,
    *,
    device: str,
    batch_rows: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reconstruct raw-space W_U at one checkpoint and its matrix EV.

    x_norm: (K, V, d) preprocessed crosscoder inputs; native_step: (V, d) raw
    snapshot at this step; stats_step: {"scale": (1, 1), "mean": (1, d)} or None.
    Returns recon (V, d) fp32 on CPU and {matrix_ev, matrix_sse, matrix_var}.
    """
    K, V, d = x_norm.shape
    thr = inference.jumprelu_threshold(sd, device=device)  # (D,)
    dec_norm = inference.decoder_norms(sd, step_idx, device=device)  # (D,)
    bias = inference.encoder_bias_total(sd, device=device)  # (D,)
    out = torch.empty((V, d), dtype=torch.float32)
    native_mean = native_step.float().mean(dim=0, keepdim=True)  # (1, d)
    if stats_step is not None:
        scale = stats_step["scale"].to(device).float()  # (1, 1)
        mean = stats_step["mean"].to(device).float()  # (1, d)
    sse = 0.0
    var = 0.0
    for start in range(0, V, batch_rows):
        stop = min(start + batch_rows, V)
        pre = bias.expand(stop - start, -1).clone()  # (B, D)
        for k in range(K):
            pre += inference.encoder_preacts(sd, x_norm[k, start:stop].to(device).float(), k, device=device)
        acts, _ = inference.jumprelu_feature_acts(pre, thr, dec_norm)  # (B, D)
        recon = inference.decode_step(sd, acts, step_idx, device=device)  # (B, d), preprocessed space
        if stats_step is not None:
            recon = inference.invert_preprocess(recon, mean, scale)
        recon = recon.cpu().float()
        out[start:stop] = recon
        native = native_step[start:stop].float()
        sse += float((recon - native).pow(2).sum().item())
        var += float((native - native_mean).pow(2).sum().item())
        del pre, acts, recon
        _empty_cache(device)
    matrix = {
        "matrix_ev": 1.0 - sse / var if var > 0 else float("nan"),
        "matrix_sse": sse,
        "matrix_var": var,
    }
    return out, matrix


def load_ids_seqs(eval_tokens: Path, n_sequences: int | None) -> torch.Tensor:
    """First ``n_sequences`` full SEQ_LEN-token sequences of the eval corpus -> (N, SEQ_LEN) long."""
    data = torch.load(eval_tokens, map_location="cpu", weights_only=False)
    ids = data["ids"].detach().cpu().to(torch.long)
    n_full = (len(ids) // SEQ_LEN) * SEQ_LEN
    ids_seqs = ids[:n_full].view(-1, SEQ_LEN)
    if n_sequences is not None:
        ids_seqs = ids_seqs[:n_sequences]
    return ids_seqs


def functional_metrics(
    h_ln: torch.Tensor,
    ids_seqs: torch.Tensor,
    native_wu: torch.Tensor,
    recon_wu: torch.Tensor,
    *,
    device: str,
    batch_seqs: int,
) -> dict[str, float]:
    """Logit / next-token agreement of recon vs native readout on cached hidden states.

    h_ln: (N_cache, T, d) pre-readout hidden states (first N rows used);
    ids_seqs: (N, T) token ids (position t predicts t+1); native_wu, recon_wu: (V, d).
    logit_r2 is 1 - SSE / total variance over every logit; centered_logit_r2
    first removes each position's mean logit (shift-invariant shape). The total
    variance is merged across batches from per-batch centered sums (Chan et al.),
    not as sum(x^2) - sum(x)^2 / n, which cancels catastrophically in float32
    when the mean logit is large relative to its spread.
    """
    h_ln = h_ln[: ids_seqs.shape[0]]
    targets = ids_seqs[:, 1:]  # (N, T-1)
    W_native = native_wu.to(device).float()
    W_recon = recon_wu.to(device).float()

    nll_native_sum = nll_recon_sum = kl_sum = 0.0
    residual_sse = native_mean = native_m2 = 0.0
    centered_residual_sse = centered_native_var = 0.0
    top5_overlap_sum = 0.0
    n_logits = n_tok = 0
    top1_native = top1_recon = top5_native = top5_recon = top1_agree = 0

    for start in range(0, ids_seqs.shape[0], batch_seqs):
        stop = min(start + batch_seqs, ids_seqs.shape[0])
        h = h_ln[start:stop, :-1, :].to(device).float()  # (b, T-1, d)
        tgt = targets[start:stop].to(device)  # (b, T-1)
        logits_native = h @ W_native.T  # (b, T-1, V)
        logits_recon = h @ W_recon.T
        logp_native = F.log_softmax(logits_native, dim=-1)
        logp_recon = F.log_softmax(logits_recon, dim=-1)
        p_native = logp_native.exp()

        nll_native_sum += float((-logp_native.gather(-1, tgt.unsqueeze(-1))).sum().item())
        nll_recon_sum += float((-logp_recon.gather(-1, tgt.unsqueeze(-1))).sum().item())
        kl_sum += float((p_native * (logp_native - logp_recon)).sum().item())

        residual_sse += float((logits_recon - logits_native).pow(2).sum().item())
        n_b = int(logits_native.numel())
        mean_b = float(logits_native.mean().item())
        m2_b = float((logits_native - mean_b).pow(2).sum().item())
        n_ab = n_logits + n_b
        delta = mean_b - native_mean
        native_m2 += m2_b + delta * delta * n_logits * n_b / n_ab
        native_mean += delta * n_b / n_ab
        n_logits = n_ab
        centered_native = logits_native - logits_native.mean(dim=-1, keepdim=True)
        centered_recon = logits_recon - logits_recon.mean(dim=-1, keepdim=True)
        centered_residual_sse += float((centered_recon - centered_native).pow(2).sum().item())
        centered_native_var += float(centered_native.pow(2).sum().item())

        top5_n = logp_native.topk(5, dim=-1).indices  # (b, T-1, 5)
        top5_r = logp_recon.topk(5, dim=-1).indices
        top1_native += int((top5_n[..., 0] == tgt).sum().item())
        top1_recon += int((top5_r[..., 0] == tgt).sum().item())
        top5_native += int((top5_n == tgt.unsqueeze(-1)).any(dim=-1).sum().item())
        top5_recon += int((top5_r == tgt.unsqueeze(-1)).any(dim=-1).sum().item())
        top1_agree += int((top5_n[..., 0] == top5_r[..., 0]).sum().item())
        overlap = (top5_n.unsqueeze(-1) == top5_r.unsqueeze(-2)).sum(dim=(-1, -2))  # (b, T-1)
        top5_overlap_sum += float((overlap.float() / 5.0).sum().item())

        n_tok += int(tgt.numel())
        del logits_native, logits_recon, logp_native, logp_recon, p_native, centered_native, centered_recon
        _empty_cache(device)

    logit_r2 = 1.0 - residual_sse / native_m2 if native_m2 > 0 else float("nan")
    centered_logit_r2 = 1.0 - centered_residual_sse / centered_native_var if centered_native_var > 0 else float("nan")
    nll_native = nll_native_sum / n_tok
    nll_recon = nll_recon_sum / n_tok
    return {
        "n_predictions": int(n_tok),
        "logit_r2": float(logit_r2),
        "centered_logit_r2": float(centered_logit_r2),
        "kl_native_to_recon": float(kl_sum / n_tok),
        "nll_native": float(nll_native),
        "nll_recon": float(nll_recon),
        "delta_nll": float(nll_recon - nll_native),
        "ppl_native": float(math.exp(nll_native)),
        "ppl_recon": float(math.exp(nll_recon)),
        "top1_native": float(top1_native / n_tok),
        "top1_recon": float(top1_recon / n_tok),
        "top5_native": float(top5_native / n_tok),
        "top5_recon": float(top5_recon / n_tok),
        "top1_agreement": float(top1_agree / n_tok),
        "top5_overlap": float(top5_overlap_sum / n_tok),
    }


def load_hln(hln_dir: Path, step: int, native_step: torch.Tensor) -> tuple[Path, torch.Tensor, torch.Tensor]:
    """(cache path, h_LN (N, T, d), native W_U (V, d)) for one step; W_U_orig falls back to the snapshot."""
    cache_path = hln_dir / f"hLN_step{step}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} missing; build it with scripts/extract/build_hln_cache.py --out-dir {hln_dir}"
        )
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    native_wu = cached.get("W_U_orig")
    if native_wu is None:
        native_wu = native_step.float()
    return cache_path, cached["h_LN"], native_wu


def prepare_crosscoder(args: argparse.Namespace):
    """Load checkpoint + snapshots; returns (ckpt path, cp, target steps, snapshots, x_norm, stats, mode)."""
    ckpt = args.ckpt or release_path(args.model, "W_U", dim=args.d_sae, seed=args.seed)
    cp = load_checkpoint(ckpt)
    if cp.steps is None or cp.model_name is None:
        raise RuntimeError(f"checkpoint missing model_name or steps: {ckpt}")
    target_steps = parse_steps(args.steps, cp.steps)
    mode = cp.training.get("input_preprocess") or "none"
    snapshots = cast_snapshots(load_snapshots(cp.model_name, cp.steps), args.snapshot_dtype)  # (K, V, d)
    x_norm, stats = preprocess_snapshots(snapshots.float(), mode=mode)  # (K, V, d) fp32
    gc.collect()
    print(f"[setup] ckpt={ckpt} preprocess={mode} snapshots={tuple(snapshots.shape)} {snapshots.dtype}", flush=True)
    return ckpt, cp, target_steps, snapshots, x_norm, stats, mode


def stats_at(stats: dict | None, idx: int) -> dict[str, torch.Tensor] | None:
    return None if stats is None else {"scale": stats["scale"][idx], "mean": stats["mean"][idx]}


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    seed_everything(args.seed)
    log_run_provenance(args.seed)
    device = args.device or get_device()
    out_dir = args.out_dir or OUT_ROOT / f"{args.model}_d{args.d_sae}_seed{args.seed}"
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    ckpt, cp, target_steps, snapshots, x_norm, stats, mode = prepare_crosscoder(args)
    step_to_idx = {step: i for i, step in enumerate(cp.steps)}
    ids_seqs = load_ids_seqs(args.eval_tokens, args.max_seqs)
    print(f"[setup] eval {ids_seqs.shape[0]} sequences x {SEQ_LEN}; steps={target_steps}", flush=True)

    def shard_path(step: int) -> Path:
        return shard_dir / f"step{step}.json"

    for step in iter_undone(target_steps, shard_path, label="step"):
        idx = step_to_idx[step]
        native_step = snapshots[idx]
        recon_wu, matrix = reconstruct_one_step(
            cp.state_dict,
            x_norm,
            native_step,
            stats_at(stats, idx),
            idx,
            device=device,
            batch_rows=args.batch_rows,
        )
        cache_path, h_ln, native_wu = load_hln(args.hln_dir, step, native_step)
        functional = functional_metrics(h_ln, ids_seqs, native_wu, recon_wu, device=device, batch_seqs=args.batch_seqs)
        row = {
            "model": args.model,
            "d_sae": args.d_sae,
            "seed": args.seed,
            "step": step,
            "hln_cache": str(cache_path),
            **matrix,
            **functional,
        }
        atomic_write_json(shard_path(step), row)
        print(
            f"[step {step}] matrix EV={row['matrix_ev']:.4f} logit R2={row['logit_r2']:.4f} "
            f"KL={row['kl_native_to_recon']:.4f} dNLL={row['delta_nll']:.4f} top1={row['top1_agreement']:.4f}",
            flush=True,
        )
        del recon_wu, h_ln, native_wu
        gc.collect()
        _empty_cache(device)

    n_done = sum(1 for s in target_steps if shard_path(s).exists())
    if n_done < len(target_steps):
        print(f"[partial] {n_done}/{len(target_steps)} shards present; re-run to complete", flush=True)
        return
    n_rows = aggregate_json_shards(shard_dir, out_dir / "readout_functional_fidelity.csv", key="step")
    atomic_write_json(
        out_dir / "readout_functional_fidelity_metadata.json",
        {
            "ckpt_path": str(ckpt),
            "model_name": cp.model_name,
            "steps": cp.steps,
            "preprocess": mode,
            "d_sae": int(cp.state_dict["W_D"].shape[1]),
            "model": args.model,
            "seed": args.seed,
            "target_steps": target_steps,
            "snapshot_dtype": args.snapshot_dtype,
            "eval_tokens": str(args.eval_tokens),
            "hln_dir": str(args.hln_dir),
            "max_seqs": args.max_seqs,
            "batch_rows": args.batch_rows,
            "batch_seqs": args.batch_seqs,
            "device": device,
            "git_commit": git_commit(),
            "elapsed_s": round(time.time() - t0, 2),
        },
    )
    print(f"[done] aggregated {n_rows} rows -> {out_dir}", flush=True)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared with the sibling strata audit."""
    parser.add_argument("--model", choices=["pythia-160m", "pythia-1b", "pythia-6.9b"], required=True)
    parser.add_argument("--d-sae", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Crosscoder checkpoint; default is the released <model>/W_U/.../d<d-sae>/seed<seed>.safetensors "
        "(pass seed0-sparse.safetensors explicitly for the selected Pythia-6.9B dictionary).",
    )
    parser.add_argument("--steps", default="all", help="Comma-separated checkpoint steps, or 'all' (default).")
    parser.add_argument(
        "--hln-dir",
        type=Path,
        required=True,
        help="Directory of hLN_step<N>.pt caches (scripts/extract/build_hln_cache.py --out-dir).",
    )
    parser.add_argument("--eval-tokens", type=Path, default=DEFAULT_EVAL_TOKENS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-seqs", type=int, default=32, help="Eval sequences (32 x 511 = 16,352 predictions).")
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument("--batch-seqs", type=int, default=2)
    parser.add_argument("--device", default=None, help="Default: cuda > mps > cpu.")
    parser.add_argument(
        "--snapshot-dtype",
        choices=sorted(SNAPSHOT_DTYPES),
        default="fp32",
        help="Storage dtype for the K-stack of W_U snapshots. bf16 halves RAM for 1B / 6.9B.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
