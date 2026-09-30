"""Token-stratified fidelity audit of the reconstructed W_U readout (Appendix D.2 prose).

The aggregate fidelity numbers of readout_functional_fidelity.py (matrix EV,
logit R^2, KL, delta NLL) can hide whether reconstruction quality is uniform
across token classes. This audit reruns the same streaming reconstruction and
accumulates residuals per stratum, never holding more than one (V, d)
reconstruction at a time.

Strata (computed once from the eval corpus + tokenizer):
    freq_decile   0..9 by corpus log-frequency over tokens seen in the corpus, 0 = rarest
    length_bin    visible (stripped) length "1", "2-3", "4-7", "8+"; whitespace-only -> bin 0
    coarse_class  readout.dynamics.temporal_patch.coarse_class_of of the decoded token

Per (step, scheme, bin) row:
    matrix_ev, matrix_sse, matrix_var, res_norm_mean, n_rows
        row-level reconstruction against the native readout, restricted to the stratum's rows
    nll_native, nll_recon, delta_nll, ppl_native, ppl_recon, kl_to_native, top1_agreement, n_targets
        averaged over predictions whose TARGET token is in the stratum

Outputs (resume-safe per step):
  <out-dir>/shards/step<N>.json   list of rows
  <out-dir>/strata_summary.csv
  <out-dir>/strata_metadata.json  bin names per scheme + provenance

Usage (the Appendix D.2 check, Pythia-1B d=24576 seed 0):
  uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/readout_functional_fidelity_strata.py \\
      --model pythia-1b --d-sae 24576 --seed 0 --steps all --snapshot-dtype bf16 \\
      --hln-dir "$UM_SSD_ROOT/derived/readout_edit_timing_pythia1b"
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import readout_functional_fidelity as rff
import torch
import torch.nn.functional as F

from readout.core.data import get_device
from readout.core.repro import git_commit, log_run_provenance, seed_everything
from readout.core.resume import aggregate_json_shards, atomic_write_json, iter_undone
from readout.dynamics.temporal_patch import coarse_class_of

LENGTH_BIN_EDGES = [(1, 1), (2, 3), (4, 7), (8, 10**9)]


def length_bin_names() -> list[str]:
    return [f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**8 else f"{lo}+") for lo, hi in LENGTH_BIN_EDGES]


def length_to_bin(length: int) -> int:
    for i, (lo, hi) in enumerate(LENGTH_BIN_EDGES):
        if lo <= length <= hi:
            return i
    return -1


def freq_decile_strata(ids_flat: torch.Tensor, V: int) -> torch.Tensor:
    """(V,) int32 decile 0..9 of corpus log-frequency among seen tokens (0 = rarest); unseen -> -1.

    Tied frequencies are ordered by torch.argsort (not guaranteed stable), as in the original audit.
    """
    counts = torch.zeros(V, dtype=torch.long)
    counts.scatter_add_(0, ids_flat.long(), torch.ones_like(ids_flat, dtype=torch.long))
    log_freq = torch.log1p(counts.float())
    seen_idx = (counts > 0).nonzero(as_tuple=False).squeeze(-1)
    order = log_freq[seen_idx].argsort()
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(seen_idx), dtype=rank.dtype)
    decile = (rank.float() / max(len(seen_idx) - 1, 1) * 10).clamp(max=9.999).long()
    out = torch.full((V,), -1, dtype=torch.int32)
    out[seen_idx] = decile.to(torch.int32)
    return out


def text_strata(decoded: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """(length_bin (V,), coarse_class (V,), coarse class names in first-seen order) from decoded tokens."""
    V = len(decoded)
    length_strat = torch.full((V,), -1, dtype=torch.int32)
    coarse_strat = torch.full((V,), -1, dtype=torch.int32)
    coarse_to_idx: dict[str, int] = {}
    for v, text in enumerate(decoded):
        text = text if text is not None else ""
        stripped = text.strip()
        length_strat[v] = length_to_bin(len(stripped)) if stripped else 0
        name = coarse_class_of(text)
        coarse_strat[v] = coarse_to_idx.setdefault(name, len(coarse_to_idx))
    return length_strat, coarse_strat, list(coarse_to_idx)


def build_strata(tokenizer, ids_flat: torch.Tensor, V: int) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """{scheme: (V,) int32 stratum index, -1 = skip} and {scheme: bin names}."""
    length_strat, coarse_strat, coarse_names = text_strata([tokenizer.decode([v]) for v in range(V)])
    strata = {
        "freq_decile": freq_decile_strata(ids_flat, V),
        "length_bin": length_strat,
        "coarse_class": coarse_strat,
    }
    names = {
        "freq_decile": [str(b) for b in range(10)],
        "length_bin": length_bin_names(),
        "coarse_class": coarse_names,
    }
    return strata, names


def stratified_matrix_residuals(
    recon: torch.Tensor, native: torch.Tensor, strata: dict[str, torch.Tensor]
) -> dict[str, dict[int, dict[str, float]]]:
    """Per-stratum SSE / variance / mean residual norm of recon (V, d) vs native (V, d).

    The variance is about the FULL-vocabulary mean row, so per-stratum EVs are
    comparable with the aggregate matrix EV.
    """
    res_sq = (recon.float() - native.float()).pow(2).sum(dim=-1)  # (V,)
    res_norm = res_sq.sqrt()
    var_per_row = (native.float() - native.float().mean(dim=0, keepdim=True)).pow(2).sum(dim=-1)  # (V,)
    out: dict[str, dict[int, dict[str, float]]] = {}
    for scheme, idx in strata.items():
        per_bin: dict[int, dict[str, float]] = {}
        for b in idx.unique().tolist():
            if b < 0:
                continue
            mask = idx == b
            per_bin[int(b)] = {
                "n_rows": int(mask.sum().item()),
                "matrix_sse": float(res_sq[mask].sum().item()),
                "matrix_var": float(var_per_row[mask].sum().item()),
                "res_norm_mean": float(res_norm[mask].mean().item()),
            }
        out[scheme] = per_bin
    return out


def stratified_functional(
    h_ln: torch.Tensor,
    ids_seqs: torch.Tensor,
    W_native: torch.Tensor,
    W_recon: torch.Tensor,
    strata: dict[str, torch.Tensor],
    *,
    device: str,
    batch_seqs: int,
) -> dict[str, dict[int, dict[str, float]]]:
    """Per-stratum NLL / KL / top-1 agreement, keyed by the TARGET token's stratum.

    h_ln: (N_cache, T, d); ids_seqs: (N, T); W_native, W_recon: (V, d).
    """
    targets_all = ids_seqs[:, 1:]  # (N, T-1)
    Wn = W_native.to(device).float()
    Wr = W_recon.to(device).float()
    strata_dev = {scheme: idx.to(device).long() for scheme, idx in strata.items()}
    accum: dict[str, dict[int, dict[str, float]]] = {scheme: {} for scheme in strata}

    h_ln = h_ln[: ids_seqs.shape[0]]
    for s in range(0, h_ln.shape[0], batch_seqs):
        h = h_ln[s : s + batch_seqs, :-1, :].to(device).float()  # (b, T-1, d)
        tgt = targets_all[s : s + batch_seqs].to(device)  # (b, T-1)
        logp_n = F.log_softmax(h @ Wn.T, dim=-1)  # (b, T-1, V)
        logp_r = F.log_softmax(h @ Wr.T, dim=-1)
        nll_n = -logp_n.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll_r = -logp_r.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        kl = (logp_n.exp() * (logp_n - logp_r)).sum(-1)
        agree = logp_n.argmax(dim=-1) == logp_r.argmax(dim=-1)

        flat = {
            "nll_native_sum": nll_n.reshape(-1),
            "nll_recon_sum": nll_r.reshape(-1),
            "kl_sum": kl.reshape(-1),
            "top1_agree_sum": agree.reshape(-1).float(),
        }
        flat_tgt = tgt.reshape(-1)
        for scheme, idx in strata_dev.items():
            tgt_strat = idx[flat_tgt]
            for b in tgt_strat.unique().tolist():
                if b < 0:
                    continue
                m = tgt_strat == b
                sums = accum[scheme].setdefault(int(b), {})
                for key, vals in flat.items():
                    sums[key] = sums.get(key, 0.0) + float(vals[m].sum().item())
                sums["n_targets"] = sums.get("n_targets", 0.0) + float(int(m.sum().item()))
        del logp_n, logp_r
        rff._empty_cache(device)

    out: dict[str, dict[int, dict[str, float]]] = {}
    for scheme, bins in accum.items():
        per_bin: dict[int, dict[str, float]] = {}
        for b, sums in bins.items():
            n = max(sums.get("n_targets", 0.0), 1.0)
            nll_native = sums["nll_native_sum"] / n
            nll_recon = sums["nll_recon_sum"] / n
            per_bin[b] = {
                "n_targets": int(sums.get("n_targets", 0.0)),
                "nll_native": nll_native,
                "nll_recon": nll_recon,
                "delta_nll": nll_recon - nll_native,
                "ppl_native": math.exp(nll_native),
                "ppl_recon": math.exp(nll_recon),
                "kl_to_native": sums["kl_sum"] / n,
                "top1_agreement": sums["top1_agree_sum"] / n,
            }
        out[scheme] = per_bin
    return out


def strata_rows(
    args: argparse.Namespace,
    step: int,
    matrix_strat: dict[str, dict[int, dict[str, float]]],
    functional_strat: dict[str, dict[int, dict[str, float]]],
) -> list[dict]:
    """Flatten per-scheme matrix + functional dicts to CSV rows (functional cols empty for untargeted bins)."""
    rows = []
    for scheme, mat in matrix_strat.items():
        fun = functional_strat.get(scheme, {})
        for b, mat_row in mat.items():
            ev = 1.0 - mat_row["matrix_sse"] / mat_row["matrix_var"] if mat_row["matrix_var"] > 0 else float("nan")
            rows.append(
                {
                    "model": args.model,
                    "d_sae": args.d_sae,
                    "seed": args.seed,
                    "step": step,
                    "scheme": scheme,
                    "bin": int(b),
                    "n_rows": mat_row["n_rows"],
                    "matrix_ev": ev,
                    "matrix_sse": mat_row["matrix_sse"],
                    "matrix_var": mat_row["matrix_var"],
                    "res_norm_mean": mat_row["res_norm_mean"],
                    **fun.get(b, {}),
                }
            )
    return rows


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    t0 = time.time()
    seed_everything(args.seed)
    log_run_provenance(args.seed)
    device = args.device or get_device()
    out_dir = args.out_dir or rff.OUT_ROOT / f"{args.model}_d{args.d_sae}_seed{args.seed}" / "strata"
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    ckpt, cp, target_steps, snapshots, x_norm, stats, _mode = rff.prepare_crosscoder(args)
    step_to_idx = {s: i for i, s in enumerate(cp.steps)}
    ids_seqs = rff.load_ids_seqs(args.eval_tokens, args.max_seqs)
    # Frequency deciles use the WHOLE eval corpus, not just the scored shard.
    ids_flat = torch.load(args.eval_tokens, map_location="cpu", weights_only=False)["ids"].long()
    V = int(snapshots.shape[1])
    print(f"[setup] building strata for V={V} tokens (single pass)", flush=True)
    strata, bin_names = build_strata(AutoTokenizer.from_pretrained(cp.model_name), ids_flat, V)

    def shard_path(step: int) -> Path:
        return shard_dir / f"step{step}.json"

    for step in iter_undone(target_steps, shard_path, label="step"):
        idx = step_to_idx[step]
        native_step = snapshots[idx]
        recon_wu, matrix_global = rff.reconstruct_one_step(
            cp.state_dict,
            x_norm,
            native_step,
            rff.stats_at(stats, idx),
            idx,
            device=device,
            batch_rows=args.batch_rows,
        )
        _cache_path, h_ln, native_wu = rff.load_hln(args.hln_dir, step, native_step)
        matrix_strat = stratified_matrix_residuals(recon_wu, native_wu, strata)
        functional_strat = stratified_functional(
            h_ln, ids_seqs, native_wu, recon_wu, strata, device=device, batch_seqs=args.batch_seqs
        )
        rows = strata_rows(args, step, matrix_strat, functional_strat)
        atomic_write_json(shard_path(step), rows)
        print(
            f"[step {step}] aggregate matrix EV={matrix_global['matrix_ev']:.4f} | strata rows: {len(rows)}",
            flush=True,
        )
        del recon_wu, h_ln, native_wu
        gc.collect()
        rff._empty_cache(device)

    n_done = sum(1 for s in target_steps if shard_path(s).exists())
    if n_done < len(target_steps):
        print(f"[partial] {n_done}/{len(target_steps)} step shards present", flush=True)
        return
    n = aggregate_json_shards(shard_dir, out_dir / "strata_summary.csv", key="step")
    atomic_write_json(
        out_dir / "strata_metadata.json",
        {
            "model": args.model,
            "d_sae": args.d_sae,
            "seed": args.seed,
            "ckpt_path": str(ckpt),
            "target_steps": target_steps,
            "schemes": bin_names,
            "n_freq_seen_tokens": int((strata["freq_decile"] >= 0).sum().item()),
            "eval_tokens": str(args.eval_tokens),
            "hln_dir": str(args.hln_dir),
            "max_seqs": args.max_seqs,
            "snapshot_dtype": args.snapshot_dtype,
            "device": device,
            "git_commit": git_commit(),
            "elapsed_s": round(time.time() - t0, 2),
        },
    )
    print(f"[done] aggregated {n} stratum-rows -> {out_dir / 'strata_summary.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rff.add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
