"""Pythia-1B sparse-feature causal pilot.

This is a deliberately narrow first pass:

* checkpoint: W_U at step 1000
* concept: non_latin_scripts
* hidden states: h_512 and h_1000 caches from the existing readout-swap grid
* metric: matched-control concept-mass logit contrast

The script ranks features by a first-order attribution to concept mass on a
selection split, then evaluates top-k ablation and preserve-only edits on a
held-out split. It only materializes reconstructed rows for concept and
matched-control token ids, so it avoids re-encoding the full 1B vocabulary.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from readout.core.data import write_csv
from readout.core.paths import ssd_path

REPO = Path(__file__).resolve().parents[4]
import readout.dynamics.temporal_patch as TPM  # noqa: E402

CKPT_PATH = ssd_path(
    "hf_release",
    "parameter-trajectory-crosscoders",
    "pythia-1b",
    "W_U",
    "cross-snapshot-32",
    "d24576",
    "seed0.safetensors",
)
CONFIG_PATH = CKPT_PATH.with_suffix(".config.json")
AGG_PATH = ssd_path("derived", "aggregates", "aggregates_pythia-1b_d24576_seed0.pt")
OUT_DIR = REPO / "results/experiments/sparse_feature_causal_tests/1b_nonlatin_step1000"


@dataclass
class RowRecon:
    rows: np.ndarray
    acts: torch.Tensor
    W_full: torch.Tensor
    W_bias: torch.Tensor


@dataclass
class CrosscoderPieces:
    steps: list[int]
    snapshot_idx: int
    W_D: torch.Tensor
    b_D: torch.Tensor
    thr: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    mean_i: torch.Tensor
    scale_i: torch.Tensor


def device_default() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_tpm() -> None:
    TPM.ACTIVE_CFG = TPM.CFG_PYTHIA_1B
    TPM.ACTIVE_D_SAE = 24576
    TPM._refresh_aliases()


def load_pieces(step: int, device: str) -> CrosscoderPieces:
    cfg = json.loads(CONFIG_PATH.read_text())
    steps = list(cfg["steps"])
    snap_idx = steps.index(step)
    agg = torch.load(AGG_PATH, map_location="cpu", weights_only=False)
    ps = agg["preprocess_stats"]

    with safe_open(str(CKPT_PATH), framework="pt", device="cpu") as f:
        return CrosscoderPieces(
            steps=steps,
            snapshot_idx=snap_idx,
            W_D=f.get_slice("W_D")[snap_idx].float().to(device),
            b_D=f.get_slice("b_D")[snap_idx].float().to(device),
            thr=f.get_tensor("activation_function.log_jumprelu_threshold").exp().float().to(device),
            mean=ps["mean"].float().to(device),
            scale=ps["scale"].float().to(device),
            mean_i=ps["mean"][snap_idx].squeeze(0).float().to(device),
            scale_i=ps["scale"][snap_idx].squeeze().float().to(device),
        )


def load_eval_sequences() -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(TPM.CORPUS_TOKENS, map_location="cpu", weights_only=False)
    ids = data["ids"]
    n_full = (len(ids) // TPM.SEQ_LEN) * TPM.SEQ_LEN
    return ids[:n_full], ids[:n_full].view(-1, TPM.SEQ_LEN)


def concept_rows(
    meta: TPM.TokenMeta,
    concept: TPM.ConceptDef,
    pool: TPM.PoolState,
    n_concept_tokens: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    in_all = np.where(concept.member_mask)[0]
    if len(in_all) > n_concept_tokens:
        w = meta.log_freq[in_all] + 1e-3
        w = w / w.sum()
        in_C = np.sort(rng.choice(in_all, size=n_concept_tokens, replace=False, p=w))
    else:
        in_C = in_all
    null_ids: list[int] = []
    seen: set[int] = set()
    for v in in_C:
        distractors = TPM.sample_matched_distractors(int(v), concept, meta, pool, n_match=8, rng=rng)
        for d in distractors.tolist():
            d = int(d)
            if d in seen or concept.member_mask[d]:
                continue
            null_ids.append(d)
            seen.add(d)
            if len(null_ids) >= len(in_C):
                break
        if len(null_ids) >= len(in_C):
            break
    if len(null_ids) < len(in_C):
        off = np.where(~concept.member_mask)[0]
        off = np.array([int(v) for v in off if int(v) not in seen], dtype=np.int64)
        w = meta.log_freq[off] + 1e-3
        w = w / w.sum()
        extra = rng.choice(
            off,
            size=min(len(in_C) - len(null_ids), len(off)),
            replace=False,
            p=w,
        )
        null_ids.extend(int(v) for v in extra)
    null_C = np.asarray(null_ids, dtype=np.int64)
    return in_C.astype(np.int64), null_C.astype(np.int64)


def encode_reconstruct_rows(row_ids: np.ndarray, pieces: CrosscoderPieces, device: str) -> RowRecon:
    """Canonical cross-snapshot encode for a small row subset.

    The crosscoder encoder first sums preactivations from every snapshot head.
    For a target decoder head, JumpReLU firing uses that shared preactivation
    multiplied by the target decoder norm. This matches `scripts/eval/recompute_metrics.py`.
    """
    R = len(row_ids)
    D = pieces.W_D.shape[0]
    accumulated = torch.zeros(R, D, dtype=torch.float32, device=device)

    with safe_open(str(CKPT_PATH), framework="pt", device="cpu") as f:
        for head_idx, step in enumerate(pieces.steps):
            W_E = f.get_slice("W_E")[head_idx].float().to(device)
            b_E = f.get_slice("b_E")[head_idx].float().to(device)
            W_rows = TPM.load_snapshot(step)[row_ids].to(device)
            mean_h = pieces.mean[head_idx].squeeze(0)
            scale_h = pieces.scale[head_idx].squeeze()
            x_proc = (W_rows - mean_h) / scale_h
            accumulated += x_proc @ W_E + b_E
            del W_E, b_E, W_rows, x_proc
            if device == "mps":
                torch.mps.empty_cache()

    decoder_norm = pieces.W_D.norm(dim=-1).clamp_min(1e-12)
    fired = (accumulated * decoder_norm) > pieces.thr
    acts = accumulated * fired.to(accumulated.dtype)
    W_proc = acts @ pieces.W_D + pieces.b_D
    W_full = W_proc * pieces.scale_i + pieces.mean_i
    W_bias = pieces.b_D.unsqueeze(0) * pieces.scale_i + pieces.mean_i
    return RowRecon(
        rows=row_ids,
        acts=acts,
        W_full=W_full,
        W_bias=W_bias.expand_as(W_full),
    )


def collect_h_splits(
    h_step: int,
    ids_seqs: torch.Tensor,
    concept: TPM.ConceptDef,
    max_contexts: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed + h_step)
    cached = TPM.cache_or_build_hLN(h_step, ids_seqs)
    h, _target_ids = TPM.collect_concept_contexts(
        cached["h_LN"], ids_seqs, concept, max_per_concept=max_contexts, rng=rng
    )
    del cached
    if h.shape[0] < 2:
        raise RuntimeError(f"Not enough contexts for h_step={h_step}: {h.shape[0]}")
    perm = torch.randperm(h.shape[0], generator=torch.Generator().manual_seed(seed + h_step))
    split = h.shape[0] // 2
    return h[perm[:split]].contiguous(), h[perm[split:]].contiguous()


def concept_mass_from_rows(
    h_cpu: torch.Tensor,
    W_rows: torch.Tensor,
    n_in: int,
    device: str,
) -> float:
    h = h_cpu.to(device)
    Wc = W_rows[:n_in]
    Wn = W_rows[n_in:]
    value = torch.logsumexp(h @ Wc.T, dim=-1) - torch.logsumexp(h @ Wn.T, dim=-1)
    return float(value.mean().detach().cpu())


def grad_rows_for_concept_mass(
    h_cpu: torch.Tensor,
    W_rows: torch.Tensor,
    n_in: int,
    device: str,
) -> torch.Tensor:
    h = h_cpu.to(device)
    Wc = W_rows[:n_in]
    Wn = W_rows[n_in:]
    pc = torch.softmax(h @ Wc.T, dim=-1)
    pn = torch.softmax(h @ Wn.T, dim=-1)
    grad_c = pc.T @ h / h.shape[0]
    grad_n = -(pn.T @ h) / h.shape[0]
    return torch.cat([grad_c, grad_n], dim=0)


def feature_scores(
    row_recon: RowRecon,
    grad_rows: torch.Tensor,
    pieces: CrosscoderPieces,
) -> torch.Tensor:
    # score_f = sum_v a[v,f] * <grad_W[v], scale * W_D[f]>
    projected = grad_rows @ pieces.W_D.T
    if pieces.scale_i.ndim == 0:
        projected = projected * pieces.scale_i
    else:
        # Scale is scalar for current W_U runs; keep this branch for completeness.
        projected = (grad_rows * pieces.scale_i.unsqueeze(0)) @ pieces.W_D.T
    return (row_recon.acts * projected).sum(dim=0).detach().cpu()


def edited_rows(row_recon: RowRecon, pieces: CrosscoderPieces, features: list[int], mode: str) -> torch.Tensor:
    if not features:
        return row_recon.W_full
    idx = torch.as_tensor(features, dtype=torch.long, device=row_recon.acts.device)
    delta = row_recon.acts[:, idx] @ pieces.W_D[idx]
    delta = delta * pieces.scale_i
    if mode == "ablate":
        return row_recon.W_full - delta
    if mode == "only":
        return row_recon.W_bias + delta
    raise ValueError(mode)


def matched_controls(
    selected: list[int],
    rates: np.ndarray,
    decoder_norms: np.ndarray,
    step_idx: int,
    n_controls: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    rate = rates[step_idx]
    norm = decoder_norms[step_idx]
    norm_edges = np.quantile(norm, np.linspace(0, 1, 11))
    rate_edges = np.quantile(rate, np.linspace(0, 1, 11))
    nb = np.clip(np.digitize(norm, norm_edges[1:-1]), 0, 9)
    rb = np.clip(np.digitize(rate, rate_edges[1:-1]), 0, 9)
    selected_set = set(selected)
    controls: list[list[int]] = []
    for _ in range(n_controls):
        used = set(selected_set)
        out: list[int] = []
        for j in selected:
            mask = (nb == nb[j]) & (rb == rb[j])
            candidates = np.where(mask)[0]
            candidates = np.array([int(c) for c in candidates if int(c) not in used])
            if candidates.size == 0:
                dist = ((norm - norm[j]) / (norm.std() + 1e-12)) ** 2 + ((rate - rate[j]) / (rate.std() + 1e-12)) ** 2
                for u in used:
                    dist[u] = np.inf
                candidates = np.argsort(dist)[:64]
            pick = int(rng.choice(candidates))
            out.append(pick)
            used.add(pick)
        controls.append(out)
    return controls


def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=device_default())
    ap.add_argument("--snapshot-step", type=int, default=1000)
    ap.add_argument("--rank-h-step", type=int, default=1000)
    ap.add_argument("--eval-h-steps", type=int, nargs="+", default=[512, 1000])
    ap.add_argument("--concept", default="non_latin_scripts")
    ap.add_argument("--max-contexts", type=int, default=128)
    ap.add_argument("--n-concept-tokens", type=int, default=256)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument(
        "--score-mode",
        choices=["positive", "abs"],
        default="positive",
        help="Rank features by signed positive attribution or absolute attribution.",
    )
    ap.add_argument("--n-controls", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    t0 = time.time()
    configure_tpm()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[pilot] device={args.device} snapshot={args.snapshot_step} concept={args.concept}",
        flush=True,
    )

    pieces = load_pieces(args.snapshot_step, args.device)
    step_idx = pieces.steps.index(args.snapshot_step)
    agg = torch.load(AGG_PATH, map_location="cpu", weights_only=False)
    rates = agg["rates"].numpy()
    decoder_norms = agg["decoder_norms"].numpy()

    ids, ids_seqs = load_eval_sequences()
    tok = TPM.load_tokenizer()
    W_snapshot = TPM.load_snapshot(args.snapshot_step)
    W_terminal = TPM.load_snapshot(143000)
    meta = TPM.build_token_meta(tok, ids, V_explicit=W_snapshot.shape[0])
    concepts = TPM.build_concept_registry(meta)
    concept = concepts[args.concept]
    pool = TPM.build_pools(meta, W_terminal)
    in_C, null_C = concept_rows(meta, concept, pool, args.n_concept_tokens, args.seed)
    row_ids = np.concatenate([in_C, null_C]).astype(np.int64)
    n_in = len(in_C)
    print(
        f"[pilot] rows: in_C={len(in_C)} null={len(null_C)} union={len(row_ids)}",
        flush=True,
    )

    row_recon = encode_reconstruct_rows(row_ids, pieces, args.device)
    h_splits: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for h_step in sorted(set([args.rank_h_step] + args.eval_h_steps)):
        h_splits[h_step] = collect_h_splits(h_step, ids_seqs, concept, args.max_contexts, args.seed)
        print(
            f"[pilot] h{h_step}: select={h_splits[h_step][0].shape[0]} eval={h_splits[h_step][1].shape[0]}",
            flush=True,
        )

    h_rank = h_splits[args.rank_h_step][0]
    grad_rows = grad_rows_for_concept_mass(h_rank, row_recon.W_full, n_in, args.device)
    scores = feature_scores(row_recon, grad_rows, pieces)
    ranking_scores = scores.abs() if args.score_mode == "abs" else scores
    order = torch.argsort(ranking_scores, descending=True).tolist()
    top_max = order[: max(args.ks)]
    top_condition_prefix = f"top_{args.score_mode}_attr"

    rows: list[dict] = []
    for h_step in args.eval_h_steps:
        h_eval = h_splits[h_step][1]
        orig = concept_mass_from_rows(h_eval, W_snapshot[row_ids].to(args.device), n_in, args.device)
        full = concept_mass_from_rows(h_eval, row_recon.W_full, n_in, args.device)
        bias = concept_mass_from_rows(h_eval, row_recon.W_bias, n_in, args.device)
        rows.append(
            {
                "h_step": h_step,
                "k": 0,
                "condition": "baseline",
                "control_id": -1,
                "concept_mass": full,
                "drop_from_full": 0.0,
                "recovery_from_bias": full - bias,
                "orig_concept_mass": orig,
                "full_concept_mass": full,
                "bias_concept_mass": bias,
                "features": "",
            }
        )
        for k in args.ks:
            feats = top_max[:k]
            W_ablate = edited_rows(row_recon, pieces, feats, "ablate")
            W_only = edited_rows(row_recon, pieces, feats, "only")
            cm_ablate = concept_mass_from_rows(h_eval, W_ablate, n_in, args.device)
            cm_only = concept_mass_from_rows(h_eval, W_only, n_in, args.device)
            rows.extend(
                [
                    {
                        "h_step": h_step,
                        "k": k,
                        "condition": f"{top_condition_prefix}_ablate",
                        "control_id": -1,
                        "concept_mass": cm_ablate,
                        "drop_from_full": full - cm_ablate,
                        "recovery_from_bias": cm_ablate - bias,
                        "orig_concept_mass": orig,
                        "full_concept_mass": full,
                        "bias_concept_mass": bias,
                        "features": " ".join(map(str, feats)),
                    },
                    {
                        "h_step": h_step,
                        "k": k,
                        "condition": f"{top_condition_prefix}_only",
                        "control_id": -1,
                        "concept_mass": cm_only,
                        "drop_from_full": full - cm_only,
                        "recovery_from_bias": cm_only - bias,
                        "orig_concept_mass": orig,
                        "full_concept_mass": full,
                        "bias_concept_mass": bias,
                        "features": " ".join(map(str, feats)),
                    },
                ]
            )
            for ci, ctrl in enumerate(
                matched_controls(
                    feats,
                    rates,
                    decoder_norms,
                    step_idx,
                    args.n_controls,
                    args.seed + k,
                )
            ):
                Wc_ablate = edited_rows(row_recon, pieces, ctrl, "ablate")
                Wc_only = edited_rows(row_recon, pieces, ctrl, "only")
                cm_c_ablate = concept_mass_from_rows(h_eval, Wc_ablate, n_in, args.device)
                cm_c_only = concept_mass_from_rows(h_eval, Wc_only, n_in, args.device)
                rows.extend(
                    [
                        {
                            "h_step": h_step,
                            "k": k,
                            "condition": "matched_ablate",
                            "control_id": ci,
                            "concept_mass": cm_c_ablate,
                            "drop_from_full": full - cm_c_ablate,
                            "recovery_from_bias": cm_c_ablate - bias,
                            "orig_concept_mass": orig,
                            "full_concept_mass": full,
                            "bias_concept_mass": bias,
                            "features": " ".join(map(str, ctrl)),
                        },
                        {
                            "h_step": h_step,
                            "k": k,
                            "condition": "matched_only",
                            "control_id": ci,
                            "concept_mass": cm_c_only,
                            "drop_from_full": full - cm_c_only,
                            "recovery_from_bias": cm_c_only - bias,
                            "orig_concept_mass": orig,
                            "full_concept_mass": full,
                            "bias_concept_mass": bias,
                            "features": " ".join(map(str, ctrl)),
                        },
                    ]
                )

    write_csv(args.out_dir / "summary.csv", rows)
    torch.save(
        {
            "scores": scores,
            "top_features": top_max,
            "in_C": in_C,
            "null_C": null_C,
            "row_ids": row_ids,
            "config": jsonable_args(args),
        },
        args.out_dir / "raw.pt",
    )
    manifest = {
        **jsonable_args(args),
        "ckpt_path": str(CKPT_PATH),
        "aggregate_path": str(AGG_PATH),
        "n_in_C": int(len(in_C)),
        "n_null_C": int(len(null_C)),
        "top_features": top_max[:20],
        "elapsed_s": round(time.time() - t0, 2),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[pilot] wrote {args.out_dir / 'summary.csv'}", flush=True)
    print(f"[pilot] elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
