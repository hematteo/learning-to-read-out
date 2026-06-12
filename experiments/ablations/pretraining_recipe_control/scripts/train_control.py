#!/usr/bin/env python
"""From-scratch Pythia-style trainer for the readout-recipe control.

Faithful to the Pythia recipe on the modern torch stack:
  - GPTNeoX architecture (parallel residual, rotary 0.25, untied W_U)
  - small_init / wang_init initialisation
  - AdamW betas (0.9, 0.95), eps 1e-8, wd 0.1 (excluded from LN + biases)
  - faithful cosine+warmup schedule defined over Pythia's 143k horizon, stopped early
  - fp16 + dynamic loss scaling, grad clip 1.0
  - a separate optimiser param group for the output readout W_U (embed_out.weight)
    with its own LR multiplier (1.0 = baseline; 0.25 / 4.0 for the C2/C3 controls)

Suite features:
  - TOKEN budget (--max-tokens): stops at the budget and writes a COMPLETE sentinel.
  - TOKEN-milestone checkpointing: dense early steps + log-spaced token milestones,
    so checkpoints land at identical TOKEN points across conditions (and batches).
  - configurable --warmup-steps (C1 long-warmup).
  - per-slot wall guard (--max-hours): exits cleanly WITHOUT COMPLETE so a
    wrapper or batch scheduler can resubmit a continuation before hitting a
    walltime limit.
  - held-out validation loss + final-LN gain norm + W_U row-norm geometry.

Resumable: re-running reloads checkpoints/latest.pt and continues the identical
deterministic data order. Checkpoints + state are written atomically.
"""

import argparse
import bisect
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

MODEL_SIZES = {
    "14M": dict(
        num_hidden_layers=6,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=512,
    ),
    "31M": dict(
        num_hidden_layers=6,
        hidden_size=256,
        num_attention_heads=8,
        intermediate_size=1024,
    ),
    "70M": dict(
        num_hidden_layers=6,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048,
    ),
}
VOCAB = 50304  # Pythia padded vocab (GPT-NeoX-20B tokenizer)
SEQLEN = 2048
DECAY_ITERS = 143000  # Pythia full horizon (faithful cosine defined over this)
MIN_LR_FRAC = 0.1  # min_lr 1e-4 = 0.1 * peak 1e-3
EARLY_STEPS = {0, 1, 2, 4, 8, 16, 32, 64}  # very-early updates, always checkpointed


def build_model(size):
    h = MODEL_SIZES[size]
    cfg = GPTNeoXConfig(
        vocab_size=VOCAB,
        hidden_size=h["hidden_size"],
        num_hidden_layers=h["num_hidden_layers"],
        num_attention_heads=h["num_attention_heads"],
        intermediate_size=h["intermediate_size"],
        max_position_embeddings=SEQLEN,
        rotary_pct=0.25,
        rotary_emb_base=10000,
        use_parallel_residual=True,
        tie_word_embeddings=False,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        use_cache=False,
        attn_implementation="sdpa",
    )
    model = GPTNeoXForCausalLM(cfg)
    apply_pythia_init(model, cfg)
    return model, cfg


def apply_pythia_init(model, cfg):
    """small_init for most weights; wang_init for the residual-output projections."""
    d, n = cfg.hidden_size, cfg.num_hidden_layers
    small_std = math.sqrt(2.0 / (5.0 * d))
    wang_std = 2.0 / (n * math.sqrt(d))
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            if name.endswith("attention.dense.weight") or name.endswith("mlp.dense_4h_to_h.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=wang_std)
            else:
                torch.nn.init.normal_(p, mean=0.0, std=small_std)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    for name, p in model.named_parameters():
        if name.endswith(".bias"):
            torch.nn.init.zeros_(p)


def make_param_groups(model, wd, readout_lr_mult, peak_lr, wu_weight_decay=None):
    """3 groups: decay (2-D weights), no_decay (LN + biases), readout (W_U, own LR)."""
    wu_wd = wd if wu_weight_decay is None else wu_weight_decay
    decay, no_decay, readout = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "embed_out.weight":
            readout.append(p)
        elif p.dim() < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": wd, "lr": peak_lr, "name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "lr": peak_lr, "name": "no_decay"},
        {
            "params": readout,
            "weight_decay": wu_wd,
            "lr": peak_lr * readout_lr_mult,
            "name": "readout",
        },
    ]


def make_lr_factor(warmup, decay=DECAY_ITERS, min_frac=MIN_LR_FRAC):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        if step >= decay:
            return min_frac
        prog = (step - warmup) / (decay - warmup)
        return min_frac + 0.5 * (1 - min_frac) * (1 + math.cos(math.pi * prog))

    return f


def token_milestones(max_tokens):
    """Log-spaced token milestones (1M, 2M, 4M, ...), batch-independent."""
    ms = set()
    v = 1 << 20
    cap = int(max_tokens) if max_tokens > 0 else (1 << 35)
    while v < cap:
        ms.add(v)
        v *= 2
    if max_tokens > 0:
        ms.add(int(max_tokens))
    return sorted(ms)


class PackedData:
    """uint16 token memmap -> deterministic permuted stream of train blocks; last
    val_blocks held out for validation."""

    def __init__(self, path, seqlen, data_seed, val_blocks=0, sequential=False):
        self.toks = np.memmap(path, dtype=np.uint16, mode="r")
        self.seqlen = seqlen
        self.data_seed = data_seed
        self.sequential = sequential  # read blocks in file order (e.g. exact Pythia preshuffled order)
        total = len(self.toks) // seqlen
        self.val_blocks = max(0, min(val_blocks, total - 1))
        self.n_blocks = total - self.val_blocks  # train blocks: [0, n_blocks)
        self._val_start = self.n_blocks  # val blocks: [n_blocks, total)
        if self.n_blocks < 1:
            raise ValueError(f"data slice too small: {len(self.toks)} tokens")
        self._cache_epoch = None
        self._cache_perm = None

    def _perm(self, epoch):
        if epoch != self._cache_epoch:
            self._cache_perm = (
                np.arange(self.n_blocks)
                if self.sequential
                else np.random.default_rng(self.data_seed + epoch).permutation(self.n_blocks)
            )
            self._cache_epoch = epoch
        return self._cache_perm

    def _read(self, idxs):
        sl = self.seqlen
        rows = np.stack([np.asarray(self.toks[i * sl : (i + 1) * sl], dtype=np.int64) for i in idxs])
        return torch.from_numpy(rows)

    def batch(self, start_block, micro_batch, device):
        idxs = []
        cur = start_block
        while len(idxs) < micro_batch:
            epoch, within = divmod(cur, self.n_blocks)
            perm = self._perm(epoch)
            take = perm[within : within + (micro_batch - len(idxs))]
            idxs.extend(int(i) for i in take)
            cur += len(take)
        return self._read(idxs).to(device, non_blocking=True), start_block + micro_batch

    def val_chunks(self, micro_batch):
        for s in range(0, self.val_blocks, micro_batch):
            n = min(micro_batch, self.val_blocks - s)
            yield self._read([self._val_start + s + j for j in range(n)])


@torch.no_grad()
def eval_val(model, data, micro_batch, device, amp_dtype, use_amp):
    if data.val_blocks == 0:
        return float("nan")
    model.eval()
    tot, n = 0.0, 0
    for rows in data.val_chunks(micro_batch):
        x = rows.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss = model(input_ids=x, labels=x).loss
        tot += loss.item() * x.size(0)
        n += x.size(0)
    model.train()
    return tot / max(1, n)


def geom_stats(model):
    w = model.embed_out.weight.detach().float()  # (vocab, hidden)
    rn = w.norm(dim=1)
    try:
        lnf = model.gpt_neox.final_layer_norm.weight.detach().float().norm().item()
    except AttributeError:
        lnf = float("nan")
    return dict(
        wu_row_norm_mean=rn.mean().item(),
        wu_row_norm_std=rn.std().item(),
        wu_row_norm_p50=rn.median().item(),
        wu_row_norm_max=rn.max().item(),
        lnf_gain_norm=lnf,
    )


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-size", default="31M", choices=list(MODEL_SIZES))
    ap.add_argument("--readout-lr-mult", type=float, default=1.0)
    ap.add_argument("--peak-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument(
        "--wu-weight-decay",
        type=float,
        default=-1.0,
        help="<0 = same as --weight-decay (C4 sets 0)",
    )
    ap.add_argument("--warmup-steps", type=int, default=1430)
    ap.add_argument("--global-batch", type=int, default=1024, help="sequences per optimiser step")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument(
        "--max-tokens",
        type=float,
        default=0.0,
        help="token budget; writes COMPLETE when reached",
    )
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument(
        "--max-hours",
        type=float,
        default=0.0,
        help="per-slot wall guard (no COMPLETE -> chain continues)",
    )
    ap.add_argument("--val-blocks", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1234)
    ap.add_argument(
        "--sequential-data",
        action="store_true",
        help="read blocks in file order (no permutation); use for the exact "
        "Pythia preshuffled slice to preserve Pythia's released token order",
    )
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--latest-every", type=int, default=100)
    args = ap.parse_args()

    assert args.global_batch % args.micro_batch == 0, "global must be divisible by micro"
    accum = args.global_batch // args.micro_batch
    tps = args.global_batch * SEQLEN  # tokens per optimiser step
    wu_wd = None if args.wu_weight_decay < 0 else args.wu_weight_decay
    out = Path(args.out_dir)
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, cfg = build_model(args.model_size)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    groups = make_param_groups(model, args.weight_decay, args.readout_lr_mult, args.peak_lr, wu_wd)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=make_lr_factor(args.warmup_steps))
    use_amp = args.precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))

    data = PackedData(
        args.data,
        SEQLEN,
        args.data_seed,
        val_blocks=args.val_blocks,
        sequential=args.sequential_data,
    )
    milestones = token_milestones(args.max_tokens)

    # ---- resume ----
    step, cursor, loss_ema = 0, 0, None
    latest = out / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step, cursor, loss_ema = ck["step"], ck["cursor"], ck.get("loss_ema")
        torch.set_rng_state(ck["torch_rng"].cpu())
        print(
            f"[resume] step={step} tokens={step * tps / 1e9:.3f}B cursor={cursor} "
            f"train_blocks={data.n_blocks} val_blocks={data.val_blocks}",
            flush=True,
        )
    else:
        print(
            f"[start] model={args.model_size} params={n_params / 1e6:.1f}M readout_lr_mult={args.readout_lr_mult} "
            f"wu_wd={wu_wd if wu_wd is not None else args.weight_decay} warmup={args.warmup_steps} "
            f"global_batch={args.global_batch} micro={args.micro_batch} accum={accum} tps={tps} "
            f"max_tokens={args.max_tokens:.0f} precision={args.precision} "
            f"train_blocks={data.n_blocks} val_blocks={data.val_blocks}",
            flush=True,
        )
        (out / "config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "n_params": n_params,
                    "tokens_per_step": tps,
                    "wu_weight_decay_effective": wu_wd if wu_wd is not None else args.weight_decay,
                },
                indent=2,
            )
        )

    ms_ptr = bisect.bisect_right(milestones, step * tps)

    metrics_csv = out / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv.write_text(
            "step,tokens,train_loss,val_loss,lr_body,lr_readout,grad_norm,tok_per_s,"
            "wu_row_norm_mean,wu_row_norm_std,wu_row_norm_p50,wu_row_norm_max,lnf_gain_norm\n"
        )

    def save_ckpt(tag, val_loss):
        d = out / "ckpts" / f"step{tag}"
        d.mkdir(parents=True, exist_ok=True)
        atomic_save(
            {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
            d / "model_fp16.pt",
        )
        s = geom_stats(model)
        lr = sched.get_last_lr()
        (d / "metrics.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "tokens": step * tps,
                    "val_loss": val_loss,
                    "lr_body": lr[0],
                    "lr_readout": lr[2],
                    **s,
                },
                indent=2,
            )
        )

    def save_latest():
        atomic_save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "cursor": cursor,
                "loss_ema": loss_ema,
                "torch_rng": torch.get_rng_state(),
            },
            latest,
        )

    def checkpoint(grad_norm):
        vloss = eval_val(model, data, args.micro_batch, device, amp_dtype, use_amp)
        save_ckpt(step, vloss)
        with open(metrics_csv, "a") as f:
            s = geom_stats(model)
            lr = sched.get_last_lr()
            toks = step * tps
            f.write(
                f"{step},{toks},{loss_ema:.5f},{vloss:.5f},{lr[0]:.3e},{lr[2]:.3e},{grad_norm:.3f},"
                f"{0:.0f},{s['wu_row_norm_mean']:.5f},{s['wu_row_norm_std']:.5f},"
                f"{s['wu_row_norm_p50']:.5f},{s['wu_row_norm_max']:.5f},{s['lnf_gain_norm']:.5f}\n"
            )

    if step == 0 and not (out / "ckpts" / "step0").exists():
        save_ckpt(0, eval_val(model, data, args.micro_batch, device, amp_dtype, use_amp))

    model.train()
    t0, tok0, start_wall = time.time(), step * tps, time.time()
    while True:
        if args.max_steps and step >= args.max_steps:
            break
        if args.max_tokens and step * tps >= args.max_tokens:
            (out / "COMPLETE").write_text(str(step * tps))
            print(
                f"[complete] reached budget {args.max_tokens:.0f} tokens at step {step}",
                flush=True,
            )
            break
        if args.max_hours and (time.time() - start_wall) / 3600.0 >= args.max_hours:
            print(
                f"[slot-stop] reached max_hours={args.max_hours} at step {step} "
                f"(no COMPLETE -> continuation will resume)",
                flush=True,
            )
            break

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            x, cursor = data.batch(cursor, args.micro_batch, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model(input_ids=x, labels=x).loss / accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if not math.isfinite(loss_acc):
            print(f"[WARN] non-finite loss at step {step}: {loss_acc}", flush=True)
        loss_ema = loss_acc if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_acc

        if step % args.log_every == 0:
            toks = step * tps
            dt = time.time() - t0
            tpss = (toks - tok0) / dt if dt > 0 else 0.0
            lr = sched.get_last_lr()
            print(
                f"step {step:6d} | loss {loss_acc:.4f} (ema {loss_ema:.4f}) | "
                f"lr_body {lr[0]:.2e} lr_wu {lr[2]:.2e} | gnorm {grad_norm:.2f} | "
                f"{tpss / 1e3:.1f}k tok/s | {toks / 1e9:.3f}B tok",
                flush=True,
            )
            t0, tok0 = time.time(), toks

        do_ckpt = step in EARLY_STEPS
        toks_now = step * tps
        while ms_ptr < len(milestones) and toks_now >= milestones[ms_ptr]:
            do_ckpt = True
            ms_ptr += 1
        if do_ckpt and not (out / "ckpts" / f"step{step}").exists():
            checkpoint(grad_norm)
        if step % args.latest_every == 0:
            save_latest()

    save_latest()
    if not (out / "ckpts" / f"step{step}").exists():
        checkpoint(float("nan"))
    print(
        f"[done] step={step} tokens={step * tps / 1e9:.3f}B complete={(out / 'COMPLETE').exists()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
