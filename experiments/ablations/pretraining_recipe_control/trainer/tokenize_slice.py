#!/usr/bin/env python
"""Stream the Pile, tokenize with the GPT-NeoX-20B tokenizer, pack into a uint16 memmap.

Deterministic (seeded shuffle buffer), resumable (atomic finalise + .done marker; a
completed slice is reused). Output is a flat .bin of uint16 token ids (vocab < 65536).
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=float, required=True)
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--dataset", default="monology/pile-uncopyrighted")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shuffle-buffer", type=int, default=10000)
    ap.add_argument("--batch-texts", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    target = int(args.target_tokens)
    done = out.with_suffix(out.suffix + ".done")
    if done.exists() and out.exists():
        print(
            f"[skip] {out} already complete ({done.read_text().strip()} tokens)",
            flush=True,
        )
        return

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    mm = np.memmap(tmp, dtype=np.uint16, mode="w+", shape=(target,))
    write_ptr = 0
    texts = []

    def flush():
        nonlocal write_ptr
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        flat = []
        for ids in enc:
            flat.extend(ids)
            flat.append(eos)
        arr = np.asarray(flat, dtype=np.uint16)
        take = min(len(arr), target - write_ptr)
        # `mm` is the enclosing scope's memmap; ruff flags it F821 because of
        # the later `del mm`, but flush() is only ever called before that del.
        mm[write_ptr : write_ptr + take] = arr[:take]  # noqa: F821
        write_ptr += take
        texts.clear()

    # Streaming the Pile over many minutes is fragile (transient httpx/connection
    # drops kill the iterator). Retry: on any error, discard the partial buffer and
    # restart the stream with a fresh shuffle seed so we APPEND new docs from where
    # write_ptr already is. The produced .bin is a fixed artifact (tokenize once,
    # reuse across all conditions), so per-run retry variation does not matter.
    last = 0
    attempt, max_attempts = 0, 12
    while write_ptr < target and attempt < max_attempts:
        attempt += 1
        try:
            ds = load_dataset(args.dataset, split=args.split, streaming=True)
            ds = ds.shuffle(seed=args.seed + attempt - 1, buffer_size=args.shuffle_buffer)
            for ex in ds:
                texts.append(ex["text"])
                if len(texts) >= args.batch_texts:
                    flush()
                    if write_ptr - last >= 50_000_000:
                        print(
                            f"  tokenized {write_ptr / 1e6:.0f}M / {target / 1e6:.0f}M",
                            flush=True,
                        )
                        last = write_ptr
                    if write_ptr >= target:
                        break
            if write_ptr < target and texts:
                flush()
        except Exception as e:  # noqa: BLE001 -- any stream/network error: restart
            texts.clear()
            wait = min(60, 5 * attempt)
            print(
                f"[retry {attempt}/{max_attempts}] stream error at {write_ptr / 1e6:.0f}M: "
                f"{type(e).__name__}: {e} -- restarting in {wait}s",
                flush=True,
            )
            time.sleep(wait)

    if write_ptr < target:
        raise RuntimeError(f"only tokenized {write_ptr}/{target} after {attempt} attempts")

    mm.flush()
    del mm
    os.replace(tmp, out)
    done.write_text(str(write_ptr))
    print(f"[done] wrote {write_ptr} tokens to {out}", flush=True)


if __name__ == "__main__":
    main()
