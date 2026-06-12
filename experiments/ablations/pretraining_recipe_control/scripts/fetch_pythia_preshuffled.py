#!/usr/bin/env python3
"""Extract the first N tokens of the *exact* Pythia preshuffled Pile as a flat
uint16 .bin, drop-in compatible with train_control.py.

Why this works without the .idx:
  The Pythia preshuffled dataset is a GPT-NeoX MMapIndexedDataset whose `.bin`
  is just the concatenated uint16 token ids (vocab 50304 < 65536), already in
  Pythia's global preshuffled training order. The 30 GB `document-0000k-of-00020.bin`
  shards are a raw BYTE split of that single stream (unshard_memmap.py merges them
  by plain concatenation), so shard 0 starts at token 0 and any byte-prefix of it
  is a valid token-prefix. The `.idx` only stores document boundaries, which our
  trainer ignores -- it reads the .bin as a flat uint16 memmap and packs SEQLEN
  blocks. So we range-download only the first 2*N bytes of shard 0; no .idx, no
  reindexing, no 602 GB download.

Output is byte-identical in format to tokenize_slice.py: a flat uint16 .bin plus
a `<out>.done` marker. Point train_control.py --data at it.

Resumable: a Range download appends from the current file size; a finished .done
slice is reused. Unauthenticated HF access is fine (set HF_TOKEN for faster pulls).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from huggingface_hub import get_session, hf_hub_url

CHUNK_BYTES = 64 * 1024 * 1024  # 64 MB per request range


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--target-tokens",
        type=float,
        default=10e9,
        help="number of uint16 tokens to extract (default 10B)",
    )
    ap.add_argument(
        "--repo",
        default="EleutherAI/pile-standard-pythia-preshuffled",
        help="use the -deduped variant only if the analysis uses *-deduped models",
    )
    ap.add_argument("--filename", default="document-00000-of-00020.bin")
    args = ap.parse_args()

    target = int(args.target_tokens)
    nbytes = target * 2  # uint16
    out = args.out
    done = out.with_suffix(out.suffix + ".done")
    tmp = out.with_suffix(out.suffix + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)

    if done.exists() and out.exists() and out.stat().st_size == nbytes:
        print(f"[skip] {out} already complete ({done.read_text().strip()} tokens)")
        return

    url = hf_hub_url(args.repo, args.filename, repo_type="dataset")
    sess = get_session()

    # shard 0 must be large enough to hold the requested prefix (best-effort guard;
    # header name / client kwargs vary across requests vs httpx-based hf_hub)
    try:
        head = sess.head(url, follow_redirects=True, timeout=60)
        shard_len = int(
            head.headers.get("x-linked-size") or head.headers.get("Content-Length") or 0
        )
        if shard_len and shard_len < nbytes:
            raise SystemExit(
                f"requested {nbytes} bytes but shard is only {shard_len}; "
                f"need a multi-shard fetch for >{shard_len // 2} tokens"
            )
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 -- guard only; range GETs below still validate
        print(f"[warn] shard-size HEAD check skipped: {type(e).__name__}: {e}")

    start = tmp.stat().st_size if tmp.exists() else 0
    start -= start % 2  # keep token alignment on resume
    mode = "r+b" if start else "wb"
    print(
        f"[fetch] {args.repo}/{args.filename}\n"
        f"        {target / 1e9:.1f}B tokens = {nbytes / 1e9:.1f} GB"
        f"{f' (resume from {start / 1e9:.2f} GB)' if start else ''} -> {out}",
        flush=True,
    )

    with open(tmp, mode) as f:
        if start:
            f.seek(start)
        pos = start
        while pos < nbytes:
            end = min(pos + CHUNK_BYTES, nbytes) - 1
            r = sess.get(url, headers={"Range": f"bytes={pos}-{end}"}, timeout=120)
            r.raise_for_status()
            f.write(r.content)
            pos += len(r.content)
            if (pos // CHUNK_BYTES) % 16 == 0 or pos >= nbytes:
                print(
                    f"  {pos / 1e9:6.2f} / {nbytes / 1e9:.2f} GB "
                    f"({pos / nbytes * 100:5.1f}%)",
                    flush=True,
                )

    os.replace(tmp, out)

    # sanity: valid token range + decodable
    mm = np.memmap(out, dtype=np.uint16, mode="r")
    assert len(mm) == target, (len(mm), target)
    hi = int(mm[:5_000_000].max())
    assert hi < 50304, f"token id {hi} >= 50304 (vocab) -- format mismatch"
    done.write_text(str(target))
    print(f"[done] wrote {target} tokens to {out} (sample max id {hi})", flush=True)
    print(f"data = {out}")


if __name__ == "__main__":
    main()
