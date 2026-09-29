#!/usr/bin/env python3
"""Build the held-out evaluation slice for the recipe-control swap / probe analyses.

The arms train on the first ``--train-tokens`` (10e9) tokens of the Pythia
preshuffled shard ``document-00000-of-00020.bin`` in sequential order. This
range-fetches a small slice starting at ``--start-token`` (default = the train
budget, i.e. immediately after the trained region), so the eval is
in-distribution but provably unseen by every condition (they share the same
first-10B slice). The shard is 30 GB = 15B tokens, so [10B, 10B+2M) exists.
Byte-identical format to the trainer's data (flat uint16, no ``.idx``).
Writes ``<out>`` and ``<out>.sha256`` (provenance of the slice).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from huggingface_hub import get_session, hf_hub_url
from rc_common import default_eval_bin

from readout.probes.recipe_control_models import VOCAB

CHUNK = 64 * 1024 * 1024


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=default_eval_bin())
    ap.add_argument("--start-token", type=float, default=10e9)
    ap.add_argument("--eval-tokens", type=float, default=2e6)
    ap.add_argument("--train-tokens", type=float, default=10e9, help="guard: start-token must be >= this")
    ap.add_argument("--repo", default="EleutherAI/pile-standard-pythia-preshuffled")
    ap.add_argument("--filename", default="document-00000-of-00020.bin")
    args = ap.parse_args()

    start_tok, n_tok = int(args.start_token), int(args.eval_tokens)
    if start_tok < int(args.train_tokens):
        raise SystemExit(f"start-token {start_tok} < train-tokens {int(args.train_tokens)}: overlaps training")
    start_byte, nbytes = start_tok * 2, n_tok * 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    url = hf_hub_url(args.repo, args.filename, repo_type="dataset")
    sess = get_session()
    print(f"[heldout] {args.repo}/{args.filename} tokens [{start_tok}, {start_tok + n_tok}) -> {args.out}", flush=True)

    buf, pos, end = bytearray(), start_byte, start_byte + nbytes
    while pos < end:
        hi = min(pos + CHUNK, end) - 1
        r = sess.get(url, headers={"Range": f"bytes={pos}-{hi}"}, timeout=120)
        r.raise_for_status()
        buf.extend(r.content)
        pos += len(r.content)
    if len(buf) != nbytes:
        raise SystemExit(f"short read: {len(buf)} bytes != {nbytes}")

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_bytes(bytes(buf))
    tmp.replace(args.out)

    mm = np.memmap(args.out, dtype=np.uint16, mode="r")
    hi_id = int(mm.max())
    if len(mm) != n_tok or hi_id >= VOCAB:
        raise SystemExit(f"slice check failed: {len(mm)} tokens, max id {hi_id}")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(
        f"{digest}  {args.out.name}  tokens={n_tok} start={start_tok} max_id={hi_id}\n"
    )
    print(f"[heldout] wrote {n_tok} tokens (max id {hi_id}) sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
