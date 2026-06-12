"""CLI front door for per-snapshot firing-rate extraction.

Thin wrapper around ``src.crosscoder.extract_rates.main`` so rate extraction
can be invoked as a script (the library module has no ``__main__`` block per
the src/ layout contract; it stays importable for its in-repo consumers).
Computes per-snapshot per-feature firing rates for a trained crosscoder
checkpoint against a W_U snapshot cache.

Usage:
  uv run python scripts/extract/extract_rates.py \
      --ckpts <checkpoint.pt> --cache-dir <snapshot_dir> --output <rates.pt> \
      --device cpu
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from src.crosscoder.extract_rates import main  # noqa: E402

if __name__ == "__main__":
    main()
