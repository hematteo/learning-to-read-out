"""CLI front door for parameter-trajectory crosscoder training.

Thin wrapper around ``readout.crosscoder.wu_adapter.main`` so the trainer can be
invoked as a script (the library module has no ``__main__`` block per the
src/ layout contract). Trains a trajectory crosscoder on a cache of extracted
W_U snapshots; see docs/REPRODUCE.md for the figure-by-figure reproduction map and
the flags below.

Usage:
  uv run python scripts/train/train_crosscoder.py \
      --model EleutherAI/pythia-160m --expansion-factor 32.0 \
      --batch-size 1024 --lr 5e-5 --n-epochs 300 --seed 0 \
      --input-preprocess center_scale --amp-dtype fp32 \
      --tanh-stretch 1.0 \
      --output local_snapshots/wu_cc_pythia160m_seed0.pt

The settings-of-record for each published crosscoder are in configs/runs/*.yaml
(the published W_U runs use --tanh-stretch 1.0; the CLI default 4.0 is the
W_E value).
"""

from __future__ import annotations

from readout.crosscoder.wu_adapter import main  # noqa: E402

if __name__ == "__main__":
    main()
