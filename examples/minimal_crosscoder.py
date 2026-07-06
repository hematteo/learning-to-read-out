"""Minimal, CPU-only, seeded W_U trajectory-crosscoder example.

Synthesizes a tiny fake stack of unembedding (W_U) snapshots, preprocesses,
trains a small crosscoder for a handful of steps, and prints explained
variance (EV) and mean L0. No downloads, no data files, no GPU.

Run from the repo root:
    uv run python examples/minimal_crosscoder.py

Mirrors the CPU call pattern of tests/test_crosscoder_smoke.py: every entry
point (build_crosscoder, train, quick_quality) is passed device="cpu"
explicitly (their default auto-selects cuda/mps/cpu).
"""

import logging
import warnings

# @torch.autocast(device_type="cuda") at import time; harmless on CPU.
warnings.filterwarnings("ignore", message="CUDA is not available")
# torch.distributed logs a redirects notice at import on macOS/Windows; not
# relevant to this single-process CPU demo.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch

from readout.core.repro import seed_everything
from readout.crosscoder.wu_adapter import (
    build_crosscoder,
    preprocess_snapshots,
    quick_quality,
    train,
)

# Seed Python / NumPy / torch (CPU) from one integer for a reproducible run.
seed_everything(0)

# Tiny synthetic problem. Real runs use K=32 snapshots, V~50k vocab, d~768.
N_SNAPSHOTS = 8  # K: number of W_U training snapshots stacked on the hook axis
VOCAB = 256  # V: vocabulary rows, treated as the crosscoder's "batch" axis
D_MODEL = 64  # d: model hidden dim (W_U row width)
N_ATOMS = 16  # rows are sparse mixes of this many shared directions
EXPANSION_FACTOR = 4.0  # d_sae = expansion_factor * d_model = 256 features

# Fake W_U snapshot stack with structure a sparse dictionary can actually capture:
# each row is a sparse mix of a few shared directions ("atoms"), reweighted slightly
# per checkpoint (a smooth readout drift). Real W_U snapshots are low-rank and evolve
# gradually; without this structure a sparse dictionary has nothing to reconstruct.
atoms = torch.randn(N_ATOMS, D_MODEL)  # (n_atoms, d_model) shared dictionary
support = (torch.rand(VOCAB, N_ATOMS) < 0.25).float()  # (V, n_atoms) ~4 active atoms/row
coeffs = torch.randn(VOCAB, N_ATOMS) * support  # (V, n_atoms) sparse per-row loadings
scales = torch.linspace(0.8, 1.2, N_SNAPSHOTS).view(N_SNAPSHOTS, 1, 1)  # per-checkpoint drift
snapshots = scales * (coeffs @ atoms).unsqueeze(0)  # (n_snapshots, vocab, d_model)
snapshots += 0.05 * torch.randn_like(snapshots)  # small observation noise

# Per-snapshot row-mean centering, exactly as wu_adapter expects (stats invertible).
snapshots, _stats = preprocess_snapshots(snapshots, mode="center")  # (K, V, d), dict|None

# Untrained baseline: build a tiny crosscoder on CPU and score it as-is, to
# contrast with the trained model below. (build_crosscoder / train / quick_quality
# default device=None, which auto-selects cuda/mps/cpu — pass device="cpu"
# everywhere for a deterministic CPU run.)
baseline = build_crosscoder(
    N_SNAPSHOTS,
    D_MODEL,
    expansion_factor=EXPANSION_FACTOR,
    device="cpu",
)
ev_untrained = quick_quality(baseline, snapshots, batch_size=64, device="cpu")["explained_variance"]

# Train on CPU. batch_size 64 over V=256 -> 4 steps/epoch, 40 epochs -> 160 steps.
# train() builds and returns its own crosscoder, so use its return value.
crosscoder = train(
    snapshots,
    expansion_factor=EXPANSION_FACTOR,
    n_epochs=40,
    batch_size=64,
    device="cpu",
    seed=0,
    log_every=40,  # ~4 progress lines
)

# Score reconstruction quality over the full snapshot stack (device="cpu"!).
metrics = quick_quality(crosscoder, snapshots, batch_size=64, device="cpu")

# explained_variance in (-inf, 1], mean_l0 in [0, d_sae]; d_sae = 256 here.
print(
    f"FINAL  EV: untrained={ev_untrained:.4f} -> trained={metrics['explained_variance']:.4f}  "
    f"L0={metrics['mean_l0']:.2f}  "
    f"(d_sae={metrics['d_sae']}, K={metrics['n_snapshots']}, V={metrics['n_rows']})"
)
