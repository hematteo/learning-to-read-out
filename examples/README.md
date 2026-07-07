# Minimal crosscoder example

A small, CPU-only, fully seeded sanity check for the W_U trajectory-crosscoder
code path. It synthesizes a tiny stack of unembedding (`W_U`) snapshots with
realistic low-rank structure (no downloads, no data files, no GPU), per-snapshot
row-mean centers it, builds and trains a small crosscoder, and reports how much
reconstruction quality training buys.

## What it does

1. `seed_everything(0)` — seeds Python / NumPy / torch (CPU) for reproducibility.
2. Synthesizes `snapshots` of shape `(8, 256, 64)` = `(n_snapshots, vocab, d_model)`,
   where each row is a sparse mix of 16 shared "atoms" reweighted slightly per
   checkpoint — the kind of low-rank, gradually drifting structure a sparse
   dictionary can capture (real `W_U` snapshots behave this way).
3. `preprocess_snapshots(..., mode="center")` — per-snapshot row-mean centering.
4. `build_crosscoder(..., device="cpu")` — a tiny untrained crosscoder
   (`expansion_factor=4.0`, so `d_sae = 256`) scored as a baseline.
5. `train(..., device="cpu", n_epochs=40, batch_size=64)` — ~160 optimizer steps.
6. `quick_quality(..., device="cpu")` — explained variance (EV) and mean L0,
   before vs. after training.

Every entry point is passed `device="cpu"` explicitly (their default
auto-selects cuda/mps/cpu), mirroring `tests/test_crosscoder_smoke.py`.

## How to run

From the repo root (the script imports the installed `readout` package —
`make install` / `uv sync` provides it; no `sys.path` edits):

```bash
uv run python examples/minimal_crosscoder.py
```

Runs deterministically in a few seconds on CPU.

## Expected output

A few training log lines, then a single final summary line of the form:

```
FINAL  EV: untrained=<~0.00> -> trained=<~0.80>  L0=<float>  (d_sae=256, K=8, V=256)
```

`EV` is the explained variance (in `(-inf, 1]`; higher is better): it climbs from
~0 for the untrained crosscoder to ~0.8 after training, showing the dictionary
learns to reconstruct the structured snapshots. `L0` is the mean number of active
features per row (in `[0, d_sae] = [0, 256]`). Because everything is seeded, the
printed numbers are reproducible across runs.
