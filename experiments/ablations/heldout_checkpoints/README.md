# heldout_checkpoints

Held-out checkpoint reconstruction: train a crosscoder on a sparse snapshot grid,
then evaluate explained variance on snapshots NOT in that grid. This validates that
the crosscoder generalizes off the training-snapshot subset (i.e. captures continuous
trajectory structure, not just a fixed snapshot set).

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| `fig:app-heldout-checkpoints` | `heldout_ev.csv` / `heldout_ev.pt` (pythia-160m) | `experiments/ablations/heldout_checkpoints/scripts/eval_heldout.py` |
| `fig:app-heldout-checkpoints-1b` | `heldout_ev.csv` / `heldout_ev.pt` (pythia-1b) | `experiments/ablations/heldout_checkpoints/scripts/eval_heldout.py` |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim
Held-out checkpoint reconstruction: train a crosscoder on a sparse snapshot grid,
then evaluate explained variance on snapshots NOT in that grid. Validates that the
crosscoder generalizes off the training-snapshot subset (i.e. captures continuous
trajectory structure, not just a fixed snapshot set).

## Reproduce
**Evaluate held-out EV** — uses the released 32-snapshot crosscoder and
evaluates reconstruction on snapshots outside the cross-32 grid. Run once per
model (160m / 1b), pointing at that model's released crosscoder and full
snapshot set:
```
uv run python experiments/ablations/heldout_checkpoints/scripts/eval_heldout.py \
  --crosscoder <crosscoder.safetensors> \
  --snapshot-dir <snapshots/pythia-{160m,1b}/> \
  --out-dir <results-dir> \
  --model-slug EleutherAI_pythia-{160m,1b}
```
This writes `heldout_ev.csv` and `heldout_ev.pt` (see "Outputs").

## Inputs (SSD canonical paths)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d8192/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/snapshots/pythia-{160m,1b}/` — full snapshot set (incl. snapshots not in cross-32 grid)

## Outputs
`scripts/eval_heldout.py` writes, into its `--out-dir`:
- `heldout_ev.csv` — per-checkpoint explained variance on held-out folds (direct,
  endpoint, PCA, plus raw/norm ceilings).
- `heldout_ev.pt` — same rows plus `trained_steps`, `held_out`, ceiling EVs,
  residuals, `pca_rank`, and the source crosscoder path.

These regenerated metrics (gitignored — not shipped in this code-only release)
are the reproducible artifacts. This repo ships no
figure-rendering code; the paper figures are rendered in the separate paper
LaTeX tree from these metrics.

## Layout
| path | role |
|---|---|
| `scripts/eval_heldout.py`   | computes per-checkpoint EV on held-out folds |
