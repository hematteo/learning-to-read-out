# heldout_checkpoints

Do the trajectory crosscoders memorize their 32 anchor checkpoints? This
experiment reconstructs `W_U` at eight Pythia checkpoints outside the anchor
schedule (steps 10k, 23k, 33k, 43k, 73k, 93k, 113k, and 123k) with encoder
and decoder heads interpolated from the neighboring anchors, and compares
them with nearest-head, raw-interpolation, endpoint, and rank-50 PCA
references and the in-sample ceiling. It supports
`sec:app-heldout-checkpoints` ("Held-out Checkpoint Reconstruction").

## Paper results

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:app-heldout-checkpoints-1b` | Interpolated heads of the Pythia-1B d=24,576 seed-0 crosscoder track the in-sample ceiling within 0.01 EV at six held-out steps and deviate only at 113k and 123k. | `scripts/eval_heldout.py` | `heldout_ev.csv`, `heldout_ev.pt` |
| `sec:app-heldout-checkpoints` (text) | For the Pythia-160M d=8,192 seed-0 crosscoder, synthetic heads track the in-sample ceiling within 0.011 EV on all eight held-out checkpoints. | `scripts/eval_heldout.py` | `heldout_ev.csv` |

## Run

1. Evaluate the Pythia-1B dictionary:
   ```bash
   uv run python experiments/ablations/heldout_checkpoints/scripts/eval_heldout.py \
       --crosscoder ${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors \
       --snapshot-dir ${UM_SSD_ROOT}/snapshots/pythia-1b \
       --out-dir results/experiments/ablations/heldout_checkpoints/pythia-1b \
       --model-slug EleutherAI_pythia-1b
   ```
2. Evaluate the Pythia-160M dictionary:
   ```bash
   uv run python experiments/ablations/heldout_checkpoints/scripts/eval_heldout.py \
       --crosscoder ${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d8192/seed0.safetensors \
       --snapshot-dir ${UM_SSD_ROOT}/snapshots/pythia-160m \
       --out-dir results/experiments/ablations/heldout_checkpoints/pythia-160m \
       --model-slug EleutherAI_pythia-160m
   ```

## Inputs

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d8192/seed0.safetensors`
- `${UM_SSD_ROOT}/snapshots/pythia-{1b,160m}/`: the 32 anchor snapshots plus the eight held-out steps.

## Outputs

Under each `--out-dir`:

- `heldout_ev.csv`: EV per held-out checkpoint for the interpolated heads and each reference (nearest head, raw `W_U` interpolation, endpoint interpolation, rank-50 PCA), with the in-sample EV of the neighboring anchors.
- `heldout_ev.pt`: the same rows plus `trained_steps`, `held_out`, ceiling EVs, residuals, `pca_rank`, and the source crosscoder path.

## Notes

- The PCA reference uses rank 50 (`--pca-rank`); `--seed` seeds its randomized fit.
