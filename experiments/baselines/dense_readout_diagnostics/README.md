# dense_readout_diagnostics

Is raw `W_U` geometry dominated by a mean row shared across the vocabulary,
and when do its dominant dense subspaces move? This experiment computes the
mean-direction audit that motivates the `center_scale` preprocessing and the
adjacent-checkpoint movement of the top centered singular subspaces. It
supports the mean-direction check (`sec:app-mean-direction-preprocessing`)
and the dense baselines (`sec:app-linear-baselines`) of the dictionary
validation appendix.

## Paper results

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:app-mean-direction-preprocessing` | The mean row grows and aligns with the raw top right singular vector after early training, inducing a positive row-cosine floor that centering removes. | `scripts/build_mean_direction_audit.py` | `results/experiments/dense_readout_diagnostics/mean_direction_metrics.{csv,pt}` |
| `fig:app-mean-direction-spectral-gap` | Centering lowers the top spectral gap sigma_1/sigma_2 at late checkpoints in Pythia-1B, Pythia-6.9B, and OLMo-2-7B. | `scripts/build_mean_direction_audit.py` | `results/experiments/dense_readout_diagnostics/mean_direction_metrics.csv` |
| `fig:app-dense-reorg-pythia` | Top-10 subspace drift peaks on the checkpoint pair ending at step 256 in Pythia-160M and 1B and at step 512 in Pythia-6.9B. | `scripts/build_dense_reorganization_timing.py` | `results/experiments/dense_readout_diagnostics/dense_reorganization/{subspace_metrics,sv_stability}.csv` |

## Run

1. Compute the mean-direction audit (top two right singular directions by block power iteration):
   ```bash
   uv run python experiments/baselines/dense_readout_diagnostics/scripts/build_mean_direction_audit.py
   ```
2. Compute adjacent-checkpoint subspace movement (randomized PCA of centered `W_U` on the 32-checkpoint schedule):
   ```bash
   uv run python experiments/baselines/dense_readout_diagnostics/scripts/build_dense_reorganization_timing.py
   ```

## Inputs

- `${UM_SSD_ROOT}/snapshots/pythia-{160m,1b,6.9b}/EleutherAI_pythia-*_step*_wu.pt`
- `${UM_SSD_ROOT}/snapshots/OLMo-2-1124-7B/allenai_OLMo-2-1124-7B_step*_wu.pt`

## Outputs

- `results/experiments/dense_readout_diagnostics/mean_direction_metrics.{csv,pt}`, `mean_direction_provenance.json`: per (model, step) mean-row norm ratio, alignment with the raw top singular vector, raw and centered spectral gap, stable rank, and mean pairwise row cosine.
- `results/experiments/dense_readout_diagnostics/dense_reorganization/subspace_metrics.csv`, `sv_stability.csv`, `top_svd_vectors.pt`, `provenance.json`: chordal Grassmann distance per adjacent pair and subspace rank, and top-vector stability.
- `figures/dense_readout_diagnostics/dense_reorganization_timing.{csv,pt}`, `dense_reorganization_timing_olmo.csv`: the same subspace metrics as tables.

## Notes

- Both scripts default to Pythia-160M, 1B, 6.9B, and OLMo-2-7B (`--models`). The dense reorganization figure shows only the three Pythia models, and the mean-direction figures omit Pythia-160M.
- OLMo-2-7B uses its own 32-checkpoint stage-1 schedule.
- `--copy-to-paper` (off by default) also writes the timing tables into a local `paper/` tree.
