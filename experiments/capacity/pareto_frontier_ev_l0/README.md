# pareto_frontier_ev_l0

How do reconstruction and sparsity trade off across the Pythia-160M
dictionaries (widths, seeds, `W_U` versus `W_E`, architecture variants, and
sparsity weights)? The two paper figures plot aggregate EV against mean L0,
and those values come from the dictionary inventory that `crosscoder_main`
regenerates. The scripts in this directory add per-snapshot EV and L0 for the
same fits. It supports the SAE baseline and sparsity-penalty subsections of
`sec:app-instrument-validation`.

## Paper results

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:app-instrument-pareto` | `W_U` dictionaries reach high EV at sparse activation counts with small seed spread, and `W_E` dictionaries reach lower EV than `W_U` dictionaries of the same width. | `experiments/crosscoders/crosscoder_main/scripts/appendix_validation/recompute_quality.py` | `experiments/crosscoders/crosscoder_main/derived/appendix_validation/full_inventory.csv` |
| `fig:app-lambda-sweep` | Raising lambda_1 from 0.4 to 1.8 lowers mean L0 from about 160 to about 23 and EV from 0.75 to 0.53. | `experiments/crosscoders/crosscoder_main/scripts/appendix_validation/recompute_quality.py` (`--filter lambda-sweep` for the sweep fits) | `experiments/crosscoders/crosscoder_main/derived/appendix_validation/lambda_sweep.csv`, `full_inventory.csv` |

The commands for both are in the `crosscoder_main` README (Run, step 1).

## Run

1. Per-snapshot EV and L0 for eight Pythia-160M fits, seed 0 each (`W_U` d=8,192, 16,384, and 24,576; the 16-checkpoint d=8,192 fit; BatchTopK and gated-retuned d=8,192; `W_E` d=8,192 and 24,576):
   ```bash
   uv run python experiments/capacity/pareto_frontier_ev_l0/scripts/compute_ev_l0_phase2.py
   ```
2. Per-snapshot (L0, EV) for the Pythia-160M `W_U` fits across seeds (d=8,192 seeds 0 to 4, d=16,384, d=24,576 seeds 0 to 2, the 16-checkpoint d=8,192 fit):
   ```bash
   uv run python experiments/capacity/pareto_frontier_ev_l0/scripts/compute_l0_ev_phase_plane.py
   ```
3. Final-snapshot SAE EV against crosscoder EV, as a text table (point `--sae-dir`, `--cc-d8192`, and `--cc-d16384` at your checkpoints):
   ```bash
   uv run python experiments/capacity/pareto_frontier_ev_l0/scripts/compute_sae_pareto_table.py
   ```

## Inputs

- Step 1: `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/{W_U,W_E}/...` (the eight released checkpoints listed in the script), `${UM_SSD_ROOT}/snapshots/`, and the `W_E` snapshot cache `${UM_SSD_ROOT}/we_snapshots/`.
- Step 2: checkpoints under `${UM_SSD_ROOT}/derived/rates/{wu-d8192-multiseed,wu-d24576-multiseed,wu-d8192-cs16}/` and `${UM_SSD_ROOT}/run4_dsae16384/`, plus `${UM_SSD_ROOT}/snapshots/` (`--release-root` changes the root).

## Outputs

- `results/experiments/capacity/pareto_frontier_ev_l0/ev_l0_phase2.json`: per-fit, per-snapshot EV and L0, NRMSE, EV mean, min, and max, and active and dead feature counts.
- `figures/pareto_frontier_ev_l0/l0_ev_phase_plane.csv` (`label, family, d_sae, snap_idx, step, L0, EV`) and `l0_ev_phase_plane_cache.pt`.
- `experiments/capacity/pareto_frontier_ev_l0/derived/sae_pareto.txt`.

## Notes

- The paper does not plot the per-snapshot outputs of steps 1 to 3.
- `scripts/extract_rates_phase2.sh` computes firing rates for checkpoints placed under `${UM_SSD_ROOT}/cluster_results/<run>/`; none of the steps above read them.
