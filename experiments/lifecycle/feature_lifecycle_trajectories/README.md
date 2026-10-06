# feature_lifecycle_trajectories

When does the readout reorganize, and do its sparse features share one
trajectory? This experiment computes per-feature decoder-norm trajectories,
rule-based lifecycle profiles, population diagnostics, the reorganization
window with its circular-shift nulls and bootstrap, and the stability of these
results across 21 Pythia dictionary fits. It supports `sec:readout-lifecycle`
("The Readout Reorganizes Early") and the lifecycle appendices
(`sec:app-lifecycle-diagnostics`, `sec:app-temporal-localization`,
`sec:app-lifecycle-stability`).

## Paper results

Outputs are under `results/experiments/lifecycle/feature_lifecycle_trajectories/`.

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:main-selected-normalized-trajectories` | Peak-normalized decoder-norm trajectories of every active feature in the four selected dictionaries: many features decay early in Pythia-160M and 1B, nearly all persist in Pythia-6.9B, and most emerge late in OLMo-2-7B. | `scripts/compute_normalized_trajectories.py` | `selected_decoder_norm_trajectories.{csv,pt}`, `normalized_trajectories.csv` |
| `fig:app-lifecycle-profile-composition` | Profile shares under the rules of `tab:lifecycle-profile-rules`. | `scripts/compute_lifecycle_profile_composition.py` | `selected_lifecycle_profile_composition_refined_{summary,features}.csv`, `selected_lifecycle_profile_composition_refined.pt` |
| `fig:app-selected-population-lifecycle-diagnostics` | Fraction of active features peaking at each checkpoint and total decoder-norm mass; the text adds the share of peaks inside the window and the mean rho_f of features that peak by step 128. | `scripts/build_population_diagnostics.py` | `selected_population_lifecycle_diagnostics_{peak_counts,norm_mass,window_peaks,early_peak_decay}.csv`, `selected_population_lifecycle_diagnostics.pt` |
| `fig:app-reorganization-window-metrics` | The reorganization score peaks between step 512 and step 1.6k in all three Pythia models, with p=0.005 at 160M and 1B and p=0.04 at 6.9B. | `scripts/find_reorganization_steps.py` | `reorganization_window_metrics_selected.{csv,pt}`, `reorganization_window_summary_selected.csv` |
| `tab:app-reorganization-window-stats` | Selected window, circular-shift p value, and how many of 400 feature resamples select each window, per model. | `scripts/find_reorganization_steps.py` | `reorganization_window_summary_selected.csv`, `reorganization_window_bootstrap_counts_selected.csv` |
| `tab:app-lifecycle-stability` | 20 of the 21 Pythia dictionaries select the same window; profile shares vary with width and sparsity. | `scripts/lifecycle_stability.py` | `lifecycle_stability/{pythia160m_widths_seeds,pythia160m_lambda_sweep,pythia1b_widths,pythia69b_sparsity,validation}.csv` |

## Run

1. Compute decoder-norm trajectories; this also writes the per-run `<key>_decoder_norms.npy` caches that later steps and `olmo_matched_checkpoint_window` read:
   ```bash
   uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/compute_normalized_trajectories.py
   ```
2. Assign lifecycle profiles (reads `selected_decoder_norm_trajectories.pt`):
   ```bash
   uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/compute_lifecycle_profile_composition.py
   ```
3. Compute peak counts and decoder-norm mass per checkpoint:
   ```bash
   uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/build_population_diagnostics.py
   ```
4. Find reorganization windows, null p values, and bootstrap counts:
   ```bash
   uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/find_reorganization_steps.py
   ```
5. Repeat the profile and window analysis for every Pythia dictionary fit (its `validate` stage needs steps 2 and 4):
   ```bash
   uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/lifecycle_stability.py --stage all \
       --out-dir results/experiments/lifecycle/feature_lifecycle_trajectories/lifecycle_stability
   ```

## Inputs

Selected dictionaries (steps 1 to 4):

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/olmo-2-7b/W_U/cross-snapshot-32/d32768/seed0.safetensors`
- `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-{1b_d24576,160m_d24576}_seed0.pt`
- `${UM_SSD_ROOT}/derived/rates/wu-d24576-multiseed/decoder_norms_dsae24576_seed0.npy`, `${UM_SSD_ROOT}/derived/rates/wu-d8192-multiseed/decoder_norms_all_seeds.npy`, `${UM_SSD_ROOT}/derived/rates/wu-1b-d24576/decoder_norms_dsae24576_seed0.npy`
- `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/{p69b_sparse,olmo27b}_d32768_seed0.pt`: Pythia-6.9B and OLMo-2-7B activation rates (curated; see `crosscoder_main`).

Other fits (step 5):

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d{8192,16384,24576}/seed*.safetensors` (+ `.config.json`; seeds 0 to 4 at d=8,192, 0 to 2 at d=24,576, 0 at d=16,384)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/lambda-sweep/d8192/lam{0p40,1p00,1p20,1p35,1p80}_seed*.safetensors` (+ `.config.json`; seeds 0 to 2 at lambda_1 = 1.35)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d{8192,16384,24576}/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0.safetensors`
- `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-1b_d{8192,16384,24576}_seed0.pt`, `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-6.9b_d32768_seed0.pt`
- `${UM_SSD_ROOT}/snapshots/pythia-160m/EleutherAI_pythia-160m_step*_wu.pt`

## Outputs

Under `results/experiments/lifecycle/feature_lifecycle_trajectories/`:

- `selected_decoder_norm_trajectories{,_full}.{csv,pt}`: decoder-norm trajectory quantiles per snapshot; `normalized_trajectories.csv`: per-(panel, feature, snapshot) normalized norms for Pythia-160M and 1B.
- `selected_lifecycle_profile_composition{,_refined}_{summary,features}.csv` and matching `.pt`: profile shares.
- `selected_population_lifecycle_diagnostics_*.csv` and `.pt`: peak counts, norm mass, window peaks, early-peak decay.
- `reorganization_step_pair_metrics_selected.csv`, `reorganization_step_peaks_selected.csv`, `reorganization_window_metrics_selected.{csv,pt}`, `reorganization_window_summary_selected.csv`, `reorganization_bootstrap_selected.csv`, `reorganization_window_bootstrap_counts_selected.csv`: adjacent-pair metrics, window scores, null p values, and bootstrap resamples.
- `lifecycle_stability/*.csv`: profile shares, active count, median peak step, selected window, and null p per fit; `validation.csv` recomputes the four selected dictionaries with the same code.

Decoder-norm and decoder-geometry caches (`<key>_decoder_norms.npy`,
`<key>_adjacent_rotation_radians.npy`, `<key>_cos_to_terminal.npy`) go to
`figures/feature_lifecycle_trajectories/section52_lifecycle/cache/`.

## Notes

- The profile rules of `tab:lifecycle-profile-rules` are implemented in `src/readout/dynamics/lifecycle.py` (`classify_profiles_refined`). `compute_lifecycle_profile_composition.py` writes the same refined-rule shares under two stems; the paper uses the `_refined` files.
- The decoder-geometry caches are derived from the released checkpoints on first run (`scripts/lifecycle_common.py`). `lifecycle_stability.py` derives the geometry of the other fits from their released `W_D`, and their activation rates with `readout.crosscoder.extract_rates.compute_rates_canonical` from the checkpoint and the Pythia-160M `W_U` snapshots; Pythia-1B and 6.9B norms and rates come from the aggregates.
- The selected Pythia-6.9B row of `tab:app-lifecycle-stability` (lambda_1 = 0.6) is the `pythia69b_d32768` row of `lifecycle_stability/validation.csv`.
- Step 5 takes about 25 minutes on a 24 GB laptop CPU, dominated by the activation-rate pass over the 16 Pythia-160M fits; the d=24,576 fits need about 15 GB of RAM. `--stage {validate,pythia1b,pythia160m,lambda,pythia69b}` runs one block, and `--device mps` speeds up the encoder pass.
