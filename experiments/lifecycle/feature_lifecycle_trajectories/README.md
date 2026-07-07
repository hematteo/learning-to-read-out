# feature_lifecycle_trajectories

Per-feature normalized trajectories that split readout features into emergence,
maturation, and quiescence populations. This experiment computes metrics only: it
persists the data behind the lifecycle figures of the "Output Readout Develops
Through a Sparse Lifecycle" section (§5.2 + appendix B) and renders no figures
itself.

## Figures produced

| Thesis label | Metric file | Producing script |
|---|---|---|
| `fig:main-section52-selected-heatmaps` | `selected_decoder_norm_trajectories.{csv,pt}` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_normalized_trajectories.py` |
| `fig:main-selected-normalized-trajectories` | `selected_decoder_norm_trajectories.{csv,pt}`, `normalized_trajectories.csv` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_normalized_trajectories.py` |
| `fig:app-selected-normalized-trajectories` | `selected_decoder_norm_trajectories_full.{csv,pt}` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_normalized_trajectories.py` |
| `fig:app-selected-decoder-norm-heatmaps` | `selected_decoder_norm_trajectories.{csv,pt}` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_normalized_trajectories.py` |
| `fig:lr-trajectories` | `selected_decoder_norm_trajectories.{csv,pt}` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_normalized_trajectories.py` |
| `fig:app-lifecycle-profile-composition` | `selected_lifecycle_profile_composition_{summary,features}.{csv,pt}` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_lifecycle_profile_composition.py` |
| `fig:app-selected-population-lifecycle-diagnostics` | `selected_lifecycle_profile_composition_refined_{summary,features}.{csv,pt}` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_lifecycle_profile_composition.py` |
| `fig:app-metric-lifecycle-combined` | `selected_lifecycle_profile_composition_{summary,features}.{csv,pt}` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_lifecycle_profile_composition.py` |
| `fig:app-wishbone-manual-corner-split` | `wishbone/selected_wishbone_manual_corner_split.pt`, `wishbone/selected_wishbone_summary.csv` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_selected_wishbone.py` |
| `fig:app-wishbone-pca` | `wishbone/selected_wishbone_k2_pca_clusters.pt`, `wishbone/*_scores_metrics.csv` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_selected_wishbone.py` |
| `fig:app-prepost-reorganization-peakstep` | `reorganization_step_peaks_selected.csv`, `reorganization_step_pair_metrics_selected.csv` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/find_reorganization_steps.py` |
| `fig:app-reorganization-window-metrics` | `reorganization_window_metrics_selected.{csv,pt}`, `reorganization_window_summary_selected.csv` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/find_reorganization_steps.py` |
| `fig:lr-reorg-window` | `reorganization_window_metrics_selected.{csv,pt}` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/find_reorganization_steps.py` |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim
The output readout develops through sparse lifecycle profiles: decoder-norm
trajectories include early-decaying, late-emerging, and persistent populations,
with rare sharply transitional mid-training profiles. Evidence is strongest for
Pythia-160M and Pythia-1B; sparse Pythia-6.9B and OLMo are used as scale and
cross-family checks.

## Reproduce (in order; scripts bootstrap the repo onto `sys.path`)
1. Decoder-norm trajectories (bootstrap — also writes the per-run
   `<key>_decoder_norms.npy` caches under
   `figures/feature_lifecycle_trajectories/section52_lifecycle/cache/` that
   steps 3–4 consume): `uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_normalized_trajectories.py`
2. Profile composition (consumes `selected_decoder_norm_trajectories.pt`): `uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_lifecycle_profile_composition.py`
3. Wishbone scores/clusters: `uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_selected_wishbone.py`
4. Reorganization steps: `uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/find_reorganization_steps.py`

Steps 3–4 also need the decoder-geometry caches
(`<key>_adjacent_rotation_radians.npy`, `<key>_cos_to_terminal.npy`) in the
same cache dir; on first run they are derived automatically from the released
crosscoder checkpoints listed under Inputs (see `scripts/lifecycle_common.py`).

## Inputs (SSD canonical paths)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/olmo-2-7b/W_U/cross-snapshot-32/d32768/seed0.safetensors`
- `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/olmo27b_d32768_seed0.{json,csv}`
- `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-{1b_d24576,160m_d24576}_seed0.pt`

## Outputs
Scripts write metrics to `results/experiments/lifecycle/feature_lifecycle_trajectories/`
(and its `wishbone/` subdir). Thesis figures are rendered in the separate thesis
LaTeX tree from these regenerated metrics (gitignored — not shipped in this
code-only release).
- `selected_decoder_norm_trajectories{,_full}.{csv,pt}` — decoder-norm trajectory quantiles per snapshot (§5.2 main + appendix)
- `normalized_trajectories.csv` — per-(panel, feature, snapshot) normalized norms for 160M/1B
- `selected_lifecycle_profile_composition{,_refined}_{summary,features}.csv` + matching `.pt` — profile-class composition
- `wishbone/*_scores_metrics.csv`, `wishbone/*_wishbone.pt`, `wishbone/selected_wishbone_summary.csv`, `wishbone/selected_wishbone_k2_pca_clusters.pt`, `wishbone/selected_wishbone_manual_corner_split.pt` — wishbone scores/clusters per run
- `reorganization_step_pair_metrics_selected.csv`, `reorganization_step_peaks_selected.csv`, `reorganization_window_metrics_selected.{csv,pt}`, `reorganization_window_summary_selected.csv`, `reorganization_bootstrap_selected.csv` — reorganization-step peaks and window metrics

## Layout
| path | role |
|---|---|
| `scripts/` | lifecycle metric computation (no figure rendering): `plot_normalized_trajectories.py` (decoder-norm trajectories), `plot_lifecycle_profile_composition.py` (profile composition), `plot_selected_wishbone.py` (wishbone scores/clusters), `find_reorganization_steps.py` (reorganization-step peaks/windows). The shared `dynamics` core library is imported from `src/readout/dynamics/`. |
