# feature_lifecycle_trajectories

Per-feature normalized trajectories that split readout features into emergence,
maturation, and quiescence populations. This experiment computes metrics only: it
persists the data behind the lifecycle results of Section 4.1 ("The Output Readout
Develops Through a Sparse Lifecycle") and Appendix E (lifecycle diagnostics,
temporal localization, and stability across dictionary fits) and renders no figures
itself.

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| `fig:main-selected-normalized-trajectories` (Fig. 2) | `selected_decoder_norm_trajectories.{csv,pt}`, `normalized_trajectories.csv` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_normalized_trajectories.py` |
| `tab:lifecycle-profile-rules` (Table E.1) | rules only (hand-typeset); implemented in `src/readout/dynamics/lifecycle.py` (`classify_profiles_refined`) | — |
| `fig:app-lifecycle-profile-composition` (Fig. E.1) | `selected_lifecycle_profile_composition_refined_{summary,features}.csv`, `selected_lifecycle_profile_composition_refined.pt` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_lifecycle_profile_composition.py` |
| `fig:app-selected-population-lifecycle-diagnostics` (Fig. E.2), and the adjacent text's window-peak share and mean $\rho_f$ of features peaking by step 128 | `selected_population_lifecycle_diagnostics_{peak_counts,norm_mass,window_peaks,early_peak_decay}.csv`, `selected_population_lifecycle_diagnostics.pt` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/build_population_diagnostics.py` |
| `tab:app-reorganization-window-stats` (Table E.2) | `reorganization_window_summary_selected.csv` (windows, $p$), `reorganization_window_bootstrap_counts_selected.csv` (resamples per window) | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/find_reorganization_steps.py` |
| `fig:app-reorganization-window-metrics` (Fig. E.4) | `reorganization_window_metrics_selected.{csv,pt}`, `reorganization_window_summary_selected.csv` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/find_reorganization_steps.py` |
| `tab:app-lifecycle-stability` (Table E.3) | `lifecycle_stability/{pythia160m_widths_seeds,pythia160m_lambda_sweep,pythia1b_widths,pythia69b_sparsity,validation}.csv` | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/lifecycle_stability.py` |

The paper uses the refined profile ruleset throughout; the un-suffixed
`selected_lifecycle_profile_composition_*` files hold the legacy ruleset and are
not cited. Table E.3's selected Pythia-6.9B row (λ₁ = 0.6) is the
`pythia69b_d32768` row of `lifecycle_stability/validation.csv`. See
[`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim
The output readout develops through sparse lifecycle profiles: decoder-norm
trajectories include early-decaying, late-emerging, and persistent populations,
with rare sharply transitional mid-training profiles. Evidence is strongest for
Pythia-160M and Pythia-1B; sparse Pythia-6.9B and OLMo are used as scale and
cross-family checks. Readout change concentrates in a reorganization window at
steps 512–1.6k in all three Pythia models; 20 of the 21 Pythia dictionary fits
select that window, while the profile shares vary with dictionary width and
sparsity.

## Reproduce (in order, from the repo root)
1. Decoder-norm trajectories (bootstrap — also writes the per-run
   `<key>_decoder_norms.npy` caches under
   `figures/feature_lifecycle_trajectories/section52_lifecycle/cache/` that
   the later steps consume): `uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_normalized_trajectories.py`
2. Profile composition (consumes `selected_decoder_norm_trajectories.pt`): `uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_lifecycle_profile_composition.py`
3. Population diagnostics (peak counts and decoder-norm mass): `uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/build_population_diagnostics.py`
4. Reorganization windows, null $p$ values, and bootstrap resample counts: `uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/find_reorganization_steps.py`
5. Stability across the 21 Pythia dictionary fits (needs steps 2 and 4 for its
   `validate` stage; about 25 min on a 24 GB laptop CPU, dominated by the canonical-rate
   encoder pass over the 16 Pythia-160M fits, whose width-24,576 fits need ~15 GB RAM; `--stage {validate,pythia1b,pythia160m,lambda,pythia69b}` runs one table block, `--device mps` speeds up the encoder pass):
   `uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/lifecycle_stability.py --stage all --out-dir results/experiments/lifecycle/feature_lifecycle_trajectories/lifecycle_stability`
6. Wishbone scores/clusters (auxiliary, not in the camera-ready paper): `uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/plot_selected_wishbone.py`

Steps 4–6 also need the decoder-geometry caches
(`<key>_adjacent_rotation_radians.npy`, `<key>_cos_to_terminal.npy`) in the
same cache dir; on first run they are derived automatically from the released
crosscoder checkpoints listed under Inputs (see `scripts/lifecycle_common.py`).
`lifecycle_stability.py` derives the geometry of the other fits directly from
their released `W_D`, and their activation rates with
`readout.crosscoder.extract_rates.compute_rates_canonical` from the checkpoint and
the Pythia-160M `W_U` snapshots (Pythia-1B and Pythia-6.9B norms and rates come
from the SSD aggregates).

## Inputs (SSD canonical paths)
Selected dictionaries (steps 1–4):
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/olmo-2-7b/W_U/cross-snapshot-32/d32768/seed0.safetensors`
- `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-{1b_d24576,160m_d24576}_seed0.pt`
- `${UM_SSD_ROOT}/derived/rates/wu-d24576-multiseed/decoder_norms_dsae24576_seed0.npy`, `${UM_SSD_ROOT}/derived/rates/wu-d8192-multiseed/decoder_norms_all_seeds.npy`, `${UM_SSD_ROOT}/derived/rates/wu-1b-d24576/decoder_norms_dsae24576_seed0.npy`
- `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/{p69b_sparse,olmo27b}_d32768_seed0.pt` (6.9B sparse and OLMo-2-7B activation rates)

Other fits of Table E.3 (step 5):
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d{8192,16384,24576}/seed*.safetensors` (+ `.config.json`; seeds 0–4 at d8192, 0–2 at d24576, 0 at d16384)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/lambda-sweep/d8192/lam{0p40,1p00,1p20,1p35,1p80}_seed*.safetensors` (+ `.config.json`; seeds 0–2 at λ₁ = 1.35)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d{8192,16384,24576}/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0.safetensors`
- `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-1b_d{8192,16384,24576}_seed0.pt`, `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-6.9b_d32768_seed0.pt`
- `${UM_SSD_ROOT}/snapshots/pythia-160m/EleutherAI_pythia-160m_step*_wu.pt`

## Outputs
Scripts write metrics to `results/experiments/lifecycle/feature_lifecycle_trajectories/`
(and its `wishbone/` and `lifecycle_stability/` subdirs). Paper figures are rendered
in the separate paper LaTeX tree from these regenerated metrics (gitignored — not
shipped in this code-only release).
- `selected_decoder_norm_trajectories{,_full}.{csv,pt}` — decoder-norm trajectory quantiles per snapshot (Fig. 2)
- `normalized_trajectories.csv` — per-(panel, feature, snapshot) normalized norms for 160M/1B
- `selected_lifecycle_profile_composition{,_refined}_{summary,features}.csv` + matching `.pt` — profile-class composition (`_refined` is the paper's ruleset)
- `selected_population_lifecycle_diagnostics_{peak_counts,norm_mass}.csv` + `.pt` — peak-step feature fractions and active decoder-norm mass per step (Fig. E.2)
- `reorganization_step_pair_metrics_selected.csv`, `reorganization_step_peaks_selected.csv`, `reorganization_window_metrics_selected.{csv,pt}`, `reorganization_window_summary_selected.csv`, `reorganization_bootstrap_selected.csv`, `reorganization_window_bootstrap_counts_selected.csv` — reorganization-step peaks, window metrics, null $p$ values, and bootstrap resamples per window (Fig. E.4, Table E.2)
- `lifecycle_stability/{pythia160m_widths_seeds,pythia160m_lambda_sweep,pythia1b_widths,pythia69b_sparsity}.csv` — profile fractions, active count, median peak step, selected window and null $p$ per dictionary fit; `lifecycle_stability/validation.csv` — the four selected dictionaries recomputed by the same code beside the regenerated selected-dictionary CSVs (Table E.3)
- `wishbone/*_scores_metrics.csv`, `wishbone/*_wishbone.pt`, `wishbone/selected_wishbone_summary.csv`, `wishbone/selected_wishbone_k2_pca_clusters.pt`, `wishbone/selected_wishbone_manual_corner_split.pt` — wishbone scores/clusters per run (auxiliary)

## Layout
| path | role |
|---|---|
| `scripts/` | lifecycle metric computation (no figure rendering): `plot_normalized_trajectories.py` (decoder-norm trajectories), `plot_lifecycle_profile_composition.py` (profile composition), `build_population_diagnostics.py` (peak counts and norm mass), `find_reorganization_steps.py` (reorganization-step peaks/windows and bootstrap counts), `lifecycle_stability.py` (profile fractions and windows across dictionary fits), `plot_selected_wishbone.py` (wishbone scores/clusters); `lifecycle_common.py` holds the shared run table and cache loaders. The profile rules live in `src/readout/dynamics/lifecycle.py`. |
