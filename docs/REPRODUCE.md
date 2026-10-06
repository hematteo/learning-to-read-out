# Reproducing the paper

There are three levels, from cheapest to most expensive.

1. **Redraw the figures** with `make figures` (CPU, about 30 s). The scripts in
   [`paper_figures/`](../paper_figures/README.md) redraw 14 of the paper's data
   figures, including Figures 2 to 5, from the 4.9 MB of inputs in
   `paper_figures/data/`.
2. **Recompute those inputs** from the released dictionaries and checkpoints with
   the analysis scripts listed below. They read the released artifacts from
   `UM_SSD_ROOT` (see [DATA.md](DATA.md)); the larger models need a GPU.
3. **Retrain** the dictionaries (`scripts/train/train_crosscoder.py`, settings in
   `configs/runs/`) or the 31M runs
   (`experiments/ablations/pretraining_recipe_control/`).

[`experiments.yaml`](../experiments.yaml) is the manifest that links each
experiment to the figures and tables it backs; each experiment's README lists its
commands in order. The quickest check of the code itself is
`make install && make test`, which needs no data.

## Figure index

Alphabetical by label; deep-link as `docs/REPRODUCE.md#<label>`. The third column
is the figure's file in the paper's LaTeX source. This table is generated from
`experiments.yaml` by `scripts/audit/gen_reproduce_index.py`, and CI fails if they
drift; edit the manifest, not the table.

<!-- BEGIN GENERATED FIGURE INDEX (scripts/audit/gen_reproduce_index.py) -->
| Label | Experiment | Rendered figure (paper LaTeX tree) |
|---|---|---|
| <a id="fig:app-ae-capacity"></a>`fig:app-ae-capacity` | contrastive_readout_swap | paper/figures/availability_expression/rpt_capacity.pdf |
| <a id="fig:app-ae-grid"></a>`fig:app-ae-grid` | contrastive_readout_swap | paper/figures/availability_expression/rpt_triple_grid.pdf |
| <a id="fig:app-aligned-readout-swap"></a>`fig:app-aligned-readout-swap` | temporal_localization_patching | results/experiments/causal/temporal_localization_patching/aligned_swap_grid/summary.csv |
| <a id="fig:app-concat-pca"></a>`fig:app-concat-pca` | crosscoder_main | paper/figures/dense_baselines/concatenated_trajectory_pca.pdf |
| <a id="fig:app-contrastive-readout-lag"></a>`fig:app-contrastive-readout-lag` | contrastive_readout_swap | paper/figures/readout_coordination/readout_lag_narrative_69b.pdf |
| <a id="fig:app-dense-reorg-pythia"></a>`fig:app-dense-reorg-pythia` | dense_readout_diagnostics | paper/figures/dense_readout/dense_reorganization_timing.pdf |
| <a id="fig:app-fidelity-curves"></a>`fig:app-fidelity-curves` | crosscoder_main | paper/figures/readout_functional_fidelity/fidelity_curves.pdf |
| <a id="fig:app-heldout-checkpoints-1b"></a>`fig:app-heldout-checkpoints-1b` | heldout_checkpoints | paper/figures/heldout_checkpoints/heldout_reconstruction_1b.pdf |
| <a id="fig:app-instrument-pareto"></a>`fig:app-instrument-pareto` | crosscoder_main | paper/figures/instrument_pareto/instrument_pareto.pdf |
| <a id="fig:app-lambda-sweep"></a>`fig:app-lambda-sweep` | crosscoder_main | paper/figures/instrument_pareto/sparsity_penalty_sensitivity.pdf |
| <a id="fig:app-lifecycle-profile-composition"></a>`fig:app-lifecycle-profile-composition` | feature_lifecycle_trajectories | paper/figures/feature_lifecycle/selected_lifecycle_profile_composition.pdf |
| <a id="fig:app-mean-direction-preprocessing"></a>`fig:app-mean-direction-preprocessing` | dense_readout_diagnostics | paper/figures/dense_readout/mean_direction_preprocessing.pdf |
| <a id="fig:app-mean-direction-spectral-gap"></a>`fig:app-mean-direction-spectral-gap` | dense_readout_diagnostics | paper/figures/dense_readout/mean_direction_spectral_gap_by_model.pdf |
| <a id="fig:app-olmo-window-population"></a>`fig:app-olmo-window-population` | olmo_matched_checkpoint_window | paper/figures/feature_lifecycle/olmo_matched_checkpoint_window_population_lifecycle_diagnostics.pdf |
| <a id="fig:app-pca-static"></a>`fig:app-pca-static` | crosscoder_main | paper/figures/dense_baselines/static_pca_baseline.pdf |
| <a id="fig:app-readout-swap-english-aggregate"></a>`fig:app-readout-swap-english-aggregate` | language_stratified_readout_swap | paper/figures/readout_coordination/english_aggregate_swap.pdf |
| <a id="fig:app-readout-swap-languages"></a>`fig:app-readout-swap-languages` | language_stratified_readout_swap | paper/figures/readout_coordination/language_stratified_swap.pdf |
| <a id="fig:app-readout-swap-nll-kl-grid"></a>`fig:app-readout-swap-nll-kl-grid` | temporal_localization_patching | paper/figures/readout_coordination/readout_swap_nll_kl_grid.png |
| <a id="fig:app-recipe-control-basin"></a>`fig:app-recipe-control-basin` | pretraining_recipe_control | paper/figures/recipe_control/F2_trajectory_basin.pdf |
| <a id="fig:app-recipe-control-explag"></a>`fig:app-recipe-control-explag` | pretraining_recipe_control | paper/figures/recipe_control/F6_expression_lag.pdf |
| <a id="fig:app-recipe-control-geometry"></a>`fig:app-recipe-control-geometry` | pretraining_recipe_control | paper/figures/recipe_control/F1_geometry_trajectories.pdf |
| <a id="fig:app-recipe-control-reorg"></a>`fig:app-recipe-control-reorg` | pretraining_recipe_control | paper/figures/recipe_control/reorg_window.pdf |
| <a id="fig:app-recipe-control-trajectories"></a>`fig:app-recipe-control-trajectories` | pretraining_recipe_control | paper/figures/recipe_control/normalized_trajectories.pdf |
| <a id="fig:app-reorganization-window-metrics"></a>`fig:app-reorganization-window-metrics` | crosscoder_olmo, feature_lifecycle_trajectories | paper/figures/feature_lifecycle/reorganization_window_metrics_selected.pdf |
| <a id="fig:app-selected-population-lifecycle-diagnostics"></a>`fig:app-selected-population-lifecycle-diagnostics` | feature_lifecycle_trajectories | paper/figures/feature_lifecycle/selected_population_lifecycle_diagnostics.pdf |
| <a id="fig:app-snapshot-fidelity"></a>`fig:app-snapshot-fidelity` | persnap_sae_baseline | paper/figures/persnap_sae/snapshot_reconstruction_fidelity.pdf |
| <a id="fig:app-sparse-feature-causal-curves"></a>`fig:app-sparse-feature-causal-curves` | sparse_feature_causal_tests | paper/figures/sparse_feature_causal_tests/control_corrected_ablation_curves.pdf |
| <a id="fig:app-sparse-feature-causal-specificity-k32"></a>`fig:app-sparse-feature-causal-specificity-k32` | sparse_feature_causal_tests | paper/figures/sparse_feature_causal_tests/specificity_heatmaps_k32.pdf |
| <a id="fig:app-wordnet-supersense-probes-160m-inventory"></a>`fig:app-wordnet-supersense-probes-160m-inventory` | concept_evolution_validation | paper/figures/vocabulary_family_probes/wordnet_supersense_probe_heatmap_160m.png |
| <a id="fig:app-wordnet-supersense-probes-1b"></a>`fig:app-wordnet-supersense-probes-1b` | concept_evolution_validation | paper/figures/vocabulary_family_probes/wordnet_supersense_probe_heatmap_1b.pdf |
| <a id="fig:main-multimodel-validation"></a>`fig:main-multimodel-validation` | crosscoder_main, crosscoder_olmo | paper/figures/multimodel_validation/instrument_validation_summary.pdf |
| <a id="fig:main-recipe-dose-response"></a>`fig:main-recipe-dose-response` | pretraining_recipe_control | paper/figures/recipe_control/timing_dose_response.pdf |
| <a id="fig:main-selected-normalized-trajectories"></a>`fig:main-selected-normalized-trajectories` | crosscoder_olmo, feature_lifecycle_trajectories | paper/figures/feature_lifecycle/selected_decoder_norm_trajectories.pdf |
| <a id="fig:main-sva-availability-expression"></a>`fig:main-sva-availability-expression` | contrastive_readout_swap | paper/figures/readout_coordination/sva_availability_expression_69b.pdf |
| <a id="fig:main-wordnet-matched-controls"></a>`fig:main-wordnet-matched-controls` | concept_evolution_validation | paper/figures/vocabulary_family_probes/wordnet_matched_controls_160m.pdf |
| <a id="fig:method-cross-snapshot-schematic"></a>`fig:method-cross-snapshot-schematic` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="fig:overview"></a>`fig:overview` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:app-ae-binary"></a>`tab:app-ae-binary` | contrastive_readout_swap | (produced; see experiments.yaml / tables below) |
| <a id="tab:app-ae-early"></a>`tab:app-ae-early` | contrastive_readout_swap | (produced; see experiments.yaml / tables below) |
| <a id="tab:app-ae-openvocab"></a>`tab:app-ae-openvocab` | contrastive_readout_swap | (produced; see experiments.yaml / tables below) |
| <a id="tab:app-claim-tests"></a>`tab:app-claim-tests` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:app-contrastive-localisation-ledger"></a>`tab:app-contrastive-localisation-ledger` | contrastive_task_feature_rescue | results/experiments/causal/contrastive_task_feature_rescue/run0_pythia1b_s143000_h1000_pos/shards/sva__h1000__s143000.json |
| <a id="tab:app-design-rationale"></a>`tab:app-design-rationale` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:app-lifecycle-stability"></a>`tab:app-lifecycle-stability` | feature_lifecycle_trajectories | (produced; see experiments.yaml / tables below) |
| <a id="tab:app-readout-swap-languages"></a>`tab:app-readout-swap-languages` | language_stratified_readout_swap | paper/figures/readout_coordination/language_swap_table.tex |
| <a id="tab:app-recipe-control-summary"></a>`tab:app-recipe-control-summary` | pretraining_recipe_control | results/experiments/ablations/pretraining_recipe_control/lifecycle_seed0/lifecycle_summary.csv |
| <a id="tab:app-reorganization-window-stats"></a>`tab:app-reorganization-window-stats` | feature_lifecycle_trajectories | (produced; see experiments.yaml / tables below) |
| <a id="tab:app-task-construction"></a>`tab:app-task-construction` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:app-terminology-glossary"></a>`tab:app-terminology-glossary` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:app-wordnet-supersense-1b"></a>`tab:app-wordnet-supersense-1b` | concept_evolution_validation | experiments/probes/concept_evolution_validation/derived/wordnet_supersense_1b/wordnet_supersense_probe_pos_summary_1b.csv |
| <a id="tab:checkpoint-suite-comparison"></a>`tab:checkpoint-suite-comparison` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:compute-resources"></a>`tab:compute-resources` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:evaluation-controls"></a>`tab:evaluation-controls` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:full-inventory"></a>`tab:full-inventory` | crosscoder_main | experiments/crosscoders/crosscoder_main/derived/appendix_validation/full_inventory.csv |
| <a id="tab:lifecycle-profile-rules"></a>`tab:lifecycle-profile-rules` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:method-comparison"></a>`tab:method-comparison` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:repro-dictionary-hparams"></a>`tab:repro-dictionary-hparams` | crosscoder_main | (produced; see experiments.yaml / tables below) |
| <a id="tab:task-protocol"></a>`tab:task-protocol` | (paper LaTeX tree only) | (paper LaTeX tree only) |
<!-- END GENERATED FIGURE INDEX -->

## Pipeline

1. **Extract** snapshots from the public Hugging Face checkpoints. The trainer
   extracts Pythia `W_U` snapshots on a cache miss
   (`readout.crosscoder.wu_adapter.load_wu_snapshot`); OLMo-2 is extracted with
   `experiments/crosscoders/crosscoder_olmo/scripts/extract_wu_olmo.py`, and
   `scripts/extract/` holds the other extraction helpers (`W_E`, final hidden
   states, firing rates, the OLMo evaluation corpus). Extracted snapshots are also
   released at `hf.co/datasets/hematteo/wu-crosscoder-snapshots`.
2. **Train** a trajectory crosscoder (`scripts/train/train_crosscoder.py`). The
   paper's dictionaries are released at
   `hf.co/hematteo/parameter-trajectory-crosscoders`.
3. **Analyze**: each experiment's scripts compute and save the metrics behind its
   figures (CSV, JSON, or `.pt`) under the gitignored `results/` and `figures/`
   trees. `paper_figures/data/README.md` names the script that regenerates each
   file `make figures` reads.

## Figure to experiment to metrics

Ordered as in the paper.

### Main text

| Paper item | Experiment | Metrics / scripts | Needs GPU | Needs SSD data |
|---|---|---|---|---|
| Figure 1 (`fig:overview`) | paper only; margins from `contrastive_readout_swap` | TikZ schematic; its two margins are per-example outputs of `run_swap_grid.py` on Pythia-1B (hidden state from step 1000, readouts 1000 and 143k, no alignment) | yes | snapshots + hidden states |
| Table 1 (`tab:task-protocol`) | paper only | hand-typeset; the tasks are built by `build_task_datasets.py` and `readout.probes.contrastive_tasks` | no | none |
| Figure 2 (`fig:main-selected-normalized-trajectories`) | `feature_lifecycle_trajectories`, `crosscoder_olmo` | `compute_normalized_trajectories.py` saves the decoder norm trajectories | no | crosscoders |
| Figure 3 (`fig:main-wordnet-matched-controls`) | `concept_evolution_validation` | `run_wordnet_matched_controls.py` | no | `W_U` snapshots |
| Figure 4 (`fig:main-sva-availability-expression`) | `contrastive_readout_swap` | left panel: `run_swap_grid.py` (Pythia-6.9B, 32 checkpoints, `--wu-round bf16`) and `run_controlled_hidden_probes.py`; right panel: `build_task_datasets.py --task-set availability`, `extract_hidden_dense.py`, `run_availability_probes.py --group-split` | yes | snapshots + hidden states |
| Section 4.4 (sparse feature ablation) | `contrastive_task_feature_rescue` | `run_feature_attribution.py` (Pythia-1B hidden states from step 1000, reconstructed step-143k readout) | yes | crosscoders + hidden states |
| Figure 5 (`fig:main-recipe-dose-response`) | `pretraining_recipe_control` | `train_control.py` (validation loss in each run's `metrics.csv`); `train_control_crosscoder.py` and `analyze_lifecycle.py` for seeds 0 to 2, pooled by `aggregate_seed_replication.py` into `lifecycle_multiseed_table.csv` | for training and fits | trained 31M checkpoints (released) |
| Table 2 (`tab:method-comparison`) | paper only | qualitative comparison | | |

### Appendices A to C (terminology, method details, reproducibility)

| Paper item | Experiment | Metrics / scripts | Needs GPU | Needs SSD data |
|---|---|---|---|---|
| Table C.2 (`tab:repro-dictionary-hparams`) | `crosscoder_main` | settings of record in `configs/runs/` | | |
| Figure B.1, Tables A.1 to A.3, B.1, C.1, C.3 | paper only | schematic and hand-typeset tables | | |

### Appendix D (instrument validation)

| Paper item | Experiment | Metrics / scripts | Needs GPU | Needs SSD data |
|---|---|---|---|---|
| Figure D.1, Table D.1 (`fig:main-multimodel-validation`, `tab:full-inventory`) | `crosscoder_main`, `crosscoder_olmo` | `appendix_validation/recompute_quality.py` (Pythia-160M and 1B rows), `appendix_validation/eval_persnap_fidelity.py` and `main_1b/eval_persnap_1b.py` (per-checkpoint curves); see Known deviations for the Pythia-6.9B and OLMo-2-7B rows | for retrain | trained crosscoders |
| Figures D.2, D.3, D.9 (`fig:app-mean-direction-preprocessing`, `fig:app-mean-direction-spectral-gap`, `fig:app-dense-reorg-pythia`) | `dense_readout_diagnostics` | `build_mean_direction_audit.py`, `build_dense_reorganization_timing.py` | no | `W_U` snapshots |
| Figure D.4 (`fig:app-fidelity-curves`) and the strata audit | `crosscoder_main` | `appendix_validation/readout_functional_fidelity.py` (held-out matrix EV, logit $R^2$, KL, $\Delta$NLL, top-1/5 per checkpoint) and `readout_functional_fidelity_strata.py` (the token stratum check in the Appendix D text); the 1B and 6.9B runs need about 25 GB RAM | no | crosscoders + snapshots |
| Figures D.5, D.11 (`fig:app-instrument-pareto`, `fig:app-lambda-sweep`) | `crosscoder_main` | `appendix_validation/recompute_quality.py` (`full_inventory.csv`; `--filter lambda-sweep` for `lambda_sweep.csv`); the architecture variants are trained by `scripts/train/train_arch_sweep.py` | for retrain | trained crosscoders |
| Figure D.6 (`fig:app-snapshot-fidelity`) | `persnap_sae_baseline`, `crosscoder_main` | `train_persnap_saes.py` (the per-checkpoint SAEs), then `appendix_validation/eval_persnap_fidelity.py` | for retrain | per-checkpoint SAEs |
| Figures D.7, D.8 (`fig:app-pca-static`, `fig:app-concat-pca`) | `crosscoder_main` | `appendix_validation/baselines/eval_pca_static.py`, `eval_concat_pca.py` | no | `W_U` snapshots |
| Figure D.10 (`fig:app-heldout-checkpoints-1b`) | `heldout_checkpoints` | `eval_heldout.py` | for retrain | snapshots + crosscoders |

### Appendix E (lifecycle and reorganization window)

| Paper item | Experiment | Metrics / scripts | Needs GPU | Needs SSD data |
|---|---|---|---|---|
| Figure E.1 (`fig:app-lifecycle-profile-composition`) | `feature_lifecycle_trajectories` | `compute_lifecycle_profile_composition.py` (refined ruleset, `*_refined_*` files) | no | crosscoders |
| Table E.1 (`tab:lifecycle-profile-rules`) | paper only | rules implemented in `readout.dynamics.lifecycle.classify_profiles_refined` | | |
| Figure E.2 (`fig:app-selected-population-lifecycle-diagnostics`) and the adjacent window-peak and $\rho_f$ numbers | `feature_lifecycle_trajectories` | `build_population_diagnostics.py` | no | decoder-norm caches |
| Table E.2, Figure E.4 (`tab:app-reorganization-window-stats`, `fig:app-reorganization-window-metrics`) | `feature_lifecycle_trajectories`, `crosscoder_olmo` | `find_reorganization_steps.py` (windows, circular-shift $p$ values, bootstrap resample counts) | no | crosscoders + aggregates |
| Figure E.3 (`fig:app-olmo-window-population`) and the late-start profile shares | `olmo_matched_checkpoint_window` | `train_late_start.py` (optional; the dictionary is released), `compute_late_start_vs_olmo.py` | for retrain | snapshots + crosscoders |
| Table E.3 (`tab:app-lifecycle-stability`) | `feature_lifecycle_trajectories` | `lifecycle_stability.py --stage all` (21 dictionary fits; about 25 min and 15 GB RAM on CPU) | no | released crosscoders + aggregates |

### Appendix F (vocabulary family probes)

| Paper item | Experiment | Metrics / scripts | Needs GPU | Needs SSD data |
|---|---|---|---|---|
| Figures F.1, F.2, Table F.1 (`fig:app-wordnet-supersense-probes-160m-inventory`, `-1b`, `tab:app-wordnet-supersense-1b`) | `concept_evolution_validation` | `run_wordnet_supersense_probe.py` | no | `W_U` snapshots |
| Lemma-grouped splits (Appendix F text) | `concept_evolution_validation` | `run_wordnet_supersense_probe.py --split lemma` (splits grouped by lemma; also writes the split-leakage statistics) | no | `W_U` snapshots |

These probe scripts read WordNet 3.0 through NLTK.
Before the first run, download the corpus once:
`python -c "import nltk; nltk.download('wordnet')"` (one-time, into `~/nltk_data/`).

### Appendix G (availability and expression)

| Paper item | Experiment | Metrics / scripts | Needs GPU | Needs SSD data |
|---|---|---|---|---|
| Figure G.1 (`fig:app-contrastive-readout-lag`) | `contrastive_readout_swap` | the Figure 4 left-panel run (`run_swap_grid.py`, `run_controlled_hidden_probes.py`) | yes | snapshots + hidden states |
| Figures G.2, G.3, Tables G.2-G.4 (`fig:app-ae-capacity`, `fig:app-ae-grid`, `tab:app-ae-binary`, `tab:app-ae-openvocab`, `tab:app-ae-early`) | `contrastive_readout_swap` | `build_task_datasets.py --task-set availability`, `extract_hidden_dense.py`, `run_availability_probes.py` for Pythia-6.9B and the two Pythia-1B runs (commands in the experiment README) | yes (one 48 GB GPU at fp32 for 6.9B) | snapshots + hidden states; BLiMP from the Hub; function-vector word lists in `$FV_DATA_DIR` |
| Table G.1 (`tab:app-task-construction`) | paper only | hand-typeset templates and word lists; implemented in `readout.probes.availability_tasks` and its sibling modules | | |

`run_availability_probes.py` builds its grouped cross-validation folds with the
tie order of the x86-64 NumPy sort used for the paper, so the paper's folds, and
its fold-dependent intervals, come out the same on any platform. (Sklearn's
`GroupKFold` alone would give different folds on arm64, e.g. Apple Silicon.)

### Appendix H (readout swaps and localization)

| Paper item | Experiment | Metrics / scripts | Needs GPU | Needs SSD data |
|---|---|---|---|---|
| Figures H.1, H.4 (`fig:app-readout-swap-nll-kl-grid`, `fig:app-aligned-readout-swap`) | `temporal_localization_patching` | `temporal_patch_grid.py`, `run_aligned_swap_grid.py`; the Pythia eval corpus is rebuilt by `build_eval_corpus_pythia.py` (also released) | yes | snapshots + hidden states |
| Table H.1, Figures H.2, H.3 (`tab:app-readout-swap-languages`, `fig:app-readout-swap-english-aggregate`, `fig:app-readout-swap-languages`) | `language_stratified_readout_swap` | `run_language_stratified_swap.py` (per-language NLL, paired block bootstrap, argmin probabilities) | yes (about 5.5 h on one A40) | snapshots + hidden states + released eval corpus |
| Table H.2 (`tab:app-contrastive-localisation-ledger`) | `contrastive_task_feature_rescue` | `run_feature_attribution.py` (ablate, preserve, and sign columns), then `run_converse_intervention.py` on the same held-out half (hidden-projection columns) | yes | crosscoders + hidden states |
| Figures H.5, H.6 (`fig:app-sparse-feature-causal-curves`, `-specificity-k32`) | `sparse_feature_causal_tests` | eight `run_1b_pilot.py` runs, two `run_specificity_from_pilots.py` passes, then `aggregate_causal_tests.py` | yes | crosscoders + aggregates |

### Appendix I (readout learning rate control)

| Paper item | Experiment | Metrics / scripts | Needs GPU | Needs SSD data |
|---|---|---|---|---|
| Figure I.1, Table I.1, Figure I.2 (`fig:app-recipe-control-trajectories`, `tab:app-recipe-control-summary`, `fig:app-recipe-control-reorg`) | `pretraining_recipe_control` | `train_control_crosscoder.py`, `analyze_lifecycle.py` (seed 0 for tables and per-feature figures, seeds 0-2 for timing), `aggregate_seed_replication.py` | for the fits | trained 31M checkpoints (released) |
| Figure I.3 (`fig:app-recipe-control-geometry`) | `pretraining_recipe_control` | `analyze_readout_geometry.py` | no | trained 31M checkpoints |
| Figures I.4, I.5 (`fig:app-recipe-control-basin`, `fig:app-recipe-control-explag`) | `pretraining_recipe_control` | `fetch_heldout_slice.py`, `run_trajectory_swap.py`, `run_expression_lag.py` | yes (CPU works at reduced `--eval-tokens`) | trained 31M checkpoints + held-out slice |

## Known deviations

- **Section 4.4 and Table H.2.** The release seeds the localization split and the
  matched controls separately for each task, and builds the simple SVA items in
  interleaved order. The paper's run drew both from one sequential generator, so
  the release scores a different held-out half and its values differ from the
  paper's.
- **Figure 4, left panel.** The native and best swept series have no surviving run
  output. `paper_figures/` ships them as values transcribed from the original
  figure; the probe series is run output.
- **Figure D.1 and Table D.1.** The Pythia-6.9B and OLMo-2-7B rows and their
  per-checkpoint curves come from evaluation files that no release script
  regenerates; `recompute_quality.py` recomputes their aggregate EV and L0.
- **Figure H.4.** The paper's aligned swap grid scores the first 32 sequences
  (16,352 targets). Build 32-sequence caches with
  `scripts/extract/build_hln_cache.py --max-seqs 32` before running
  `run_aligned_swap_grid.py`; without them the script scores the full corpus.

## Notes

- `make audit` (`scripts/audit/check_layout.py`) checks that every experiment
  directory has a manifest entry and that every path the manifest names exists.
  The paper source is not shipped, so the label coverage check against it is
  skipped.
- Items marked "paper only" are listed under `paper_only_labels` in
  `experiments.yaml`: the schematics (Figure 1, Figure B.1) and the hand-typeset
  tables (Tables 1, 2, A.1 to A.3, B.1, C.1, C.3, E.1, G.1). No experiment
  produces them, but the numbers they quote come from the experiments above.
- The temporal patch library (`readout.dynamics.temporal_patch`) runs from public
  assets only; its evaluation corpus resolves to the released copy under
  `$UM_SSD_ROOT/hf_release/.../evaluation/eval-corpus/eval_tokens.pt`.
