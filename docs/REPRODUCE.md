# Reproducing the paper

Three levels, from cheapest to most expensive; run every command from the repository root.

1. **Redraw the figures**: `make figures` redraws 14 data figures from the 4.9 MB in `paper_figures/data/` (CPU, about 30 s).
2. **Recompute from the released artifacts**: the scripts in `reproduce/<section>/` read the released dictionaries
   and checkpoints under `UM_SSD_ROOT` and write metrics (CSV, JSON, `.pt`, no figures) under the gitignored
   `results/<section>/`. The larger models need a GPU.
3. **Retrain** the dictionaries ([Training the dictionaries](#training-the-dictionaries)) or the 31M runs
   ([Recipe control](#recipe-control-section-45-appendix-h)).

## Setup

```bash
make install                          # uv sync --extra dev --extra figures (Python 3.11 or 3.12)
make test                             # CPU only, no data
export UM_SSD_ROOT=/path/to/storage   # default <repo>/local_snapshots (src/readout/core/paths.py)
uv run python -c "import nltk; nltk.download('wordnet')"   # WordNet 3.0, into ~/nltk_data or $NLTK_DATA
hf download hematteo/parameter-trajectory-crosscoders \
    --revision fb7ee860b9257f125ddbac7ff3c793b35fdcce8d \
    --local-dir "$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders"
hf download hematteo/readout-recipe-control --local-dir "${UM_SSD_ROOT}/runs"
```

| Hugging Face repository | Contents |
|---|---|
| `hematteo/parameter-trajectory-crosscoders` | dictionaries (MIT; `.safetensors` + `.config.json`), aggregates (`derived/aggregates/`), activation rates (`derived/rates/`), attribution artifacts, evaluation corpus; `index.json` lists every file |
| `hematteo/readout-recipe-control` | the 31M checkpoints: the four compared conditions and `warmup_long` |
| `datasets/hematteo/wu-crosscoder-snapshots` | pre-extracted `W_U` snapshots (optional; the trainer extracts them on a cache miss) |

On Linux, `pyproject.toml` pins torch 2.7.1+cu118, the version of the paper's runs. The scripts read aggregates
and rates from `$UM_SSD_ROOT/derived/`, so make the release's `derived/` directory available there. The
evaluation corpus is `$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt`
(Wikipedia, CC BY-SA; 173,229 targets under the Pythia tokenizer); every `--eval-tokens` defaults to it except
that of `scripts/extract/build_hln_cache.py`, which requires it.

```text
$UM_SSD_ROOT/
  snapshots/<model>/<slug>_step<N>_wu.pt          W_U snapshots; <slug> = HF id with / replaced by _
  hf_release/parameter-trajectory-crosscoders/    released dictionaries and evaluation corpus
  derived/aggregates/, derived/rates/             aggregates and activation rates
  derived/readout_edit_timing*/hLN_step<t>.pt     hidden-state caches of the fidelity check (App. C)
  readout_edit_timing*/hLN_step<t>.pt             hidden-state caches of the readout swap grids (App. G)
  pile_slice/, runs/<condition>/                  recipe-control token slices and 31M checkpoints
```

`<model>` is `pythia-160m`, `pythia-1b`, `pythia-6.9b`, or `OLMo-2-1124-7B`. Optional overrides: `WU_SNAP_DIR`,
`WU_CACHE_DIR` (the trainer's snapshot cache), `FV_DATA_DIR` (function-vector lists, default
`data/function_vectors/`), `HF_HOME`, `READOUT_COMMIT` (provenance without `.git`). The checkpoint schedules
are `DEFAULT_STEPS_32` (Pythia) and `OLMO_STEPS_32` (OLMo-2-7B, no step 0) in `src/readout/core/model_specs.py`.

| Asset | Source | License | Needed for |
|---|---|---|---|
| Pythia 160M, 1B, 6.9B | `EleutherAI/pythia-{160m,1b,6.9b}` | Apache-2.0 | snapshots, hidden states |
| OLMo-2-1124-7B | `allenai/OLMo-2-1124-7B` | Apache-2.0 | the OLMo-2-7B dictionary |
| WordNet 3.0 | Princeton WordNet, through NLTK | WordNet 3.0 license | Figure 3, Appendix E |
| Wikipedia slices | Wikimedia 2023-11-01 dumps | CC BY-SA 4.0 / GFDL | the evaluation corpus |
| BLiMP | `nyu-mll/blimp` | see its dataset card | availability suite (Figure 4, Appendix F) |
| Function-vector word lists | `github.com/ericwtodd/function_vectors`, `dataset_files/abstractive/` | MIT | availability suite |
| GPT-NeoX-20B tokenizer, Pile | `EleutherAI/gpt-neox-20b`, `EleutherAI/pile-standard-pythia-preshuffled` (prefix) | Apache-2.0, MIT | retraining the 31M runs |

| Workload | Hardware |
|---|---|
| figures, tests, lifecycle, lexical probes, dense diagnostics | CPU (`lifecycle_stability.py`: about 25 min, 15 GB RAM) |
| fidelity check | Pythia-160M about 15 min on a laptop CPU; Pythia-1B about 25 GB RAM |
| Pythia-6.9B availability extraction | one 48 GB GPU at fp32 |
| per-language swap grid | 5.5 h on one A40; 1.8 GB of Pythia-1B hidden-state cache per checkpoint |
| one 31M pretraining run | one A40, about 21 h |
| OLMo-2-7B dictionary | extraction: 14 GB download per checkpoint, 50 GB of snapshots; training: four A100/H100-class GPUs (40 to 80 GB), 4 to 8 h |

## Redrawing the figures

```bash
make figures   # uv run --extra figures python paper_figures/make_all.py; writes paper_figures/out/
```

Each script also runs alone and needs only matplotlib, numpy, and pandas.
[`paper_figures/data/README.md`](../paper_figures/data/README.md) lists each data file and the `reproduce/`
script that regenerates it.

| Figure | Label | Script in `paper_figures/` |
|---|---|---|
| 2 | `fig:main-selected-normalized-trajectories` | `lifecycle_trajectories.py` |
| 3 | `fig:main-wordnet-matched-controls` | `wordnet_controls.py` |
| 4 | `fig:main-sva-availability-expression` | `sva_availability.py` |
| 5 | `fig:main-recipe-dose-response` | `recipe_dose_response.py` |
| C.1 | `fig:main-multimodel-validation` | `instrument_validation.py` |
| C.3 | `fig:app-fidelity-curves` | `fidelity_curves.py` |
| C.6 | `fig:app-dense-reorg-pythia` | `dense_reorganization_timing.py` |
| D.1 | `fig:app-lifecycle-profile-composition` | `lifecycle_profile_composition.py` |
| G.2 | `fig:app-readout-swap-languages` | `readout_swap_languages.py` (`readout_swap_languages.pdf`) |
| G.3 | `fig:app-aligned-readout-swap` | `aligned_swap_grid.py` |
| H.1 | `fig:app-recipe-control-reorg` | `recipe_reorganization_window.py` |

With matplotlib 3.10 (the `figures` extra), every page matches the paper's PDF pixel for pixel at 60 and 150 dpi,
except Figure 2, whose float16 trajectories shift rasterized lines by a few gray levels.

## Lifecycle (Section 4.1, Appendix D)

Figure 2, the reorganization windows of Section 4.1, Figures D.1 and D.2, Tables D.2 and D.3, and the timing
statistics of Appendix D.2. CPU only.

```bash
uv run python reproduce/lifecycle/compute_normalized_trajectories.py      # 1. Figure 2; caches in results/lifecycle/cache/
uv run python reproduce/lifecycle/compute_lifecycle_profile_composition.py # 2. Figure D.1
uv run python reproduce/lifecycle/build_population_diagnostics.py          # 3. Appendix D.2 statistics
uv run python reproduce/lifecycle/find_reorganization_steps.py             # 4. windows, nulls, bootstrap
uv run python reproduce/lifecycle/lifecycle_stability.py --stage all \
    --out-dir results/lifecycle/lifecycle_stability            # 5. Table D.3; its validate stage needs 2 and 4
```

| Paper item | Output, under `results/lifecycle/` |
|---|---|
| Figure 2 (`fig:main-selected-normalized-trajectories`) | `selected_decoder_norm_trajectories{,_full}.{csv,pt}`, `normalized_trajectories.csv` |
| Figure D.1 (`fig:app-lifecycle-profile-composition`) | `selected_lifecycle_profile_composition_refined_{summary,features}.csv` |
| Appendix D.2 statistics (peaks inside the window, decay of early-peaking features, decoder norm mass) | `selected_population_lifecycle_diagnostics_{peak_counts,norm_mass,window_peaks,early_peak_decay}.csv` |
| Section 4.1 windows, Table D.2, Figure D.2 (`tab:app-reorganization-window-stats`, `fig:app-reorganization-window-metrics`) | `reorganization_window_{metrics,summary,bootstrap_counts}_selected.csv` |
| Table D.3 (`tab:app-lifecycle-stability`) | `lifecycle_stability/*.csv` (selected Pythia-6.9B: row `pythia69b_d32768` of `validation.csv`) |

Step 4 reads curated Pythia-6.9B and OLMo-2-7B rates that are not shipped ([Known deviations](#known-deviations)).

**Late-start control** (Appendix D.3, `sec:app-olmo-checkpoint-window-control`):
a Pythia-1B dictionary on a 32-checkpoint schedule starting at step 256, as OLMo-2-7B's starts at 150.

```bash
# 1. Train, or use the released pythia-1b/W_U/cross-snapshot-32-matched-window/d24576/seed0.safetensors
#    (defaults match configs/runs/pythia-1b_wu_d24576_seed0_olmo-matched-window.yaml; no CUDA: try --device mps)
uv run python reproduce/lifecycle/train_late_start.py --d-sae 24576 --n-epochs 300 --device cuda
# 2. Save its firing rates as results/lifecycle/olmo_window/pythia1b_wu_late_start32_dsae24576_seed0_rates.pt
#    (dict with rates (K, D), steps, optional norms; see Known deviations). 3. Run lifecycle step 1, then:
uv run python reproduce/lifecycle/compute_late_start_vs_olmo.py
```

Outputs, under `results/lifecycle/olmo_window/`: `late_start_profile_fractions.csv` (profile shares),
`comparison_summary.csv` (activation-rate shares of the late-start and OLMo-2-7B dictionaries), and
`comparison_feature_scores.csv` (per-feature peak steps).

## Lexical (Section 4.2, Appendix E)

Figure 3, Figure E.1 (`fig:app-wordnet-supersense-probes-160m-inventory`), the Pythia-1B replication of
Appendix E.5, and the lemma-grouped split of Appendix E.6 (`sec:app-wordnet-lemma-split`). CPU is enough
(`--device cpu`).

```bash
P=reproduce/lexical/run_wordnet_supersense_probe.py
uv run python $P --model pythia-160m        # 35 supersense categories, one-vs-rest, every checkpoint
uv run python $P --model pythia-1b
uv run python reproduce/lexical/run_wordnet_matched_controls.py   # metadata baseline and matched nulls, Pythia-160M
uv run python $P --model pythia-160m --split lemma                # lemma-grouped split
uv run python $P --model pythia-1b --split lemma
for s in 1 2 3 4; do for sp in row lemma; do                      # split noise: seeds 1 to 4, six checkpoints
  uv run python $P --model pythia-160m --split $sp --seed $s --steps 0,256,512,1000,2000,143000
done; done
```

| Paper item | Output, under `results/lexical/` |
|---|---|
| Figure 3 (`fig:main-wordnet-matched-controls`) | `wordnet_matched_controls_160m/wordnet_matched_control_summary_160m.csv` |
| Figure E.1, onset and largest-gain counts | `wordnet_supersense_160m/wordnet_supersense_probe_trajectory_160m.csv` |
| Appendix E.5, Pythia-1B onsets | `wordnet_supersense_1b/wordnet_supersense_probe_{trajectory,pos_summary}_1b.csv` |
| Appendix E.6, lemma-split comparison | `final_acc`, `emergence_step`, `max_gain_pair` of `wordnet_supersense_probe_summary_<suffix>.csv`, `_lemma` vs row directory |

Directories gain `_lemma` and `_seed<N>` (non-zero seed). The leakage count is the sum of
`n_test_pos_with_train_sibling` over the sum of `n_test_positive` in the row split's
`wordnet_supersense_split_stats_<suffix>.csv`. The matched nulls were not rerun under the lemma split.

## Availability (Section 4.3, Appendix F, Figure 1)

Figure 4, Figures F.1 and F.2, Tables F.2 to F.4, the terminal-readout accuracies of Appendix G.4
(`sec:app-terminal-swap-alignment`), and the margins of Figure 1. Step 1 also feeds Section 4.4. Needs a GPU.

```bash
# 1. Pythia-1B localization inventories and swap grid (add --save-per-example for the Figure 1 margins)
uv run python reproduce/availability/build_task_datasets.py \
    --model EleutherAI/pythia-1b --out-dir results/availability/datasets/pythia-1b
uv run python reproduce/availability/run_swap_grid.py --model pythia-1b \
    --datasets-dir results/availability/datasets/pythia-1b --out-dir results/availability/run0_pythia1b \
    --alignments none scale row_norm procrustes

# 2. Pythia-6.9B 32-checkpoint SVA and IOI trajectories (bf16 forward pass, bf16-rounded W_U)
RUN=results/availability/run_pythia-6.9b_probe_trajectory
DS=results/availability/datasets/pythia-6.9b_trajectory
STEPS32="0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 5000 6000 7000 8000 9000 14000 21000 27000 34000 \
  47000 61000 75000 89000 102000 116000 130000 143000"
uv run python reproduce/availability/build_task_datasets.py \
    --model EleutherAI/pythia-6.9b --families sva ioi_role_balanced --n-max-per-family 2000 \
    --rng-seed 0 --no-norm-match --out-dir $DS
uv run python reproduce/availability/run_swap_grid.py \
    --model pythia-6.9b --datasets-dir $DS --families sva ioi_role_balanced \
    --h-steps $STEPS32 --alignments none --device cuda --dtype bf16 --wu-round bf16 --out-dir $RUN
uv run python reproduce/availability/run_controlled_hidden_probes.py \
    --hidden-dir $RUN/hidden --datasets-dir $DS --families sva ioi_role_balanced \
    --stem probe_trajectory_69b --out-dir $RUN/controlled

# 3. Pythia-6.9B availability suite: 20 families, 24 checkpoints, no row-norm matching
FAMS="sva sva_across_pp sva_subject_rc sva_object_rc ioi_role_balanced copy_control fixed_token_control \
  blimp_det_noun_agreement blimp_anaphor_agreement blimp_npi fv_antonym fv_country_capital fv_en_fr \
  fv_present_past induction numeric_gt_v2 numeric_lt_v2 numeric_tf_v2 relational_facts relational_facts_balanced"
STEPS="0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 5000 6000 8000 10000 16000 32000 64000 128000 143000"
OUT=results/availability/availability
uv run python reproduce/availability/build_task_datasets.py \
    --task-set availability --model EleutherAI/pythia-6.9b --families $FAMS \
    --n-max-per-family 2000 --rng-seed 0 --no-norm-match --out-dir $OUT/datasets/pythia-6.9b
uv run python reproduce/availability/extract_hidden_dense.py \
    --hf-model EleutherAI/pythia-6.9b --datasets-dir $OUT/datasets/pythia-6.9b --families $FAMS \
    --h-steps $STEPS --out-dir $OUT/run_pythia-6.9b/hidden --device cuda --dtype fp32 --batch-size 8
uv run python reproduce/availability/run_availability_probes.py \
    --model EleutherAI/pythia-6.9b --hidden-dir $OUT/run_pythia-6.9b/hidden \
    --datasets-dir $OUT/datasets/pythia-6.9b --families $FAMS --swept-steps $STEPS --group-split \
    --device cuda --out-dir $OUT/run_pythia-6.9b/probes

# 4. Pythia-1B column of Table F.2 and rows of Table F.4: agreement run (attractor always mismatched) and breadth run
AGREE="sva sva_across_pp sva_subject_rc sva_object_rc ioi_role_balanced copy_control fixed_token_control"
BREADTH="blimp_det_noun_agreement blimp_anaphor_agreement blimp_npi fv_antonym fv_country_capital fv_en_fr \
  fv_present_past induction numeric_gt_v2 numeric_lt_v2 numeric_tf_v2 relational_facts relational_facts_balanced"
uv run python reproduce/availability/build_task_datasets.py \
    --task-set availability --subject-rc-attractor mismatched --model EleutherAI/pythia-1b --families $AGREE \
    --n-max-per-family 2000 --rng-seed 0 --no-norm-match --out-dir $OUT/datasets/pythia-1b
uv run python reproduce/availability/build_task_datasets.py \
    --task-set availability --model EleutherAI/pythia-1b --families $BREADTH \
    --n-max-per-family 2000 --rng-seed 0 --no-norm-match --out-dir $OUT/datasets/pythia-1b_breadth
for RUN in pythia-1b:"$AGREE" pythia-1b_breadth:"$BREADTH"; do
  NAME=${RUN%%:*}; F=${RUN#*:}
  uv run python reproduce/availability/extract_hidden_dense.py \
      --hf-model EleutherAI/pythia-1b --datasets-dir $OUT/datasets/$NAME --families $F \
      --h-steps $STEPS --out-dir $OUT/run_$NAME/hidden --device cuda --dtype fp32 --batch-size 16
done
uv run python reproduce/availability/run_availability_probes.py \
    --model EleutherAI/pythia-1b --hidden-dir $OUT/run_pythia-1b/hidden --datasets-dir $OUT/datasets/pythia-1b \
    --families $AGREE --swept-steps $STEPS --group-split --device cuda --out-dir $OUT/run_pythia-1b/probes
uv run python reproduce/availability/run_availability_probes.py \
    --model EleutherAI/pythia-1b --hidden-dir $OUT/run_pythia-1b_breadth/hidden \
    --datasets-dir $OUT/datasets/pythia-1b_breadth --families $BREADTH --swept-steps $STEPS --group-split \
    --stratified-prefixes numeric --unprobed-prefixes blimp --device cuda --out-dir $OUT/run_pythia-1b_breadth/probes
```

| Paper item | Step | Output, under `results/availability/` |
|---|---|---|
| Figure 4 left (`fig:main-sva-availability-expression`), probe | 2 | `run_pythia-6.9b_probe_trajectory/controlled/probe_trajectory_69b.csv` (`probe_name = sva_number_lemma_grouped`, `feature_source = hidden`) |
| Figure 4 left, native and best swept readout | 2 | `run_pythia-6.9b_probe_trajectory/summary.csv` (native: `s_step = h_step`; best swept: [Known deviations](#known-deviations)) |
| Figure F.1 (`fig:app-contrastive-readout-lag`) | 2 | the same two files, families `sva` and `ioi_role_balanced` |
| Figure 4 right | 3 | `availability/run_pythia-6.9b/probes/probe_summary.csv` (`h_step = 143000`, `probe_acc_1d`, `native_readout_acc`, the four `sva*` families) |
| Figure F.2, Tables F.2 to F.4 (`fig:app-ae-grid`, `tab:app-ae-binary`, `tab:app-ae-openvocab`, `tab:app-ae-early`) | 3, 4 | `availability/run_<run>/probes/probe_summary.csv` |
| Appendix G.4 terminal-readout accuracies (`sec:app-terminal-swap-alignment`) | 1 | `run0_pythia1b/summary.csv` (`h_step = 1000`, `s_step = 143000`, per alignment) |
| Figure 1 (`fig:overview`) margins | 1 | `run0_pythia1b/shards/sva__a-none__h1000__s<s>.pt` (`margins`, `margins_native`, `y_plus`, `y_minus`; `s` = 1000, 143000) |

Notes: step 1 filters answer pairs on terminal row norms (needs the step-143000 `W_U` snapshot); steps 3 and 4
need `nyu-mll/blimp` and the four function-vector files in `$FV_DATA_DIR`. Native expression applies `LN_t` and
`W_U^t` to the pre-LayerNorm state `z_t`; each swept candidate `s` (the `--swept-steps` minus 143000) applies
`LN_s` and `W_U^s`. `run_availability_probes.py` resumes per cell and reproduces sklearn's `GroupKFold` with the
x86-64 NumPy tie order, so folds match on every platform, including arm64.

## Localization (Section 4.4, Appendix G)

Section 4.4 and Table G.2 (`tab:app-contrastive-localisation-ledger`). Run
[availability](#availability-section-43-appendix-f-figure-1) step 1 first. Needs a GPU.

```bash
# Rank features by attribution on half the items; ablate or keep the top K on the other half
uv run python reproduce/localization/run_feature_attribution.py --model pythia-1b --d-sae 24576 --seed 0 \
    --datasets-dir results/availability/datasets/pythia-1b --hidden-dir results/availability/run0_pythia1b/hidden \
    --snapshot-step 143000 --h-step 1000 --rank-by pos --K 8 16 32 64 128 256 --split-frac 0.5 --split-seed 0 \
    --out-dir results/localization/run0_pythia1b_s143000_h1000_pos
# Hidden-projection columns of Table G.2 (same held-out half)
uv run python reproduce/localization/run_converse_intervention.py --model pythia-1b \
    --datasets-dir results/availability/datasets/pythia-1b --hidden-dir results/availability/run0_pythia1b/hidden \
    --attribution-dir results/localization/run0_pythia1b_s143000_h1000_pos \
    --snapshot-step 143000 --h-step 1000 --K 8 16 32 64 128 256 \
    --out-dir results/localization/converse_pythia1b_s143000_h1000_pos
```

| Paper item | Output, under `results/localization/` |
|---|---|
| Section 4.4; Table G.2 *recon*, *ablate*, *preserve-only*, *sign* | `run0_pythia1b_s143000_h1000_pos/shards/<family>__h1000__s143000.json`, `per_K["8"]`: `summary_{recon,ablate_top,ablate_ctrl_sign_matched,preserve_top,preserve_ctrl_sign_matched}.accuracy`; original readout `summary_native_s` |
| Table G.2 *orig.* (greater-than, IOI, relational facts) | the `none`, h=1000, s=143000 cell of `results/availability/run0_pythia1b/summary.csv` |
| Table G.2 *hidden projection* | `converse_pythia1b_s143000_h1000_pos/<family>__h1000__s143000.json`: `baseline.native_t.accuracy`, `per_K["8"].summary_proj_native_t.accuracy` |

The families are `sva`, `numeric_gt`, `ioi`, `relational_facts`. A family whose attribution `.json`/`.pt` shard
pair exists is skipped. The release scores a different held-out half than the paper ([Known deviations](#known-deviations)).

## Recipe control (Section 4.5, Appendix H)

Figure 5, the geometry, swap, and probe numbers of Section 4.5, and Appendix H (Figures H.1 to H.4, Table H.1):
four 31M Pythia-architecture runs that differ only in the `W_U` learning-rate multiplier or a shorter warmup.
Steps 2 and 3 can be replaced by the `hematteo/readout-recipe-control` download.

```bash
uv run python reproduce/recipe_control/selfcheck.py   # 1. model API, init, param groups (CPU, no data)
# 2. First 10B tokens of the Pythia preshuffled Pile, in Pythia's order
uv run python reproduce/recipe_control/fetch_pythia_preshuffled.py \
    --out ${UM_SSD_ROOT}/pile_slice/pythia_preshuffled.bin --target-tokens 10e9
# 3. Train one condition (baseline shown). The others change --out-dir ${UM_SSD_ROOT}/runs/<condition> and:
#    wu_lr_0p25: --readout-lr-mult 0.25; wu_lr_4x: --readout-lr-mult 4.0; warmup_short: --warmup-steps 715;
#    warmup_long (released, not compared: never reaches peak LR in 4769 steps): --warmup-steps 5720
uv run python reproduce/recipe_control/train_control.py \
    --data ${UM_SSD_ROOT}/pile_slice/pythia_preshuffled.bin --sequential-data \
    --out-dir ${UM_SSD_ROOT}/runs/baseline --model-size 31M --global-batch 1024 --max-tokens 10e9 \
    --readout-lr-mult 1.0 --warmup-steps 1430
# 4-5. Three crosscoders per condition (one GPU) and their lifecycle statistics (CPU)
for s in 0 1 2; do
  uv run python reproduce/recipe_control/train_control_crosscoder.py \
      --ckpt-root ${UM_SSD_ROOT}/runs --conditions wu_lr_0p25 baseline wu_lr_4x warmup_short --seed $s \
      --out-dir results/recipe_control/crosscoders
  uv run python reproduce/recipe_control/analyze_lifecycle.py \
      --ckpt-root ${UM_SSD_ROOT}/runs --cc-dir results/recipe_control/crosscoders \
      --conditions wu_lr_0p25 baseline wu_lr_4x warmup_short --seed $s --d-sae 8192 \
      --out-dir results/recipe_control/lifecycle_seed$s
done
# 6. Pool the three fits
R=results/recipe_control
uv run python reproduce/recipe_control/aggregate_seed_replication.py \
    --seed-stat $R/lifecycle_seed0/lifecycle_stats.json --seed-stat $R/lifecycle_seed1/lifecycle_stats.json \
    --seed-stat $R/lifecycle_seed2/lifecycle_stats.json --seeds 0 1 2 \
    --out-json $R/lifecycle_multiseed_summary.json --out-csv $R/lifecycle_multiseed_table.csv
# 7. W_U geometry (CPU)
uv run python reproduce/recipe_control/analyze_readout_geometry.py \
    --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline warmup_short wu_lr_0p25 wu_lr_4x \
    --out-csv results/recipe_control/readout_geometry_pythia.csv
# 8. The 2M-token held-out slice right after the trained 10B tokens (network, 4 MB)
uv run python reproduce/recipe_control/fetch_heldout_slice.py \
    --out ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin --start-token 10e9 --eval-tokens 2e6
# 9-10. Within-run readout swap grid and frequency-class probe (one GPU, minutes)
uv run python reproduce/recipe_control/run_trajectory_swap.py \
    --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short \
    --h-steps 256 512 1024 2048 4769 --eval-bin ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin \
    --eval-tokens 120000 --out-dir results/recipe_control/trajectory
uv run python reproduce/recipe_control/run_expression_lag.py \
    --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short \
    --h-steps 256 512 1024 2048 4769 --eval-bin ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin \
    --eval-tokens 400000 --max-positions 20000 --out-dir results/recipe_control/expression_lag
```

| Paper item | Output, under `results/recipe_control/` unless stated |
|---|---|
| Figure 5 left (`fig:main-recipe-dose-response`) | `lifecycle_multiseed_table.csv` |
| Figure 5 right | `${UM_SSD_ROOT}/runs/<condition>/metrics.csv` (validation loss) |
| Table H.1, Figure H.1 (`tab:app-recipe-control-summary`, `fig:app-recipe-control-reorg`) | `lifecycle_seed0/{lifecycle_summary.csv,reorg_window.csv}` |
| Figures H.2, H.3, H.4 (`fig:app-recipe-control-geometry`, `fig:app-recipe-control-basin`, `fig:app-recipe-control-explag`) | `readout_geometry_pythia.csv`, `trajectory/trajectory_swap_all.csv`, `expression_lag/expression_lag.csv` |

Notes: one A40 per pretraining run, about 21 h; it resumes from `latest.pt`, and `--max-hours` stops it cleanly.
Each run keeps 16 checkpoints; the analysis scripts map requested steps to the nearest one on disk. CPU smoke
test of step 4: add `--device cpu --expansion-factor 0.5 --n-epochs 2 --batch-size 4096 --conditions baseline`.
Steps 9 and 10 run on CPU at reduced size, e.g. `--eval-tokens 8192 --seq-len 256 --device cpu`.

## Readout swaps (Appendix G)

Figure G.1 (`fig:app-readout-swap-nll-kl-grid`), the late-checkpoint anomaly of Appendix G.3
(`sec:app-late-checkpoint-anomaly`), the Pythia-160M replication, Table G.1 (`tab:app-readout-swap-languages`),
and Figures G.2 and G.3.

```bash
# Optional: rebuild the evaluation corpus token for token from the 2023-11-01 Wikipedia dumps (network)
uv run python reproduce/readout_swaps/build_eval_corpus_pythia.py --output results/readout_swaps/eval_tokens_pythia.pt
# Pythia-1B swap grid (Figure G.1); also the parity reference of the per-language run
uv run python reproduce/readout_swaps/temporal_patch_grid.py \
    --model pythia-1b --seed 0 --h-eval-steps 256 512 1000 2000 3000 14000 47000 143000 \
    --out-dir results/readout_swaps/temporal_patch_grid/pythia-1b_seed0
# Pythia-160M replication (default --h-eval-steps 256 512 1000 2000 143000)
uv run python reproduce/readout_swaps/temporal_patch_grid.py \
    --model pythia-160m --seed 0 --out-dir results/readout_swaps/temporal_patch_grid/pythia-160m_seed0
# Per-language grid (Table G.1, Figure G.2; resumable)
uv run python reproduce/readout_swaps/run_language_stratified_swap.py \
    --model pythia-1b --seed 0 --h-eval-steps 256 512 1000 2000 3000 14000 47000 143000 \
    --block-tokens 256 --bootstrap-reps 2000 --bootstrap-seed 0 \
    --out-dir results/readout_swaps/languages/pythia-1b_seed0 \
    --reference-summary results/readout_swaps/temporal_patch_grid/pythia-1b_seed0/summary_global.csv
# Aligned swaps (Figure G.3); read the Figure G.3 item of Known deviations first
uv run python reproduce/readout_swaps/run_aligned_swap_grid.py --model pythia-1b --out-dir results/readout_swaps/aligned_swap_grid
```

| Paper item | Output, under `results/readout_swaps/` |
|---|---|
| Figure G.1, Pythia-160M replication | `temporal_patch_grid/pythia-{1b,160m}_seed0/summary_global.csv` |
| Table G.1 | `languages/pythia-1b_seed0/`: `summary.csv` (interval at `s*`), `argmin_summary.csv`, `manifest.json` (target counts) |
| Figure G.2 (`fig:app-readout-swap-languages`), English comparison of Appendix G.2 | `languages/pythia-1b_seed0/summary.csv` (`language` = `all`, `en`, per language), `english_only_summary.csv` |
| Figure G.3 (`fig:app-aligned-readout-swap`) | `aligned_swap_grid/summary.csv` |

Notes: the per-language gain is `-delta_nll` with interval `[-delta_nll_ci_high, -delta_nll_ci_low]`; the `all`
stratum must match the reference grid within 1e-5 nats (`reference_parity.json`). Hidden states are cached
under `${UM_SSD_ROOT}/readout_edit_timing_pythia1b/` (1B) and `${UM_SSD_ROOT}/readout_edit_timing/` (160M),
written on a cache miss. Without the snapshot mirror, add `--readout-cache-dir <scratch>/readouts`;
`--hln-cache-dir <scratch>/hln --delete-hln-cache-after-step` keeps the hidden-state caches on local scratch.

## Validation (Appendix C)

Figures C.1 to C.8 and Table C.1 (`sec:app-instrument-validation`).

```bash
REL="$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders"
V=results/validation/appendix_validation
# 1. EV, mean L0, dead rate of every released dictionary and SAE baseline
uv run python reproduce/validation/recompute_quality.py \
    --ckpt-root "$REL" --snap-dir "$UM_SSD_ROOT/snapshots" --out-csv $V/full_inventory.csv
uv run python reproduce/validation/recompute_quality.py \
    --ckpt-root "$REL" --snap-dir "$UM_SSD_ROOT/snapshots" --filter lambda-sweep --out-csv $V/lambda_sweep.csv
# 2. Per-checkpoint EV and L0 (Pythia-160M; reads full_inventory.csv), then the Pythia-1B width sweep
uv run python reproduce/validation/eval_persnap_fidelity.py
uv run python reproduce/validation/eval_persnap_1b.py
# 3. Pre-readout hidden-state caches for the fidelity check (fp32 for 160M and 1B, bf16 default for 6.9B)
EVAL="$REL/evaluation/eval-corpus/eval_tokens.pt"
uv run python scripts/extract/build_hln_cache.py --model pythia-160m --steps all --eval-tokens "$EVAL" \
    --out-dir "$UM_SSD_ROOT/derived/readout_edit_timing" --save-dtype fp32
uv run python scripts/extract/build_hln_cache.py --model pythia-1b --steps all --eval-tokens "$EVAL" \
    --out-dir "$UM_SSD_ROOT/derived/readout_edit_timing_pythia1b" --save-dtype fp32 --max-seqs 64
uv run python scripts/extract/build_hln_cache.py --model pythia-6.9b --steps all --eval-tokens "$EVAL" \
    --out-dir "$UM_SSD_ROOT/derived/readout_edit_timing_pythia69b"
# 4. Next-token fidelity per model (first 32 sequences), then the token-stratified check (64) on Pythia-1B
F=reproduce/validation
uv run python $F/readout_functional_fidelity.py --model pythia-160m --d-sae 8192 --seed 0 --steps all \
    --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing
uv run python $F/readout_functional_fidelity.py --model pythia-1b --d-sae 24576 --seed 0 --steps all \
    --snapshot-dtype bf16 --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing_pythia1b
uv run python $F/readout_functional_fidelity.py --model pythia-6.9b --d-sae 32768 --seed 0 --steps all \
    --snapshot-dtype bf16 --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing_pythia69b \
    --ckpt "$REL/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors"
uv run python $F/readout_functional_fidelity_strata.py --model pythia-1b --d-sae 24576 --seed 0 --steps all \
    --max-seqs 64 --snapshot-dtype bf16 --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing_pythia1b
# 5. Dense PCA baselines; checkpoints are discovered from the snapshot directory unless --steps is given
uv run python reproduce/validation/eval_pca_static.py --models pythia-160m pythia-1b pythia-6.9b olmo-2-7b
uv run python reproduce/validation/eval_concat_pca.py --models pythia-160m pythia-1b pythia-6.9b olmo-2-7b
# 6. Mean-direction audit and adjacent-checkpoint subspace movement (CPU)
uv run python reproduce/validation/build_mean_direction_audit.py
uv run python reproduce/validation/build_dense_reorganization_timing.py
# 7. Held-out checkpoints; --snapshot-dir must also hold steps 10k, 23k, 33k, 43k, 73k, 93k, 113k, 123k
uv run python reproduce/validation/eval_heldout.py \
    --crosscoder "$REL/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors" --model-slug EleutherAI_pythia-1b \
    --snapshot-dir ${UM_SSD_ROOT}/snapshots/pythia-1b --out-dir results/validation/heldout/pythia-1b
uv run python reproduce/validation/eval_heldout.py \
    --crosscoder "$REL/pythia-160m/W_U/cross-snapshot-32/d8192/seed0.safetensors" --model-slug EleutherAI_pythia-160m \
    --snapshot-dir ${UM_SSD_ROOT}/snapshots/pythia-160m --out-dir results/validation/heldout/pythia-160m
# 8. Optional: retrain the 32 per-snapshot SAEs (to $UM_SSD_ROOT/wu_crosscoder/per_snap_sae/) and compare
uv run python reproduce/validation/train_persnap_saes.py
uv run python reproduce/validation/persnap_vs_crosscoder.py
uv run python reproduce/validation/build_persnap_comparison.py
uv run python reproduce/validation/compute_cc_vs_persnap_gap.py
```

| Paper item | Output, under `results/validation/` |
|---|---|
| Figure C.1 (`fig:main-multimodel-validation`) A, Table C.1 (`tab:full-inventory`), Figure C.8 (`fig:app-lambda-sweep`) | `appendix_validation/full_inventory.csv`, `appendix_validation/lambda_sweep.csv` |
| Figure C.1 B, C | `appendix_validation/persnap_fidelity.csv` (Pythia-160M), `main_1b/persnap_1b_d<d_sae>_seed0.csv` (Pythia-1B); Pythia-6.9B and OLMo-2-7B curated |
| Figure C.4 (`fig:app-snapshot-fidelity`) | `appendix_validation/persnap_fidelity.csv` (released per-snapshot SAEs) |
| Figure C.2 (`fig:app-mean-direction-preprocessing`), spectral-gap check of Appendix C.1 | `dense/mean_direction_metrics.csv` |
| Figure C.3 (`fig:app-fidelity-curves`), token-stratified check of Appendix C.2 | `appendix_validation/readout_functional_fidelity/<model>_d<d_sae>_seed0/readout_functional_fidelity.csv`, `strata/strata_summary.csv` |
| Figure C.5 (`fig:app-pca-static`), concatenated PCA of Appendix C.4 | `appendix_validation/{pca_static,concat_pca}.csv` |
| Figure C.6 (`fig:app-dense-reorg-pythia`) | `dense/dense_reorganization/{subspace_metrics,sv_stability}.csv` |
| Figure C.7 (`fig:app-heldout-checkpoints-1b`), Pythia-160M numbers of Appendix C.5 | `heldout/<model>/heldout_ev.csv` |

The paper's concatenated PCA uses 45 Pythia-160M, 32 Pythia-1B and OLMo-2-7B, and 15 Pythia-6.9B checkpoints.

## Training the dictionaries

Each `configs/runs/<id>.yaml` records one dictionary's hyperparameters (Table B.2,
`tab:repro-dictionary-hparams`) and its `reported:` EV and L0. Nothing loads these files: pass each field as the
flag of the same name. CLI defaults that differ from the published values: `--n-epochs 20`, `--batch-size 2048`,
`--input-preprocess none`, `--l1-warmup-fraction 0.0`, `--tanh-stretch 4.0`. Where records disagree, the
checkpoint's embedded training dict is primary.

| Config | Dictionary | Released checkpoint, under `hf_release/parameter-trajectory-crosscoders/` |
|---|---|---|
| `pythia-160m_wu_d24576_seed0.yaml` | Pythia-160M `W_U`, d=24,576 (selected) | `pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors` |
| `pythia-1b_wu_d24576_seed0.yaml` | Pythia-1B `W_U`, d=24,576 (selected) | `pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors` |
| `pythia-6.9b_wu_d32768_seed0_sparse.yaml` | Pythia-6.9B `W_U`, d=32,768, lambda_1=0.6 (selected) | `pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors` |
| `pythia-6.9b_wu_d32768_seed0_dense.yaml` | Pythia-6.9B `W_U`, d=32,768, lambda_1=0.3 (comparison) | `pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0.safetensors` |
| `olmo-2-7b_wu_d32768_seed0.yaml` | OLMo-2-7B `W_U`, d=32,768 | `olmo-2-7b/W_U/cross-snapshot-32/d32768/seed0.safetensors` |
| `pythia-1b_wu_d24576_seed0_olmo-matched-window.yaml` | Pythia-1B late-start control | `pythia-1b/W_U/cross-snapshot-32-matched-window/d24576/seed0.safetensors` |
| `pythia-160m_we_d8192_seed0.yaml` | Pythia-160M `W_E`, d=8,192 | `pythia-160m/W_E/cross-snapshot-32/d8192/` |

```bash
# Pythia-160M W_U, d=24,576 (one GPU or CPU); first extracts any missing W_U snapshot into --cache-dir
uv run python scripts/train/train_crosscoder.py \
    --model EleutherAI/pythia-160m --expansion-factor 32.0 --batch-size 1024 --lr 5e-5 --n-epochs 300 --seed 0 \
    --input-preprocess center_scale --amp-dtype fp32 --tanh-stretch 1.0 --l1-warmup-fraction 0.05 \
    --output local_snapshots/wu_cc_pythia160m_seed0.pt
uv run python scripts/extract/extract_we_pythia.py --model EleutherAI/pythia-160m   # W_E snapshots (own cache)
```

`--cache-dir` defaults to `$UM_SSD_ROOT/snapshots/pythia-160m` (or `WU_CACHE_DIR`) for every `--model`; point it
at `$UM_SSD_ROOT/snapshots/<model>` for the others. Pythia-6.9B and OLMo-2-7B train head-parallel over four GPUs
with `scripts/train/train_distributed.py`. `scripts/extract/extract_rates.py` computes per-snapshot firing rates.

### OLMo-2-7B

```bash
# 1. W_U (lm_head) at the 32 stage-1 checkpoints (script default), in the layout the analysis scripts read
uv run python scripts/extract/extract_wu_olmo.py --cache-dir "$UM_SSD_ROOT/snapshots/OLMo-2-1124-7B"
# 2. d_sae=32,768, 300 epochs, batch 1024, head-parallel over four GPUs
uv run torchrun --nproc-per-node=4 --nnodes=1 scripts/train/train_distributed.py \
    --model allenai/OLMo-2-1124-7B --cache-dir "$UM_SSD_ROOT/snapshots/OLMo-2-1124-7B" \
    --steps 150 600 700 850 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 14000 21000 27000 34000 47000 110000 173000 236000 299000 362000 425000 488000 614000 677000 740000 803000 866000 928000 \
    --d-sae 32768 --batch-size 1024 --lr 1e-5 --jumprelu-lr-factor 0.3 --l1-coefficient 0.3 --tanh-stretch 1.0 \
    --init-threshold 0.1 --decoder-transpose-init 1.0 --input-preprocess center_scale --l1-warmup-fraction 0.1 \
    --lr-warmup-fraction 0.10 --lr-decay-fraction 0.20 --optimizer adam --amp-dtype fp32 --n-epochs 300 --seed 0 \
    --output "$UM_SSD_ROOT/runs/olmo/cc_olmo27b_dsae32768_seed0.pt"
```

Then rerun the [lifecycle](#lifecycle-section-41-appendix-d) commands. The OLMo-2-7B EV, L0, and activation
rates the paper uses are the curated `large_evals/` files ([Known deviations](#known-deviations)).

## Known deviations

- **Section 4.4 and Table G.2.** The release seeds the localization split and the matched controls separately
  for each task, and builds the simple SVA items in interleaved order. The paper's run drew both from one
  sequential generator, so the release scores a different held-out half and its values differ from the paper's.
  (The swap-grid accuracies of availability step 1 do not depend on item order.)
- **Figure 4, left panel.** The native and best swept series have no surviving run output. `paper_figures/`
  ships them as values transcribed from the original figure; the probe series is run output.
- **Figures 4 (left) and F.1, best swept readout.** No release script computes the best swept readout of the
  trajectories (`B_env`, the maximum over readouts per task and hidden-state checkpoint, excluding the native
  and terminal readouts; `sec:app-evaluation-details`). Take the maximum accuracy over the `alignment = none`
  rows of `run_pythia-6.9b_probe_trajectory/summary.csv` whose `s_step` is neither `h_step` nor 143000.
- **Figure C.1, Table C.1, and the curated `large_evals/` files.** The Pythia-6.9B and OLMo-2-7B rows and their
  per-checkpoint curves come from evaluation files that no release script regenerates; `recompute_quality.py`
  recomputes their aggregate EV and L0. The same curated files, under
  `results/validation/appendix_validation/large_evals/` and not shipped, supply the Pythia-6.9B and OLMo-2-7B
  activation rates that `find_reorganization_steps.py` (Figure D.2, Table D.2, and the windows quoted in
  Section 4.1) and `compute_late_start_vs_olmo.py` (Appendix D.3) read.
- **Late-start control (Appendix D.3).** No release script writes the late-start firing-rate file that `compute_late_start_vs_olmo.py`
  reads. `scripts/extract/extract_rates.py` computes the rates but saves them under
  `rates_per_seed[<checkpoint stem>]`, while the comparison reads a top-level `rates` key, so the tensor has to
  be stored under `rates` together with `steps`.
- **Figure G.3.** The paper's aligned swap grid scores the first 32 sequences (16,352 targets). Build
  32-sequence caches with `scripts/extract/build_hln_cache.py --max-seqs 32` before running
  `run_aligned_swap_grid.py`; without them the script scores the full corpus. `run_aligned_swap_grid.py` has no
  cache-directory option: it reads `${UM_SSD_ROOT}/readout_edit_timing_pythia1b/`, where `temporal_patch_grid.py`
  and `run_language_stratified_swap.py` keep their full-corpus caches, and `build_hln_cache.py` skips any step
  whose `hLN_step<t>.pt` already exists there. For the same reason, its `summary.csv` is not a valid parity
  reference for the per-language run.
- **Appendix F on CPU.** The label-shuffle and random-label nulls can differ from the published values by about
  one item, and the full-vocabulary mean ranks by near-tie flips (rank-1 rates agree).
- **OLMo-2-7B optimizer.** `tab:repro-dictionary-hparams` lists SparseAdam for OLMo-2-7B, while the
  [training command](#olmo-2-7b) passes `--optimizer adam` and the settings of record record `adam`; the
  provenance note in `configs/runs/olmo-2-7b_wu_d32768_seed0.yaml` explains the uncertainty.

## Paper items without a script

Drawn or typeset by hand in the paper source; the numbers they quote come from the commands above.

- Schematics: Figure 1 (`fig:overview`; margins from [availability](#availability-section-43-appendix-f-figure-1)
  step 1) and Figure A.1 (`fig:method-cross-snapshot-schematic`).
- Tables 1 (`tab:task-protocol`; task registries `src/readout/probes/contrastive_tasks.py` and
  `src/readout/probes/availability_tasks.py`), A.1 (`tab:evaluation-controls`), B.1
  (`tab:checkpoint-suite-comparison`), B.3 (`tab:compute-resources`), D.1 (`tab:lifecycle-profile-rules`; rules
  in `classify_profiles_refined`, `src/readout/dynamics/lifecycle.py`), F.1 (`tab:app-task-construction`;
  templates in `readout.probes.availability_tasks` and sibling modules), and I.1 (`tab:method-comparison`).
