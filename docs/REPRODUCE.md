# Reproducing the paper

There are three levels, from cheapest to most expensive.

1. **Redraw the figures** with `make figures` (CPU, about 30 s). The scripts in
   `paper_figures/` redraw 14 of the paper's data figures, including Figures 2 to
   5, from the 4.9 MB of inputs in `paper_figures/data/`
   ([Redrawing the figures](#redrawing-the-figures)).
2. **Recompute those inputs** from the released dictionaries and checkpoints with
   the analysis scripts in `experiments/`. They read the released artifacts from
   `UM_SSD_ROOT` ([Setup and data](#setup-and-data)); the larger models need a GPU.
   [Main-text results](#main-text-results) and [Appendix results](#appendix-results)
   give the commands for each paper result in order.
3. **Retrain** the dictionaries (`scripts/train/train_crosscoder.py` with the
   [settings of record](#settings-of-record)) or the 31M runs
   ([Figure 5](#figure-5-readout-learning-rate)).

Run every command from the repository root. The analysis scripts compute and save
metrics (CSV, JSON, `.pt`) under the gitignored `results/` and `figures/` trees and
each experiment's `derived/` directory; they do not draw figures, and no outputs
ship with the repository. [`experiments.yaml`](../experiments.yaml) maps each
experiment to the figures and tables it backs; the commands in its `entry` lists
are the ones given below. `make install && make test` checks the code without data,
and `make audit` checks the manifest and the [figure index](#figure-index).

## Setup and data

### Environment

```bash
make install   # uv sync --extra dev --extra figures
make test      # CPU only, no data
```

Python 3.11 or 3.12. The tests and figure scripts run on CPU on Linux, macOS, and
Windows. Dictionaries above Pythia-160M and the 31M pretraining runs target Linux
with CUDA 11.8; `pyproject.toml` pins Linux torch to the cu118 wheels (torch
2.7.1+cu118, the version of the paper's runs). Hugging Face downloads use anonymous
public access; `HF_HOME` sets the cache location.

### Storage root

Analysis scripts resolve their inputs under `UM_SSD_ROOT` (default
`<repo>/local_snapshots`, set in `src/readout/core/paths.py`):

```bash
export UM_SSD_ROOT=/path/to/storage
```

```text
$UM_SSD_ROOT/
  snapshots/<model>/<slug>_step<N>_wu.pt           extracted W_U matrices
  hf_release/parameter-trajectory-crosscoders/     released dictionaries (.safetensors), evaluation corpus
  derived/aggregates/aggregates_<model>_d<N>_seed<S>.pt
  derived/rates/                                   activation-rate sidecars
  derived/readout_edit_timing*/hLN_step<t>.pt      hidden-state caches of the fidelity check
  readout_edit_timing*/hLN_step<t>.pt              hidden-state caches of the readout swap grids
  pile_slice/, runs/<condition>/                   recipe-control token slices and 31M checkpoints
```

`<model>` is `pythia-160m`, `pythia-1b`, `pythia-6.9b`, or `OLMo-2-1124-7B`;
`<slug>` is the Hugging Face id with `/` replaced by `_`, as in
`snapshots/pythia-1b/EleutherAI_pythia-1b_step1000_wu.pt`. The scripts read
aggregates and rates from `$UM_SSD_ROOT/derived/`; in the crosscoder release they
are under `derived/aggregates/` and `derived/rates/`.

Optional overrides, all defaulting under `UM_SSD_ROOT`:

| Variable | Read by | Overrides |
|---|---|---|
| `WU_SNAP_DIR` | `src/readout/core/model_specs.py` | snapshot directory (default `<ssd>/snapshots`) |
| `WU_CACHE_DIR` | `src/readout/crosscoder/wu_adapter.py`, `src/readout/crosscoder/checkpoint_loaders.py` | the trainer's `W_U` snapshot cache |
| `CLUSTER_SCRATCH` | `src/readout/crosscoder/checkpoint_loaders.py` | scratch directory searched by run discovery (default `~/scratch`) |
| `DYNAMICS_DATA_ROOT` | `src/readout/dynamics/discovery.py` | training-run tree the analysis table discovers runs under (default `<ssd>/wu_crosscoder`) |
| `DYNAMICS_DERIVED_ROOT` | `src/readout/dynamics/derive.py` | per-run derived-artifact cache (default `experiments/crosscoders/crosscoder_main/derived`) |
| `READOUT_COMMIT` | `src/readout/core/repro.py` | git commit recorded in provenance when running without `.git` |
| `HF_HOME`, `HF_HUB_CACHE` | Hugging Face libraries and extraction scripts | Hugging Face cache locations |

### Downloads

| Hugging Face repository | Contents |
|---|---|
| `hematteo/parameter-trajectory-crosscoders` | trained crosscoders with `.config.json` and `.md` training sidecars, per-model aggregates (`derived/aggregates/`), activation-rate sidecars (`derived/rates/`), attribution artifacts, and the held-out evaluation corpus; `index.json` lists every file |
| `hematteo/readout-recipe-control` | the 31M recipe-control checkpoints: the four compared conditions and `warmup_long` |
| `datasets/hematteo/wu-crosscoder-snapshots` | pre-extracted `W_U` snapshots |

```bash
hf download hematteo/parameter-trajectory-crosscoders \
    --revision fb7ee860b9257f125ddbac7ff3c793b35fdcce8d \
    --local-dir "$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders"
hf download hematteo/readout-recipe-control --local-dir "${UM_SSD_ROOT}/runs"
```

The pinned revision is the verified release commit; drop `--revision` for the
latest. The dictionaries are released under MIT, and each can be retrained from
public checkpoints ([Settings of record](#settings-of-record)). Original model
checkpoints and raw Wikipedia dumps are not redistributed.

**Evaluation corpus.** The held-out corpus is
`$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt`
(Wikipedia-derived, CC BY-SA; its README there gives provenance), 173,229
next-token targets under the Pythia tokenizer. Scripts that take `--eval-tokens`
default to it, except `scripts/extract/build_hln_cache.py`, which requires the flag;
pass `--eval-tokens` to use another token tensor.
`build_eval_corpus_pythia.py` rebuilds it token for token
([Appendix H](#h-readout-swaps-and-localization)).

**`W_U` snapshots.** The trainer extracts them on demand: on a cache miss,
`readout.crosscoder.wu_adapter.load_wu_snapshot` downloads each Pythia `step{N}`
revision and saves its `embed_out.weight` as `<slug>_step<N>_wu.pt`, so running
the trainer fills the cache:

```bash
uv run python scripts/train/train_crosscoder.py --model EleutherAI/pythia-160m \
    --output /tmp/cc_160m.pt   # extracts any missing W_U snapshots first
```

CPU is enough for Pythia-160M; larger models want a GPU. The trainer's
`--cache-dir` defaults to `$UM_SSD_ROOT/snapshots/pythia-160m` (or `WU_CACHE_DIR`)
for every `--model`, so point it at `$UM_SSD_ROOT/snapshots/<model>` for the other
models. OLMo-2 uses a different revision format and `lm_head`; extract it with
`extract_wu_olmo.py` ([OLMo-2-7B dictionary](#crosscoder_olmo-olmo-2-7b-dictionary)).
`scripts/extract/` holds the other extraction helpers: `extract_we_pythia.py`
(`W_E` input embeddings for the Appendix D `W_E` dictionaries, in a separate cache),
`build_hln_cache.py` (final hidden states), `extract_rates.py` (firing rates), and
`build_eval_corpus_olmo.py` (the OLMo evaluation corpus).

**Checkpoint schedules** (`src/readout/core/model_specs.py`; the paper's
reproducibility appendix lists the same steps):

- Pythia, `DEFAULT_STEPS_32` (the 32-checkpoint schedule of Ge et al.): 0, 1, 2, 4,
  8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
  9000, 14000, 21000, 27000, 34000, 47000, 61000, 75000, 89000, 102000, 116000,
  130000, 143000.
- OLMo-2-7B stage 1, `OLMO_STEPS_32`: 150, 600, 700, 850, 900, 1000, 2000, 3000,
  4000, 5000, 6000, 7000, 8000, 9000, 14000, 21000, 27000, 34000, 47000, 110000,
  173000, 236000, 299000, 362000, 425000, 488000, 614000, 677000, 740000, 803000,
  866000, 928000. OLMo-2-7B releases no step-0 checkpoint; step 150 is its earliest.

### External assets and licenses

| Asset | Source | License | Needed for |
|---|---|---|---|
| Pythia 160M, 1B, 6.9B checkpoints and tokenizer | `EleutherAI/pythia-{160m,1b,6.9b}` | Apache-2.0 | snapshots, hidden states |
| OLMo-2-1124-7B checkpoints and tokenizer | `allenai/OLMo-2-1124-7B` | Apache-2.0 | the OLMo-2-7B dictionary |
| WordNet 3.0 | Princeton WordNet, through NLTK | WordNet 3.0 license | Figure 3, Appendix F |
| Wikipedia slices | Wikimedia 2023-11-01 dumps | CC BY-SA 4.0 / GFDL | the evaluation corpus |
| BLiMP | `nyu-mll/blimp` (Hugging Face Hub) | see its dataset card | the availability suite (Figure 4, Appendix G) |
| Function-vector word lists | `github.com/ericwtodd/function_vectors`, `dataset_files/abstractive/` | MIT | the availability suite (Figure 4, Appendix G) |
| GPT-NeoX-20B tokenizer | `EleutherAI/gpt-neox-20b` | Apache-2.0 | retraining the 31M runs |
| Pile pretraining corpus | `EleutherAI/pile-standard-pythia-preshuffled` for the exact Pythia order (byte-range prefix, no full 602 GB download), or `monology/pile-uncopyrighted` (slice tokenized locally) | MIT (Pile compilation) | retraining the 31M runs |

The last two rows are needed only to retrain the 31M runs. The tokenized slice is a
flat `uint16` `.bin` built by `fetch_pythia_preshuffled.py`
([Figure 5](#figure-5-readout-learning-rate)).

The WordNet probes read WordNet 3.0 through NLTK. Download it once, into
`~/nltk_data/` or the directory in `NLTK_DATA`; until then the probes raise
`LookupError` or `ModuleNotFoundError`:

```bash
uv run python -c "import nltk; nltk.download('wordnet')"
```

### Disk

- OLMo-2-7B extraction downloads about 14 GB per checkpoint (the extractor clears
  the Hugging Face cache between checkpoints) and keeps about 50 GB of snapshots for
  the 32 checkpoints.
- The Pythia-1B hidden-state caches of the readout swap grids take 1.8 GB per
  checkpoint.
- The recipe-control data are a byte-range prefix of the preshuffled Pile and a
  4 MB held-out slice.

## Settings of record

Each `configs/runs/<id>.yaml` records the training hyperparameters of one
dictionary in the paper; Table C.2 (`tab:repro-dictionary-hparams`) typesets them.
Field names mirror the flags of `scripts/train/train_crosscoder.py`
(`expansion_factor`, `batch_size`, `lr`, `n_epochs`, `seed`, `input_preprocess`,
`amp_dtype`, `l1_coefficient`, `tanh_stretch`, `init_threshold`, and the warmup and
decay fractions).

| File | Dictionary | Released checkpoint, under `hf_release/parameter-trajectory-crosscoders/` |
|---|---|---|
| `pythia-160m_wu_d24576_seed0.yaml` | Pythia-160M `W_U`, d=24,576 (selected) | `pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors` |
| `pythia-1b_wu_d24576_seed0.yaml` | Pythia-1B `W_U`, d=24,576 (selected) | `pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors` |
| `pythia-6.9b_wu_d32768_seed0_sparse.yaml` | Pythia-6.9B `W_U`, d=32,768, lambda_1=0.6 (selected) | `pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors` |
| `pythia-6.9b_wu_d32768_seed0_dense.yaml` | Pythia-6.9B `W_U`, d=32,768, lambda_1=0.3 (comparison) | `pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0.safetensors` |
| `olmo-2-7b_wu_d32768_seed0.yaml` | OLMo-2-7B `W_U`, d=32,768 | `olmo-2-7b/W_U/cross-snapshot-32/d32768/seed0.safetensors` |
| `pythia-1b_wu_d24576_seed0_olmo-matched-window.yaml` | Pythia-1B late-start control | `pythia-1b/W_U/cross-snapshot-32-matched-window/d24576/seed0.safetensors` |
| `pythia-160m_we_d8192_seed0.yaml` | Pythia-160M `W_E`, d=8,192 | `pythia-160m/W_E/cross-snapshot-32/d8192/` |

- Nothing loads these files. To retrain, pass every field as a flag to
  `scripts/train/train_crosscoder.py`, or to
  `experiments/crosscoders/crosscoder_olmo/scripts/train_distributed.py` for the
  four-GPU head-parallel Pythia-6.9B and OLMo-2-7B runs. Pythia-160M trains on one
  GPU and is the cheapest end-to-end path. Several CLI defaults are not the
  published values: `--n-epochs 20`, `--batch-size 2048`, `--input-preprocess none`,
  `--l1-warmup-fraction 0.0`, and `--tanh-stretch 4.0` (the `W_E` value).
- `reported:` lists the published EV and L0, to cross-check a retrain. `source:`
  names where each value came from. Values that no shipped file determines (the
  per-model `lr` of the Pythia-1B and 6.9B runs, and the selected 6.9B run's
  `l1_coefficient` and `init_threshold`) come from the released checkpoints'
  embedded training dicts and `.config.json` sidecars.
- Where two records of a run disagree, the checkpoint's embedded training dict is
  the primary record; the provenance note of the OLMo-2-7B file explains its case.
- The `-sparse` suffix separates the selected Pythia-6.9B dictionary from the
  default-lambda comparison `seed0.safetensors` in the same directory. A retrained
  copy belongs at that path; the rescue scripts also accept an explicit checkpoint
  flag.

## Redrawing the figures

```bash
make figures   # uv run --extra figures python paper_figures/make_all.py
```

This writes all of them into `paper_figures/out/`; to draw one, run its script,
e.g. `python paper_figures/wordnet_controls.py`. The scripts need only matplotlib,
numpy, and pandas (no GPU, model downloads, or external drives); the `figures`
extra pins matplotlib 3.10, with which every figure was checked against the
camera-ready PDFs. [`paper_figures/data/README.md`](../paper_figures/data/README.md)
lists each data file, the rows and columns it keeps, and the experiment script that
regenerates it.

| Paper label | Script | Data files | Status |
| --- | --- | --- | --- |
| `fig:main-selected-normalized-trajectories` (Figure 2) | `lifecycle_trajectories.py` | `lifecycle_*.npz` | matches the paper; trajectories are stored as float16, which shifts the rasterized lines by a few gray levels |
| `fig:main-wordnet-matched-controls` (Figure 3) | `wordnet_controls.py` | `wordnet_matched_controls_160m.csv` | identical |
| `fig:main-sva-availability-expression` (Figure 4) | `sva_availability.py` | `sva_hidden_probe_trajectory_69b.csv`, `sva_native_swept_transcribed_69b.csv`, `availability_probe_summary_69b.csv` | matches; the native and best-swept series of the left panel are transcribed from the original figure (no run output) |
| `fig:main-recipe-dose-response` (Figure 5) | `recipe_dose_response.py` | `recipe_lifecycle_multiseed.csv`, `recipe_val_loss.csv` | identical |
| `fig:main-multimodel-validation` (Figure D.1) | `instrument_validation.py` | `instrument_inventory.csv`, `persnap_fidelity_160m.csv`, `persnap_fidelity_1b_d24576.csv`, `large_dictionary_summary.csv`, `large_dictionary_per_snapshot.csv` | identical |
| `fig:app-fidelity-curves` (Figure D.4) | `fidelity_curves.py` | `fidelity_curves.csv` | identical |
| `fig:app-dense-reorg-pythia` (Figure D.9) | `dense_reorganization_timing.py` | `dense_subspace_metrics.csv`, `dense_sv_stability.csv` | identical |
| `fig:app-lifecycle-profile-composition` (Figure E.1) | `lifecycle_profile_composition.py` | `lifecycle_*.npz` | identical |
| `fig:app-selected-population-lifecycle-diagnostics` (Figure E.2) | `lifecycle_population_diagnostics.py` | `lifecycle_*.npz` | identical |
| `fig:app-ae-capacity` (Figure G.2) | `probe_fits.py` | `availability_probe_summary_69b.csv` | identical |
| `fig:app-readout-swap-english-aggregate` (Figure H.2) | `readout_swap_languages.py` (writes `readout_swap_aggregate.pdf`) | `language_swap_summary.csv` | identical |
| `fig:app-readout-swap-languages` (Figure H.3) | `readout_swap_languages.py` (writes `readout_swap_languages.pdf`) | `language_swap_summary.csv` | identical |
| `fig:app-aligned-readout-swap` (Figure H.4) | `aligned_swap_grid.py` | `aligned_swap_summary_1b.csv` | identical |
| `fig:app-recipe-control-reorg` (Figure I.2) | `recipe_reorganization_window.py` | `recipe_reorg_window_seed0.csv` | identical |

"Identical" means the rendered page matches the paper's PDF pixel for pixel at 60
and 150 dpi. `lifecycle_profile_composition.py` imports the profile rules from
`readout.dynamics.lifecycle` (numpy only), and `_common.py` adds `src/` to the
import path, so the package need not be installed.

The other appendix figures are not redrawn here. The schematics (Figure 1, Figure
B.1) and the hand-typeset tables (Tables 1, 2, A.1 to A.3, B.1, C.1, C.3, E.1, G.1)
are listed under `paper_only_labels` in `experiments.yaml`; no experiment produces
them, but the numbers they quote come from the experiments below.

## Main-text results

### Figure 2: the readout reorganizes early

Paper: `sec:readout-lifecycle` (Section 4.1). Computes the per-feature
decoder-norm trajectories of the four selected dictionaries, their lifecycle
profiles, and the reorganization window with circular-shift nulls and a feature
bootstrap.

Experiment: `experiments/lifecycle/feature_lifecycle_trajectories/`. The OLMo-2-7B
dictionary comes from [`crosscoder_olmo`](#crosscoder_olmo-olmo-2-7b-dictionary);
the dictionary-quality numbers at the start of the section are Figure D.1
([Appendix D](#d-dictionary-validation)).

```bash
# 1. Decoder-norm trajectories; also writes the per-run <key>_decoder_norms.npy caches
uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/compute_normalized_trajectories.py
# 2. Lifecycle profiles (reads selected_decoder_norm_trajectories.pt)
uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/compute_lifecycle_profile_composition.py
# 3. Peak counts and decoder-norm mass per checkpoint
uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/build_population_diagnostics.py
# 4. Reorganization windows, null p values, and bootstrap counts
uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/find_reorganization_steps.py
```

| Paper item | Step | Output, under `results/experiments/lifecycle/feature_lifecycle_trajectories/` |
|---|---|---|
| Figure 2 (`fig:main-selected-normalized-trajectories`) | 1 | `selected_decoder_norm_trajectories{,_full}.{csv,pt}` (trajectory quantiles per snapshot); `normalized_trajectories.csv` (normalized norms per panel, feature, and snapshot, Pythia-160M and 1B) |
| Window and p values in the text; Table E.2 and Figure E.4 (`tab:app-reorganization-window-stats`, `fig:app-reorganization-window-metrics`) | 4 | `reorganization_window_metrics_selected.{csv,pt}`, `reorganization_window_summary_selected.csv`, `reorganization_window_bootstrap_counts_selected.csv`; also `reorganization_step_pair_metrics_selected.csv`, `reorganization_step_peaks_selected.csv`, `reorganization_bootstrap_selected.csv` |
| Profile shares (Figure E.1) and population diagnostics (Figure E.2) | 2, 3 | [Appendix E](#e-lifecycle-profiles-and-reorganization-timing) |

Step 1 writes the decoder-norm and decoder-geometry caches (`<key>_decoder_norms.npy`,
`<key>_adjacent_rotation_radians.npy`, `<key>_cos_to_terminal.npy`) to
`figures/feature_lifecycle_trajectories/section52_lifecycle/cache/`, deriving the
geometry from the released checkpoints on first run; later steps and the late-start
control read them.

Inputs: the four selected dictionaries, the Pythia aggregates and rate sidecars,
and the curated Pythia-6.9B and OLMo-2-7B activation rates that step 4 reads; the
full list is under [`feature_lifecycle_trajectories`](#feature_lifecycle_trajectories)
in Appendix E. No GPU is needed. Deviation: the curated rate files are not
regenerated by any release script ([Known deviations](#known-deviations)).

### Figure 3: lexical organization in the readout rows

Paper: `sec:vocabulary-family-readout` (Section 4.2). Probes the 35 WordNet
supersense (lexicographer file) categories one-vs-rest on the centered `W_U` rows of
Pythia-160M and Pythia-1B at every checkpoint, with a metadata-only baseline and
matched-random and label-shuffle nulls on Pythia-160M.

Experiment: `experiments/probes/concept_evolution_validation/`. Download WordNet
first ([External assets](#external-assets-and-licenses)).

```bash
# Probe every category at every checkpoint (row split)
uv run python experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py --model pythia-160m
uv run python experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py --model pythia-1b
# Matched controls on Pythia-160M
uv run python experiments/probes/concept_evolution_validation/scripts/run_wordnet_matched_controls.py
```

| Paper item | Script | Output, under `experiments/probes/concept_evolution_validation/derived/` |
|---|---|---|
| Figure 3 (`fig:main-wordnet-matched-controls`) | `run_wordnet_matched_controls.py` | `wordnet_matched_controls_160m/wordnet_matched_control_summary_160m.csv` |
| Separability and largest-gain counts in the text; Figure F.1 | `run_wordnet_supersense_probe.py --model pythia-160m` | `wordnet_supersense_160m/wordnet_supersense_probe_trajectory_160m.csv` |
| Pythia-1B onsets in the text; Figure F.2, Table F.1 | `run_wordnet_supersense_probe.py --model pythia-1b` | `wordnet_supersense_1b/wordnet_supersense_probe_{trajectory,pos_summary}_1b.csv` |
| Lemma-grouped numbers in the text | | [Appendix F](#f-wordnet-supersense-probes) |

Each probe run also writes `wordnet_supersense_audit_<suffix>.csv`,
`wordnet_supersense_probe_trajectory_<suffix>.json`,
`wordnet_supersense_probe_summary_<suffix>.csv`,
`wordnet_supersense_probe_metadata_<suffix>.json` (records `split` and `seed`), and
`wordnet_supersense_split_stats_<suffix>.csv` (per-category test support and
same-lemma train/test leakage). The matched controls also write
`wordnet_matched_control_{null_samples,match_quality,audit}_160m.csv` and
`wordnet_matched_control_metadata_160m.json`.

Inputs: `${UM_SSD_ROOT}/snapshots/pythia-{160m,1b}/EleutherAI_pythia-*_step*_wu.pt`;
the evaluation corpus (`--eval-tokens`), whose token counts feed the metadata
baseline and the matched nulls; WordNet 3.0. No GPU is needed; `--device cpu` runs
the probes on CPU.

### Figure 4 and Section 4.3: availability before expression

Paper: `sec:readout-reorganization-decodes` (Section 4.3). For single-token,
two-choice tasks, scores the contrastive margin of each hidden-state checkpoint
under each checkpoint's readout (readout swap) and compares linear probes on the
final hidden state (availability) with native and swept readout accuracy
(expression) on Pythia-1B and Pythia-6.9B.

Experiment: `experiments/probes/contrastive_readout_swap/`. Outputs are under
`results/experiments/probes/contrastive_readout_swap/`.

```bash
# 1. Pythia-1B localization inventories and swap grid (also used by Figure 1, Section 4.4, Table H.2)
uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
    --model EleutherAI/pythia-1b --out-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b
uv run python experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py --model pythia-1b \
    --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
    --out-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b \
    --alignments none scale row_norm procrustes

# 2. Pythia-6.9B 32-checkpoint SVA and IOI trajectories (bf16 forward pass, bf16-rounded W_U)
RUN=results/experiments/probes/contrastive_readout_swap/run_pythia-6.9b_probe_trajectory
DS=results/experiments/probes/contrastive_readout_swap/datasets/pythia-6.9b_trajectory
STEPS32="0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 5000 6000 7000 8000 9000 14000 21000 27000 34000 \
  47000 61000 75000 89000 102000 116000 130000 143000"
uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
    --model EleutherAI/pythia-6.9b --families sva ioi_role_balanced --n-max-per-family 2000 \
    --rng-seed 0 --no-norm-match --out-dir $DS
uv run python experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py \
    --model pythia-6.9b --datasets-dir $DS --families sva ioi_role_balanced \
    --h-steps $STEPS32 --alignments none --device cuda --dtype bf16 --wu-round bf16 --out-dir $RUN
uv run python experiments/probes/contrastive_readout_swap/scripts/run_controlled_hidden_probes.py \
    --hidden-dir $RUN/hidden --datasets-dir $DS --families sva ioi_role_balanced \
    --stem probe_trajectory_69b --out-dir $RUN/controlled

# 3. Pythia-6.9B availability suite: 20 families, 24 checkpoints, no row-norm matching
FAMS="sva sva_across_pp sva_subject_rc sva_object_rc ioi_role_balanced copy_control fixed_token_control \
  blimp_det_noun_agreement blimp_anaphor_agreement blimp_npi fv_antonym fv_country_capital fv_en_fr \
  fv_present_past induction numeric_gt_v2 numeric_lt_v2 numeric_tf_v2 relational_facts relational_facts_balanced"
STEPS="0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 5000 6000 8000 10000 16000 32000 64000 128000 143000"
OUT=results/experiments/probes/contrastive_readout_swap/availability
uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
    --task-set availability --model EleutherAI/pythia-6.9b --families $FAMS \
    --n-max-per-family 2000 --rng-seed 0 --no-norm-match --out-dir $OUT/datasets/pythia-6.9b
uv run python experiments/probes/contrastive_readout_swap/scripts/extract_hidden_dense.py \
    --hf-model EleutherAI/pythia-6.9b --datasets-dir $OUT/datasets/pythia-6.9b --families $FAMS \
    --h-steps $STEPS --out-dir $OUT/run_pythia-6.9b/hidden --device cuda --dtype fp32 --batch-size 8
uv run python experiments/probes/contrastive_readout_swap/scripts/run_availability_probes.py \
    --model EleutherAI/pythia-6.9b --hidden-dir $OUT/run_pythia-6.9b/hidden \
    --datasets-dir $OUT/datasets/pythia-6.9b --families $FAMS --swept-steps $STEPS --group-split \
    --device cuda --out-dir $OUT/run_pythia-6.9b/probes
```

| Paper item | Step | Output |
|---|---|---|
| Figure 4 left (`fig:main-sva-availability-expression`): probe | 2 | `run_pythia-6.9b_probe_trajectory/controlled/probe_trajectory_69b.csv` (`probe_name = sva_number_lemma_grouped`, `feature_source = hidden`, `acc`) |
| Figure 4 left: native and best swept readout | 2 | `run_pythia-6.9b_probe_trajectory/summary.csv` (native: rows with `s_step = h_step`; best swept: see [Known deviations](#known-deviations)) |
| Figure 4 right | 3 | `availability/run_pythia-6.9b/probes/probe_summary.csv` (`h_step = 143000`, `probe_acc_1d`, `native_readout_acc`; families `sva`, `sva_across_pp`, `sva_subject_rc`, `sva_object_rc`) |
| Terminal-readout and Procrustes accuracies in the text (`sec:app-terminal-swap-alignment`) | 1 | `run0_pythia1b/summary.csv` (`h_step = 1000`, `s_step = 143000`, per alignment) |
| The 20-family suite at both scales | 3 | [Appendix G](#g-availability-and-expression-across-task-families) |

Notes:

- Step 1's builder writes every family of `contrastive_tasks.TASK_BUILDERS`,
  including `sva`, `numeric_gt`, `ioi`, and `relational_facts`, and filters answer
  pairs on terminal row norms, which needs the step-143000 `W_U` snapshot. It builds
  the paper's 320 SVA items, interleaved by number. Add `--save-per-example` to its
  `run_swap_grid.py` call to also save the per-item margins of
  [Figure 1](#figure-1-the-overview-margins).
- `extract_hidden_dense.py` writes the final-position hidden states after (`h`) and
  before (`z`) the final LayerNorm, and `<slug>_step<S>_{wu,lnf}.pt` snapshots under
  `${UM_SSD_ROOT}/snapshots/pythia-6.9b/`.
- Swap-grid margins are `h_t . W_U^s[y+] - h_t . W_U^s[y-]`, with `h_t` the hidden
  state after checkpoint `t`'s final LayerNorm and no second normalization.
- In `run_availability_probes.py`
  (`readout.probes.availability_expression.native_and_best_swept`), native
  expression applies checkpoint `t`'s final LayerNorm and `W_U` to the pre-LayerNorm
  state `z_t`, and each swept candidate `s` applies its own final LayerNorm with its
  `W_U`, `LN_s(z_t) . W_U^s`. The candidates are the `--swept-steps` checkpoints
  minus the terminal step 143000, so the native checkpoint is included.
  `best_swept_readout_acc` is the in-sample maximum over candidates;
  `best_swept_acc_heldout` picks the candidate on a random half of the items (seed
  `--seed`) and reports its accuracy on the other half.
- `run_availability_probes.py` is CPU-only apart from the full-vocabulary rank
  computations (`--device`) and resumes per cell. Its grouped folds
  (`group_kfold_splits`) reproduce sklearn's `GroupKFold` with the tie order of the
  x86-64 NumPy sort used for the paper, so the folds and the fold-dependent
  intervals come out the same on any platform (sklearn's own `GroupKFold` gives
  different folds on arm64, e.g. Apple Silicon).
- Table 1 (`tab:task-protocol`) is typeset by hand. The task registries are
  `src/readout/probes/contrastive_tasks.py` (localization inventories) and
  `src/readout/probes/availability_tasks.py` (availability suite); probe labels,
  group keys, folds, and readout metrics are in
  `src/readout/probes/availability_expression.py`.
- Released but not used above: `launch_swap_extraction.sh` and
  `run_4gpu_extraction.sh` (multi-GPU extraction wrappers),
  `extract_wu_hidden_standalone.py` (standalone `W_U` and hidden-state extraction),
  and `run_step6_reporting.py` (held-out best-readout selection with a paired
  bootstrap).

Inputs:

- `${UM_SSD_ROOT}/snapshots/pythia-{1b,6.9b}/EleutherAI_pythia-*_step*_wu.pt`: the
  32-step schedule for step 2; `extract_hidden_dense.py` writes missing availability
  steps and the `_lnf.pt` final-LayerNorm parameters.
- The Hugging Face checkpoints of each model, forwarded once per hidden-state step.
- For step 3: `nyu-mll/blimp` from the Hugging Face Hub, and
  `$FV_DATA_DIR/{antonym,english-french,present-past,country-capital}.json` from
  `github.com/ericwtodd/function_vectors` (`dataset_files/abstractive/`, MIT).
  `FV_DATA_DIR` defaults to `data/function_vectors/` under the repository root (not
  shipped); the builder raises if a file is missing.

Outputs:

- `datasets/<run>/<family>.jsonl` (+ control sidecars and `summary.json`):
  tokenizer-specific task datasets.
- `run_swap_grid.py`, under `--out-dir`: `hidden/<family>_h<step>.pt`
  (final-position hidden states after the final LayerNorm),
  `shards/<family>__a-<alignment>__h<h>__s<s>.json` (resume markers; with
  `--save-per-example`, a matching `.pt` holding `margins`, `margins_native`,
  `y_plus`, `y_minus`), `manifest.json`, and `summary.csv` (mean margin, accuracy,
  and their change from the native cell per family, alignment, `h_step`, `s_step`).
- `run_controlled_hidden_probes.py`: `<stem>.csv` and `<stem>.pt`, probe accuracies
  on hidden states, prompt-token features, and Gaussian features.
- `availability/run_<run>/hidden/<family>_{h,z}<step>.pt` and
  `availability/run_<run>/probes/probe_summary.{csv,pt}` (+
  `probe_summary.provenance.json`).

Hardware: a GPU; step 2 runs a bf16 forward pass on CUDA, and step 3's extraction
needs one 48 GB GPU at fp32. Deviation: the left panel's native and best swept
series ([Known deviations](#known-deviations)).

### Section 4.4: a few readout directions carry task margins

Paper: `sec:sparse-coordinate-localization` (Section 4.4). With step-1000 Pythia-1B
hidden states decoded through the reconstructed terminal readout of the released
d=24,576 crosscoder, ranks features by attribution to each task margin on half of
the items, then ablates the top K, or keeps only them, on the other half, against
random features matched on decoder norm, firing rate, and attribution sign.

Experiment: `experiments/causal/contrastive_task_feature_rescue/`. Run
[Figure 4 step 1](#figure-4-and-section-43-availability-before-expression) first; it
builds the datasets and caches the step-1000 hidden states.

```bash
# Terminal readout, top-K by positive attribution, 50/50 selection and scoring split
uv run python experiments/causal/contrastive_task_feature_rescue/scripts/run_feature_attribution.py \
    --model pythia-1b --d-sae 24576 --seed 0 \
    --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
    --hidden-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden \
    --snapshot-step 143000 --h-step 1000 --rank-by pos --K 8 16 32 64 128 256 --split-frac 0.5 --split-seed 0 \
    --out-dir results/experiments/causal/contrastive_task_feature_rescue/run0_pythia1b_s143000_h1000_pos
```

The section's accuracies, and the *recon*, *ablate*, and *preserve-only* columns
(top-8 and *sign*) of Table H.2, are in
`run0_pythia1b_s143000_h1000_pos/shards/<family>__h1000__s143000.json` under
`results/experiments/causal/contrastive_task_feature_rescue/`, at `per_K["8"]`:
`summary_recon`, `summary_ablate_top`, `summary_ablate_ctrl_sign_matched`,
`summary_preserve_top`, `summary_preserve_ctrl_sign_matched` (`.accuracy`). The
original terminal readout is `per_K["8"].summary_native_s`. The families are `sva`,
`numeric_gt` (greater-than completion), `ioi`, and `relational_facts`.

| Output, under `--out-dir` | Content |
|---|---|
| `shards/<family>__h<h>__s<s>.json` | per-family attribution summary and top-K interventions with matched controls |
| `shards/<family>__h<h>__s<s>.pt` | full attribution vector and ranking |
| `reconstruction_ev.json` | EV of the reconstructed readout |
| `specificity_matrix.{json,pt}` | top-K ablations chosen for one family, scored on every family |
| `manifest.json` | inputs, per-family split, git commit |

Inputs:
`${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`;
`${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step*_wu.pt`;
`datasets/pythia-1b/<family>.jsonl` and `run0_pythia1b/hidden/<family>_h<step>.pt`
from Figure 4 step 1. Needs a GPU.

Notes:

- Attribution is `attr_f = mean_n (a_{y+,f} - a_{y-,f}) * (h_n . D_f) * scale_t`
  over the selection half, where `a` are the crosscoder's feature activations on the
  answer rows at `--snapshot-step`. Each control feature is the nearest unused
  neighbor of a selected feature in the plane of log decoder norm and log firing
  rate; the sign-matched control draws only from features with positive
  attribution.
- Shards are resumable: a family whose `.json` and `.pt` shard pair exists is
  skipped; delete the pair to recompute.
- Library code: `src/readout/probes/contrastive_tasks.py`,
  `src/readout/probes/readout_swap.py`.

Deviation: the release scores a different held-out half than the paper
([Known deviations](#known-deviations)).

### Figure 5: readout learning rate

Paper: `sec:recipe-control` (Section 4.5). Four 31M Pythia-architecture models share
initialization, data, and data order and differ only in the learning rate
multiplier of `W_U` (0.25x, 1x, 4x) or, in the fourth run, a shorter warmup; three
trajectory crosscoder fits per condition give the lifecycle statistics.

Experiment: `experiments/ablations/pretraining_recipe_control/`. The analysis
scripts read the checkpoints from `${UM_SSD_ROOT}/runs/<condition>/`, trained in
steps 2 and 3 or downloaded ([Downloads](#downloads)); they take `--ckpt-root` and
`--conditions`, which defaults to the four compared conditions.

```bash
# 1. Check the model API, initialization, parameter groups, and forward pass (CPU, no GPU, no data)
uv run python experiments/ablations/pretraining_recipe_control/scripts/selfcheck.py
# 2. First 10B tokens of the Pythia preshuffled Pile, in Pythia's order
uv run python experiments/ablations/pretraining_recipe_control/scripts/fetch_pythia_preshuffled.py \
    --out ${UM_SSD_ROOT}/pile_slice/pythia_preshuffled.bin --target-tokens 10e9
# 3. Train one condition on one GPU (baseline shown; other conditions in the table below)
uv run python experiments/ablations/pretraining_recipe_control/scripts/train_control.py \
    --data ${UM_SSD_ROOT}/pile_slice/pythia_preshuffled.bin --sequential-data \
    --out-dir ${UM_SSD_ROOT}/runs/baseline --model-size 31M --global-batch 1024 --max-tokens 10e9 \
    --readout-lr-mult 1.0 --warmup-steps 1430
# 4. Three trajectory crosscoders per condition, seeds 0 to 2 (one GPU; resumable, skips outputs that exist)
for s in 0 1 2; do
  uv run python experiments/ablations/pretraining_recipe_control/scripts/train_control_crosscoder.py \
      --ckpt-root ${UM_SSD_ROOT}/runs --conditions wu_lr_0p25 baseline wu_lr_4x warmup_short --seed $s \
      --out-dir results/experiments/ablations/pretraining_recipe_control/crosscoders
done
# 5. Lifecycle statistics per seed (CPU)
for s in 0 1 2; do
  uv run python experiments/ablations/pretraining_recipe_control/scripts/analyze_lifecycle.py \
      --ckpt-root ${UM_SSD_ROOT}/runs --cc-dir results/experiments/ablations/pretraining_recipe_control/crosscoders \
      --conditions wu_lr_0p25 baseline wu_lr_4x warmup_short --seed $s --d-sae 8192 \
      --out-dir results/experiments/ablations/pretraining_recipe_control/lifecycle_seed$s
done
# 6. Pool the three fits
R=results/experiments/ablations/pretraining_recipe_control
uv run python experiments/ablations/pretraining_recipe_control/scripts/aggregate_seed_replication.py \
    --seed-stat $R/lifecycle_seed0/lifecycle_stats.json --seed-stat $R/lifecycle_seed1/lifecycle_stats.json \
    --seed-stat $R/lifecycle_seed2/lifecycle_stats.json --seeds 0 1 2 \
    --out-json $R/lifecycle_multiseed_summary.json --out-csv $R/lifecycle_multiseed_table.csv
```

The other conditions change only `--out-dir` and these flags of step 3:

| Condition | Flags |
|---|---|
| `baseline` | `--readout-lr-mult 1.0 --warmup-steps 1430` |
| `wu_lr_0p25` | `--readout-lr-mult 0.25 --warmup-steps 1430` |
| `wu_lr_4x` | `--readout-lr-mult 4.0 --warmup-steps 1430` |
| `warmup_short` | `--readout-lr-mult 1.0 --warmup-steps 715` |
| `warmup_long` (released, not compared in the paper) | `--readout-lr-mult 1.0 --warmup-steps 5720` |

| Paper item | Output |
|---|---|
| Figure 5 left (`fig:main-recipe-dose-response`) | `results/experiments/ablations/pretraining_recipe_control/lifecycle_multiseed_table.csv` (step 6) |
| Figure 5 right | `${UM_SSD_ROOT}/runs/<condition>/metrics.csv` (validation loss, written by `train_control.py`) |
| Geometry, swap, and frequency class probe numbers in the text | [Appendix I](#i-readout-learning-rate-control) |

Outputs, under `results/experiments/ablations/pretraining_recipe_control/`; every
script also writes `provenance_<script>.json` (git hash, environment, seed,
arguments) next to its outputs:

- `crosscoders/cc_<condition>_d8192_seed<seed>.pt`: one crosscoder per condition and
  seed (`state_dict`, `config`, `steps`, `quality` with EV, mean L0, and dead rate,
  and the `center_scale` preprocessing statistics).
- `lifecycle_seed<seed>/`: `lifecycle_stats.json`, `lifecycle_summary.csv`,
  `median_trajectories.csv`, `peak_step_population.csv`, `reorg_window.csv`,
  `feature_lifecycles_<condition>.csv`, `feature_trajectories.pt`.
- `lifecycle_multiseed_summary.json`, `lifecycle_multiseed_table.csv`: per-seed
  values, means, sample standard deviations, ranges, and per-seed ordering checks.

Inputs:

- `${UM_SSD_ROOT}/pile_slice/pythia_preshuffled.bin`: flat `uint16` GPT-NeoX-20B
  token ids, the first 10B tokens of shard 0 of
  `EleutherAI/pile-standard-pythia-preshuffled`. The trainer needs no other input.
- `${UM_SSD_ROOT}/runs/<condition>/`: `config.json`, `metrics.csv`, and
  `ckpts/step<N>/{model_fp16.pt,metrics.json}` at 16 steps.

Notes:

- Compute (`tab:compute-resources`): one A40 GPU per pretraining run, about 21 hours
  per run. Step 4 takes tens of minutes per seed for the four conditions.
- Each condition keeps 16 checkpoints, at steps 0, 1, 2, 4, ..., 2048, about 2479,
  4096, and 4769. The step near 2479 differs slightly across conditions (2464 to
  2481 for the four compared runs); the analysis scripts glob `ckpts/step*/` and map
  requested steps to the nearest checkpoint on disk.
- `warmup_long` warms up over 5720 steps, longer than the 4769-step budget, so it
  never reaches the peak learning rate; the paper leaves it out. Add it to
  `--conditions` to analyze it.
- The trainer builds Hugging Face `GPTNeoXForCausalLM` (the Pythia architecture) and
  follows Pythia's optimizer recipe (`sec:app-recipe-control-setup`). It is a
  Pythia-style proxy, not a reproduction of the public Pythia run. It resumes from
  `latest.pt` and writes a `COMPLETE` sentinel at the token budget; `--max-hours`
  stops it cleanly before a time limit.
- The crosscoder recipe is the Pythia-160M column of `tab:repro-dictionary-hparams`
  mapped to d_model=256 (d_sae=8192, lambda_1=0.3, 300 epochs, `center_scale`). CPU
  smoke test: add
  `--device cpu --expansion-factor 0.5 --n-epochs 2 --batch-size 4096 --conditions baseline`
  to step 4. `--d-sae` in step 5 must match the crosscoder filenames; `--no-rates`
  skips the firing-rate pass.
- Library code: checkpoint loading and the model rebuild in
  `src/readout/probes/recipe_control_models.py`, the swap kernel in
  `src/readout/probes/readout_swap.py`, and the probe in
  `src/readout/probes/availability_probe.py`, with CPU tests under `tests/`.
  `rc_common.py` holds shared helpers.

### Figure 1: the overview margins

Paper: `fig:overview`. Figure 1 is a TikZ schematic; its two margins are
per-example outputs of `run_swap_grid.py` on Pythia-1B (hidden state from step 1000,
readouts from steps 1000 and 143000, no alignment). Run
[Figure 4 step 1](#figure-4-and-section-43-availability-before-expression) with
`--save-per-example` added to the `run_swap_grid.py` call, and read `margins`,
`margins_native`, `y_plus`, and `y_minus` from
`results/experiments/probes/contrastive_readout_swap/run0_pythia1b/shards/sva__a-none__h1000__s<s>.pt`.
Needs a GPU.

## Appendix results

### D. Dictionary validation

Paper: `sec:app-instrument-validation`. Reconstruction quality of every dictionary,
next-token fidelity of the reconstructed readout, dense and per-snapshot baselines,
held-out checkpoints, and the training of the OLMo-2-7B dictionary.

| Paper item | Experiment: script |
|---|---|
| Figure D.1 (`fig:main-multimodel-validation`) panel A; Table D.1 (`tab:full-inventory`) | `crosscoder_main`: `recompute_quality.py` |
| Figure D.1 panels B and C | `crosscoder_main`: `eval_persnap_fidelity.py` (Pythia-160M), `eval_persnap_1b.py` (Pythia-1B); Pythia-6.9B and OLMo-2-7B are curated ([Known deviations](#known-deviations)) |
| Figures D.2, D.3 (`fig:app-mean-direction-preprocessing`, `fig:app-mean-direction-spectral-gap`) | `dense_readout_diagnostics`: `build_mean_direction_audit.py` |
| Figure D.4 (`fig:app-fidelity-curves`) and the token-stratified check in the text | `crosscoder_main`: `readout_functional_fidelity.py`, `readout_functional_fidelity_strata.py` |
| Figures D.5, D.11 (`fig:app-instrument-pareto`, `fig:app-lambda-sweep`) | `crosscoder_main`: `recompute_quality.py` (`--filter lambda-sweep` for the sweep fits) |
| Figure D.6 (`fig:app-snapshot-fidelity`) | `persnap_sae_baseline`: `train_persnap_saes.py`, then `crosscoder_main`: `recompute_quality.py` and `eval_persnap_fidelity.py` |
| Figures D.7, D.8 (`fig:app-pca-static`, `fig:app-concat-pca`) | `crosscoder_main`: `eval_pca_static.py`, `eval_concat_pca.py` |
| Figure D.9 (`fig:app-dense-reorg-pythia`) | `dense_readout_diagnostics`: `build_dense_reorganization_timing.py` |
| Figure D.10 (`fig:app-heldout-checkpoints-1b`) and the Pythia-160M text | `heldout_checkpoints`: `eval_heldout.py` |
| Figure 2, Figure D.1, Figure E.4 (OLMo-2-7B dictionary) | `crosscoder_olmo` |

#### crosscoder_main

Experiment: `experiments/crosscoders/crosscoder_main/`. Dictionary inventory,
per-checkpoint EV and L0, next-token fidelity of the reconstructed readout, and PCA
baselines.

```bash
# 1. EV, mean L0, and dead rate of every released dictionary and SAE baseline (step 2 reads the per-snapshot SAE rows)
uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/recompute_quality.py \
    --ckpt-root "$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders" --snap-dir "$UM_SSD_ROOT/snapshots" \
    --out-csv experiments/crosscoders/crosscoder_main/derived/appendix_validation/full_inventory.csv
uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/recompute_quality.py \
    --ckpt-root "$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders" --snap-dir "$UM_SSD_ROOT/snapshots" \
    --filter lambda-sweep --out-csv experiments/crosscoders/crosscoder_main/derived/appendix_validation/lambda_sweep.csv
# 2. Per-checkpoint EV and L0 of the Pythia-160M dictionaries and SAE baselines
uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/eval_persnap_fidelity.py
# 3. Per-checkpoint FVU, L0, and dead rate of the Pythia-1B width sweep (d=8,192, 16,384, 24,576)
uv run python experiments/crosscoders/crosscoder_main/scripts/main_1b/eval_persnap_1b.py
# 4. Pre-readout hidden-state caches for the fidelity check, one call per model (Pythia-1B shown)
uv run python scripts/extract/build_hln_cache.py --model pythia-1b --steps all \
    --eval-tokens "$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt" \
    --out-dir "$UM_SSD_ROOT/derived/readout_edit_timing_pythia1b" --save-dtype fp32
# 5. Next-token fidelity per model, then the token-stratified check on Pythia-1B
F=experiments/crosscoders/crosscoder_main/scripts/appendix_validation
uv run python $F/readout_functional_fidelity.py --model pythia-160m --d-sae 8192 --seed 0 --steps all \
    --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing
uv run python $F/readout_functional_fidelity.py --model pythia-1b --d-sae 24576 --seed 0 --steps all \
    --snapshot-dtype bf16 --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing_pythia1b
uv run python $F/readout_functional_fidelity.py --model pythia-6.9b --d-sae 32768 --seed 0 --steps all \
    --snapshot-dtype bf16 --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing_pythia69b \
    --ckpt ${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors
uv run python $F/readout_functional_fidelity_strata.py --model pythia-1b --d-sae 24576 --seed 0 --steps all \
    --max-seqs 64 --snapshot-dtype bf16 --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing_pythia1b
# 6. Dense PCA baselines (both scripts default to --models pythia-160m)
uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/baselines/eval_pca_static.py --models pythia-160m pythia-1b pythia-6.9b olmo-2-7b
uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/baselines/eval_concat_pca.py --models pythia-160m pythia-1b pythia-6.9b olmo-2-7b
```

Step 4 stores fp32 caches for Pythia-160M and 1B and uses the bf16 default for
Pythia-6.9B; each call's `--out-dir` is the `--hln-dir` that step 5 passes for that
model. The fidelity runs read the first 32 sequences (16,352 predictions), the
default of `build_hln_cache.py`; the strata check reads 64, so build the Pythia-1B
caches with `--max-seqs 64`.

Outputs, under `experiments/crosscoders/crosscoder_main/derived/` unless stated:

- `appendix_validation/full_inventory.csv`, `appendix_validation/lambda_sweep.csv`
  (+ `.provenance.json`): canonical and training-formula EV, MSE, L0, and dead rate
  per checkpoint file. `full_inventory.csv` also holds the lambda_1 = 0.3 anchor
  fits, the architecture variants, and the SAE baselines.
- `appendix_validation/persnap_fidelity.csv`: `family, step, ev, fvu, l0` for the
  Pythia-160M d=8,192 and d=24,576 dictionaries, the 16-checkpoint dictionary, the
  per-snapshot SAEs, and the final-snapshot SAE evaluated at every step.
- `main_1b/persnap_1b_d<d_sae>_seed0.csv`: per-checkpoint FVU, L0, and dead rate for
  Pythia-1B.
- `appendix_validation/readout_functional_fidelity/<model>_d<d_sae>_seed0/`:
  `readout_functional_fidelity.csv` (held-out matrix EV, logit R^2, KL to the native
  readout, delta NLL, top-1 and top-5 agreement per checkpoint), per-step `shards/`,
  metadata, and `strata/strata_summary.csv` for the stratified check.
- `appendix_validation/baselines/raw/pca_static_spectra.pt`: per-snapshot singular
  spectra from `eval_pca_static.py`.
- `figures/crosscoder_main/appendix_validation/{pca_static,concat_pca}.{csv,pt}`
  (from the repository root): PCA EV by rank.

Inputs:

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/`: released
  dictionaries and SAE baselines, including
  `pythia-160m/W_U/{cross-snapshot-32,cross-snapshot-16,lambda-sweep,architecture-comparison,final-snapshot-saes}/`,
  `pythia-1b/W_U/cross-snapshot-32/d*/seed0.safetensors`, and
  `pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0{,-sparse}.safetensors`.
- `${UM_SSD_ROOT}/snapshots/<model>/<slug>_step<N>_wu.pt`, the evaluation corpus,
  and the step-4 caches `${UM_SSD_ROOT}/derived/readout_edit_timing*/hLN_step*.pt`.
- `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/`:
  curated Pythia-6.9B and OLMo-2-7B EV, L0, per-checkpoint values, and activation
  rates ([Known deviations](#known-deviations)).

Notes:

- The Pythia-160M fidelity curve uses the d=8,192 seed-0 dictionary, not the
  selected d=24,576 one.
- Memory: the Pythia-1B and 6.9B fidelity runs hold all 32 snapshots and the
  dictionary in memory (about 25 GB for Pythia-1B). The Pythia-160M run takes about
  15 minutes on a laptop CPU.
- The PCA baselines discover checkpoints from the snapshot directory unless
  `--steps` is given. The paper's concatenated PCA uses 45 Pythia-160M, 32 Pythia-1B
  and OLMo-2-7B, and 15 Pythia-6.9B checkpoints.
- `scripts/train/train_arch_sweep.py` trains the architecture variants of Figure
  D.5 and Table D.1.
- Also released, with no paper figure or table mapped to their outputs:
  `build_analysis_table.py`, `eval_per_snap.py` (per-snapshot EV and L0), and
  `eval_endpoint_linear.py`.

#### crosscoder_olmo (OLMo-2-7B dictionary)

Experiment: `experiments/crosscoders/crosscoder_olmo/`. Extracts `W_U` from the 32
stage-1 checkpoints of OLMo-2-1124-7B and trains the d=32,768 trajectory crosscoder
used as the cross-architecture dictionary. Skip it by downloading the released
`olmo-2-7b/W_U/cross-snapshot-32/d32768/seed0.safetensors` ([Downloads](#downloads)).

```bash
# 1. Extract W_U at the 32 checkpoints of the paper's OLMo-2-7B schedule (the script default)
uv run python experiments/crosscoders/crosscoder_olmo/scripts/extract_wu_olmo.py --cache-dir "$WU_CACHE"
# 2. Single-GPU pilot fit (d_sae=16,384, 100 epochs) and its verdict
WU_CACHE="$WU_CACHE" bash experiments/crosscoders/crosscoder_olmo/scripts/run_olmo_pilot.sh
uv run python experiments/crosscoders/crosscoder_olmo/scripts/eval_decision_rules.py \
    --crosscoder /workspace/results/olmo_cross_family/pilot/cc_olmo27b_dsae16384_seed0.pt \
    --wu-cache "$WU_CACHE" --stage pilot \
    --out /workspace/results/olmo_cross_family/pilot/verdict_seed0.json
# 3. Production dictionary: d_sae=32,768, 300 epochs, batch 1024, head-parallel over four GPUs (train_distributed.py)
WU_CACHE="$WU_CACHE" bash experiments/crosscoders/crosscoder_olmo/scripts/run_olmo_production.sh
```

Then compute the OLMo-2-7B lifecycle metrics with the
[Figure 2](#figure-2-the-readout-reorganizes-early) commands.

- The launchers read `WU_CACHE` (default `/workspace/wu_cache_olmo`), `OUT_DIR`
  (default `/workspace/results/olmo_cross_family/{pilot,production}`), and `SEED`
  from the environment; the production launcher also reads `NPROC` (default 4) and
  `PILOT_VERDICT` (default: the verdict path above). The dense diagnostics and the
  PCA baselines read OLMo-2-7B snapshots from
  `${UM_SSD_ROOT}/snapshots/OLMo-2-1124-7B/`.
- The production launcher exits unless the pilot verdict sets `decision.proceed`.
  `eval_decision_rules.py` sets it when the per-feature rate-rotation correlation
  exceeds 0.5 and at least two of three secondary gates pass: a reduced-baseline
  CUSUM permutation test with p < 0.01, reconstruction EV of at least 0.55, and a
  dead-feature rate below 30% at the final snapshot. The paper reports neither these
  gates nor the Gaussian-snapshot control (`run_null_control.sh`).
- Settings of record: `configs/runs/olmo-2-7b_wu_d32768_seed0.yaml`
  ([Known deviations](#known-deviations) for its optimizer).

Inputs: the `allenai/OLMo-2-1124-7B` stage-1 revisions from the Hugging Face Hub
(one fp16 download of about 14 GB per checkpoint; the extractor clears the Hugging
Face cache between checkpoints) and about 50 GB for `$WU_CACHE`.

Outputs: `$WU_CACHE/allenai_OLMo-2-1124-7B_step<N>_wu.pt` (fp32 `W_U`, from
`extract_wu_olmo.py`); `$OUT_DIR/cc_olmo27b_dsae16384_seed0.pt` and
`verdict_seed0.json` for the pilot; `$OUT_DIR/cc_olmo27b_dsae32768_seed0.pt` and
`train_seed0.log` for production. The OLMo-2-7B EV, L0, and activation-rate files the
paper uses are the curated `large_evals/` files.

Hardware: the pilot runs on one A100-80GB in about 3 hours. Production
(`tab:compute-resources`) needs four A100/H100-class GPUs with 40 to 80 GB each, for
4 to 8 hours excluding extraction.

#### dense_readout_diagnostics

Experiment: `experiments/baselines/dense_readout_diagnostics/`. The mean-direction
audit behind the `center_scale` preprocessing (`sec:app-mean-direction-preprocessing`)
and the adjacent-checkpoint movement of the top centered singular subspaces
(`sec:app-linear-baselines`).

```bash
# Mean-direction audit (top two right singular directions by block power iteration)
uv run python experiments/baselines/dense_readout_diagnostics/scripts/build_mean_direction_audit.py
# Adjacent-checkpoint subspace movement (randomized PCA of centered W_U on the 32-checkpoint schedule)
uv run python experiments/baselines/dense_readout_diagnostics/scripts/build_dense_reorganization_timing.py
```

Outputs, under `results/experiments/dense_readout_diagnostics/`:

- `mean_direction_metrics.{csv,pt}`, `mean_direction_provenance.json`: per model and
  step, the mean-row norm ratio, alignment with the raw top singular vector, raw and
  centered spectral gap, stable rank, and mean pairwise row cosine (Figures D.2,
  D.3).
- `dense_reorganization/subspace_metrics.csv`, `sv_stability.csv`,
  `top_svd_vectors.pt`, `provenance.json`: chordal Grassmann distance per adjacent
  pair and subspace rank, and top-vector stability (Figure D.9).
- `figures/dense_readout_diagnostics/dense_reorganization_timing.{csv,pt}` and
  `dense_reorganization_timing_olmo.csv` (from the repository root): the same
  subspace metrics as tables.

Inputs: `${UM_SSD_ROOT}/snapshots/pythia-{160m,1b,6.9b}/EleutherAI_pythia-*_step*_wu.pt`
and `${UM_SSD_ROOT}/snapshots/OLMo-2-1124-7B/allenai_OLMo-2-1124-7B_step*_wu.pt`
(OLMo-2-7B on its own 32-checkpoint schedule). No GPU is needed.

Notes: both scripts default to Pythia-160M, 1B, 6.9B, and OLMo-2-7B (`--models`);
Figure D.9 shows only the three Pythia models, and Figures D.2 and D.3 omit
Pythia-160M. `--copy-to-paper` (off by default) also writes the timing tables into a
local `paper/` tree.

#### persnap_sae_baseline

Experiment: `experiments/baselines/persnap_sae_baseline/`. Trains 32 per-snapshot
JumpReLU SAEs on Pythia-160M `W_U` (d_sae=8,192, one per checkpoint of the
32-step schedule) to compare with the checkpoint-spanning dictionaries.

```bash
# Train the per-snapshot SAEs (defaults: EleutherAI/pythia-160m, d_sae=8,192, lambda_1=0.3, 100 epochs)
uv run python experiments/baselines/persnap_sae_baseline/scripts/train_persnap_saes.py
# Optional: compare the SAEs with the crosscoders checkpoint by checkpoint
uv run python experiments/baselines/persnap_sae_baseline/scripts/persnap_vs_crosscoder.py
uv run python experiments/baselines/persnap_sae_baseline/scripts/build_persnap_comparison.py
uv run python experiments/baselines/persnap_sae_baseline/scripts/compute_cc_vs_persnap_gap.py
```

Figure D.6 takes its per-checkpoint EV and L0 from [`crosscoder_main`](#crosscoder_main)
steps 1 and 2 (`persnap_fidelity.csv`), which read the released per-snapshot SAEs.
`train_persnap_saes.py` writes retrained SAEs to
`${UM_SSD_ROOT}/wu_crosscoder/per_snap_sae/wu_sae_dsae8192_step<step>.pt`, outside
the release tree those steps read.

Inputs: `${UM_SSD_ROOT}/snapshots/pythia-160m/EleutherAI_pythia-160m_step*_wu.pt`;
`${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d{8192,24576}/seed0.safetensors`;
the released SAEs in
`${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/per-snapshot-saes/d8192/`
(default of `compute_cc_vs_persnap_gap.py`); and
`${UM_SSD_ROOT}/derived/rates/wu-d8192-persnap/train.log`, the SAE training log that
`build_persnap_comparison.py` parses for EV and L0 (`--sae-log`).

Outputs of the optional comparisons: per-snapshot recovery `R = 1 - MSE/Var` and
mean L0 on stdout (`persnap_vs_crosscoder.py`);
`figures/persnap_sae_baseline/crosscoder_vs_persnap.json`, the per-snapshot SAE EV
and L0 and the per-snapshot EV of the d=24,576 seed-0 crosscoder under the key
`crosscoder_t46_d24576_seed0` (`build_persnap_comparison.py`); and
`figures/persnap_sae_baseline/crosscoder_vs_persnap_gap.txt`, per-snapshot
crosscoder EV, SAE EV, and their gap with mean, max, and min
(`compute_cc_vs_persnap_gap.py`, `--out-dir`).

#### heldout_checkpoints

Experiment: `experiments/ablations/heldout_checkpoints/`. Reconstructs `W_U` at
eight Pythia checkpoints outside the anchor schedule (steps 10k, 23k, 33k, 43k,
73k, 93k, 113k, and 123k) with encoder and decoder heads interpolated from the
neighboring anchors, and compares them with nearest-head, raw-interpolation,
endpoint, and rank-50 PCA references and the in-sample ceiling
(`sec:app-heldout-checkpoints`).

```bash
# Pythia-1B dictionary (Figure D.10)
uv run python experiments/ablations/heldout_checkpoints/scripts/eval_heldout.py \
    --crosscoder ${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors \
    --snapshot-dir ${UM_SSD_ROOT}/snapshots/pythia-1b \
    --out-dir results/experiments/ablations/heldout_checkpoints/pythia-1b \
    --model-slug EleutherAI_pythia-1b
# Pythia-160M dictionary (text)
uv run python experiments/ablations/heldout_checkpoints/scripts/eval_heldout.py \
    --crosscoder ${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d8192/seed0.safetensors \
    --snapshot-dir ${UM_SSD_ROOT}/snapshots/pythia-160m \
    --out-dir results/experiments/ablations/heldout_checkpoints/pythia-160m \
    --model-slug EleutherAI_pythia-160m
```

Inputs: the two dictionaries in the commands, and `${UM_SSD_ROOT}/snapshots/pythia-{1b,160m}/`
holding the 32 anchor snapshots plus the eight held-out steps (the script does not
download missing snapshots).

Outputs, under each `--out-dir`: `heldout_ev.csv` (EV per held-out checkpoint for
the interpolated heads and each reference, with the in-sample EV of the neighboring
anchors) and `heldout_ev.pt` (the same rows plus `trained_steps`, `held_out`, ceiling
EVs, residuals, `pca_rank`, and the source crosscoder path). `--pca-rank` (default
50) sets the PCA rank, and `--seed` seeds its randomized fit.

### E. Lifecycle profiles and reorganization timing

Paper: `sec:app-lifecycle-diagnostics`, with `sec:app-temporal-localization`,
`sec:app-lifecycle-stability`, and `sec:app-olmo-checkpoint-window-control`.

| Paper item | Source |
|---|---|
| Figure E.1 (`fig:app-lifecycle-profile-composition`) | [Figure 2](#figure-2-the-readout-reorganizes-early) step 2: `selected_lifecycle_profile_composition_refined_{summary,features}.csv`, `selected_lifecycle_profile_composition_refined.pt` |
| Table E.1 (`tab:lifecycle-profile-rules`) | typeset by hand; the rules are `classify_profiles_refined` in `src/readout/dynamics/lifecycle.py` |
| Figure E.2 (`fig:app-selected-population-lifecycle-diagnostics`) and the window-peak share and mean rho_f of features that peak by step 128 | Figure 2 step 3: `selected_population_lifecycle_diagnostics_{peak_counts,norm_mass,window_peaks,early_peak_decay}.csv`, `selected_population_lifecycle_diagnostics.pt` |
| Table E.2, Figure E.4 (`tab:app-reorganization-window-stats`, `fig:app-reorganization-window-metrics`) | Figure 2 step 4 |
| Table E.3 (`tab:app-lifecycle-stability`) | `lifecycle_stability.py`, below |
| Figure E.3 (`fig:app-olmo-window-population`) and the late-start shares in the text | `olmo_matched_checkpoint_window`, below |

The Figure 2 outputs are under
`results/experiments/lifecycle/feature_lifecycle_trajectories/`.
`compute_lifecycle_profile_composition.py` writes the same refined-rule shares under
two stems; the paper uses the `_refined` files.

#### feature_lifecycle_trajectories

Experiment: `experiments/lifecycle/feature_lifecycle_trajectories/`; steps 1 to 4 are
in [Figure 2](#figure-2-the-readout-reorganizes-early). Step 5 repeats the profile
and window analysis for every Pythia dictionary fit; its `validate` stage needs
steps 2 and 4.

```bash
uv run python experiments/lifecycle/feature_lifecycle_trajectories/scripts/lifecycle_stability.py --stage all \
    --out-dir results/experiments/lifecycle/feature_lifecycle_trajectories/lifecycle_stability
```

Outputs: `lifecycle_stability/{pythia160m_widths_seeds,pythia160m_lambda_sweep,pythia1b_widths,pythia69b_sparsity,validation}.csv`,
with profile shares, active count, median peak step, selected window, and null p
per fit. `validation.csv` recomputes the four selected dictionaries with the same
code; its `pythia69b_d32768` row is the selected Pythia-6.9B row (lambda_1 = 0.6) of
Table E.3.

Runtime: about 25 minutes on a 24 GB laptop CPU, dominated by the activation-rate
pass over the 16 Pythia-160M fits; the d=24,576 fits need about 15 GB of RAM.
`--stage {validate,pythia1b,pythia160m,lambda,pythia69b}` runs one block, and
`--device mps` speeds up the encoder pass.

Inputs, with `<release>` standing for
`${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders`. Selected dictionaries
(steps 1 to 4):

- `<release>/pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `<release>/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `<release>/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors`
- `<release>/olmo-2-7b/W_U/cross-snapshot-32/d32768/seed0.safetensors`
- `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-{1b_d24576,160m_d24576}_seed0.pt`
- `${UM_SSD_ROOT}/derived/rates/wu-d24576-multiseed/decoder_norms_dsae24576_seed0.npy`,
  `${UM_SSD_ROOT}/derived/rates/wu-d8192-multiseed/decoder_norms_all_seeds.npy`,
  `${UM_SSD_ROOT}/derived/rates/wu-1b-d24576/decoder_norms_dsae24576_seed0.npy`
- `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/{p69b_sparse,olmo27b}_d32768_seed0.pt`:
  Pythia-6.9B and OLMo-2-7B activation rates (curated, not shipped;
  [Known deviations](#known-deviations)).

Other fits (step 5):

- `<release>/pythia-160m/W_U/cross-snapshot-32/d{8192,16384,24576}/seed*.safetensors`
  (+ `.config.json`; seeds 0 to 4 at d=8,192, 0 to 2 at d=24,576, 0 at d=16,384)
- `<release>/pythia-160m/W_U/lambda-sweep/d8192/lam{0p40,1p00,1p20,1p35,1p80}_seed*.safetensors`
  (+ `.config.json`; seeds 0 to 2 at lambda_1 = 1.35)
- `<release>/pythia-1b/W_U/cross-snapshot-32/d{8192,16384,24576}/seed0.safetensors`
- `<release>/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0.safetensors`
- `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-1b_d{8192,16384,24576}_seed0.pt`,
  `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-6.9b_d32768_seed0.pt`
- `${UM_SSD_ROOT}/snapshots/pythia-160m/EleutherAI_pythia-160m_step*_wu.pt`

The decoder-geometry caches are derived from the released checkpoints on first run
(`lifecycle_common.py`). `lifecycle_stability.py` derives the geometry of the other
fits from their released `W_D`, and their activation rates with
`readout.crosscoder.extract_rates.compute_rates_canonical` from the checkpoint and
the Pythia-160M `W_U` snapshots; Pythia-1B and 6.9B norms and rates come from the
aggregates.

#### olmo_matched_checkpoint_window

Experiment: `experiments/ablations/olmo_matched_checkpoint_window/`. Tests whether
the near-absence of early-decaying features in OLMo-2-7B follows from its release
having no checkpoints before step 150: trains a Pythia-1B `W_U` crosscoder
(d_sae=24,576) on a 32-checkpoint schedule that starts at step 256, and compares its
lifecycle profiles, peak steps, and activation rates with the selected Pythia-1B
and OLMo-2-7B dictionaries.

The late-start schedule is

```text
256, 512, 1000, 2000, 3000, 4000, 5000, 6000,
7000, 8000, 9000, 10000, 13000, 14000, 21000, 23000,
27000, 33000, 34000, 43000, 47000, 53000, 61000, 73000,
83000, 89000, 93000, 102000, 113000, 123000, 130000, 143000
```

1. Train the late-start dictionary, or skip this step and use the released
   `pythia-1b/W_U/cross-snapshot-32-matched-window/d24576/seed0.safetensors`. The
   script defaults match `configs/runs/pythia-1b_wu_d24576_seed0_olmo-matched-window.yaml`.
   ```bash
   uv run python experiments/ablations/olmo_matched_checkpoint_window/scripts/train_late_start.py \
       --d-sae 24576 --n-epochs 300 --device cuda
   ```
2. Compute the per-checkpoint firing rates of the late-start dictionary and save them
   as `results/experiments/olmo_matched_checkpoint_window/pythia1b_wu_late_start32_dsae24576_seed0_rates.pt`,
   a dict with `rates` `(K, D)`, optional `norms`, and `steps`. No release script
   writes this file ([Known deviations](#known-deviations)).
3. Run Figure 2 step 1 (`compute_normalized_trajectories.py`), which writes the
   decoder-norm caches read here.
4. Compute the comparison:
   ```bash
   uv run python experiments/ablations/olmo_matched_checkpoint_window/scripts/compute_late_start_vs_olmo.py
   ```

Outputs, under `figures/olmo_matched_checkpoint_window/`:
`olmo_matched_checkpoint_window_population_lifecycle_diagnostics_{peak_counts,norm_mass}.csv`
and `olmo_matched_checkpoint_window_population_lifecycle_diagnostics.pt` (Figure
E.3), `late_start_profile_fractions.csv` (early-decaying shares in the text),
`comparison_summary.csv` (activation-rate decayer and early-dominant shares), and
sidecars the paper does not use. Step 1 writes the dictionary to
`results/experiments/olmo_matched_checkpoint_window/pythia1b_wu_late_start32_dsae24576_seed0.pt`
(default `--output`).

Inputs: `${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step*_wu.pt` at the
late-start steps (training only); the step-2 rates file;
`experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/olmo27b_d32768_seed0.pt`
(OLMo-2-7B firing rates, curated);
`figures/feature_lifecycle_trajectories/section52_lifecycle/cache/{olmo2_7b_d32768,pythia1b_d24576}_decoder_norms.npy`
from step 3.

Hardware: step 1 uses CUDA; without it, use `--device mps` and try a smaller pilot
first, for example `--model EleutherAI/pythia-160m`.

### F. WordNet supersense probes

Paper: `sec:app-vocabulary-family-probes`. Figures F.1 and F.2 and Table F.1
(`fig:app-wordnet-supersense-probes-160m-inventory`,
`fig:app-wordnet-supersense-probes-1b`, `tab:app-wordnet-supersense-1b`) come from the
[Figure 3](#figure-3-lexical-organization-in-the-readout-rows) probe runs. The
lemma-grouped splits (`sec:app-wordnet-lemma-split`) rerun the probe with
`StratifiedGroupKFold` over `decode(id).strip().lower()` for the outer split and
the inner selection of C:

```bash
P=experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py
uv run python $P --model pythia-160m --split lemma
uv run python $P --model pythia-1b --split lemma
# Split noise on Pythia-160M: seeds 1 to 4, six checkpoints, both splits
for s in 1 2 3 4; do for sp in row lemma; do
  uv run python $P --model pythia-160m --split $sp --seed $s --steps 0,256,512,1000,2000,143000
done; done
```

Output directories gain a `_lemma` suffix, and `_seed<N>` for a non-zero seed
(`derived/wordnet_supersense_{160m,1b}_lemma/` under the experiment). The lemma-split
numbers compare `final_acc`, `emergence_step`, and `max_gain_pair` in
`wordnet_supersense_probe_summary_<suffix>.csv` between the `_lemma` and row-split
directories. The leakage count is the sum of `n_test_pos_with_train_sibling` over
the sum of `n_test_positive` in the row split's
`wordnet_supersense_split_stats_<suffix>.csv`. The matched-control nulls were not
rerun under the lemma split.

### G. Availability and expression across task families

Paper: `sec:app-availability-expression`.

| Paper item | Source |
|---|---|
| Figure G.1 (`fig:app-contrastive-readout-lag`) | [Figure 4](#figure-4-and-section-43-availability-before-expression) step 2: `run_pythia-6.9b_probe_trajectory/summary.csv` and `controlled/probe_trajectory_69b.csv`, families `sva` and `ioi_role_balanced`; panel B compares the probe with the strongest prompt-token or Gaussian-feature control |
| Figures G.2, G.3, Tables G.2 to G.4 (`fig:app-ae-capacity`, `fig:app-ae-grid`, `tab:app-ae-binary`, `tab:app-ae-openvocab`, `tab:app-ae-early`) | Figure 4 step 3, `availability/run_pythia-6.9b/probes/probe_summary.csv`; the Pythia-1B columns of Table G.2 from the two runs below |
| Table G.1 (`tab:app-task-construction`) | typeset by hand; the templates and word lists are implemented in `readout.probes.availability_tasks` and its sibling modules |

The Pythia-1B columns come from two runs with the settings used for the paper. The
agreement run builds the subject-RC set with the attractor always
number-mismatched (`--subject-rc-attractor mismatched`); the breadth run uses
stratified folds for the numeric families and no probe labels for BLiMP
(`--stratified-prefixes numeric --unprobed-prefixes blimp`). Omitting these three
flags applies the Pythia-6.9B protocol at Pythia-1B. With `STEPS` and `OUT` from
Figure 4 step 3:

```bash
AGREE="sva sva_across_pp sva_subject_rc sva_object_rc ioi_role_balanced copy_control fixed_token_control"
BREADTH="blimp_det_noun_agreement blimp_anaphor_agreement blimp_npi fv_antonym fv_country_capital fv_en_fr \
  fv_present_past induction numeric_gt_v2 numeric_lt_v2 numeric_tf_v2 relational_facts relational_facts_balanced"
uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
    --task-set availability --subject-rc-attractor mismatched --model EleutherAI/pythia-1b --families $AGREE \
    --n-max-per-family 2000 --rng-seed 0 --no-norm-match --out-dir $OUT/datasets/pythia-1b
uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
    --task-set availability --model EleutherAI/pythia-1b --families $BREADTH \
    --n-max-per-family 2000 --rng-seed 0 --no-norm-match --out-dir $OUT/datasets/pythia-1b_breadth
for RUN in pythia-1b:"$AGREE" pythia-1b_breadth:"$BREADTH"; do
  NAME=${RUN%%:*}; F=${RUN#*:}
  uv run python experiments/probes/contrastive_readout_swap/scripts/extract_hidden_dense.py \
      --hf-model EleutherAI/pythia-1b --datasets-dir $OUT/datasets/$NAME --families $F \
      --h-steps $STEPS --out-dir $OUT/run_$NAME/hidden --device cuda --dtype fp32 --batch-size 16
done
uv run python experiments/probes/contrastive_readout_swap/scripts/run_availability_probes.py \
    --model EleutherAI/pythia-1b --hidden-dir $OUT/run_pythia-1b/hidden --datasets-dir $OUT/datasets/pythia-1b \
    --families $AGREE --swept-steps $STEPS --group-split --device cuda --out-dir $OUT/run_pythia-1b/probes
uv run python experiments/probes/contrastive_readout_swap/scripts/run_availability_probes.py \
    --model EleutherAI/pythia-1b --hidden-dir $OUT/run_pythia-1b_breadth/hidden \
    --datasets-dir $OUT/datasets/pythia-1b_breadth --families $BREADTH --swept-steps $STEPS --group-split \
    --stratified-prefixes numeric --unprobed-prefixes blimp --device cuda --out-dir $OUT/run_pythia-1b_breadth/probes
```

Outputs: `availability/run_pythia-1b/probes/probe_summary.csv` and
`availability/run_pythia-1b_breadth/probes/probe_summary.csv`, with the hidden states
under `availability/run_<run>/hidden/`. Inputs, folds, and the definitions of native
and swept readouts are as in Figure 4. Deviation: CPU nulls and ranks can differ
slightly ([Known deviations](#known-deviations)).

### H. Readout swaps and localization

Paper: `sec:app-readout-swap`.

| Paper item | Script | Output |
|---|---|---|
| Figure H.1 (`fig:app-readout-swap-nll-kl-grid`); late-checkpoint anomaly (`sec:app-late-checkpoint-anomaly`) | `temporal_patch_grid.py` | `temporal_patch_grid/pythia-1b_seed0/summary_global.csv` |
| Pythia-160M replication (text, `sec:app-readout-swap`) | `temporal_patch_grid.py --model pythia-160m` | `temporal_patch_grid/pythia-160m_seed0/summary_global.csv` |
| Figure H.4 (`fig:app-aligned-readout-swap`) | `run_aligned_swap_grid.py` | `aligned_swap_grid/summary.csv` |
| Table H.1 (`tab:app-readout-swap-languages`) | `run_language_stratified_swap.py` | `summary.csv` (gain interval at `s*`), `argmin_summary.csv` (`s*`, gain, argmin probability), `manifest.json` (target counts) |
| Figure H.2 (`fig:app-readout-swap-english-aggregate`) | `run_language_stratified_swap.py` | `summary.csv` (`language` in {`all`, `en`}), `english_only_summary.csv` |
| Figure H.3 (`fig:app-readout-swap-languages`) | `run_language_stratified_swap.py` | `summary.csv` (per-language rows) |
| Table H.2 (`tab:app-contrastive-localisation-ledger`) | `run_feature_attribution.py` ([Section 4.4](#section-44-a-few-readout-directions-carry-task-margins)), `run_converse_intervention.py` | see [contrastive_task_feature_rescue](#contrastive_task_feature_rescue) |
| Terminal-readout alignment (text, `sec:app-terminal-swap-alignment`) | [Figure 4](#figure-4-and-section-43-availability-before-expression) step 1 | `run0_pythia1b/summary.csv` |
| Figures H.5, H.6 (`fig:app-sparse-feature-causal-curves`, `fig:app-sparse-feature-causal-specificity-k32`) | `run_1b_pilot.py`, `run_specificity_from_pilots.py`, `aggregate_causal_tests.py` | see [sparse_feature_causal_tests](#sparse_feature_causal_tests) |

The first three rows are under
`results/experiments/causal/temporal_localization_patching/`, and the three
per-language rows under
`results/experiments/causal/language_stratified_readout_swap/pythia-1b_seed0/`.

#### temporal_localization_patching and language_stratified_readout_swap

Experiments: `experiments/causal/temporal_localization_patching/` and
`experiments/causal/language_stratified_readout_swap/`. Pythia-1B hidden states from
eight checkpoints are held fixed and decoded through each of the 32 readouts
`W_U^(s)`, recording next-token NLL and the KL to the same-checkpoint readout on the
nine-language held-out corpus. The per-language run recomputes this raw grid for
each language (English, Russian, Chinese, Japanese, Thai, Arabic, Hindi, Korean,
Bengali) and for all targets together, with paired block-bootstrap intervals and the
bootstrap probability that each readout is the NLL minimum. The aligned grid repeats
the swap after aligning the swapped readout's column mean, scale, row norms, or
orthogonal frame to the native one.

```bash
# Optional: rebuild the released evaluation corpus token for token from the 2023-11-01 Wikipedia dumps (network)
uv run python experiments/causal/temporal_localization_patching/scripts/build_eval_corpus_pythia.py \
    --output results/experiments/causal/temporal_localization_patching/eval_tokens_pythia.pt
# Pythia-1B swap grid (Figure H.1); also the parity reference of the per-language run
uv run python experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py \
    --model pythia-1b --seed 0 --h-eval-steps 256 512 1000 2000 3000 14000 47000 143000 \
    --out-dir results/experiments/causal/temporal_localization_patching/temporal_patch_grid/pythia-1b_seed0
# Pythia-160M replication (the default --h-eval-steps are 256 512 1000 2000 143000)
uv run python experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py \
    --model pythia-160m --seed 0 \
    --out-dir results/experiments/causal/temporal_localization_patching/temporal_patch_grid/pythia-160m_seed0
# Per-language grid (all 32 readouts, raw swap, 173,229 targets; resumable, completed cells are skipped)
uv run python experiments/causal/language_stratified_readout_swap/scripts/run_language_stratified_swap.py \
    --model pythia-1b --seed 0 --h-eval-steps 256 512 1000 2000 3000 14000 47000 143000 \
    --block-tokens 256 --bootstrap-reps 2000 --bootstrap-seed 0 \
    --out-dir results/experiments/causal/language_stratified_readout_swap/pythia-1b_seed0 \
    --reference-summary results/experiments/causal/temporal_localization_patching/temporal_patch_grid/pythia-1b_seed0/summary_global.csv
# Aligned swaps (none, mean, scale, row_norm, procrustes; default hidden-state checkpoints 256 512 1000 2000 143000)
uv run python experiments/causal/temporal_localization_patching/scripts/run_aligned_swap_grid.py \
    --model pythia-1b --out-dir results/experiments/causal/temporal_localization_patching/aligned_swap_grid
```

Inputs:

- The evaluation corpus (`--eval-tokens`): token ids with per-token `languages`
  labels.
- `${UM_SSD_ROOT}/snapshots/pythia-{1b,160m}/EleutherAI_pythia-*_step*_wu.pt`. Without
  them, the per-language run takes `--readout-cache-dir` to extract each `W_U` from
  its Hugging Face revision.
- Hidden states after the final LayerNorm, cached per step as
  `${UM_SSD_ROOT}/readout_edit_timing_pythia1b/hLN_step<t>.pt` (Pythia-1B) and
  `${UM_SSD_ROOT}/readout_edit_timing/hLN_step<t>.pt` (Pythia-160M) and shared by
  the three Pythia-1B scripts. On a cache miss the scripts forward the Hugging Face
  checkpoint and write the cache; a cache built from a different corpus is rejected.
  `run_aligned_swap_grid.py` scores every sequence held in the caches it finds; the
  paper scored 32 ([Known deviations](#known-deviations)).

Outputs, under each `--out-dir`:

- `temporal_patch_grid.py`: `manifest.json`, `summary_global.csv` (one row per
  `(h_t, W_U_s)`: NLL, top-1, top-5, entropy, KL to native), `summary_concept.csv`,
  `raw.pt`.
- `run_aligned_swap_grid.py`: `manifest.json`, per-cell JSON shards, `summary.csv`
  (one row per alignment, `h_t`, `W_U_s`).
- `run_language_stratified_swap.py`: `manifest.json` (configuration, corpus sha1,
  per-language target counts, provenance), `layout.pt` (bootstrap block of every
  target), `native/h<t>.pt` and `cells/h<t>_s<s>.pt` (per-block NLL sums in float64,
  the sufficient statistics for every estimate), `shards/h<t>_s<s>.json` (resume
  markers), `summary.csv` (one row per `(language, t, s)` with `nll`, `nll_native`,
  `delta_nll`, and the 95% paired interval), `english_only_summary.csv` (the `en`
  rows), `argmin_summary.csv` (per `(t, language)`: best readout, its gain, and the
  argmin probabilities of the best and native readouts), `argmin_probabilities.csv`,
  and `reference_parity.json` (agreement of the `all` stratum with the reference
  grid within 1e-5 nats, or `not_checked` without `--reference-summary`).
- `build_eval_corpus_pythia.py`: the token file (`ids`, `scripts`, `languages`,
  counts) and, with `--text-output`, the raw text per language.

Notes:

- The plotted and tabulated per-language gain is `-delta_nll` (native minus swapped
  NLL), with interval `[-delta_nll_ci_high, -delta_nll_ci_low]`. Intervals come from
  2,000 paired resamples of consecutive 256-token blocks within each stratum (seed
  0).
- A target enters a per-language estimate only when its context token has the same
  language label, which drops the 8 targets that straddle a slice boundary; the
  `all` stratum keeps every target and matches the aggregate grid. The
  `summary.csv` of `run_aligned_swap_grid.py` is not a valid parity reference,
  because the paper's alignment check scored only the first 16,352 targets.
- Hardware: the paper's per-language run took 5.5 hours on one A40 GPU. Without the
  snapshot mirror, add `--readout-cache-dir <scratch>/readouts`, and
  `--hln-cache-dir <scratch>/hln --delete-hln-cache-after-step` to keep the 1.8 GB
  per-checkpoint hidden-state caches on local scratch.
- The swap library is `src/readout/dynamics/temporal_patch.py`, shared with
  `sparse_feature_causal_tests`; it runs from public assets only.
  `summary_concept.csv` and `temporal_patch_metrics.py` (a CLI over the same
  library) back no paper result.

#### contrastive_task_feature_rescue

Table H.2 takes its *orig.*, *recon*, *ablate*, *preserve-only*, and *sign* columns
from the [Section 4.4](#section-44-a-few-readout-directions-carry-task-margins) run;
for greater-than completion, IOI, and relational facts, *orig.* is on the full
inventory, the `none`-alignment cell (h=1000, s=143000) of
`results/experiments/probes/contrastive_readout_swap/run0_pythia1b/summary.csv`. The
*hidden projection* columns project the selected decoder directions out of the
step-1000 hidden states and re-score them under the native readout `W_U^(1000)`, on
the same held-out half:

```bash
uv run python experiments/causal/contrastive_task_feature_rescue/scripts/run_converse_intervention.py \
    --model pythia-1b \
    --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
    --hidden-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden \
    --attribution-dir results/experiments/causal/contrastive_task_feature_rescue/run0_pythia1b_s143000_h1000_pos \
    --snapshot-step 143000 --h-step 1000 --K 8 16 32 64 128 256 \
    --out-dir results/experiments/causal/contrastive_task_feature_rescue/converse_pythia1b_s143000_h1000_pos
```

Outputs, under its `--out-dir`: `<family>__h<h>__s<s>.json` with `baseline` margins
under the native `W_U` at the h and s steps and, per K, projected margins, logit
shifts (`dy_plus`, `dy_minus`, `dmargin`), and `h_residual_frac` (mean of
norm(h_proj)/norm(h)); the table reads `baseline.native_t.accuracy` and
`per_K["8"].summary_proj_native_t.accuracy`. `manifest.json` records the inputs,
per-family split provenance, and git commit. The script reads the split parameters
from each attribution shard and re-derives the same per-family permutation, so both
scripts score the same held-out items. Deviation: the held-out half differs from the
paper's ([Known deviations](#known-deviations)).

#### sparse_feature_causal_tests

Experiment: `experiments/causal/sparse_feature_causal_tests/`
(`sec:app-sparse-feature-causal-tests`). For Pythia-1B and four token groups
(non-Latin script tokens, punctuation, digits, English function words), ablates the
top-k positive-attribution features of the released d=24,576 `W_U` crosscoder from
the reconstructed readout rows, under fixed hidden states, at the step-1000 and the
step-143k readout, and compares the effects with five feature sets matched on
decoder-norm and firing-rate deciles.

```bash
# 1. One selection-and-ablation pass per token group and readout (8 runs)
for c in non_latin_scripts punctuation digits function_words; do
  uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_1b_pilot.py \
    --concept $c --snapshot-step 1000 --rank-h-step 1000 --eval-h-steps 512 1000 \
    --score-mode positive --out-dir results/experiments/sparse_feature_causal_tests/1b_${c}_step1000_positive
  uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_1b_pilot.py \
    --concept $c --snapshot-step 143000 --rank-h-step 143000 --eval-h-steps 143000 \
    --score-mode positive --out-dir results/experiments/sparse_feature_causal_tests/1b_${c}_step143000_positive
done
# 2. Reuse each group's selected features on every group's metric
uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_specificity_from_pilots.py --snapshot-step 1000 --h-step 1000
uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_specificity_from_pilots.py --snapshot-step 143000 --h-step 143000
# 3. Aggregate the tables behind both figures (CPU, seconds)
uv run python experiments/causal/sparse_feature_causal_tests/scripts/aggregate_causal_tests.py
```

Inputs:
`${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`
and `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-1b_d24576_seed0.pt`
(per-feature firing-rate and norm aggregates). Steps 1 and 2 need a GPU.

Outputs, under `results/experiments/sparse_feature_causal_tests/`:

- `1b_<group>_step<step>_positive/`: per-run `summary.csv`, `raw.pt`, and manifest.
- `specificity_step<step>_h<step>/`: `specificity.csv` and, after step 3,
  `control_corrected_drop_k<k>.csv` (source-ablation drop minus mean matched-control
  drop) and `diagonal_summary.csv`. Figure H.6 reads `control_corrected_drop_k32.csv`.
- `family_expansion_summary.csv`: top-feature drop, matched-control drop mean, min,
  and max, and keep-only recoveries per group, readout, and k. Figure H.5 plots the
  top-feature drop minus the mean matched drop, with the range over the five matched
  sets.

### I. Readout learning rate control

Paper: `sec:app-recipe-control`. Experiment:
`experiments/ablations/pretraining_recipe_control/`; the training, crosscoder, and
lifecycle steps are in [Figure 5](#figure-5-readout-learning-rate).

| Paper item | Source, under `results/experiments/ablations/pretraining_recipe_control/` |
|---|---|
| Figure I.1 (`fig:app-recipe-control-trajectories`) | Figure 5 step 5, seed 0: `lifecycle_seed0/feature_trajectories.pt` |
| Table I.1 (`tab:app-recipe-control-summary`) | Figure 5 step 5, seed 0: `lifecycle_seed0/lifecycle_summary.csv` (`lifecycle_stats.json` holds the same values) |
| Figure I.2 (`fig:app-recipe-control-reorg`) | Figure 5 step 5, seed 0: `lifecycle_seed0/reorg_window.csv` |
| Figure I.3 (`fig:app-recipe-control-geometry`) | `analyze_readout_geometry.py`: `readout_geometry_pythia.csv` |
| Figure I.4 (`fig:app-recipe-control-basin`) | `run_trajectory_swap.py`: `trajectory/trajectory_swap_all.csv` (argmin-NLL readout per hidden-state checkpoint) |
| Figure I.5 (`fig:app-recipe-control-explag`) | `run_expression_lag.py`: `expression_lag/expression_lag.csv` |

The last three also give the geometry, swap, and frequency class probe numbers of
Section 4.5.

```bash
# W_U geometry across conditions and checkpoints (CPU)
uv run python experiments/ablations/pretraining_recipe_control/scripts/analyze_readout_geometry.py \
    --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline warmup_short wu_lr_0p25 wu_lr_4x \
    --out-csv results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv
# The 2M-token held-out slice that directly follows the trained 10B tokens (network, 4 MB)
uv run python experiments/ablations/pretraining_recipe_control/scripts/fetch_heldout_slice.py \
    --out ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin --start-token 10e9 --eval-tokens 2e6
# Within-run readout swap grid (one GPU, minutes; resumable per cell)
uv run python experiments/ablations/pretraining_recipe_control/scripts/run_trajectory_swap.py \
    --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short \
    --h-steps 256 512 1024 2048 4769 --eval-bin ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin \
    --eval-tokens 120000 --out-dir results/experiments/ablations/pretraining_recipe_control/trajectory
# Frequency class probe against the native and best readouts (one GPU or CPU)
uv run python experiments/ablations/pretraining_recipe_control/scripts/run_expression_lag.py \
    --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short \
    --h-steps 256 512 1024 2048 4769 --eval-bin ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin \
    --eval-tokens 400000 --max-positions 20000 \
    --out-dir results/experiments/ablations/pretraining_recipe_control/expression_lag
```

Outputs, under `results/experiments/ablations/pretraining_recipe_control/`, each with
its `provenance_<script>.json`:

- `readout_geometry_pythia.csv`: per-checkpoint `W_U` row-norm statistics, singular
  spectrum (`top1_frac`, effective and stable rank), and final LayerNorm gain.
- `trajectory/trajectory_swap_<condition>.csv`, `trajectory/trajectory_swap_all.csv`
  (+ `shards/`): NLL, delta NLL, KL to native, top-1 agreement, and centered-logit
  R^2 per condition, alignment, `h_step`, and `s_step`.
- `expression_lag/expression_lag.csv` (+ `shards/`): per condition and `h_step`,
  probe accuracy with label-shuffle and random-label nulls, `native_readout_acc`,
  `best_readout_acc`, `availability_gap`, and `readout_rescue`.

Inputs: the checkpoints under `${UM_SSD_ROOT}/runs/<condition>/` and the held-out
slice `${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin` (+ `.sha256`), which no
condition trained on. The swap grid and the probe run on CPU at reduced size, for
example `--eval-tokens 8192 --seq-len 256 --device cpu`.

## Known deviations

- **Section 4.4 and Table H.2.** The release seeds the localization split and the
  matched controls separately for each task, and builds the simple SVA items in
  interleaved order. The paper's run drew both from one sequential generator, so
  the release scores a different held-out half and its values differ from the
  paper's. (The swap-grid accuracies of Figure 4 step 1 do not depend on item
  order.)
- **Figure 4, left panel.** The native and best swept series have no surviving run
  output. `paper_figures/` ships them as values transcribed from the original
  figure; the probe series is run output.
- **Figures 4 (left) and G.1, best swept readout.** No release script computes the
  best swept readout of the trajectories (`B_env`, the maximum over readouts per
  task and hidden-state checkpoint, excluding the native and terminal readouts;
  `sec:app-evaluation-details`). Take the maximum accuracy over the
  `alignment = none` rows of `run_pythia-6.9b_probe_trajectory/summary.csv` whose
  `s_step` is neither `h_step` nor 143000.
- **Figure D.1, Table D.1, and the curated `large_evals/` files.** The Pythia-6.9B
  and OLMo-2-7B rows and their per-checkpoint curves come from evaluation files that
  no release script regenerates; `recompute_quality.py` recomputes their aggregate EV
  and L0. The same curated files, under
  `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/`
  and not shipped with the repository, supply the Pythia-6.9B and OLMo-2-7B
  activation rates that `find_reorganization_steps.py` (Figure E.4, Table E.2, and
  the windows quoted in Section 4.1) and `compute_late_start_vs_olmo.py` (Figure E.3)
  read.
- **Figure E.3.** No release script writes the late-start firing-rate file that
  `compute_late_start_vs_olmo.py` reads. `scripts/extract/extract_rates.py` computes
  the rates but saves them under `rates_per_seed[<checkpoint stem>]`, while the
  comparison reads a top-level `rates` key, so the tensor has to be stored under
  `rates` together with `steps`.
- **Figure H.4.** The paper's aligned swap grid scores the first 32 sequences
  (16,352 targets). Build 32-sequence caches with
  `scripts/extract/build_hln_cache.py --max-seqs 32` before running
  `run_aligned_swap_grid.py`; without them the script scores the full corpus.
  `run_aligned_swap_grid.py` has no cache-directory option: it reads
  `${UM_SSD_ROOT}/readout_edit_timing_pythia1b/`, where `temporal_patch_grid.py` and
  `run_language_stratified_swap.py` keep their full-corpus caches, and
  `build_hln_cache.py` skips any step whose `hLN_step<t>.pt` already exists there.
- **Appendix G on CPU.** The label-shuffle and random-label nulls can differ from
  the published values by about one item, and the full-vocabulary mean ranks by
  near-tie flips (rank-1 rates agree).
- **OLMo-2-7B optimizer.** `tab:repro-dictionary-hparams` lists SparseAdam for
  OLMo-2-7B, while `run_olmo_production.sh` passes `--optimizer adam` and the
  settings of record record `adam`; the provenance note in
  `configs/runs/olmo-2-7b_wu_d32768_seed0.yaml` explains the uncertainty.

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
| <a id="tab:app-ae-binary"></a>`tab:app-ae-binary` | contrastive_readout_swap | (produced; see the sections above) |
| <a id="tab:app-ae-early"></a>`tab:app-ae-early` | contrastive_readout_swap | (produced; see the sections above) |
| <a id="tab:app-ae-openvocab"></a>`tab:app-ae-openvocab` | contrastive_readout_swap | (produced; see the sections above) |
| <a id="tab:app-claim-tests"></a>`tab:app-claim-tests` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:app-contrastive-localisation-ledger"></a>`tab:app-contrastive-localisation-ledger` | contrastive_task_feature_rescue | results/experiments/causal/contrastive_task_feature_rescue/run0_pythia1b_s143000_h1000_pos/shards/sva__h1000__s143000.json |
| <a id="tab:app-design-rationale"></a>`tab:app-design-rationale` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:app-lifecycle-stability"></a>`tab:app-lifecycle-stability` | feature_lifecycle_trajectories | (produced; see the sections above) |
| <a id="tab:app-readout-swap-languages"></a>`tab:app-readout-swap-languages` | language_stratified_readout_swap | paper/figures/readout_coordination/language_swap_table.tex |
| <a id="tab:app-recipe-control-summary"></a>`tab:app-recipe-control-summary` | pretraining_recipe_control | results/experiments/ablations/pretraining_recipe_control/lifecycle_seed0/lifecycle_summary.csv |
| <a id="tab:app-reorganization-window-stats"></a>`tab:app-reorganization-window-stats` | feature_lifecycle_trajectories | (produced; see the sections above) |
| <a id="tab:app-task-construction"></a>`tab:app-task-construction` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:app-terminology-glossary"></a>`tab:app-terminology-glossary` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:app-wordnet-supersense-1b"></a>`tab:app-wordnet-supersense-1b` | concept_evolution_validation | experiments/probes/concept_evolution_validation/derived/wordnet_supersense_1b/wordnet_supersense_probe_pos_summary_1b.csv |
| <a id="tab:checkpoint-suite-comparison"></a>`tab:checkpoint-suite-comparison` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:compute-resources"></a>`tab:compute-resources` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:evaluation-controls"></a>`tab:evaluation-controls` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:full-inventory"></a>`tab:full-inventory` | crosscoder_main | experiments/crosscoders/crosscoder_main/derived/appendix_validation/full_inventory.csv |
| <a id="tab:lifecycle-profile-rules"></a>`tab:lifecycle-profile-rules` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:method-comparison"></a>`tab:method-comparison` | (paper LaTeX tree only) | (paper LaTeX tree only) |
| <a id="tab:repro-dictionary-hparams"></a>`tab:repro-dictionary-hparams` | crosscoder_main | (produced; see the sections above) |
| <a id="tab:task-protocol"></a>`tab:task-protocol` | (paper LaTeX tree only) | (paper LaTeX tree only) |
<!-- END GENERATED FIGURE INDEX -->
