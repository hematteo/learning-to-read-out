# Data and external assets

No datasets, model checkpoints, or trained dictionaries are bundled in this
repository — they are large and are all publicly derivable. This file lists
what each stage needs and where to get it.

## Storage root: `UM_SSD_ROOT`

Analysis scripts resolve their inputs under a single environment variable,
`UM_SSD_ROOT` (default `./local_snapshots`), via `src/core/paths.py`. Point
it anywhere before running:

```bash
export UM_SSD_ROOT=/path/to/storage
```

The canonical layout under that root is:

```
$UM_SSD_ROOT/
  snapshots/<model>/<model>_step<N>_wu.pt          # extracted W_U matrices
  hf_release/parameter-trajectory-crosscoders/...  # trained crosscoders (.safetensors)
  derived/aggregates/aggregates_<model>_d<N>_seed<S>.pt
```

The selected **sparse 6.9B** dictionary (λ=0.6) lives in the canonical layout
as `pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors` — the
`-sparse` suffix distinguishes it from the default-λ comparison run
`seed0.safetensors` in the same directory. If you retrain it (settings of
record: `configs/runs/pythia-6.9b_wu_d32768_seed0_sparse.yaml`), place your
checkpoint at that path; the rescue scripts also accept an explicit
checkpoint flag.

## Released artifacts (Hugging Face)

The trained trajectory crosscoders (with `.config.json`/`.md` training sidecars),
the per-model aggregate tensors (`derived/aggregates/`), the activation-rate
sidecars (`derived/rates/`), the attribution artifacts, and the held-out eval
token corpus are distributed at
**https://huggingface.co/matteohe/parameter-trajectory-crosscoders**
(`index.json` there is the machine-readable inventory). Download into
`$UM_SSD_ROOT` to skip retraining:

```bash
hf download matteohe/parameter-trajectory-crosscoders \
    --revision fb7ee860b9257f125ddbac7ff3c793b35fdcce8d \
    --local-dir "$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders"
```

(The pinned revision is the verified release commit; drop `--revision` for the
latest.)

The recipe-control 31M models live separately at
**https://huggingface.co/matteohe/readout-recipe-control**, and pre-extracted
W_U snapshot caches at
**https://huggingface.co/datasets/matteohe/wu-crosscoder-snapshots**.
Independently of the release, every crosscoder is regenerable from scratch:
each retrains from public checkpoints with the settings of record in
`configs/runs/`.

## Source models and checkpoints (Hugging Face, public)

| Asset | Source | Licence |
|---|---|---|
| Pythia 160M / 1B / 6.9B checkpoints + tokenizer | `EleutherAI/pythia-{160m,1b,6.9b}` | Apache-2.0 |
| OLMo-2-1124-7B checkpoints + tokenizer | `allenai/OLMo-2-1124-7B` | Apache-2.0 |
| WordNet 3.0 (supersense probes) | Princeton WordNet | WordNet 3.0 licence |
| Wikipedia slices (readout-swap eval) | Wikimedia 2023-11-01 dumps | CC BY-SA 4.0 / GFDL |
| GPT-NeoX-20B tokenizer (`pretraining_recipe_control` only) | `EleutherAI/gpt-neox-20b` | Apache-2.0 |
| Pile pretraining corpus (`pretraining_recipe_control` only) | `monology/pile-uncopyrighted` (slice tokenized locally), or `EleutherAI/pile-standard-pythia-preshuffled` for the exact Pythia order (byte-range prefix, no full 602 GB download) | MIT (Pile compilation) |

The WordNet 3.0 supersense probes read the corpus through NLTK
(`src/probes/concept_gazetteer.py`). After installing dependencies, download the
corpus once with `python -c "import nltk; nltk.download('wordnet')"` (it lands in
`~/nltk_data/`); the probe experiment raises `LookupError`/`ModuleNotFoundError`
until this is done.

The last two rows are needed **only** to retrain the `pretraining_recipe_control`
ablation from scratch (`experiments/ablations/pretraining_recipe_control/`); none
of the paper figures depend on them. The tokenized slice is a local artifact
(flat `uint16` `.bin`); build it with that experiment's `trainer/tokenize_slice.py`
or its `scripts/fetch_pythia_preshuffled.py`
(both under `experiments/ablations/pretraining_recipe_control/`).

Checkpoint schedules are fixed in `src/core/model_specs.py`
(`DEFAULT_STEPS_32`, the 32-checkpoint Ge et al. schedule). The thesis
reproducibility appendix records the exact step lists.

## Reproducing the inputs

1. **W_U snapshots** (`$UM_SSD_ROOT/snapshots/<model>/`): produced on demand by
   the trajectory trainer. On a cache miss `src.crosscoder.wu_adapter.load_wu_snapshot`
   downloads each Pythia `step{N}` revision, saves its `embed_out.weight`
   unembedding as `{slug}_step{N}_wu.pt`, and reuses it thereafter — so simply
   running the trainer populates them:
   ```bash
   uv run python scripts/train/train_crosscoder.py --model EleutherAI/pythia-160m \
       --output /tmp/cc_160m.pt   # extracts any missing W_U snapshots first
   ```
   CPU is sufficient for 160M; larger models want a GPU. OLMo-2 uses a different
   revision format and `lm_head`, so pre-extract it with
   `experiments/crosscoders/crosscoder_olmo/scripts/extract_wu_olmo.py`.

   (Note: `scripts/extract/extract_we_pythia.py` extracts the **W_E** input
   embeddings for the `crosscoder_we` companion into a *separate* cache — it is
   not the W_U path.)

2. **Trained crosscoders** (`$UM_SSD_ROOT/hf_release/...`): produced by
   `scripts/train/train_crosscoder.py` (see `REPRODUCE.md`). The 6.9B and
   OLMo runs use 4-GPU head-parallel training; 160M trains on a single GPU and
   is the cheapest end-to-end path.

3. **Aggregates** (`$UM_SSD_ROOT/derived/aggregates/...`): derived metric
   tensors built by the extract/eval scripts once snapshots and crosscoders
   exist.

## Intervention and eval-token scripts

A group of intervention, per-snapshot, and lifecycle-edit evaluation scripts
(under `experiments/causal/`, `experiments/lifecycle/feature_lifecycle_trajectories/`,
and a few `scripts/extract/` helpers) take an `--eval-tokens` tensor of held-out
evaluation tokens. The corpus used in the paper ships with the released
artifacts at `evaluation/eval-corpus/eval_tokens.pt` (Wikipedia-derived,
CC-BY-SA; see its README there for provenance):

```bash
--eval-tokens "$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt"
```

Scripts whose defaults reference a legacy `_archive/...` path accept this flag
to override; you can also supply your own token tensor.

## Derived metrics (regenerated, not shipped)

This is a code-only release: it ships **no** derived metrics. When an experiment
runs, its small CSV/JSON metric summaries are written **co-located with the
experiment that owns them** (and gitignored), under:

    experiments/<...>/<experiment_id>/derived/...

For example, the WordNet probe / matched-control tables are written under
`experiments/probes/concept_evolution_validation/derived/`, resolved via
`src.core.paths.concept_evolution_derived_dir(...)` rather than a hardcoded path.
Regenerate them by running that experiment's scripts (see `REPRODUCE.md`).

## Not redistributed

Original model checkpoints and raw Wikipedia dumps are not redistributed.
Trained dictionaries are not bundled in this code repository — they are
released separately on Hugging Face (see "Released artifacts" above) under MIT.

## Secrets

There are no credentials, API keys, or tokens in this repository. Hugging Face
downloads use anonymous public access; set `HF_HOME` to control the cache
location if needed.
