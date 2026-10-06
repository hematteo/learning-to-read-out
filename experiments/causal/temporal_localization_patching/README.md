# temporal_localization_patching

How does a fixed hidden state score under readouts from other checkpoints?
Pythia-1B hidden states from eight checkpoints are held fixed and decoded
through each of the 32 released readouts `W_U^(s)`, recording next-token NLL
and the KL to the same-checkpoint readout on a nine-language held-out
corpus. A second script repeats the swap after aligning the swapped readout's
column mean, scale, row norms, or orthogonal frame to the native one. It
supports the readout swap appendix (`sec:app-readout-swap`).

## Paper results

Outputs are under `results/experiments/causal/temporal_localization_patching/`.

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:app-readout-swap-nll-kl-grid` | In the token-weighted aggregate, the step-1000 readout is the NLL optimum for hidden states from steps 256, 512, and 1000, while later hidden states prefer nearby or same-checkpoint readouts. | `scripts/temporal_patch_grid.py` | `temporal_patch_grid/pythia-1b_seed0/summary_global.csv` |
| `sec:app-late-checkpoint-anomaly` (text) | Step-116k readout anomaly: fixed step-47k states score 2.50, 5.55, and 2.95 nats under the step-102k, 116k, and 130k readouts; step-143k states score 1.93, 4.06, and 1.78. | `scripts/temporal_patch_grid.py` | `temporal_patch_grid/pythia-1b_seed0/summary_global.csv` |
| `sec:app-readout-swap` (text) | The Pythia-160M replication at hidden-state checkpoints 256, 512, 1000, 2000, and 143000 shows the same pattern without the step-116k spike. | `scripts/temporal_patch_grid.py --model pythia-160m` | `temporal_patch_grid/pythia-160m_seed0/summary_global.csv` |
| `fig:app-aligned-readout-swap` | Each hidden-state checkpoint keeps its minimum-NLL readout under all four alignments, except a one-checkpoint shift under scale alignment for step-1000 states. | `scripts/run_aligned_swap_grid.py` | `aligned_swap_grid/summary.csv` |

## Run

1. Optional: rebuild the released evaluation corpus token for token from the 2023-11-01 Wikipedia dumps (network):
   ```bash
   uv run python experiments/causal/temporal_localization_patching/scripts/build_eval_corpus_pythia.py \
       --output results/experiments/causal/temporal_localization_patching/eval_tokens_pythia.pt
   ```
2. Score the Pythia-1B swap grid:
   ```bash
   uv run python experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py \
       --model pythia-1b --seed 0 --h-eval-steps 256 512 1000 2000 3000 14000 47000 143000 \
       --out-dir results/experiments/causal/temporal_localization_patching/temporal_patch_grid/pythia-1b_seed0
   ```
3. Score the Pythia-160M replication (the default `--h-eval-steps` are 256 512 1000 2000 143000):
   ```bash
   uv run python experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py \
       --model pythia-160m --seed 0 \
       --out-dir results/experiments/causal/temporal_localization_patching/temporal_patch_grid/pythia-160m_seed0
   ```
4. Score the aligned swaps (alignments `none`, `mean`, `scale`, `row_norm`, `procrustes`; default hidden-state checkpoints 256 512 1000 2000 143000):
   ```bash
   uv run python experiments/causal/temporal_localization_patching/scripts/run_aligned_swap_grid.py \
       --model pythia-1b --out-dir results/experiments/causal/temporal_localization_patching/aligned_swap_grid
   ```

## Inputs

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt`: the evaluation corpus (173,229 next-token targets for the Pythia tokenizer).
- `${UM_SSD_ROOT}/snapshots/pythia-{1b,160m}/EleutherAI_pythia-*_step*_wu.pt`: readouts.
- Hidden states after the final LayerNorm, cached per step as `${UM_SSD_ROOT}/readout_edit_timing_pythia1b/hLN_step<t>.pt` (Pythia-1B) and `${UM_SSD_ROOT}/readout_edit_timing/hLN_step<t>.pt` (Pythia-160M). On a cache miss the script forwards the Hugging Face checkpoint and writes the cache; a cache built from a different corpus is rejected.

## Outputs

Under each `--out-dir`:

- `temporal_patch_grid.py`: `manifest.json`, `summary_global.csv` (one row per `(h_t, W_U_s)`: NLL, top-1, top-5, entropy, KL to native), `summary_concept.csv`, `raw.pt`.
- `run_aligned_swap_grid.py`: `manifest.json`, per-cell JSON shards, `summary.csv` (one row per alignment, `h_t`, `W_U_s`).
- `build_eval_corpus_pythia.py`: the token file (`ids`, `scripts`, `languages`, counts) and, with `--text-output`, the raw text per language.

## Notes

- The paper's alignment check scored the first 16,352 targets (32 sequences, English, Russian, and Chinese), so its NLL values and its step-256 optimum differ from the nine-language aggregate. `run_aligned_swap_grid.py` scores every sequence held in the `hLN_step<t>.pt` cache it finds; the 32-sequence caches come from `scripts/extract/build_hln_cache.py` (default `--max-seqs 32`), and on a cache miss the script caches the full corpus.
- The swap library is `src/readout/dynamics/temporal_patch.py`, shared with `language_stratified_readout_swap` and `sparse_feature_causal_tests`.
- `summary_concept.csv` and `scripts/temporal_patch_metrics.py` (a CLI over the same library) back no paper result.
