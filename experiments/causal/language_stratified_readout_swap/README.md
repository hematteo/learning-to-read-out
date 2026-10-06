# language_stratified_readout_swap

Which languages drive the aggregate readout swap? The Pythia-1B raw swap grid
of `temporal_localization_patching` is recomputed separately for each of the
nine languages of the evaluation corpus (English, Russian, Chinese, Japanese,
Thai, Arabic, Hindi, Korean, Bengali) and for all targets together, with
paired block-bootstrap intervals and the bootstrap probability that each
readout is the NLL minimum. It supports `sec:app-readout-swap-languages`
("Corpus Composition and Per-Language Swaps").

## Paper results

Outputs are under `results/experiments/causal/language_stratified_readout_swap/pythia-1b_seed0/`.

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `tab:app-readout-swap-languages` | Per-language best readout `s*` and gain over the native readout with 95% intervals; the token-weighted aggregate is 97.7% non-English. | `scripts/run_language_stratified_swap.py` | `summary.csv` (gain interval at `s*`), `argmin_summary.csv` (`s*`, gain, argmin probability), `manifest.json` (target counts) |
| `fig:app-readout-swap-english-aggregate` | English gains little at its best readout, and the aggregate's best readout for step-256 states is worse than the native readout on English text. | `scripts/run_language_stratified_swap.py` | `summary.csv` (`language` in {`all`, `en`}), `english_only_summary.csv` |
| `fig:app-readout-swap-languages` | For step-256 and step-512 hidden states every language prefers an intermediate readout to both its native and the terminal readout; for step-512 states all nine languages share the step-1000 optimum. | `scripts/run_language_stratified_swap.py` | `summary.csv` (per-language rows) |

The plotted and tabulated gain is `-delta_nll` (native minus swapped NLL),
with interval `[-delta_nll_ci_high, -delta_nll_ci_low]`.

## Run

1. Optional: score the raw swap grid over the same eight hidden-state checkpoints, which step 2 uses as its parity reference:
   ```bash
   uv run python experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py \
       --model pythia-1b --seed 0 --h-eval-steps 256 512 1000 2000 3000 14000 47000 143000 \
       --out-dir results/experiments/causal/temporal_localization_patching/temporal_patch_grid/pythia-1b_seed0
   ```
2. Score the stratified grid (all 32 readouts, raw swap, 173,229 targets; resumable, completed cells are skipped):
   ```bash
   uv run python experiments/causal/language_stratified_readout_swap/scripts/run_language_stratified_swap.py \
       --model pythia-1b --seed 0 --h-eval-steps 256 512 1000 2000 3000 14000 47000 143000 \
       --block-tokens 256 --bootstrap-reps 2000 --bootstrap-seed 0 \
       --out-dir results/experiments/causal/language_stratified_readout_swap/pythia-1b_seed0 \
       --reference-summary results/experiments/causal/temporal_localization_patching/temporal_patch_grid/pythia-1b_seed0/summary_global.csv
   ```

## Inputs

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt` (`--eval-tokens`): token ids with per-token `languages` labels; rebuilt token for token by `experiments/causal/temporal_localization_patching/scripts/build_eval_corpus_pythia.py`.
- `${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step*_wu.pt`, or `--readout-cache-dir` to extract each `W_U` from its Hugging Face revision.
- `${UM_SSD_ROOT}/readout_edit_timing_pythia1b/hLN_step*.pt`: hidden states after the final LayerNorm, built from the Hugging Face checkpoint on first use and shared with `temporal_patch_grid.py`.

## Outputs

Under `--out-dir`:

- `manifest.json`: configuration, corpus sha1, per-language target counts, provenance.
- `layout.pt`: bootstrap block of every target.
- `native/h<t>.pt`, `cells/h<t>_s<s>.pt`: per-block NLL sums (float64), the sufficient statistics for every estimate.
- `shards/h<t>_s<s>.json`: per-cell rows (resume markers).
- `summary.csv`: one row per `(language, t, s)` with `nll`, `nll_native`, `delta_nll`, and the 95% paired interval.
- `english_only_summary.csv`: the `en` rows of `summary.csv`.
- `argmin_summary.csv`: one row per `(t, language)` with the best readout, its gain, and the bootstrap argmin probabilities of the best and native readouts.
- `argmin_probabilities.csv`: argmin probability of every candidate readout.
- `reference_parity.json`: agreement of the `all` stratum with the reference grid (tolerance 1e-5 nats), or `not_checked` without `--reference-summary`.

## Notes

- A target enters a per-language estimate only when its context token has the same language label, which drops the 8 targets that straddle a slice boundary; the `all` stratum keeps every target and matches the aggregate grid.
- Intervals come from 2,000 paired resamples of consecutive 256-token blocks within each stratum (seed 0).
- The paper's run took 5.5 hours on one A40 GPU. Without the snapshot mirror, add `--readout-cache-dir <scratch>/readouts`, and `--hln-cache-dir <scratch>/hln --delete-hln-cache-after-step` to keep the 1.8 GB per-checkpoint hidden-state caches on node-local scratch.
- The `summary.csv` of `run_aligned_swap_grid.py` is not a valid parity reference: the paper's alignment check scored only the first 16,352 targets.
