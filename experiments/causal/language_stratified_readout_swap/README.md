# language_stratified_readout_swap

Language-stratified version of the raw Pythia-1B readout-swap grid. Hidden
states from checkpoint `t` are held fixed and scored under the readout
`W_U^(s)` of each of the 32 released checkpoints, with next-token NLL reported
for the whole nine-language eval corpus (`all`) and separately for English,
Russian, Chinese, Japanese, Thai, Arabic, Hindi, Korean, and Bengali.

Language labels come from the per-token `languages` list in the released
`eval_tokens.pt`. A target enters a per-language estimate only when its context
token has the same label, which drops the 8 targets that straddle a slice
boundary of the concatenated corpus; the `all` stratum keeps every target and
so matches the aggregate swap grid of `temporal_localization_patching`.

Uncertainty is a paired block bootstrap over consecutive 256-token blocks
within each stratum (2,000 resamples, seed 0): each `(t, s, language)` cell
gets a 95% interval for ΔNLL against the native readout, and each
`(t, language)` row gets the bootstrap probability that each readout is the
NLL minimum.

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| `tab:app-readout-swap-languages` | `summary.csv` (gain interval at `s*`), `argmin_summary.csv` (`s*`, gain, argmin probability), `manifest.json` (target counts) | [`scripts/run_language_stratified_swap.py`](scripts/run_language_stratified_swap.py) |
| `fig:app-readout-swap-english-aggregate` | `summary.csv` (`language` ∈ {`all`, `en`}; also `english_only_summary.csv`) | [`scripts/run_language_stratified_swap.py`](scripts/run_language_stratified_swap.py) |
| `fig:app-readout-swap-languages` | `summary.csv` (per-language rows) | [`scripts/run_language_stratified_swap.py`](scripts/run_language_stratified_swap.py) |

The gain plotted and tabulated is `-delta_nll` (native minus swapped NLL); its
interval is `[-delta_nll_ci_high, -delta_nll_ci_low]`. See
[`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim
The aggregate swap-grid NLL is 97.7% non-English text. Per language, the step-256
optimum and the size of the gain vary, but for step-256 and step-512 hidden
states every language prefers an intermediate readout to both its native and
the terminal readout; for step-512 states all nine languages share the
step-1000 optimum.

## Reproduce

Confirmatory configuration: Pythia-1B, hidden states `t ∈ {256, 512, 1000,
2000, 3000, 14000, 47000, 143000}`, all 32 readouts, raw swap (no alignment),
173,229 targets. The camera-ready run took 5.5 hours on one A40 GPU; it is
resume-safe (completed cells are skipped on rerun).

```bash
uv run python experiments/causal/language_stratified_readout_swap/scripts/run_language_stratified_swap.py \
    --model pythia-1b --seed 0 \
    --h-eval-steps 256 512 1000 2000 3000 14000 47000 143000 \
    --block-tokens 256 --bootstrap-reps 2000 --bootstrap-seed 0 \
    --out-dir results/experiments/causal/language_stratified_readout_swap/pythia-1b_seed0 \
    --reference-summary results/experiments/causal/temporal_localization_patching/temporal_patch_grid/pythia-1b_seed0/summary_global.csv
```

On a machine without the snapshot mirror, add
`--readout-cache-dir <scratch>/readouts` to extract each `W_U` from its Hugging
Face revision (only the float32 readout is kept), and
`--hln-cache-dir <scratch>/hln --delete-hln-cache-after-step` to keep the
1.8 GB-per-checkpoint hidden-state caches on node-local scratch.

The `all`-stratum parity check (`reference_parity.json`) compares against the
raw swap grid over the same eight hidden-state steps (tolerance 1e-5 nats);
without that file it records `not_checked`:

```bash
uv run python experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py \
    --model pythia-1b --seed 0 \
    --h-eval-steps 256 512 1000 2000 3000 14000 47000 143000 \
    --out-dir results/experiments/causal/temporal_localization_patching/temporal_patch_grid/pythia-1b_seed0
```

The aligned-swap grid's `summary.csv` is not a valid reference: it was run on
the first 16,352 targets only.

## Inputs
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt`
  (`--eval-tokens`; rebuilt token-for-token by
  `experiments/causal/temporal_localization_patching/scripts/build_eval_corpus_pythia.py`)
- `${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step*_wu.pt`
  (or `--readout-cache-dir` for Hugging Face extraction)
- `${UM_SSD_ROOT}/readout_edit_timing_pythia1b/hLN_step*.pt` (final-norm hidden
  states; built from the Hugging Face checkpoint on first use and shared with
  `temporal_patch_grid.py`)

## Outputs (metrics-only)
Under `--out-dir`:
- `manifest.json`: run configuration, corpus sha1, per-language target counts, provenance
- `layout.pt`: bootstrap block assignment of every target
- `native/h<t>.pt`, `cells/h<t>_s<s>.pt`: per-block NLL sums (float64), the sufficient statistics for every estimate
- `shards/h<t>_s<s>.json`: per-cell rows (resume markers)
- `summary.csv`: one row per `(language, t, s)`: `nll`, `nll_native`, `delta_nll`, 95% paired interval
- `english_only_summary.csv`: the `en` rows of `summary.csv`
- `argmin_summary.csv`: one row per `(t, language)`: best readout, gain over native, bootstrap argmin probabilities of the best and native readouts
- `argmin_probabilities.csv`: argmin probability of every candidate readout
- `reference_parity.json`: `all`-stratum agreement with the reference grid

## Layout
| path | role |
|---|---|
| `scripts/run_language_stratified_swap.py` | driver; hidden-state caching and snapshot loading come from `readout.dynamics.temporal_patch` (shared with `temporal_localization_patching`) |
