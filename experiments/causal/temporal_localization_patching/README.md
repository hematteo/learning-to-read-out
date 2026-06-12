# temporal_localization_patching

Temporal patching showing that readout reorganization concentrates near step 1,000:
patching experiments replace the readout at step T with the readout at step T' and
record the behavioral effect, isolating where in training the reorganization occurs.

## Figures produced

| Thesis label | Metric file | Producing script |
|---|---|---|
| `fig:main-readout-coordination` | `summary_global.csv`, `summary_concept.csv` | `experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py` |
| `fig:app-readout-swap-nll-kl-grid` | `summary_global.csv` | `experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py` |
| `fig:app-readout-swap-concept-mass` | `summary_concept.csv` | `experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py` |
| `fig:app-readout-swap-target-nll` | `summary_concept.csv` | `experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py` |
| `fig:app-aligned-readout-swap` | `aligned_swap_grid/summary.csv` | `experiments/causal/temporal_localization_patching/scripts/run_aligned_swap_grid.py` |
| `fig:app-readout-swap-160m-target-nll` | `summary_concept.csv` | `experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py` |
| `fig:app-readout-swap-family-logit-mass` | `summary_concept.csv` | `experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py` |
| `fig:lr-swap` | `summary_global.csv` | `experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py` |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim
Readout reorganization is temporally localized near step 1,000: temporal patching
experiments (replace the readout at step T with the readout at step T') show that
swapping in the early-window readout produces most of the behavioral effect.

## Reproduce (representative; scripts bootstrap the repo onto `sys.path`)
- Temporal patch grid:    `uv run python experiments/causal/temporal_localization_patching/scripts/temporal_patch_grid.py`
- Aligned swap grid:      `uv run python experiments/causal/temporal_localization_patching/scripts/run_aligned_swap_grid.py`
- Step-1k feature rescue: `uv run python experiments/causal/temporal_localization_patching/scripts/run_step1000_feature_rescue.py`

## Inputs (SSD canonical paths)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step*_wu.pt`

## Outputs (metrics-only)
This experiment computes metrics only: it computes and persists the metrics behind the figures but
ships no figure-rendering code. Thesis figures are rendered in the separate thesis LaTeX
tree from these metrics. Per `--out-dir`, the scripts persist:
- `temporal_patch_grid.py` -> `manifest.json`, `summary_global.csv` (one row per `(h_t, W_U_s)`), `summary_concept.csv` (one row per `(h_t, W_U_s, concept)`), `raw.pt`
- `temporal_patch_metrics.py` (shared library; its `__main__` driver) -> `manifest.json`, `subsets.json`, `summary.csv`, `selectivity.csv`, `paired_vs_random.csv`, `raw.pt`
- `run_aligned_swap_grid.py` -> `manifest.json`, per-cell JSON shards, aggregated `summary.csv`
- `run_step1000_feature_rescue.py` -> `manifest.json`, per-cell JSON shards, aggregated `rescue_summary.csv`

## Layout
| path | role |
|---|---|
| `scripts/` | temporal-patching code: `temporal_patch_metrics` is the shared library imported by `temporal_patch_grid` and the `run_*` swap drivers (`run_aligned_swap_grid`, `run_step1000_feature_rescue`). It loads the archived intervention helper lazily, so importing it never requires the non-shipped `_archive/` tree. The sibling experiment `sparse_feature_causal_tests` imports `temporal_patch_metrics` from here. |
