# contrastive_task_feature_rescue

Causal companion to `contrastive_readout_swap`: sparse readout-feature
attribution and paired top-K interventions (ablate / preserve, against a
decoder-norm + firing-rate matched random control) that localize the
contrastive readout-rescue effect to a small set of `W_U` crosscoder features
in the trajectory crosscoder gauge, with a specificity matrix testing whether
ablations chosen for one task family transfer to the others.

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| `fig:app-contrastive-task-localization` | per-family shards `run0_pythia1b_s1000_h1000/shards/<family>__h1000__s1000.json` (ablate top-8 / keep-only / project-out margins); the figure is drawn in the paper tree from these tabulated values | [`experiments/causal/contrastive_task_feature_rescue/scripts/run_feature_attribution.py`](scripts/run_feature_attribution.py) |
| `tab:main-localization-ledger` | same per-family shards as below (the main-text ledger reports SVA, numeric comparison, IOI, and relational facts) | [`experiments/causal/contrastive_task_feature_rescue/scripts/run_feature_attribution.py`](scripts/run_feature_attribution.py) |
| `tab:app-contrastive-localisation-ledger` | `run0_pythia1b_s1000_h1000/shards/<family>__h<h>__s<s>.json` (e.g. `sva__h1000__s1000.json`) | [`experiments/causal/contrastive_task_feature_rescue/scripts/run_feature_attribution.py`](scripts/run_feature_attribution.py) |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim
For task families with a positive readout-rescue (see
`experiments/probes/contrastive_readout_swap/`), a small set of crosscoder
features in the reconstructed `W_U` carry the signal:

- **ablate top-k** (zero out top-k features in `W_recon`) drops the
  contrastive margin substantially below decoder-norm + firing-rate matched
  random-feature controls;
- **preserve top-k** (zero everything except top-k) recovers most of the
  margin, again above the matched control;
- the **specificity matrix** is diagonal-heavy: ablations chosen for one
  family do not transfer to the others.

This is the readout-local analogue of the Ge et al. (2026) sparse feature
attribution argument, but formulated in the trajectory crosscoder
gauge of this paper.

## Method (one cell = one task family at one (h_step, snapshot_step))

For each example with answers `(y+, y-)`:

1. Encode all `V` vocab rows through the crosscoder at `snapshot_step` →
   per-vocab feature activations `a_{v,f}` and decoder rows `D_f`.
2. Per-feature attribution to the contrastive margin:
   ```
   attr_f = mean_n  (a_{y+_n, f} - a_{y-_n, f}) * (h_n · D_f * scale_t)
   ```
3. Top-k by `attr_f`; matched random control by 1-NN in (log decoder-norm,
   log firing-rate) plane.
4. Paired interventions: ablate top-k vs preserve top-k vs matched controls.

The reconstruction matches `experiments/causal/temporal_localization_patching/scripts/run_step1000_feature_rescue.py`
(same `encode_at_step`, `feature_contribution`, and preprocessing-aware
decode), so feature indices are directly comparable across the two
experiments.

## Reproduce

Prerequisite — Exp 1's hidden states must be cached on disk:

```bash
uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
    --model EleutherAI/pythia-1b \
    --out-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b

uv run python experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py \
    --model pythia-1b \
    --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
    --h-steps 1000 \
    --out-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b
```

Then attribute and intervene:

```bash
uv run python experiments/causal/contrastive_task_feature_rescue/scripts/run_feature_attribution.py \
    --model pythia-1b --d-sae 24576 --seed 0 \
    --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
    --hidden-dir   results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden \
    --snapshot-step 1000 --h-step 1000 \
    --K 8 16 32 64 128 256 \
    --out-dir results/experiments/causal/contrastive_task_feature_rescue/run0_pythia1b_s1000_h1000
```

This persists the attribution + intervention metrics (see Outputs below).
This repo ships no figure-rendering code; the paper top-K curves and
specificity matrix are rendered in the paper LaTeX tree from these metrics.

To probe whether the rescue is concentrated at step 1k specifically, repeat
with `--snapshot-step` ∈ {256, 512, 1000, 2000, 8000, 143000} (or any cell
where Exp 1 reports a positive rescue).

## Inputs (SSD canonical paths)

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step*_wu.pt`
- Hidden-state cache from Exp 1 (`run0_pythia1b/hidden/<family>_h<step>.pt`)
- Datasets from Exp 1 (`datasets/pythia-1b/<family>.jsonl`)

## Outputs

These are the reproducible metrics behind the paper figures — regenerated by
running the steps above (gitignored; not shipped in this code-only release).

| path | role |
|---|---|
| `shards/<family>__h<h>__s<s>.json`        | per-family attribution + top-K interventions |
| `shards/<family>__h<h>__s<s>.pt`          | full attribution vector + ranking |
| `reconstruction_ev.json`                  | reconstruction explained-variance check |
| `specificity_matrix.{json,pt}`            | top-K-from-A ablation evaluated on family-B examples |
| `manifest.json`                           | inputs + git commit |

## Layout

| path | role |
|---|---|
| `scripts/run_feature_attribution.py` | encoder + attribution + paired top-K interventions; persists all metrics above |

Library helpers: [src/readout/probes/contrastive_tasks.py](../../../src/readout/probes/contrastive_tasks.py),
[src/readout/probes/readout_swap.py](../../../src/readout/probes/readout_swap.py).

## Scope notes

- The crosscoder gauge is fixed by Exp 1's snapshot step. Use the published
  `d24576/seed0` for Pythia-1B and `d8192/seed0` for Pythia-160M.
- For Pythia-6.9B the only paper-faithful instrument is the sparse / high-λ
  run at `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors`;
  pass it explicitly via `--ckpt`.
- This experiment is *secondary* to Exp 1: only run on families that show a
  positive readout-rescue in the heatmap, otherwise the attribution is being
  asked to localize an effect that does not exist.
