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
| `tab:app-contrastive-localisation-ledger` (Table H.2, App. H.5; the SVA and greater-than values are quoted in §4.4) — *orig.*, *recon*, *ablate* / *preserve-only* top-8 and *sign* columns | `run0_pythia1b_s143000_h1000_pos/shards/<family>__h1000__s143000.json`, `per_K["8"]`: `summary_native_s`, `summary_recon`, `summary_ablate_top`, `summary_ablate_ctrl_sign_matched`, `summary_preserve_top`, `summary_preserve_ctrl_sign_matched` (`.accuracy`) | [`scripts/run_feature_attribution.py`](scripts/run_feature_attribution.py) |
| `tab:app-contrastive-localisation-ledger` — *hidden projection* columns (*native*, *proj.*) | `converse_pythia1b_s143000_h1000_pos/<family>__h1000__s143000.json`: `baseline.native_t.accuracy`, `per_K["8"].summary_proj_native_t.accuracy` | [`scripts/run_converse_intervention.py`](scripts/run_converse_intervention.py) |

Families in the table: `sva`, `numeric_gt` (greater-than completion), `ioi`,
`relational_facts`. The *orig.* entries marked $^\ast$ in the paper (greater-than,
IOI, relational facts) are the native terminal readout on the full inventory,
i.e. the `none`-alignment cell (h=1000, s=143000) of
`contrastive_readout_swap/scripts/run_swap_grid.py`, not the held-out half.

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
# Table H.2 cell: step-1000 hidden states, reconstructed terminal (step-143k) readout,
# top-K by positive attribution, 50/50 selection/scoring split.
uv run python experiments/causal/contrastive_task_feature_rescue/scripts/run_feature_attribution.py \
    --model pythia-1b --d-sae 24576 --seed 0 \
    --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
    --hidden-dir   results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden \
    --snapshot-step 143000 --h-step 1000 --rank-by pos \
    --K 8 16 32 64 128 256 --split-frac 0.5 --split-seed 0 \
    --out-dir results/experiments/causal/contrastive_task_feature_rescue/run0_pythia1b_s143000_h1000_pos

# Hidden projection columns: project the saved top-K decoder rows out of h and
# re-score under the native W_U at h-step, on the attribution run's held-out half.
uv run python experiments/causal/contrastive_task_feature_rescue/scripts/run_converse_intervention.py \
    --model pythia-1b \
    --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
    --hidden-dir   results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden \
    --attribution-dir results/experiments/causal/contrastive_task_feature_rescue/run0_pythia1b_s143000_h1000_pos \
    --h-step 1000 --snapshot-step 143000 --K 8 16 32 64 128 256 \
    --out-dir results/experiments/causal/contrastive_task_feature_rescue/converse_pythia1b_s143000_h1000_pos
```

This persists the attribution + intervention metrics (see Outputs below).
The converse script reads the split parameters from each attribution shard and
re-derives the same per-family permutation, so its *native* baseline equals the
attribution shard's `per_K[K].summary_h_native` on the same held-out items.
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

`run_converse_intervention.py` writes, under its own `--out-dir`:

| path | role |
|---|---|
| `<family>__h<h>__s<s>.json` | `baseline` margins under native W_U at h- and s-step; `per_K[K]` projected margins, logit shifts (`dy_plus`, `dy_minus`, `dmargin`) and `h_residual_frac` = mean ‖h_proj‖/‖h‖ |
| `manifest.json`             | inputs, per-family split provenance, git commit |

## Layout

| path | role |
|---|---|
| `scripts/run_feature_attribution.py` | encoder + attribution + paired top-K interventions; persists all metrics above |
| `scripts/run_converse_intervention.py` | hidden-state projection of the selected decoder directions (reads the attribution shards) |

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
