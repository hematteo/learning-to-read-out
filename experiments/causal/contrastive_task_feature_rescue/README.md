# contrastive_task_feature_rescue

Do a few sparse readout directions carry the contrastive task margins? With
step-1000 Pythia-1B hidden states decoded through the reconstructed terminal
readout of the released d=24,576 crosscoder, features are ranked by
attribution to each task margin on half of the items, and the top eight are
ablated, preserved alone, or projected out of the hidden state on the other
half, against random features matched on decoder norm, firing rate, and
attribution sign. It supports `sec:sparse-coordinate-localization` ("A Few
Readout Directions Carry Task Margins") and `sec:app-contrastive-localisation`.

## Paper results

Outputs are under `results/experiments/causal/contrastive_task_feature_rescue/`.

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `tab:app-contrastive-localisation-ledger`: *recon*, *ablate* and *preserve-only* (top-8 and *sign*) | Subject-verb agreement and greater-than completion localize: ablation falls far below the sign-matched control and preserve-only stays high on eight directions. | `scripts/run_feature_attribution.py` | `run0_pythia1b_s143000_h1000_pos/shards/<family>__h1000__s143000.json`, `per_K["8"]`: `summary_recon`, `summary_ablate_top`, `summary_ablate_ctrl_sign_matched`, `summary_preserve_top`, `summary_preserve_ctrl_sign_matched` (`.accuracy`) |
| `tab:app-contrastive-localisation-ledger`: *orig.* | Original terminal readout on the held-out half; for greater-than completion, IOI, and relational facts, on the full inventory. | `scripts/run_feature_attribution.py`; full-inventory entries: `experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py` | shard `per_K["8"].summary_native_s`; the `none`-alignment cell (h=1000, s=143000) of `results/experiments/probes/contrastive_readout_swap/run0_pythia1b/summary.csv` |
| `tab:app-contrastive-localisation-ledger`: *hidden projection* | Step-1000 states scored under the native `W_U^(1000)` before and after the top eight decoder directions are projected out. | `scripts/run_converse_intervention.py` | `converse_pythia1b_s143000_h1000_pos/<family>__h1000__s143000.json`: `baseline.native_t.accuracy`, `per_K["8"].summary_proj_native_t.accuracy` |

The table's families are `sva`, `numeric_gt` (greater-than completion), `ioi`, and `relational_facts`.

## Run

1. Build the localization inventories and cache the step-1000 hidden states with `contrastive_readout_swap` (see its README, Run step 1):
   ```bash
   uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
       --model EleutherAI/pythia-1b --out-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b
   uv run python experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py --model pythia-1b \
       --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
       --out-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b \
       --alignments none scale row_norm procrustes
   ```
2. Attribute and intervene (terminal readout, top-K by positive attribution, 50/50 selection and scoring split):
   ```bash
   uv run python experiments/causal/contrastive_task_feature_rescue/scripts/run_feature_attribution.py \
       --model pythia-1b --d-sae 24576 --seed 0 \
       --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
       --hidden-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden \
       --snapshot-step 143000 --h-step 1000 --rank-by pos --K 8 16 32 64 128 256 --split-frac 0.5 --split-seed 0 \
       --out-dir results/experiments/causal/contrastive_task_feature_rescue/run0_pythia1b_s143000_h1000_pos
   ```
3. Project the selected decoder directions out of the hidden states and re-score under the native readout, on the same held-out half:
   ```bash
   uv run python experiments/causal/contrastive_task_feature_rescue/scripts/run_converse_intervention.py \
       --model pythia-1b \
       --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
       --hidden-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden \
       --attribution-dir results/experiments/causal/contrastive_task_feature_rescue/run0_pythia1b_s143000_h1000_pos \
       --snapshot-step 143000 --h-step 1000 --K 8 16 32 64 128 256 \
       --out-dir results/experiments/causal/contrastive_task_feature_rescue/converse_pythia1b_s143000_h1000_pos
   ```

## Inputs

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step*_wu.pt`
- `results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b/<family>.jsonl` and `results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden/<family>_h<step>.pt` from step 1.

## Outputs

`run_feature_attribution.py`, under its `--out-dir`:

| Path | Content |
|---|---|
| `shards/<family>__h<h>__s<s>.json` | per-family attribution summary and top-K interventions with matched controls |
| `shards/<family>__h<h>__s<s>.pt` | full attribution vector and ranking |
| `reconstruction_ev.json` | EV of the reconstructed readout |
| `specificity_matrix.{json,pt}` | top-K ablations chosen for one family, scored on every family |
| `manifest.json` | inputs, per-family split, git commit |

`run_converse_intervention.py`, under its `--out-dir`:

| Path | Content |
|---|---|
| `<family>__h<h>__s<s>.json` | `baseline` margins under the native `W_U` at the h and s steps; per K, projected margins, logit shifts (`dy_plus`, `dy_minus`, `dmargin`), and `h_residual_frac` (mean of norm(h_proj)/norm(h)) |
| `manifest.json` | inputs, per-family split provenance, git commit |

## Notes

- Attribution: `attr_f = mean_n (a_{y+,f} - a_{y-,f}) * (h_n . D_f) * scale_t` over the selection half, where `a` are the crosscoder's feature activations on the answer rows at `--snapshot-step`. Each control feature is the nearest unused neighbor of a selected feature in the plane of log decoder norm and log firing rate; the sign-matched control draws only from features with positive attribution.
- `run_converse_intervention.py` reads the split parameters from each attribution shard and re-derives the same per-family permutation, so both scripts score the same held-out items.
- Shards are resumable: a family whose `.json` and `.pt` shard pair exists is skipped; delete the pair to recompute.
- Library code: `src/readout/probes/contrastive_tasks.py`, `src/readout/probes/readout_swap.py`.
