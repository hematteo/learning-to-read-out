# contrastive_readout_swap

Is a task distinction decodable from the hidden state before the model's own
readout expresses it, and can a readout from another checkpoint express it
earlier? For single-token, two-choice tasks this experiment builds the task
datasets, scores the contrastive margin of each hidden-state checkpoint under
each readout checkpoint (readout swap), and compares linear probes on the
final hidden state (availability) with native and swept readout accuracy
(expression) on Pythia-1B and Pythia-6.9B. It supports
`sec:readout-reorganization-decodes` ("Hidden State Availability Precedes
Native Readout Expression"), the overview figure,
`sec:app-terminal-swap-alignment`, and `sec:app-availability-expression`.

## Paper results

Outputs are under `results/experiments/probes/contrastive_readout_swap/`.

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:main-sva-availability-expression`, left | On Pythia-6.9B agreement, the probe approaches ceiling by step 512 while native expression is still at chance, and the best swept readout tracks the native one. | `scripts/run_swap_grid.py`, `scripts/run_controlled_hidden_probes.py` | `run_pythia-6.9b_probe_trajectory/summary.csv` (native: rows with `s_step = h_step`; best swept readout: see Notes); `run_pythia-6.9b_probe_trajectory/controlled/probe_trajectory_69b.csv` (`probe_name = sva_number_lemma_grouped`, `feature_source = hidden`, `acc`) |
| `fig:main-sva-availability-expression`, right | At the final checkpoint, native accuracy is 1.00 on simple agreement but 0.75 across a subject relative clause, while the probe stays at ceiling. | `scripts/run_availability_probes.py` | `availability/run_pythia-6.9b/probes/probe_summary.csv` (`h_step = 143000`, `probe_acc_1d`, `native_readout_acc`; families `sva`, `sva_across_pp`, `sva_subject_rc`, `sva_object_rc`) |
| `fig:app-contrastive-readout-lag` | The 32-checkpoint Pythia-6.9B SVA and role-balanced IOI trajectories; panel B compares the probe with the strongest prompt-token or Gaussian-feature control. | `scripts/run_swap_grid.py`, `scripts/run_controlled_hidden_probes.py` | the two files of the first row, families `sva` and `ioi_role_balanced` |
| `tab:app-ae-binary`, `tab:app-ae-openvocab`, `tab:app-ae-early`, `fig:app-ae-capacity`, `fig:app-ae-grid` | The 20-family availability suite over 24 checkpoints: relative clause agreement and the numeric tasks keep a gap at the final checkpoint, while simple and across-PP agreement and IOI close it. | `scripts/run_availability_probes.py` | `availability/run_pythia-6.9b/probes/probe_summary.csv`; the Pythia-1B columns of `tab:app-ae-binary` from `availability/run_pythia-1b/probes/probe_summary.csv` and `availability/run_pythia-1b_breadth/probes/probe_summary.csv` |
| `sec:app-terminal-swap-alignment` (text) | On the 320 SVA items, step-1000 Pythia-1B hidden states score 0.800 under the native readout and 0.925 under the terminal readout, falling to 0.566 after Procrustes alignment. | `scripts/run_swap_grid.py` | `run0_pythia1b/summary.csv` (`h_step = 1000`, `s_step = 143000` per alignment) |
| `fig:overview` | One SVA item prefers `is` under the step-1000 readout and `are` under the final readout. | `scripts/run_swap_grid.py --save-per-example` | `run0_pythia1b/shards/sva__a-none__h1000__s<s>.pt` |

## Run

1. Pythia-1B localization inventories and swap grid. The builder writes every family of `contrastive_tasks.TASK_BUILDERS`, including `sva`, `numeric_gt`, `ioi`, and `relational_facts`, and filters answer pairs on terminal row norms, which needs the step-143000 `W_U` snapshot. `contrastive_task_feature_rescue` reuses the hidden-state cache; add `--save-per-example` for the per-item margins of `fig:overview`:
   ```bash
   uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
       --model EleutherAI/pythia-1b --out-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b
   uv run python experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py --model pythia-1b \
       --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
       --out-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b \
       --alignments none scale row_norm procrustes
   ```
2. Pythia-6.9B 32-checkpoint SVA and IOI trajectories. The paper scored a bf16 model: hidden states from a bf16 forward pass and bf16-rounded `W_U` snapshots (`--dtype bf16 --wu-round bf16`):
   ```bash
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
   ```
3. Pythia-6.9B availability suite (20 families, 24 checkpoints; built without row-norm matching). `extract_hidden_dense.py` writes the final-position hidden states after (`h`) and before (`z`) the final LayerNorm, and `<slug>_step<S>_{wu,lnf}.pt` snapshots under `${UM_SSD_ROOT}/snapshots/pythia-6.9b/` (one 48 GB GPU at fp32):
   ```bash
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
4. Pythia-1B availability columns, as two runs with the settings used for the paper. The agreement run builds the subject-RC set with the attractor always number-mismatched (`--subject-rc-attractor mismatched`); the breadth run uses stratified folds for the numeric families and no probe labels for BLiMP (`--stratified-prefixes numeric --unprobed-prefixes blimp`). With `STEPS` and `OUT` from step 3:
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
   Omitting these three flags applies the Pythia-6.9B protocol at Pythia-1B.

## Inputs

- `${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step*_wu.pt`
- `${UM_SSD_ROOT}/snapshots/pythia-6.9b/EleutherAI_pythia-6.9b_step*_wu.pt`: the 32-step schedule for step 2; the 24 availability steps are written by `extract_hidden_dense.py` if missing.
- `${UM_SSD_ROOT}/snapshots/pythia-{1b,6.9b}/EleutherAI_pythia-*_step*_lnf.pt`: final LayerNorm parameters, from `extract_hidden_dense.py`.
- `$FV_DATA_DIR/{antonym,english-french,present-past,country-capital}.json`: the four function-vector families, from `github.com/ericwtodd/function_vectors` (`dataset_files/abstractive/`, MIT licence). `FV_DATA_DIR` defaults to `data/function_vectors/` at the repository root; the builder raises if a file is missing.
- `nyu-mll/blimp` from the Hugging Face Hub (BLiMP families).
- Hugging Face checkpoints of each model, forwarded once per hidden-state step.

## Outputs

- `datasets/<run>/<family>.jsonl` (+ control sidecars and `summary.json`): tokenizer-specific task datasets.
- `run_swap_grid.py`, under `--out-dir`: `hidden/<family>_h<step>.pt` (final-position hidden states after the final LayerNorm), `shards/<family>__a-<alignment>__h<h>__s<s>.json` (resume markers), with `--save-per-example` a matching `.pt` holding `margins`, `margins_native`, `y_plus`, `y_minus`, `manifest.json`, and `summary.csv` (mean margin, accuracy, and their change from the native cell per family, alignment, `h_step`, `s_step`).
- `run_controlled_hidden_probes.py`: `<stem>.csv` and `<stem>.pt` with probe accuracies on hidden states, prompt-token features, and Gaussian features.
- `availability/run_<run>/hidden/<family>_{h,z}<step>.pt` and `availability/run_<run>/probes/probe_summary.{csv,pt}` (+ `probe_summary.provenance.json`) for `<run>` in `pythia-6.9b`, `pythia-1b`, `pythia-1b_breadth`.

## Notes

- Swap-grid margins are `h_t . W_U^s[y+] - h_t . W_U^s[y-]`, with `h_t` the hidden state after checkpoint `t`'s final LayerNorm and no second normalization.
- Best swept readout in the trajectories (`fig:main-sva-availability-expression` left, `fig:app-contrastive-readout-lag`): the paper defines `B_env` as the maximum over readouts per task and hidden-state checkpoint, excluding the native and terminal readouts (`sec:app-evaluation-details`). No script in this release computes it; take the maximum accuracy over the `alignment = none` rows of `summary.csv` with `s_step` not equal to `h_step` or 143000.
- Native and swept readouts in `run_availability_probes.py` (`readout.probes.availability_expression.native_and_best_swept`): native expression applies checkpoint `t`'s final LayerNorm and `W_U` to the pre-LayerNorm state `z_t`. Each swept candidate `s` applies its own final LayerNorm together with its `W_U`, `LN_s(z_t) . W_U^s`. The candidates are the `--swept-steps` checkpoints minus the terminal step 143000, so the native checkpoint is included. `best_swept_readout_acc` is the in-sample maximum over candidates; `best_swept_acc_heldout` picks the candidate on a random half of the items (seed `--seed`) and reports its accuracy on the other half.
- `run_availability_probes.py` is CPU-only apart from the full-vocabulary rank computations (`--device`); it resumes per cell.
- Probe folds come from `group_kfold_splits`, which reproduces sklearn's `GroupKFold` with the tie order of x86-64 NumPy. The paper ran on x86; on macOS arm64, sklearn's own `GroupKFold` assigns equal-size groups to different folds. On CPU the label-shuffle and random-label nulls can differ from the published values by about one item, and the full-vocabulary mean ranks by near-tie flips (rank-1 rates agree).
- `build_task_datasets.py` produces the paper's 320 SVA items (plural items first); no metric depends on item order.
- Task registries: `src/readout/probes/contrastive_tasks.py` (localization inventories) and `src/readout/probes/availability_tasks.py` (availability suite); probe labels, group keys, folds, and readout metrics are in `src/readout/probes/availability_expression.py`.
- Released but not part of the commands above: `scripts/launch_swap_extraction.sh` and `scripts/run_4gpu_extraction.sh` (multi-GPU extraction wrappers), `scripts/extract_wu_hidden_standalone.py` (standalone `W_U` and hidden-state extraction), and `scripts/run_step6_reporting.py` (held-out best-readout selection with a paired bootstrap).
