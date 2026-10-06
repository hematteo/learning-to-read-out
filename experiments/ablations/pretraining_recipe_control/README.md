# pretraining_recipe_control

Does the readout's own optimization set when the readout develops? Four 31M
Pythia-architecture models share initialization, data, and data order and
differ only in the learning rate multiplier of the output readout `W_U`
(0.25x, 1x, 4x) or, in the fourth run, in a shorter warmup. This directory
holds the trainer, the data builders, and the analyses over the released
checkpoints: `W_U` geometry, three trajectory crosscoder fits per condition
with their lifecycle statistics, the within-run readout swap, and the
frequency class probe. It supports `sec:recipe-control` ("Readout Learning
Rate Shifts Readout Development") and `sec:app-recipe-control`.

## Paper results

Outputs are under `results/experiments/ablations/pretraining_recipe_control/` unless the path says otherwise.

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:main-recipe-dose-response` | Left: the median feature peak step moves from 128 to 64 to 32 in every fit as the multiplier rises; halving the warmup leaves it at its baseline value in every fit. Peak turnover moves from step 2048 at 0.25x to 512 at 4x in every fit and is split between steps 1024 and 2048 at 1x. Right: validation loss; final losses agree within 0.06 nats. | Left: `scripts/train_control_crosscoder.py`, `scripts/analyze_lifecycle.py` (seeds 0 to 2), `scripts/aggregate_seed_replication.py`. Right: `scripts/train_control.py` | `lifecycle_multiseed_table.csv`; `${UM_SSD_ROOT}/runs/<condition>/metrics.csv` |
| `tab:app-recipe-control-summary` | Per-condition EV, L0, final validation loss, profile fractions, median peak step, and peak turnover step (seed 0). | `scripts/analyze_lifecycle.py` | `lifecycle_seed0/lifecycle_summary.csv` (`lifecycle_stats.json` holds the same values) |
| `fig:app-recipe-control-trajectories` | Per-feature normalized decoder norm rho_f(t) for each condition (seed 0). | `scripts/analyze_lifecycle.py` | `lifecycle_seed0/feature_trajectories.pt` |
| `fig:app-recipe-control-reorg` | Decoder-norm turnover peaks earlier as the readout learning rate rises; decoder rotation per checkpoint pair (seed 0). | `scripts/analyze_lifecycle.py` | `lifecycle_seed0/reorg_window.csv` |
| `fig:app-recipe-control-geometry` | `W_U` row norms, spectrum, and final LayerNorm gain over training; the final mean row norm is 1.85 at 4x and 0.93 at 0.25x. | `scripts/analyze_readout_geometry.py` | `readout_geometry_pythia.csv` |
| `fig:app-recipe-control-basin` | Lowest-NLL readout for each hidden-state checkpoint; at 0.25x the step-512 readout lowers the NLL of step-256 hidden states by 0.91 nats, and at 4x each hidden state is best decoded by its own readout. | `scripts/run_trajectory_swap.py` | `trajectory/trajectory_swap_all.csv` |
| `fig:app-recipe-control-explag` | The frequency class gap (probe minus native accuracy) is ordered 0.25x > 1x > 4x early and closes by step 1024 in every run except 0.25x, which closes by step 2048. | `scripts/run_expression_lag.py` | `expression_lag/expression_lag.csv` |

## Run

Steps 4 onward read the checkpoints from `${UM_SSD_ROOT}/runs/<condition>/`.
The analysis scripts take `--ckpt-root` and `--conditions`, which defaults to
the four compared conditions.

1. Check the model API, initialization, parameter groups, and forward pass on CPU (no GPU, no data):
   ```bash
   uv run python experiments/ablations/pretraining_recipe_control/scripts/selfcheck.py
   ```
2. Fetch the first 10B tokens of the Pythia preshuffled Pile in Pythia's order:
   ```bash
   uv run python experiments/ablations/pretraining_recipe_control/scripts/fetch_pythia_preshuffled.py \
       --out ${UM_SSD_ROOT}/pile_slice/pythia_preshuffled.bin --target-tokens 10e9
   ```
3. Train each condition on one GPU, or download the released checkpoints with `hf download hematteo/readout-recipe-control --local-dir "${UM_SSD_ROOT}/runs"`. The baseline command is
   ```bash
   uv run python experiments/ablations/pretraining_recipe_control/scripts/train_control.py \
       --data ${UM_SSD_ROOT}/pile_slice/pythia_preshuffled.bin --sequential-data \
       --out-dir ${UM_SSD_ROOT}/runs/baseline --model-size 31M --global-batch 1024 --max-tokens 10e9 \
       --readout-lr-mult 1.0 --warmup-steps 1430
   ```
   and the other conditions change only `--out-dir` and these flags:

   | Condition | Flags |
   |---|---|
   | `baseline` | `--readout-lr-mult 1.0 --warmup-steps 1430` |
   | `wu_lr_0p25` | `--readout-lr-mult 0.25 --warmup-steps 1430` |
   | `wu_lr_4x` | `--readout-lr-mult 4.0 --warmup-steps 1430` |
   | `warmup_short` | `--readout-lr-mult 1.0 --warmup-steps 715` |
   | `warmup_long` (released, not compared in the paper) | `--readout-lr-mult 1.0 --warmup-steps 5720` |

4. Compute `W_U` geometry across conditions and checkpoints (CPU):
   ```bash
   uv run python experiments/ablations/pretraining_recipe_control/scripts/analyze_readout_geometry.py \
       --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline warmup_short wu_lr_0p25 wu_lr_4x \
       --out-csv results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv
   ```
5. Fit three trajectory crosscoders per condition, seeds 0 to 2 (one GPU, tens of minutes per seed for the four conditions; resumable, skips outputs that exist):
   ```bash
   for s in 0 1 2; do
     uv run python experiments/ablations/pretraining_recipe_control/scripts/train_control_crosscoder.py \
         --ckpt-root ${UM_SSD_ROOT}/runs --conditions wu_lr_0p25 baseline wu_lr_4x warmup_short --seed $s \
         --out-dir results/experiments/ablations/pretraining_recipe_control/crosscoders
   done
   ```
6. Compute lifecycle statistics per seed (CPU):
   ```bash
   for s in 0 1 2; do
     uv run python experiments/ablations/pretraining_recipe_control/scripts/analyze_lifecycle.py \
         --ckpt-root ${UM_SSD_ROOT}/runs --cc-dir results/experiments/ablations/pretraining_recipe_control/crosscoders \
         --conditions wu_lr_0p25 baseline wu_lr_4x warmup_short --seed $s --d-sae 8192 \
         --out-dir results/experiments/ablations/pretraining_recipe_control/lifecycle_seed$s
   done
   ```
7. Pool the three fits:
   ```bash
   R=results/experiments/ablations/pretraining_recipe_control
   uv run python experiments/ablations/pretraining_recipe_control/scripts/aggregate_seed_replication.py \
       --seed-stat $R/lifecycle_seed0/lifecycle_stats.json --seed-stat $R/lifecycle_seed1/lifecycle_stats.json \
       --seed-stat $R/lifecycle_seed2/lifecycle_stats.json --seeds 0 1 2 \
       --out-json $R/lifecycle_multiseed_summary.json --out-csv $R/lifecycle_multiseed_table.csv
   ```
8. Fetch the 2M-token held-out slice that directly follows the trained 10B tokens (network, 4 MB):
   ```bash
   uv run python experiments/ablations/pretraining_recipe_control/scripts/fetch_heldout_slice.py \
       --out ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin --start-token 10e9 --eval-tokens 2e6
   ```
9. Run the within-run readout swap grid (one GPU, minutes; resumable per cell):
   ```bash
   uv run python experiments/ablations/pretraining_recipe_control/scripts/run_trajectory_swap.py \
       --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short \
       --h-steps 256 512 1024 2048 4769 --eval-bin ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin \
       --eval-tokens 120000 --out-dir results/experiments/ablations/pretraining_recipe_control/trajectory
   ```
10. Run the frequency class probe against the native and best readouts (one GPU or CPU):
    ```bash
    uv run python experiments/ablations/pretraining_recipe_control/scripts/run_expression_lag.py \
        --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short \
        --h-steps 256 512 1024 2048 4769 --eval-bin ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin \
        --eval-tokens 400000 --max-positions 20000 \
        --out-dir results/experiments/ablations/pretraining_recipe_control/expression_lag
    ```

## Inputs

- `${UM_SSD_ROOT}/pile_slice/pythia_preshuffled.bin`: flat `uint16` GPT-NeoX-20B token ids, the first 10B tokens of shard 0 of `EleutherAI/pile-standard-pythia-preshuffled` (step 2). The trainer needs no other input.
- `${UM_SSD_ROOT}/runs/<condition>/`: `config.json`, `metrics.csv`, and `ckpts/step<N>/{model_fp16.pt,metrics.json}` at 16 steps, from step 3 or from https://huggingface.co/hematteo/readout-recipe-control.
- `${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin` (+ `.sha256`): held-out slice for steps 9 and 10, unseen by every condition.

## Outputs

Under `results/experiments/ablations/pretraining_recipe_control/`; every script also writes `provenance_<script>.json` (git hash, environment, seed, arguments) next to its outputs.

- `readout_geometry_pythia.csv`: per-checkpoint `W_U` row-norm statistics, singular spectrum (`top1_frac`, effective and stable rank), and final LayerNorm gain.
- `crosscoders/cc_<condition>_d8192_seed<seed>.pt`: one crosscoder per condition and seed (`state_dict`, `config`, `steps`, `quality` with EV, mean L0, and dead rate, the `center_scale` preprocessing statistics).
- `lifecycle_seed<seed>/`: `lifecycle_stats.json`, `lifecycle_summary.csv`, `median_trajectories.csv`, `peak_step_population.csv`, `reorg_window.csv`, `feature_lifecycles_<condition>.csv`, `feature_trajectories.pt`.
- `lifecycle_multiseed_summary.json`, `lifecycle_multiseed_table.csv`: per-seed values, means, sample standard deviations, ranges, and per-seed ordering checks.
- `trajectory/trajectory_swap_<condition>.csv`, `trajectory/trajectory_swap_all.csv` (+ `shards/`): NLL, delta NLL, KL to native, top-1 agreement, and centered-logit R^2 per (condition, alignment, `h_step`, `s_step`).
- `expression_lag/expression_lag.csv` (+ `shards/`): per (condition, `h_step`) probe accuracy with label-shuffle and random-label nulls, `native_readout_acc`, `best_readout_acc`, `availability_gap`, `readout_rescue`.

## Notes

- Each condition keeps 16 checkpoints at steps 0, 1, 2, 4, ..., 2048, about 2479, 4096, and 4769. The step near 2479 is where the first 11-hour job slot ended, so it differs slightly across conditions (2464 to 2481 for the four compared runs). The analysis scripts glob `ckpts/step*/` and map requested steps to the nearest checkpoint on disk.
- `warmup_long` warms up over 5720 steps, longer than the 4769-step budget, so it never reaches the peak learning rate; the paper leaves it out. Add it to `--conditions` to analyze it.
- The trainer builds Hugging Face `GPTNeoXForCausalLM` (the Pythia architecture) and follows Pythia's optimizer recipe (`sec:app-recipe-control-setup`). It is a Pythia-style proxy, not a reproduction of the public Pythia run, and the batch-scheduler launch glue is not included. It resumes from `latest.pt` and writes a `COMPLETE` sentinel at the token budget; `--max-hours` stops cleanly before a job time limit.
- Compute (`tab:compute-resources`): one A40 GPU per run, about 21 hours per run.
- The crosscoder recipe is the Pythia-160M column of `tab:repro-dictionary-hparams` mapped to d_model=256 (d_sae=8192, lambda_1=0.3, 300 epochs, `center_scale`). CPU smoke test: add `--device cpu --expansion-factor 0.5 --n-epochs 2 --batch-size 4096 --conditions baseline` to step 5. `--d-sae` in step 6 must match the crosscoder filenames; `--no-rates` skips the firing-rate pass.
- Steps 9 and 10 run on CPU at reduced size, for example `--eval-tokens 8192 --seq-len 256 --device cpu`.
- Library code: checkpoint loading and the model rebuild in `src/readout/probes/recipe_control_models.py`, the swap kernel in `src/readout/probes/readout_swap.py`, and the probe in `src/readout/probes/availability_probe.py`, with CPU tests under `tests/`.
- Released but not reported in the paper: `scripts/run_gauge_landscape.py` (gain and readout rescaling landscape), `scripts/run_recipe_control_swap.py` (cross-condition swap at the final checkpoint), and `scripts/tokenize_slice.py` (tokenizes a fresh Pile slice in approximate order; the paper's runs used the preshuffled prefix of step 2). `scripts/rc_common.py` holds shared helpers.
