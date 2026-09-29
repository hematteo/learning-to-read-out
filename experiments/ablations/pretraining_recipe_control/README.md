# pretraining_recipe_control

The controlled training experiment behind the paper's Finding 2: matched 31M
Pythia-style runs that share initialization, tokenized data, and data order, and
differ in one recipe factor each. The paper shows that raising the output readout
learning rate moves the readout reorganization one log-spaced checkpoint bin
earlier per 4x increase, at matched validation loss, and that halving the warmup
leaves the timing unchanged. This directory holds the trainer, the data
builders, and every analysis over the released checkpoints: the $W_U$ geometry,
one trajectory crosscoder per arm with its lifecycle statistics (the timing
claim), the within-trajectory readout swap grid (compatibility basin), the
availability versus expression probe (expression lag), and the gauge analysis
(temperature conservation, loss landscape). The trained checkpoints are
released on Hugging Face.

## Figures produced

| Paper label | Metric file (under `results/experiments/ablations/pretraining_recipe_control/`) | Producing script (`scripts/`) |
|---|---|---|
| `fig:app-recipe-control-geometry` | `readout_geometry_pythia.csv` | `analyze_readout_geometry.py` |
| `fig:main-recipe-dose-response` | `lifecycle_seed0/lifecycle_summary.csv` (`median_peak_step`, `reorg_peak_step` vs `readout_lr_mult`) | `train_control_crosscoder.py` then `analyze_lifecycle.py` |
| `tab:app-recipe-control-summary` | `lifecycle_seed0/lifecycle_summary.csv` (EV, $L_0$, val loss, profile fractions, median peak, reorganization step); `lifecycle_stats.json` nested | same |
| `fig:app-recipe-control-median` | `lifecycle_seed0/median_trajectories.csv` | same |
| `fig:app-recipe-control-trajectories` | `lifecycle_seed0/feature_trajectories.pt` (per-feature `rho`, peak steps, seeded 500-feature display sample) | same |
| `fig:app-recipe-control-peakstep` | `lifecycle_seed0/peak_step_population.csv` | same |
| `fig:app-recipe-control-reorg` | `lifecycle_seed0/reorg_window.csv` | same |
| `fig:app-recipe-control-basin` | `trajectory/trajectory_swap_all.csv` (argmin-NLL readout per hidden state checkpoint) | `fetch_heldout_slice.py` then `run_trajectory_swap.py` |
| `fig:app-recipe-control-explag` | `expression_lag/expression_lag.csv` | `fetch_heldout_slice.py` then `run_expression_lag.py` |
| `fig:app-recipe-control-temperature` | `gauge/temperature_conservation.csv` | `fetch_heldout_slice.py` then `run_gauge_landscape.py` |
| `fig:app-recipe-control-gauge-landscape` | `gauge/gauge_landscape.csv` | same |

The cross-condition swap grid (`swap/swap_grid_recipe_control.csv`,
`run_recipe_control_swap.py`) is a supplement the paper discusses in prose;
`aggregate_seed_replication.py` pools the lifecycle statistics over crosscoder
seeds for the robustness check. See
[`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure to metric map.

## Claim

Under a recipe perturbation confined to the output readout, the sparse readout
reorganization moves monotonically with the readout learning rate: the median
feature peak step ($128 \to 64 \to 32$) and the peak-turnover reorganization step
($2048 \to 1024 \to 512$) each shift one log-spaced checkpoint bin earlier per
$4\times$ increase, at final validation losses matched to within $0.06$ nats,
while halving the warmup leaves both statistics on the baseline. The same
ordering appears in weight space (the slow-readout arm's early hidden states are
decoded better by later readouts; the fast-readout arm has no such lag, and this
survives the Procrustes alignment) and in probe space (the availability gap
orders $0.25\times > 1\times > 4\times$ over the early window). The $W_U$ row
norm and the final LayerNorm gain trade off along a loss-flat symmetry direction,
so the effective logit scale is conserved across arms. Claim ceiling: a
controlled recipe check, not a better pretraining algorithm or a general
developmental law.

## Conditions

Five conditions are trained at **31M parameters, global batch 1024 sequences
(2,097,152 tokens/step), to a 10B-token budget (4,769 steps)** on the first 10B
tokens of the Pythia preshuffled Pile, read in Pythia's released order
(`scripts/fetch_pythia_preshuffled.py`, `--sequential-data`), with parameter
seed 0. Same data, same order, same initialization for every arm. The paper reports four;
`warmup_long` is a fifth complete arm released alongside them.

| id | name | perturbation vs. baseline | in the paper |
|---|---|---|---|
| C0 | `baseline`     | none (reference; readout LR multiplier `m=1.0`, warmup 1430 steps) | yes |
| C1 | `wu_lr_0p25`   | output readout ($W_U$) LR multiplier `m=0.25` | yes |
| C2 | `wu_lr_4x`     | output readout ($W_U$) LR multiplier `m=4.0` | yes |
| C3 | `warmup_short` | LR warmup 1430 to 715 steps (half) | yes |
| C4 | `warmup_long`  | LR warmup 1430 to 5720 steps (4x) | no |

Each arm keeps 16 checkpoints at steps
`0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, ~2480, 4096, 4769`
(the step near 2480 is where the first 11-hour job slot ended; the trainer
checkpoints on slot stop, so this one step differs slightly across arms, from
2464 to 2497). Every analysis script discovers the steps by globbing
`ckpts/step*/` and maps requested steps to the nearest one on disk.

The trainer is a **hybrid** built on HF `GPTNeoXForCausalLM` (which *is* the
Pythia architecture) on the modern torch stack, reproducing the Pythia recipe
shape (small/Wang init, AdamW (0.9, 0.95), cosine schedule with warmup over the
143k-step horizon stopped early at the token budget, fp16 with dynamic loss
scaling, GPT-NeoX-20B tokenizer); the 2022-era GPT-NeoX/DeeperSpeed stack is
incompatible with the available cluster. The baseline is a Pythia-*style* proxy,
**not** a reproduction of the public Pythia run. See the recipe-control appendix
of the paper for the full recipe specification and what is and is not matched
to Pythia.

## Released checkpoints

All five arms, with `config.json`, `metrics.csv`, and
`ckpts/step<N>/{model_fp16.pt,metrics.json}` at the 16 steps above, are at
**https://huggingface.co/hematteo/readout-recipe-control**. Download into the
layout the analysis scripts expect (`readout.probes.recipe_control_models`
resolves `${UM_SSD_ROOT}/runs/<condition>/` by default):

```bash
hf download hematteo/readout-recipe-control --local-dir "${UM_SSD_ROOT}/runs"
```

## Result

The readout LR multiplier is load-bearing for $W_U$ geometry. At 10B tokens the
$W_U$ row norm mean separates cleanly by condition (numbers reproduced from
`readout_geometry_pythia.csv`, which `analyze_readout_geometry.py` regenerates
from the released checkpoints):

| condition | tokens | $W_U$ rn_mean | rn_max | stable_rank | top-1 σ frac |
|---|---|---|---|---|---|
| `wu_lr_0p25` | 10.0B | 0.93 | 1.73 | 6.97 | 14.4% |
| `baseline`   | 10.0B | 1.22 | 2.44 | 8.35 | 12.0% |
| `wu_lr_4x`   | 10.0B | 1.85 | 3.75 | 12.88 | 7.8% |

Faster readout learning grows $W_U$ row norms faster and spreads the spectrum;
`run_gauge_landscape.py` shows the row norm and the final LayerNorm gain trade
off along a loss-flat symmetry direction so that the effective logit scale is
conserved. The timing statistics from the per-arm trajectory crosscoders
(`analyze_lifecycle.py`, $d_{sae}=8192$, $K=16$, seed 0) are the paper's
`tab:app-recipe-control-summary`: reading down the readout LR axis
($0.25\times \to 1\times \to 4\times$) the median peak step is $128 \to 64 \to 32$
and the reorganization step $2048 \to 1024 \to 512$, with `warmup_short` on the
baseline values.

## Metrics produced

All under `results/experiments/ablations/pretraining_recipe_control/`
(gitignored, regenerated on run; not shipped in this code-only release). Each
script also writes a `provenance_<script>.json` (git hash, environment, seed,
CLI args) next to its outputs.

- `readout_geometry_pythia.csv`: per-checkpoint $W_U$ geometry across conditions
  (row norm stats `rn_mean`/`rn_max`, SVD spectrum incl. `top1_frac`, eff/stable
  rank, `lnf_norm`) vs. tokens (`analyze_readout_geometry.py`).
- `crosscoders/cc_<cond>_d8192_seed<seed>.pt`: one trajectory crosscoder per arm
  (`state_dict`, `config`, `steps`, `quality` = EV / mean $L_0$ / dead rate,
  `training`, the invertible `center_scale` `preprocess_stats`)
  (`train_control_crosscoder.py`).
- `lifecycle_seed<seed>/`: `lifecycle_stats.json` and `lifecycle_summary.csv`
  (per condition: `quality_*`, `val_loss_final`, `val_loss_at_reorg`,
  `profile_<name>` fractions, `median_peak_step`, `median_peak_tau`,
  `median_lifespan_snaps`, `reorg_peak_step`, `rotation_peak_step`,
  `max_norm_turnover`); `median_trajectories.csv` (median / IQR of the
  peak-normalized decoder norm $\rho_f(t)$ per checkpoint);
  `peak_step_population.csv` (fraction of active features peaking at each
  checkpoint, total decoder norm mass); `reorg_window.csv` (norm turnover
  $\overline{|\Delta\rho_f|}$ and decoder rotation $\overline{1-\cos}$ per
  adjacent checkpoint pair); `feature_lifecycles_<cond>.csv` (per-feature
  profile, peak, birth, lifespan); `feature_trajectories.pt` (per-condition
  `rho` and decoder norms `(K, D)`, peak steps, profiles, the seeded display
  sample, firing rates) (`analyze_lifecycle.py`).
- `lifecycle_multiseed_summary.json`, `lifecycle_multiseed_table.csv`: means,
  sample standard deviations, ranges, and per-seed readout LR ordering checks
  over crosscoder seeds (`aggregate_seed_replication.py`).
- `trajectory/trajectory_swap_<cond>.csv`, `trajectory/trajectory_swap_all.csv`
  (+ `shards/*.json`): per (condition, alignment, `h_step`, `s_step`) next-token
  NLL, `delta_nll`, KL to native, top-1 agreement, centered-logit $R^2$
  (`run_trajectory_swap.py`).
- `expression_lag/expression_lag.csv` (+ `shards/*.json`): per (condition,
  `h_step`) `probe_acc` with shuffle / random-label nulls and CI,
  `native_readout_acc`, `best_readout_acc`, `availability_gap`,
  `readout_rescue` (`run_expression_lag.py`).
- `gauge/temperature_conservation.csv` (per condition: `wu_row_norm`,
  `lnf_gain`, `norm_x_gain`, `logit_scale`, `nll`) and `gauge/gauge_landscape.csv`
  (NLL over the along-gauge $\alpha$ / off-gauge $\beta$ grid)
  (`run_gauge_landscape.py`).
- `swap/swap_grid_recipe_control.csv`: cross-condition body x readout grid under
  `none` / `row_norm` / `procrustes` / `wu_plus_ln` (`run_recipe_control_swap.py`).

This repo ships no figure-rendering code; the recipe-control figures are rendered
in the paper LaTeX tree from these files.

## Inputs (SSD canonical paths)
- A tokenized Pile slice, a flat `uint16` `.bin` of GPT-NeoX-20B token ids.
  Build one of two ways:
  - `${UM_SSD_ROOT}/pile_slice/pile_neox20b.bin`: tokenize a fresh slice from
    the live `monology/pile-uncopyrighted` mirror (faithful tokenizer +
    distribution, approximate order); or
  - the exact Pythia preshuffled order via a byte-range fetch of shard 0 of
    `EleutherAI/pile-standard-pythia-preshuffled` (no 602 GB download, no `.idx`).
- Checkpoints written by the trainer to `<out-dir>/ckpts/step<N>/model_fp16.pt`
  (the analysis input), or the released ones above, under `${UM_SSD_ROOT}/runs/<condition>/`.
  The trainer itself needs **no** SSD inputs beyond the token `.bin`.
- `${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin` (+ `.sha256`): the 2M-token
  held-out slice for the swap / probe / gauge analyses, range-fetched from the
  preshuffled shard immediately after the trained 10B tokens, so it is
  in-distribution and unseen by every arm (`fetch_heldout_slice.py`).

## Reproduce
The trainer, data prep, and analyses are self-contained (torch, transformers,
numpy, scikit-learn, datasets, huggingface_hub). The batch-scheduler launch glue
(job self-chaining, GPT-NeoX config rendering, the `readout-lr-multiplier` NeoX
patch) is **not** ported; the runs above used the hybrid HF trainer directly,
driven by CLI. Steps 5 onward read the released checkpoints from
`${UM_SSD_ROOT}/runs` and write under
`results/experiments/ablations/pretraining_recipe_control/` by default; every
script takes `--ckpt-root`, `--conditions` (default: the four paper arms; add
`warmup_long` for the fifth), `--seed`, and `--device`.

1. **CPU sanity check** (no GPU, no data; validates the GPTNeoX API, init,
   param groups, forward/backward against the installed transformers):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/selfcheck.py`

2. **Tokenize a slice** (faithful order):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/tokenize_slice.py --out ${UM_SSD_ROOT}/pile_slice/pile_neox20b.bin --target-tokens 10e9`
   or fetch the exact Pythia preshuffled prefix:
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/fetch_pythia_preshuffled.py --out ${UM_SSD_ROOT}/pile_slice/pythia_preshuffled.bin --target-tokens 10e9`
   (pass `--sequential-data` to the trainer for the preshuffled slice to preserve
   Pythia's released token order).

3. **Train each condition** (1 GPU each; `--max-tokens` makes checkpoints land at
   identical token points across conditions and batches). The arms differ only
   in the flags shown:
   - `baseline`:     `--readout-lr-mult 1.0  --warmup-steps 1430`
   - `wu_lr_0p25`:   `--readout-lr-mult 0.25 --warmup-steps 1430`
   - `wu_lr_4x`:     `--readout-lr-mult 4.0  --warmup-steps 1430`
   - `warmup_short`: `--readout-lr-mult 1.0  --warmup-steps 715`
   - `warmup_long`:  `--readout-lr-mult 1.0  --warmup-steps 5720`

   e.g. `uv run python experiments/ablations/pretraining_recipe_control/scripts/train_control.py --data ${UM_SSD_ROOT}/pile_slice/pile_neox20b.bin --out-dir ${UM_SSD_ROOT}/runs/baseline --model-size 31M --global-batch 1024 --max-tokens 10e9 --readout-lr-mult 1.0 --warmup-steps 1430`
   (resumable from `latest.pt`; writes a `COMPLETE` sentinel at the budget;
   `--max-hours` gives a per-slot wall guard for self-chaining around job limits.)
   Or skip training: `hf download hematteo/readout-recipe-control --local-dir "${UM_SSD_ROOT}/runs"`.

4. **Analyze $W_U$ geometry** across conditions and checkpoints (CPU):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/analyze_readout_geometry.py --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline warmup_short wu_lr_0p25 wu_lr_4x --out-csv results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv`
   (add `warmup_long` to `--conditions` to include the extra arm).

5. **Fit one trajectory crosscoder per arm** (one GPU, tens of minutes for the
   four arms; resumable, skips arms whose output exists):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/train_control_crosscoder.py --ckpt-root ${UM_SSD_ROOT}/runs --conditions wu_lr_0p25 baseline wu_lr_4x warmup_short --seed 0 --out-dir results/experiments/ablations/pretraining_recipe_control/crosscoders`
   The recipe is the paper's Pythia-160M dictionary column mapped to $d=256$
   ($d_{sae}=8192$, $\lambda_1=0.3$, 300 epochs, `center_scale`); see the script
   docstring. CPU smoke test: add `--device cpu --expansion-factor 0.5 --n-epochs 2 --batch-size 4096 --conditions baseline`.

6. **Lifecycle statistics** (CPU; reads the crosscoders and the checkpoints):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/analyze_lifecycle.py --ckpt-root ${UM_SSD_ROOT}/runs --cc-dir results/experiments/ablations/pretraining_recipe_control/crosscoders --conditions wu_lr_0p25 baseline wu_lr_4x warmup_short --seed 0 --d-sae 8192 --out-dir results/experiments/ablations/pretraining_recipe_control/lifecycle_seed0`
   (`--no-rates` skips the firing-rate pass; `--d-sae` must match the filenames
   from step 5.) For the seed replication, repeat steps 5 and 6 with
   `--seed 1` and `--seed 2`, then
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/aggregate_seed_replication.py --seed-stat results/experiments/ablations/pretraining_recipe_control/lifecycle_seed0/lifecycle_stats.json --seed-stat results/experiments/ablations/pretraining_recipe_control/lifecycle_seed1/lifecycle_stats.json --seed-stat results/experiments/ablations/pretraining_recipe_control/lifecycle_seed2/lifecycle_stats.json --seeds 0 1 2 --out-json results/experiments/ablations/pretraining_recipe_control/lifecycle_multiseed_summary.json --out-csv results/experiments/ablations/pretraining_recipe_control/lifecycle_multiseed_table.csv`

7. **Held-out slice** for the swap / probe / gauge analyses (network; 4 MB):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/fetch_heldout_slice.py --out ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin --start-token 10e9 --eval-tokens 2e6`

8. **Within-trajectory readout swap grid** (one GPU, minutes; resumable per cell):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/run_trajectory_swap.py --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short --h-steps 256 512 1024 2048 4769 --eval-bin ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin --eval-tokens 400000 --out-dir results/experiments/ablations/pretraining_recipe_control/trajectory`
   Cross-condition supplement (final checkpoint, with the physical-swap oracle check):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/run_recipe_control_swap.py --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short --eval-bin ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin --eval-tokens 2000000 --verify --out-csv results/experiments/ablations/pretraining_recipe_control/swap/swap_grid_recipe_control.csv`

9. **Expression lag** (availability probe vs native / best readout; one GPU or CPU):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/run_expression_lag.py --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short --h-steps 256 512 1024 2048 4769 --eval-bin ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin --eval-tokens 400000 --max-positions 20000 --out-dir results/experiments/ablations/pretraining_recipe_control/expression_lag`

10. **Gauge analysis** (temperature conservation + loss landscape):
    `uv run python experiments/ablations/pretraining_recipe_control/scripts/run_gauge_landscape.py --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short --landscape-cond baseline --eval-bin ${UM_SSD_ROOT}/pile_slice/heldout_after10B_2M.bin --eval-tokens 200000 --grid 21 --log2-range 2.0 --out-dir results/experiments/ablations/pretraining_recipe_control/gauge`

Steps 8 to 10 run on CPU at reduced `--eval-tokens` / `--grid` (e.g. `--eval-tokens 8192 --seq-len 256 --device cpu`).

## Layout
| path | role |
|---|---|
| `scripts/train_control.py`         | self-contained Pythia-style trainer; 3 optimizer groups (decay / no-decay / readout-$W_U$ with own LR mult), token-budget stopping, token-milestone checkpointing, held-out val loss + $W_U$ geometry, resumable |
| `scripts/tokenize_slice.py`        | stream + tokenize the Pile (GPT-NeoX-20B tokenizer) into a flat `uint16` `.bin`; deterministic, retry-hardened, resumable |
| `scripts/selfcheck.py`             | CPU sanity check of the model API / init / param groups / forward |
| `scripts/fetch_pythia_preshuffled.py` | byte-range fetch of the exact Pythia preshuffled Pile prefix (no `.idx`, no 602 GB download); drop-in `.bin` for the trainer |
| `scripts/analyze_readout_geometry.py` | per-checkpoint $W_U$ geometry (row norm stats, SVD spectrum, eff/stable rank) vs. $W_E$ and the final-LN gain, written to a tidy CSV |
| `scripts/train_control_crosscoder.py` | one W_U trajectory crosscoder per arm over its checkpoint trajectory (`readout.crosscoder.wu_adapter`; Pythia-160M dictionary recipe mapped to $d=256$); resumable |
| `scripts/analyze_lifecycle.py`     | per-arm lifecycle statistics of the crosscoder features (decoder norm trajectories, alive threshold, peak step, rule-based profiles, norm turnover / decoder rotation, firing rates) written as CSV / JSON / `.pt` |
| `scripts/aggregate_seed_replication.py` | pools `lifecycle_stats.json` over crosscoder seeds; per-seed readout LR ordering checks |
| `scripts/fetch_heldout_slice.py`   | range-fetch of the 2M-token held-out slice after the trained region (+ sha256) |
| `scripts/run_trajectory_swap.py`   | per-arm (h_t, s) readout swap grid under the 5-mode gauge ladder; resumable shards |
| `scripts/run_recipe_control_swap.py` | cross-condition body x readout grid at the final checkpoint (`none` / `row_norm` / `procrustes` / `wu_plus_ln`), with the physical-swap oracle (`--verify`) |
| `scripts/run_expression_lag.py`    | availability probe (k-fold logistic + nulls) vs native and best readout accuracy on the next-token frequency tier |
| `scripts/run_gauge_landscape.py`   | temperature conservation across arms and the along-gauge / off-gauge NLL landscape |
| `scripts/rc_common.py`             | sibling helpers: default paths, argument groups, held-out loader, one-forward hidden cache, CSV / provenance writers |
| `results/experiments/ablations/pretraining_recipe_control/` | all metric outputs listed above (generated on run; not shipped) |

Checkpoint loading and the model rebuild live in
`src/readout/probes/recipe_control_models.py`; the swap kernel
(`swap_cell_metrics`), the physical-swap oracle (`readout.probes.weight_swap`),
and the availability probe (`readout.probes.availability_probe`) are library
code with CPU unit tests under `tests/`.
