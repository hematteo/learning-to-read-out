# pretraining_recipe_control

The controlled training experiment behind the paper's Finding 2: matched 31M
Pythia-style runs that share initialization, tokenized data, and data order, and
differ in one recipe factor each. The paper shows that raising the output readout
learning rate moves the readout reorganization one log-spaced checkpoint bin
earlier per 4x increase, at matched validation loss, and that halving the warmup
leaves the timing unchanged. The reproducible in-repo artifact is the $W_U$
readout geometry across conditions and token-milestone checkpoints; the trained
checkpoints themselves are released on Hugging Face.

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| `fig:app-recipe-control-geometry` | `results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv` | `experiments/ablations/pretraining_recipe_control/scripts/analyze_readout_geometry.py` |

Only the $W_U$-geometry view is reproducible in-repo (from the regenerated
`readout_geometry_pythia.csv`). The other recipe-control figures
(`fig:main-recipe-dose-response`, `fig:app-recipe-control-{median,trajectories,peakstep,reorg,basin,explag,temperature,gauge-landscape}`)
and `tab:app-recipe-control-summary` need the per-condition trajectory
crosscoders (d_sae 8192, K=16), lifecycle statistics, readout swaps, and probes
over these checkpoints, which were run outside this release. See
[`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure to metric map.

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
2464 to 2497).

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
layout the analysis script expects:

```bash
hf download hematteo/readout-recipe-control --local-dir "${UM_SSD_ROOT}/runs"
```

## Result

The readout LR multiplier is load-bearing for $W_U$ geometry. At 10B tokens the
$W_U$ row-norm mean separates cleanly by condition (numbers reproduced from
`readout_geometry_pythia.csv`, which `analyze_readout_geometry.py` regenerates
from the released checkpoints):

| condition | tokens | $W_U$ rn_mean | rn_max | stable_rank | top-1 σ frac |
|---|---|---|---|---|---|
| `wu_lr_0p25` | 10.0B | 0.93 | 1.73 | 6.97 | 14.4% |
| `baseline`   | 10.0B | 1.22 | 2.44 | 8.35 | 12.0% |
| `wu_lr_4x`   | 10.0B | 1.85 | 3.75 | 12.88 | 7.8% |

Faster readout learning grows $W_U$ row norms faster and spreads the spectrum;
the paper's appendix shows the row norm and the final LayerNorm gain trade off
along a loss-flat symmetry direction so that the effective logit scale is
conserved. Whether this changes the availability/expression gap is the
downstream probe analysis over these checkpoints reported in the paper; the
training run itself produces the reloadable checkpoints and the cheap geometry.

## Metrics produced
- `results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv`:
  per-checkpoint $W_U$ geometry across conditions (row-norm stats `rn_mean`/`rn_max`,
  SVD spectrum incl. `top1_frac`, eff/stable rank, `lnf_norm`) vs. tokens, written
  by `scripts/analyze_readout_geometry.py`. This is the reproducible artifact
  behind the paper's recipe-control geometry figure (gitignored, regenerated on
  run; not shipped in this code-only release).

This repo ships no figure-rendering code; the recipe-control figures are rendered
in the paper LaTeX tree from this CSV. The downstream timing, peak-step, and
reorganization-window figures further require per-condition feature trajectories
that are not computed here.

## Inputs (SSD canonical paths)
- A tokenized Pile slice, a flat `uint16` `.bin` of GPT-NeoX-20B token ids.
  Build one of two ways:
  - `${UM_SSD_ROOT}/pile_slice/pile_neox20b.bin`: tokenize a fresh slice from
    the live `monology/pile-uncopyrighted` mirror (faithful tokenizer +
    distribution, approximate order); or
  - the exact Pythia preshuffled order via a byte-range fetch of shard 0 of
    `EleutherAI/pile-standard-pythia-preshuffled` (no 602 GB download, no `.idx`).
- Checkpoints written by the trainer to `<out-dir>/ckpts/step<N>/model_fp16.pt`
  (the analysis input), or the released ones above. The trainer itself needs
  **no** SSD inputs beyond the token `.bin`.

## Reproduce
The trainer, data prep, and analysis are self-contained (torch, transformers,
numpy, datasets, huggingface_hub). The batch-scheduler launch glue (job
self-chaining, GPT-NeoX config rendering, the `readout-lr-multiplier` NeoX
patch) is **not** ported; the runs above used the hybrid HF trainer directly,
driven by CLI.

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

4. **Analyze $W_U$ geometry** across conditions and checkpoints (CPU):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/analyze_readout_geometry.py --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline warmup_short wu_lr_0p25 wu_lr_4x --out-csv results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv`
   (add `warmup_long` to `--conditions` to include the extra arm).

## Layout
| path | role |
|---|---|
| `scripts/train_control.py`         | self-contained Pythia-style trainer; 3 optimizer groups (decay / no-decay / readout-$W_U$ with own LR mult), token-budget stopping, token-milestone checkpointing, held-out val loss + $W_U$ geometry, resumable |
| `scripts/tokenize_slice.py`        | stream + tokenize the Pile (GPT-NeoX-20B tokenizer) into a flat `uint16` `.bin`; deterministic, retry-hardened, resumable |
| `scripts/selfcheck.py`             | CPU sanity check of the model API / init / param groups / forward |
| `scripts/fetch_pythia_preshuffled.py` | byte-range fetch of the exact Pythia preshuffled Pile prefix (no `.idx`, no 602 GB download); drop-in `.bin` for the trainer |
| `scripts/analyze_readout_geometry.py` | per-checkpoint $W_U$ geometry (row-norm stats, SVD spectrum, eff/stable rank) vs. $W_E$ and the final-LN gain, written to a tidy CSV |
| `results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv` | the geometry analysis output (generated on run; not shipped) |
