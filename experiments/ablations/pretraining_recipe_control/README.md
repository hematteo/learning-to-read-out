# pretraining_recipe_control

A from-scratch sensitivity/falsification control for the Chapter 4 learning-to-read-out
results: four matched 31M Pythia-style runs, each perturbing one optimizer-recipe axis,
testing whether readout-feature timing and the availability–expression lag are stable
under or sensitive to plausible recipe perturbations. The reproducible in-repo artifact is
the $W_U$ readout geometry across conditions × token-milestone checkpoints.

## Figures produced

| Thesis label | Metric file | Producing script |
|---|---|---|
| `fig:app-recipe-control-geometry` | `results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv` | `experiments/ablations/pretraining_recipe_control/scripts/analyze_readout_geometry.py` |

Only the F1 $W_U$-geometry view is reproducible in-repo (from the regenerated
`readout_geometry_pythia.csv`). The other Appendix CC recipe-control figures and
`tab:app-recipe-control-summary` are thesis-tree-only and need per-condition feature
dictionaries that are not shipped here. See
[`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim
A from-scratch sensitivity check for the Chapter 4 learning-to-read-out results:
are readout-feature timing and the availability–expression lag *stable under*, or
*sensitive to*, plausible perturbations of a Pythia-style optimizer recipe? Four
matched conditions are trained at **31M parameters, global batch 1024 sequences,
to a 10B-token budget**, sharing one tokenized slice, seed, and data order; each
perturbs one axis vs. the `baseline` reference:

| id | name | perturbation vs. baseline |
|---|---|---|
| C0 | `baseline`    | none (reference; readout LR mult `m=1.0`, warmup 1430) |
| C1 | `long_warmup` | warmup 1430 → 2860 steps (2×) |
| C2 | `wu_lr_0p25`  | output-readout ($W_U$) LR multiplier `m=0.25` |
| C3 | `wu_lr_4x`    | output-readout ($W_U$) LR multiplier `m=4.0` |

This is the designed suite (see the recipe-control appendix). The regenerated
geometry artifact covers a subset — see **Data coverage** below.

The trainer is a **hybrid** built on HF `GPTNeoXForCausalLM` (which *is* the
Pythia architecture) on the modern torch stack, reproducing the Pythia recipe
shape (small/Wang init, AdamW (0.9, 0.95), faithful cosine+warmup over the 143k
horizon stopped early, fp16 + dynamic loss scaling, GPT-NeoX-20B tokenizer); the
2022-era GPT-NeoX/DeeperSpeed stack is incompatible with the available cluster.
The baseline is a Pythia-*style* proxy, **not** a reproduction of the public
Pythia run. See the recipe-control appendix in the thesis for the full recipe
specification, faithfulness summary, and what is/ isn't matched to Pythia.

## Result
The readout LR multiplier is load-bearing for $W_U$ geometry. At 10B tokens the
$W_U$ row-norm mean separates cleanly by condition (the numbers below are
reproduced from `readout_geometry_pythia.csv`, which
`analyze_readout_geometry.py` regenerates on run):

| condition | tokens | $W_U$ rn_mean | rn_max | stable_rank | top-1 σ frac |
|---|---|---|---|---|---|
| `wu_lr_0p25` | 10.0B | 0.93 | 1.73 | 6.97 | 14.4% |
| `baseline`   | 10.0B | 1.22 | 2.44 | 8.35 | 12.0% |
| `wu_lr_4x`   | 10.0B | 1.85 | 3.75 | 12.88 | 7.8% |

Faster readout learning grows $W_U$ row-norms faster and concentrates the
spectrum more; an early-dynamics preview also shows `wu_lr_4x` at loss ≈7.0 vs
≈9.1–9.2 for the others at ~0.3B tokens. Whether this **closes the
availability–expression lag** or merely lowers loss is a downstream probe
analysis over these checkpoints (a linear probe on final-layer hidden states vs.
the native $W_U$ margin); the training run only produces the reloadable
checkpoints and cheap geometry.

### Data coverage (what the regenerated CSV contains)
The `readout_geometry_pythia.csv` that `analyze_readout_geometry.py`
writes is **partial vs. the designed suite**. It contains the readout-LR axis in
full plus a *short*-warmup probe — it does **not** contain the designed
`long_warmup` (2×) condition:

| in CSV | warmup | reached | role |
|---|---|---|---|
| `baseline`   | 1430 | 10.0B | reference |
| `wu_lr_0p25` | 1430 | 10.0B | $W_U$-LR axis (slow) |
| `wu_lr_4x`   | 1430 | 10.0B | $W_U$-LR axis (fast) |
| `warmup_short` | 715 | 8.6B (partial) | exploratory ½× warmup; **not** the designed `long_warmup` |

So the headline $W_U$-LR finding is fully backed by data; the warmup axis is only
represented by an incomplete *short*-warmup run in the opposite direction from the
designed `long_warmup`. To reproduce that CSV exactly, run
`analyze_readout_geometry.py --conditions baseline wu_lr_0p25 wu_lr_4x warmup_short`
(the script defaults to the *designed* four, which include `long_warmup`).

## Metrics produced
- `results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv` —
  per-checkpoint $W_U$ geometry across conditions (row-norm stats `rn_mean`/`rn_max`,
  SVD spectrum incl. `top1_frac`, eff/stable rank, `lnf_norm`) vs. tokens, written
  by `scripts/analyze_readout_geometry.py`. This is the reproducible artifact
  behind the Chapter 4 recipe-control geometry figure (gitignored, regenerated on
  run — not shipped in this code-only release).

This repo ships no figure-rendering code; the recipe-control figures are rendered
in the thesis LaTeX tree from this CSV. The downstream geometry-trajectory,
timing/dose-response, peak-step, and reorg-window figures further require
per-condition feature trajectories / per-checkpoint activations that are not
committed here.

## Inputs (SSD canonical paths)
- A tokenized Pile slice — a flat `uint16` `.bin` of GPT-NeoX-20B token ids.
  Build one of two ways:
  - `${UM_SSD_ROOT}/pile_slice/pile_neox20b.bin` — tokenize a fresh slice from
    the live `monology/pile-uncopyrighted` mirror (faithful tokenizer +
    distribution, approximate order); or
  - the exact Pythia preshuffled order via a byte-range fetch of shard 0 of
    `EleutherAI/pile-standard-pythia-preshuffled` (no 602 GB download, no `.idx`).
- Checkpoints written by the trainer to `<out-dir>/ckpts/step<N>/model_fp16.pt`
  (the analysis input). The trainer itself needs **no** SSD inputs beyond the
  token `.bin`.

## Reproduce
The trainer, data prep, and analysis are self-contained (torch, transformers,
numpy, datasets, huggingface_hub). The batch-scheduler launch glue (job
self-chaining, GPT-NeoX config rendering, the `readout-lr-multiplier` NeoX
patch) is **not** ported — the runs above used the hybrid HF trainer directly,
driven by CLI.

1. **CPU sanity check** (no GPU, no data — validates the GPTNeoX API, init,
   param groups, forward/backward against the installed transformers):
   `uv run python experiments/ablations/pretraining_recipe_control/trainer/selfcheck.py`

2. **Tokenize a slice** (faithful order):
   `uv run python experiments/ablations/pretraining_recipe_control/trainer/tokenize_slice.py --out ${UM_SSD_ROOT}/pile_slice/pile_neox20b.bin --target-tokens 10e9`
   — or fetch the exact Pythia preshuffled prefix:
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/fetch_pythia_preshuffled.py --out ${UM_SSD_ROOT}/pile_slice/pythia_preshuffled.bin --target-tokens 10e9`
   (pass `--sequential-data` to the trainer for the preshuffled slice to preserve
   Pythia's released token order).

3. **Train each condition** (1 GPU each; `--max-tokens` makes checkpoints land at
   identical token points across conditions and batches). The four headline
   conditions differ only in the flags shown:
   - `baseline`:    `--readout-lr-mult 1.0  --warmup-steps 1430`
   - `long_warmup`: `--readout-lr-mult 1.0  --warmup-steps 2860`
   - `wu_lr_0p25`:  `--readout-lr-mult 0.25 --warmup-steps 1430`
   - `wu_lr_4x`:    `--readout-lr-mult 4.0  --warmup-steps 1430`

   e.g. `uv run python experiments/ablations/pretraining_recipe_control/trainer/train_control.py --data ${UM_SSD_ROOT}/pile_slice/pile_neox20b.bin --out-dir ${UM_SSD_ROOT}/runs/baseline --model-size 31M --global-batch 1024 --max-tokens 10e9 --readout-lr-mult 1.0 --warmup-steps 1430`
   (resumable from `latest.pt`; writes a `COMPLETE` sentinel at the budget;
   `--max-hours` gives a per-slot wall guard for self-chaining around job limits.)

4. **Analyze $W_U$ geometry** across conditions × checkpoints (CPU):
   `uv run python experiments/ablations/pretraining_recipe_control/scripts/analyze_readout_geometry.py --ckpt-root ${UM_SSD_ROOT}/runs --conditions baseline long_warmup wu_lr_0p25 wu_lr_4x --out-csv results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv`

## Layout
| path | role |
|---|---|
| `trainer/train_control.py`         | self-contained Pythia-style trainer; 3 optimizer groups (decay / no-decay / readout-$W_U$ with own LR mult), token-budget stopping, token-milestone checkpointing, held-out val loss + $W_U$ geometry, resumable |
| `trainer/tokenize_slice.py`        | stream + tokenize the Pile (GPT-NeoX-20B tokenizer) into a flat `uint16` `.bin`; deterministic, retry-hardened, resumable |
| `trainer/selfcheck.py`             | CPU sanity check of the model API / init / param groups / forward |
| `scripts/fetch_pythia_preshuffled.py` | byte-range fetch of the exact Pythia preshuffled Pile prefix (no `.idx`, no 602 GB download); drop-in `.bin` for the trainer |
| `scripts/analyze_readout_geometry.py` | per-checkpoint $W_U$ geometry (row-norm stats, SVD spectrum, eff/stable rank) vs. $W_E$ and the final-LN gain → tidy CSV |
| `results/experiments/ablations/pretraining_recipe_control/readout_geometry_pythia.csv` | the geometry analysis output for the four trained conditions (generated on run; not shipped) |
