# contrastive_readout_swap

Contrastive (`y+` / `y-`) readout-swap heatmaps for single-token, two-choice tasks.
For each task, the experiment scores `margin(t, s) = h_t · W_U^s[y+] - h_t · W_U^s[y-]`
across all `(h_step, s_step)` checkpoint pairs and reports the readout-rescue
relative to the native `(t, t)` cell, testing whether early hidden states already
contain task-relevant signal that a same-checkpoint readout has not yet expressed.

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| `fig:main-sva-availability-expression` | controlled hidden-state probe `<stem>.csv` (probe vs native readout accuracy per checkpoint) | `experiments/probes/contrastive_readout_swap/scripts/run_controlled_hidden_probes.py` |
| `fig:app-contrastive-readout-lag` | `results/experiments/probes/contrastive_readout_swap/run0_pythia1b/summary.csv` (+ controlled-hidden-probe `<stem>.csv`) | `experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py` |

This experiment computes and persists the swap-grid and controlled-hidden-probe
metrics that the paper's availability/expression figures summarize. The
Pythia-6.9B narrative figures (`fig:main-sva-availability-expression`,
`fig:app-contrastive-readout-lag`) and the availability/expression appendix
report (`fig:app-ae-*`, `tab:app-ae-*`) are rendered in the paper LaTeX tree from
probe-summary CSVs; this experiment generates the upstream probe signal but the
report figures themselves are not shipped or regenerable here.

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim

Task-relevant signals can exist in early hidden states *before* the native
unembedding expresses them well. For a contrastive task with answer pair
`(y+, y-)`, score the readout-rescue:

```
margin(t, s)        = h_t · W_U^s[y+] - h_t · W_U^s[y-]
readout-rescue(t, s) = margin(t, s) - margin(t, t)
```

A positive, localized rescue means the hidden state already contains
task-relevant information that the same-checkpoint readout has not yet
made visible.

## Task families

Single-token, two-choice tasks (no benchmark-format noise):

| family             | example prompt                                             | y+ / y- |
|--------------------|------------------------------------------------------------|---------|
| `sva`              | `The keys to the cabinet`                                  | ` are` / ` is` |
| `induction`        | `A B A B A` (random A, B from token pool)                  | ` B` / matched random |
| `ioi`              | `Alice and Bob went to the store. Alice gave the book to`  | ` Bob` / ` Alice` |
| `numeric_gt`       | `17 is greater than`                                       | ` 16` / ` 17` |
| `relational_facts` | `The capital of France is`                                 | ` Paris` / matched capital |
| `hypernym`         | `A robin is a type of`                                     | ` bird` / matched category |

`src/readout/probes/contrastive_tasks.py` also registers six benchmark-derived families
(`piqa`, `arc_easy`, `arc_challenge`, `sciq`, `lambada`, `winogrande`). Each
emits the standard `<family>.jsonl`, plus a `<family>_random_distractors.json`
sidecar and (for the MCQ-letter families) a `<family>_label_permutation.json`
sidecar.

Filtering: both answers must be single tokenizer tokens, leading-space matched,
and (when the terminal `W_U` is available) row-norm matched within `|log r| < 0.5`.

## Models

Primary: `EleutherAI/pythia-1b`, seed 0.
Replication: `EleutherAI/pythia-160m`.
Optional scale check: `EleutherAI/pythia-6.9b` (hidden-state extraction
fits on a single A100 at fp32 final-position only).

## Reproduce (in order)

```bash
# 1. Build per-tokenizer task datasets (jsonl, one file per family).
uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
    --model EleutherAI/pythia-1b \
    --out-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b

# 2. Score the (h_step, s_step, alignment) grid. Resume-safe per cell.
#    Writes shards/*.json (+ optional .pt with --save-per-example),
#    manifest.json, and an aggregated summary.csv on completion.
uv run python experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py \
    --model pythia-1b \
    --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
    --h-steps 256 512 1000 2000 8000 143000 \
    --alignments none scale row_norm procrustes \
    --out-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b

# 3. Optional: run leakage-controlled hidden probes on cached hidden states.
#    Writes <stem>.csv and <stem>.pt.
uv run python experiments/probes/contrastive_readout_swap/scripts/run_controlled_hidden_probes.py \
    --hidden-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b/hidden \
    --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b \
    --out-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b/figures
```

These steps compute and persist the metrics behind the experiment's figures
(`summary.csv`, per-cell shards, controlled-probe CSV/.pt). The repo ships no
figure-rendering code; paper figures are rendered in the separate thesis LaTeX
tree from these regenerated metrics (gitignored — not shipped in this
code-only release).

`run_swap_grid.py --save-per-example` additionally writes a per-cell sidecar
`shards/<family>__a-<al>__h<h>__s<s>.pt` carrying `{margins, margins_native,
y_plus, y_minus}` alongside the aggregate JSON. The JSON is written **after**
the `.pt` so it remains a valid resume marker; a stale JSON without a matching
`.pt` is auto-deleted on restart.

### Harder IOI probe

The original `ioi` probe predicts recipient identity, which is mostly a prompt
surface-feature control. `ioi_role_balanced` is a harder binary probe: the
target is whether the indirect object is the first or second listed name. For
each fixed ordered name pair, paired examples share the same bag of prompt
tokens and the same name counts.

Build the dataset:

```bash
uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
    --model EleutherAI/pythia-1b \
    --families ioi_role_balanced \
    --n-max-per-family 2000 \
    --out-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b_hard_ioi \
    --no-norm-match
```

Extract hidden states for the new prompts. On the GPU host, upload the dataset
dir and run only this family across all 32 hidden checkpoints:

```bash
FAMILIES_OVERRIDE="ioi_role_balanced" HIDDEN_STEPS_MODE=all \
    bash experiments/probes/contrastive_readout_swap/scripts/run_4gpu_extraction.sh \
    /workspace/out /workspace/datasets 4
```

After downloading the hidden cache, run:

```bash
uv run python experiments/probes/contrastive_readout_swap/scripts/run_controlled_hidden_probes.py \
    --families ioi_role_balanced \
    --hidden-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b_hard_ioi/hidden \
    --datasets-dir results/experiments/probes/contrastive_readout_swap/datasets/pythia-1b_hard_ioi \
    --stem main_claim_controlled_hidden_probe_trajectory_ioi_role_balanced \
    --out-dir results/experiments/probes/contrastive_readout_swap/run0_pythia1b_hard_ioi/figures
```

Replicate on Pythia-160M by repeating with `--model pythia-160m` (and a fresh
datasets dir, since tokenizers differ in row-norm filtering).

### Multi-GPU extraction (1B screen + 6.9B confirmatory)

The generic launcher `launch_swap_extraction.sh` covers both models:

```bash
# 1B screening run (all families, full 32-step s-grid)
bash experiments/probes/contrastive_readout_swap/scripts/launch_swap_extraction.sh \
    --model pythia-1b \
    --out-root /workspace/swap_1b_$(date -u +%Y%m%dT%H%M) \
    --build-datasets

# 6.9B confirmatory (only winners from the 1B screen)
bash experiments/probes/contrastive_readout_swap/scripts/launch_swap_extraction.sh \
    --model pythia-6.9b \
    --out-root /workspace/swap_69b_$(date -u +%Y%m%dT%H%M) \
    --families "numeric_gt piqa arc_challenge sciq" \
    --h-steps "256 512 1000 2000 8000 143000" \
    --dtype bf16 \
    --build-datasets
```

Each launcher writes `<out-root>/manifest.txt` summarising shard / sidecar /
hidden-cache counts; `<out-root>/.swap_grid_complete` is touched on success.

### Reporting (held-out best-s + paired bootstrap)

```bash
uv run python experiments/probes/contrastive_readout_swap/scripts/run_step6_reporting.py \
    --run-dir <run>/swap_grid \
    --out-dir results/experiments/probes/contrastive_readout_swap/reporting_pythia1b
# → heldout_best_s_selection.csv (dev/test best-s selection),
#   paired_bootstrap_ci.csv (paired bootstrap CI on best_s - native),
#   preselected_readout_comparison.csv.
```

## Controls

1. **Same-checkpoint baseline** — every cell is reported as `delta_*` against
   the native `(h_t, W_U^t)` cell.
2. **Future-full-model baseline** — read `(s, s)` cells from the same scored grid
   to separate hidden-state maturity from readout maturity.
3. **Matched random distractors** — pass `--use-random-distractors` to swap
   `y_minus` for a random single-token; effect should weaken if it was
   driven by answer-token frequency / norm rather than task structure.
4. **Label-permutation control** — `--use-label-permutation` swaps the
   MCQ-letter shuffle control for the benchmark-derived families.
5. **Gauge-alignment ladder** — `--alignments scale row_norm procrustes`
   matches the alignment ladder of `experiments/causal/temporal_localization_patching/scripts/run_aligned_swap_grid.py`.
   The rescue is real-not-gauge if it survives Procrustes.
6. **Prompt-corruption pairs** — `build_task_datasets.py` writes
   `<family>_corrupt.jsonl` for SVA (label flip) and IOI (entity-name swap).
   Run the swap grid on those with `--families <family>_corrupt`.
7. **Late-readout control** — the terminal step `s = 143000` column should
   not always win uniformly across `t`; if it does, the result is non-specific.

## Inputs (SSD canonical paths)

- `${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step*_wu.pt`
- `${UM_SSD_ROOT}/snapshots/pythia-160m/EleutherAI_pythia-160m_step*_wu.pt`

Hidden states at each `h_step` are forwarded once and cached under
`<out-dir>/hidden/<family>_h<step>.pt`. Reused by the sparse-feature
attribution experiment (`causal/contrastive_task_feature_rescue`).

## Outputs / Layout

| path | role |
|---|---|
| `scripts/build_task_datasets.py`  | tokenizer-specific contrastive datasets (jsonl + control sidecars + `summary.json`) |
| `scripts/run_swap_grid.py`        | (h, s, alignment, family) grid, resume-safe; writes shards + `summary.csv` |
| `scripts/run_controlled_hidden_probes.py` | leakage-controlled hidden probes with prompt-token baselines; writes `<stem>.csv` + `<stem>.pt` |
| `scripts/run_step6_reporting.py`  | held-out best-s selection + paired bootstrap from shards; writes reporting CSVs |
| `scripts/extract_wu_hidden_standalone.py` | standalone `W_U` + hidden-state extraction (`*_wu.pt`, `<family>_h<step>.pt`) |
| `scripts/run_4gpu_extraction.sh` | 4-GPU extraction wrapper; supports `FAMILIES_OVERRIDE` and `HIDDEN_STEPS_MODE=all` |
| `scripts/launch_swap_extraction.sh` | generic multi-GPU launcher for 1B/6.9B swap-grid extraction |

Library helpers: [src/readout/probes/contrastive_tasks.py](../../../src/readout/probes/contrastive_tasks.py),
[src/readout/probes/readout_swap.py](../../../src/readout/probes/readout_swap.py).

## Success criteria

This supports the stronger "capability"-adjacent framing if:
- multiple task families show positive readout-rescue at early checkpoints,
- rescue peaks near the early readout-change window (~step 1k for Pythia-1B),
  not uniformly across all future readouts,
- the effect survives the gauge-alignment ladder,
- effects are stronger on structured tasks than under matched-random distractors,
- Pythia-160M qualitatively replicates the Pythia-1B pattern.

If only SVA + induction show a positive rescue, the language downgrades to
"simple task-relevant signals." If relational facts also show it, the stronger
"capability-relevant readout formation" framing is defensible.