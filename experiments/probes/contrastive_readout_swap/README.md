# contrastive_readout_swap

Contrastive (`y+` / `y-`) readout analyses for single-token, two-choice tasks.
Two measurements share the task datasets and hidden-state caches:

- **Readout swap.** `margin(t, s) = h_t · W_U^s[y+] - h_t · W_U^s[y-]` over all
  `(h_step, s_step)` checkpoint pairs, with the readout-rescue relative to the
  native `(t, t)` cell (`run_swap_grid.py`, plus leakage-controlled hidden-state
  probes in `run_controlled_hidden_probes.py`).
- **Availability vs expression** (paper Section 4.3 and Appendix G). For 20 task
  families and 24 checkpoints: is the task feature linearly decodable from the
  final-position hidden state (availability, a single-direction shrinkage-LDA
  probe `A1` and a logistic probe `A_full`, on folds grouped by lexical item),
  and does the checkpoint's own readout `LN_t(z_t) · W_U^t` already separate the
  answers (expression `E`), compared with the best readout of any other
  checkpoint (`B`) (`run_availability_probes.py`).

## Figures produced

Camera-ready numbering: Figure 4 is in Section 4.3; the G labels are Appendix G.

| Paper label | Metric file | Producing script |
|---|---|---|
| `fig:main-sva-availability-expression` (Figure 4), left panel | `run_pythia-6.9b_probe_trajectory/summary.csv` (native `s = h` and best-over-`s` accuracy, SVA) and `run_pythia-6.9b_probe_trajectory/controlled/probe_trajectory_69b.csv` (`sva_number_lemma_grouped`, `feature_source=hidden`, `acc`) | `run_swap_grid.py`, `run_controlled_hidden_probes.py` |
| `fig:main-sva-availability-expression` (Figure 4), right panel | `availability/run_pythia-6.9b/probes/probe_summary.csv` (`h_step=143000`, `probe_acc_1d`, `native_readout_acc`; `sva`, `sva_across_pp`, `sva_subject_rc`, `sva_object_rc`) | `run_availability_probes.py` |
| `fig:app-contrastive-readout-lag` (Figure G.1) | the two left-panel files, families `sva` and `ioi_role_balanced` (panel B: the prompt-token / Gaussian rows of the controlled-probe CSV) | `run_swap_grid.py`, `run_controlled_hidden_probes.py` |
| `tab:app-ae-binary` (Table G.2), `tab:app-ae-openvocab` (Table G.3), `tab:app-ae-early` (Table G.4), `fig:app-ae-capacity` (Figure G.2), `fig:app-ae-grid` (Figure G.3) | `availability/run_pythia-6.9b/probes/probe_summary.csv`; the 1B columns of Table G.2 from `availability/run_pythia-1b{,_breadth}/probes/probe_summary.csv` | `run_availability_probes.py` |

All paths are under `results/experiments/probes/contrastive_readout_swap/`. The
figures and tables are typeset in the paper tree from these CSVs; no plotting
code ships here. See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the
full figure → metric map.

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

## Reproduce: availability vs expression (Section 4.3, Appendix G)

Task families (`readout.probes.availability_tasks`): the SVA hierarchy `sva`,
`sva_across_pp`, `sva_subject_rc`, `sva_object_rc` (attractor number independent of
the head, `tasks_sva_hierarchy`); `ioi_role_balanced`; BLiMP determiner-noun,
anaphor and NPI minimal pairs (`tasks_blimp`, downloads `nyu-mll/blimp`);
function-vector ICL tasks `fv_*` (`tasks_function_vectors`); `induction`;
head-to-head numeric comparison `numeric_{gt,lt,tf}_v2` (`tasks_numeric`);
`relational_facts{,_balanced}`; and the zero-point controls `copy_control`,
`fixed_token_control` (`tasks_controls`).

**Function-vector data.** The four `fv_*` families read
`antonym.json`, `english-french.json`, `present-past.json` and `country-capital.json`
from `github.com/ericwtodd/function_vectors` (`dataset_files/abstractive/`, MIT
licence, 530 KB) in `$FV_DATA_DIR` (default `data/function_vectors/` at the repo
root). The builder raises if a file is missing; drop the `fv_*` families to build
without them.

```bash
FAMS="sva sva_across_pp sva_subject_rc sva_object_rc ioi_role_balanced copy_control fixed_token_control \
  blimp_det_noun_agreement blimp_anaphor_agreement blimp_npi fv_antonym fv_country_capital fv_en_fr \
  fv_present_past induction numeric_gt_v2 numeric_lt_v2 numeric_tf_v2 relational_facts relational_facts_balanced"
STEPS="0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 5000 6000 8000 10000 16000 32000 64000 128000 143000"
OUT=results/experiments/probes/contrastive_readout_swap/availability

# Pythia-6.9B (all 20 families, 24 checkpoints).
# 1. Datasets. The published files were built without W_U row-norm matching.
uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
    --task-set availability --model EleutherAI/pythia-6.9b --families $FAMS \
    --n-max-per-family 2000 --rng-seed 0 --no-norm-match --out-dir $OUT/datasets/pythia-6.9b
# 2. Final-position h (post-LN) and z (pre-LN) per checkpoint, plus <slug>_step<S>_{wu,lnf}.pt
#    under ${UM_SSD_ROOT}/snapshots/pythia-6.9b/ (one 48 GB GPU at fp32).
uv run python experiments/probes/contrastive_readout_swap/scripts/extract_hidden_dense.py \
    --hf-model EleutherAI/pythia-6.9b --datasets-dir $OUT/datasets/pythia-6.9b --families $FAMS \
    --h-steps $STEPS --out-dir $OUT/run_pythia-6.9b/hidden --device cuda --dtype fp32 --batch-size 8
# 3. Probes and readouts (the 24 checkpoints are also the swept readouts).
uv run python experiments/probes/contrastive_readout_swap/scripts/run_availability_probes.py \
    --model EleutherAI/pythia-6.9b --hidden-dir $OUT/run_pythia-6.9b/hidden \
    --datasets-dir $OUT/datasets/pythia-6.9b --families $FAMS --swept-steps $STEPS --group-split \
    --device cuda --out-dir $OUT/run_pythia-6.9b/probes

# Pythia-1B, as published: two runs. The agreement run used the earlier subject-RC
# dataset (attractor always number-mismatched); the breadth run used an earlier
# runner without numeric pair groups (stratified folds) or BLiMP probe labels.
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

Omit `--subject-rc-attractor mismatched`, `--stratified-prefixes` and
`--unprobed-prefixes` for the current (6.9B) protocol at 1B. `run_availability_probes.py`
is CPU-only apart from the full-vocabulary rank GEMMs (`--device`); it resumes per
cell and writes `probe_summary.{csv,pt}` plus `probe_summary.provenance.json`.

**Reproducibility notes.** Probe folds come from `group_kfold_splits`, which
reproduces sklearn's `GroupKFold` with the tie order of x86-64 numpy (the paper
ran on x86; on macOS/arm64, sklearn's own `GroupKFold` assigns equal-size groups
to different folds). On CPU the label-shuffle and random-label logistic nulls can
differ from the published values by about one item, and the full-vocabulary mean
ranks by near-tie flips (rank-1 rates agree).

## Reproduce: Figure 4 left panel and Figure G.1

The published trajectory scored a bf16 model: hidden states from a bf16 forward
pass and bf16-rounded W_U snapshots (`--dtype bf16 --wu-round bf16`).

```bash
RUN=results/experiments/probes/contrastive_readout_swap/run_pythia-6.9b_probe_trajectory
DS=results/experiments/probes/contrastive_readout_swap/datasets/pythia-6.9b_trajectory
STEPS32="0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 5000 6000 7000 8000 9000 14000 21000 27000 34000 \
  47000 61000 75000 89000 102000 116000 130000 143000"
uv run python experiments/probes/contrastive_readout_swap/scripts/build_task_datasets.py \
    --model EleutherAI/pythia-6.9b --families sva ioi_role_balanced --n-max-per-family 2000 \
    --rng-seed 0 --no-norm-match --out-dir $DS
# Needs ${UM_SSD_ROOT}/snapshots/pythia-6.9b/*_step<S>_wu.pt for the 32 steps; forwards the model for h.
uv run python experiments/probes/contrastive_readout_swap/scripts/run_swap_grid.py \
    --model pythia-6.9b --datasets-dir $DS --families sva ioi_role_balanced \
    --h-steps $STEPS32 --alignments none --device cuda --dtype bf16 --wu-round bf16 --out-dir $RUN
uv run python experiments/probes/contrastive_readout_swap/scripts/run_controlled_hidden_probes.py \
    --hidden-dir $RUN/hidden --datasets-dir $DS --families sva ioi_role_balanced \
    --stem probe_trajectory_69b --out-dir $RUN/controlled
```

Figure 4 (left) and Figure G.1 read native accuracy from the `s_step = h_step`
rows of `summary.csv`, the best swept readout as the maximum over `s_step`, and
the probe from `acc` of the `feature_source=hidden` rows. The published `sva.jsonl`
lists the same 320 items as today's builder, all plural items first; none of the
metrics depends on item order.

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
- `${UM_SSD_ROOT}/snapshots/pythia-6.9b/EleutherAI_pythia-6.9b_step*_wu.pt` (Figure 4 left: the
  32-step schedule; availability panel: the 24 panel steps, written by `extract_hidden_dense.py` if missing)
- `${UM_SSD_ROOT}/snapshots/pythia-{1b,6.9b}/EleutherAI_pythia-*_step*_lnf.pt` (final LayerNorm, from
  `extract_hidden_dense.py`)
- `$FV_DATA_DIR/{antonym,english-french,present-past,country-capital}.json` (function-vector families)
- `nyu-mll/blimp` from the Hugging Face Hub (BLiMP families)

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
| `scripts/extract_hidden_dense.py` | per-checkpoint final-position `h`/`z` caches (`<family>_{h,z}<step>.pt`) and `_lnf.pt` / `_wu.pt` snapshots |
| `scripts/run_availability_probes.py` | availability/expression metrics per (family, checkpoint); writes `probes/probe_summary.{csv,pt}` + provenance |

Availability outputs live under `results/experiments/probes/contrastive_readout_swap/availability/`:
`datasets/<run>/<family>.jsonl`, `run_<run>/hidden/`, `run_<run>/probes/probe_summary.csv` with
`<run>` in `pythia-6.9b`, `pythia-1b`, `pythia-1b_breadth`.

Library helpers: [src/readout/probes/contrastive_tasks.py](../../../src/readout/probes/contrastive_tasks.py),
[src/readout/probes/readout_swap.py](../../../src/readout/probes/readout_swap.py),
[src/readout/probes/availability_tasks.py](../../../src/readout/probes/availability_tasks.py) (panel registry;
`tasks_sva_hierarchy`, `tasks_numeric`, `tasks_blimp`, `tasks_function_vectors`, `tasks_controls`),
[src/readout/probes/availability_expression.py](../../../src/readout/probes/availability_expression.py)
(probe labels, group keys, folds, readout metrics).

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