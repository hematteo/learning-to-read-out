# sparse_feature_causal_tests

Sparse-feature readout edits that test whether learned `W_U` crosscoder features
act mainly on the token group they were selected for, under fixed hidden states.
For Pythia-1B and four token groups (non-Latin scripts, punctuation, digits,
function words), the top-k positive-attribution features are ablated from the
reconstructed `W_U` rows, at the step-1000 and the step-143k readout, and compared
with five feature sets matched on decoder norm and firing-rate deciles
(Appendix H).

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| fig:app-sparse-feature-causal-curves | `results/experiments/sparse_feature_causal_tests/family_expansion_summary.csv` | `run_1b_pilot.py`, then `aggregate_causal_tests.py` |
| fig:app-sparse-feature-causal-specificity-k32 | `results/experiments/sparse_feature_causal_tests/specificity_step<step>_h<step>/control_corrected_drop_k32.csv` | `run_specificity_from_pilots.py`, then `aggregate_causal_tests.py` |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim

At both readouts, the ablation effect above matched controls grows from k=8 to
k=128 for every token group and is larger at the final readout than at the early
one. The top-32 features selected for one group act mostly on that group's
metric, with off-diagonal effects remaining (group-level specificity, not
monosemanticity of individual features).

## Reproduce

```bash
# 1. Pilot runs: one per token group and readout step (8 runs)
for c in non_latin_scripts punctuation digits function_words; do
  uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_1b_pilot.py \
    --concept $c --snapshot-step 1000 --rank-h-step 1000 --eval-h-steps 512 1000 \
    --score-mode positive --out-dir results/experiments/sparse_feature_causal_tests/1b_${c}_step1000_positive
  uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_1b_pilot.py \
    --concept $c --snapshot-step 143000 --rank-h-step 143000 --eval-h-steps 143000 \
    --score-mode positive --out-dir results/experiments/sparse_feature_causal_tests/1b_${c}_step143000_positive
done
# 2. Specificity: reuse each group's selected features on every group's metric
uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_specificity_from_pilots.py --snapshot-step 1000 --h-step 1000
uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_specificity_from_pilots.py --snapshot-step 143000 --h-step 143000
# 3. Aggregate into the tables behind both panels (CPU, seconds)
uv run python experiments/causal/sparse_feature_causal_tests/scripts/aggregate_causal_tests.py
```

`aggregate_causal_tests.py` writes `family_expansion_summary.csv` (drop of the
top features, matched-control drop mean/min/max, keep-only recoveries) and, per
specificity run, `control_corrected_drop_k<k>.csv` (source-ablation drop minus
the mean matched-control drop) and `diagonal_summary.csv`. On the paper's run
outputs it reproduces the tables behind both panels exactly.

## Inputs

| path | role |
|---|---|
| `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors` | released `W_U` crosscoder |
| `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-1b_d24576_seed0.pt` | per-feature firing-rate / norm aggregates |

## Outputs

| path | role |
|---|---|
| `results/experiments/sparse_feature_causal_tests/` | CSV/JSON/PT outputs (`summary.csv`, `raw.pt`, `specificity.csv`, the aggregated tables above, manifests) |
