# sparse_feature_causal_tests

Sparse-feature readout edits that test whether learned `W_U` crosscoder features
are load-bearing for family-specific output behavior under fixed hidden states.
The paper-facing audit applies Ge-style top-k feature ablations on reconstructed
`W_U` rows for Pythia-1B at the mature step-143k readout, across non-Latin
scripts, punctuation, digits, and function-word token families, with
decoder-norm/firing-rate matched controls.

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| fig:app-sparse-feature-causal-curves | `results/experiments/sparse_feature_causal_tests/.../summary.csv` | `experiments/causal/sparse_feature_causal_tests/scripts/run_1b_pilot.py` |
| fig:app-sparse-feature-causal-specificity-k32 | `results/experiments/sparse_feature_causal_tests/.../specificity.csv` | `experiments/causal/sparse_feature_causal_tests/scripts/run_specificity_from_pilots.py` |
| fig:app-contrastive-task-localization | `results/experiments/sparse_feature_causal_tests/.../{summary.csv,specificity.csv}` | `experiments/causal/sparse_feature_causal_tests/scripts/{run_1b_pilot.py,run_specificity_from_pilots.py}` |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim

Sparse readout features are behaviorally load-bearing for family-specific output
behavior under fixed hidden states. The paper-facing audit evaluates Pythia-1B
at the mature step-143k readout across non-Latin scripts, punctuation, digits,
and function-word families, using top-k feature ablations on reconstructed
`W_U` rows with decoder-norm/firing-rate matched controls. Earlier step-1000
non-Latin pilots remain historical setup runs.

## Reproduce

```bash
uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_1b_pilot.py
uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_specificity_from_pilots.py --snapshot-step 1000 --h-step 1000
uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_specificity_from_pilots.py --snapshot-step 143000 --h-step 143000
```

The entry points above compute and persist the localization metrics
(`summary.csv` / `raw.pt` from the pilot, `specificity.csv` from the specificity
pass). The `fig:app-contrastive-task-localization` main-text figure draws its localization metrics
from these same entry points. The CSV-aggregation step that the appendix
specificity panels assume is not shipped (see the "Metrics not computable in
this repo" section in [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md)).

## Inputs

| path | role |
|---|---|
| `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors` | released `W_U` crosscoder |
| `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-1b_d24576_seed0.pt` | per-feature firing-rate / norm aggregates |
| `results/experiments/sparse_feature_causal_tests/specificity_step143000_h143000/manifest.json` | run manifest for the mature-step specificity pass |

## Outputs

| path | role |
|---|---|
| `results/experiments/sparse_feature_causal_tests/` | CSV/JSON/PT outputs (`summary.csv`, `specificity.csv`, `raw.pt`, manifests) |
