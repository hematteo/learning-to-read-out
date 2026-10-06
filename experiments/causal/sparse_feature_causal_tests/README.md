# sparse_feature_causal_tests

Do readout features selected for one token group act mainly on that group?
For Pythia-1B and four token groups (non-Latin script tokens, punctuation,
digits, English function words), the top-k positive-attribution features of
the released d=24,576 `W_U` crosscoder are ablated from the reconstructed
readout rows, under fixed hidden states, at the step-1000 and the step-143k
readout. Effects are compared with five feature sets matched on decoder-norm
and firing-rate deciles. It supports `sec:app-sparse-feature-causal-tests`
("Token-Group Feature Ablations").

## Paper results

Outputs are under `results/experiments/sparse_feature_causal_tests/`.

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:app-sparse-feature-causal-curves` | For every group at both readouts, the ablation effect above matched controls is larger at k=128 than at k=8 and larger at the final readout than at the early one. | `scripts/run_1b_pilot.py`, then `scripts/aggregate_causal_tests.py` | `family_expansion_summary.csv` (top-feature drop minus mean matched drop; range over the five matched sets) |
| `fig:app-sparse-feature-causal-specificity-k32` | The top 32 features selected for one group act mostly on that group's metric, with off-diagonal effects remaining. | `scripts/run_specificity_from_pilots.py`, then `scripts/aggregate_causal_tests.py` | `specificity_step<step>_h<step>/control_corrected_drop_k32.csv` |

## Run

1. Run one selection-and-ablation pass per token group and readout (8 runs):
   ```bash
   for c in non_latin_scripts punctuation digits function_words; do
     uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_1b_pilot.py \
       --concept $c --snapshot-step 1000 --rank-h-step 1000 --eval-h-steps 512 1000 \
       --score-mode positive --out-dir results/experiments/sparse_feature_causal_tests/1b_${c}_step1000_positive
     uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_1b_pilot.py \
       --concept $c --snapshot-step 143000 --rank-h-step 143000 --eval-h-steps 143000 \
       --score-mode positive --out-dir results/experiments/sparse_feature_causal_tests/1b_${c}_step143000_positive
   done
   ```
2. Reuse each group's selected features on every group's metric:
   ```bash
   uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_specificity_from_pilots.py --snapshot-step 1000 --h-step 1000
   uv run python experiments/causal/sparse_feature_causal_tests/scripts/run_specificity_from_pilots.py --snapshot-step 143000 --h-step 143000
   ```
3. Aggregate the tables behind both figures (CPU, seconds):
   ```bash
   uv run python experiments/causal/sparse_feature_causal_tests/scripts/aggregate_causal_tests.py
   ```

## Inputs

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-1b/W_U/cross-snapshot-32/d24576/seed0.safetensors`: released `W_U` crosscoder.
- `${UM_SSD_ROOT}/derived/aggregates/aggregates_pythia-1b_d24576_seed0.pt`: per-feature firing-rate and norm aggregates.

## Outputs

Under `results/experiments/sparse_feature_causal_tests/`:

- `1b_<group>_step<step>_positive/`: per-run `summary.csv`, `raw.pt`, and manifest.
- `specificity_step<step>_h<step>/`: `specificity.csv` and, after step 3, `control_corrected_drop_k<k>.csv` (source-ablation drop minus mean matched-control drop) and `diagonal_summary.csv`.
- `family_expansion_summary.csv`: top-feature drop, matched-control drop mean, min, and max, and keep-only recoveries per group, readout, and k.
