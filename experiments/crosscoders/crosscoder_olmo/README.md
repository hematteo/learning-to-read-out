# crosscoder_olmo

Cross-family check on OLMo: corroborating evidence that the W_U lifecycle pattern is
not unique to Pythia, rather than a matched scaling replication. Paper-facing use is
limited to cross-family validation metrics and
lifecycle/timing context from the `olmo27b_d32768_seed0` sidecars; OLMo is treated as
cross-family context, not a matched Pythia scaling comparison.

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| `fig:main-multimodel-validation` | `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/olmo27b_d32768_seed0.{json,csv}` | `experiments/crosscoders/crosscoder_olmo/scripts/eval_decision_rules.py` |
| `fig:main-selected-normalized-trajectories` (OLMo context) | OLMo eval sidecars: `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/olmo27b_d32768_seed0.{json,csv}` | `experiments/crosscoders/crosscoder_olmo/scripts/eval_decision_rules.py` |
| `fig:app-reorganization-window-metrics` (OLMo context) | OLMo eval sidecars: `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/olmo27b_d32768_seed0.{json,csv}` | `experiments/crosscoders/crosscoder_olmo/scripts/eval_decision_rules.py` |

This experiment supplies upstream cross-family signal: it persists the OLMo
metric sidecars; the selected-trajectory and reorganization-window panels are
rendered in the paper LaTeX tree from those sidecars together with the Pythia
lifecycle metrics. See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the
full figure → metric map.

## Claim

Cross-family check on OLMo: corroborating evidence that the W_U lifecycle pattern is
not unique to Pythia (cross-family context, not a matched scaling replication). This
experiment computes metrics only: it persists the artifacts below but renders
no figures. Paper figures are rendered in the separate paper LaTeX tree from these
metrics.

## Reproduce

1. **Extract OLMO W_U snapshots:** `uv run python experiments/crosscoders/crosscoder_olmo/scripts/extract_wu_olmo.py`
2. **Pilot training (smoke):** `bash experiments/crosscoders/crosscoder_olmo/scripts/run_olmo_pilot.sh`
3. **Production training:** `bash experiments/crosscoders/crosscoder_olmo/scripts/run_olmo_production.sh`
4. **Null control:** `bash experiments/crosscoders/crosscoder_olmo/scripts/run_null_control.sh`
5. **Decision-rule eval:** `uv run python experiments/crosscoders/crosscoder_olmo/scripts/eval_decision_rules.py`

## Inputs

- OLMO checkpoint snapshots (extracted via `scripts/extract_wu_olmo.py`).
- Paper-facing metric sidecars:
  `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/olmo27b_d32768_seed0.{json,csv}`.
- The released crosscoder checkpoint path is not canonicalized in
  `${UM_SSD_ROOT}/crosscoders.json`; provenance follows the SSD-canonical snapshot
  paths above rather than a `/hf_release/.../olmo-*/W_U/` placeholder.

## Outputs

- W_U snapshots: `{cache_dir}/{slug}_step{N}_wu.pt` (`extract_wu_olmo.py`).
- Decision-rule verdict: `results/experiments/crosscoders/crosscoder_olmo/verdict_seed*.json` (`eval_decision_rules.py`).
- Cross-family validation / lifecycle / timing metric sidecars consumed downstream:
  `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/olmo27b_d32768_seed0.{json,csv}`.

## Layout

| path | role |
|---|---|
| `scripts/extract_wu_olmo.py`             | extract W_U from OLMO snapshots → SSD |
| `scripts/run_olmo_{pilot,production}.sh` | training launchers (multi-GPU) |
| `scripts/run_null_control.sh`            | random-init null control |
| `scripts/eval_decision_rules.py`         | gating decision: is OLMO replicating Pythia? |
| `scripts/train_distributed.py`           | distributed crosscoder training entry point |
| `SCHEDULE.md`                    | run schedule + decision-rule gates |
| `scripts/README.md`                      | scripts-level overview |
