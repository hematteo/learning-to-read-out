# crosscoder_olmo

Does the readout of a model outside the Pythia family develop the same way?
This experiment extracts `W_U` from 32 stage-1 checkpoints of
OLMo-2-1124-7B and trains the d=32,768 trajectory crosscoder used as the
cross-architecture dictionary. The OLMo-2-7B panels of the validation summary
and the lifecycle figures are computed from this dictionary by
`crosscoder_main` and `feature_lifecycle_trajectories`. It supports
`sec:readout-lifecycle` and the lifecycle appendix
(`sec:app-lifecycle-diagnostics`).

## Paper results

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:main-multimodel-validation` | The OLMo-2-7B dictionary's aggregate EV (0.853 in `tab:full-inventory`) and its per-checkpoint EV and active density. | Dictionary: `scripts/run_olmo_production.sh`. Metrics: no script in this release (curated inputs) | `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/olmo27b_d32768_seed0.json`, `olmo27b_d32768_seed0_per_snap.csv` |
| `fig:main-selected-normalized-trajectories` | OLMo-2-7B panel: most features emerge late. | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/compute_normalized_trajectories.py` | `results/experiments/lifecycle/feature_lifecycle_trajectories/selected_decoder_norm_trajectories.{csv,pt}` |
| `fig:app-reorganization-window-metrics` | OLMo-2-7B's reorganization score peaks between step 9k and step 28k without separating from the nulls (p=0.12). | `experiments/lifecycle/feature_lifecycle_trajectories/scripts/find_reorganization_steps.py` | `results/experiments/lifecycle/feature_lifecycle_trajectories/reorganization_window_metrics_selected.{csv,pt}` |

## Run

Skip steps 1 to 3 by downloading the released dictionary
`olmo-2-7b/W_U/cross-snapshot-32/d32768/seed0.safetensors` (see
[`docs/DATA.md`](../../../docs/DATA.md)).

1. Extract `W_U` at the 32 checkpoints of the paper's OLMo-2-7B schedule (the script default):
   ```bash
   uv run python experiments/crosscoders/crosscoder_olmo/scripts/extract_wu_olmo.py --cache-dir "$WU_CACHE"
   ```
2. Run the single-GPU pilot fit and write its verdict; the production launcher exits unless this verdict exists and sets `decision.proceed`:
   ```bash
   WU_CACHE="$WU_CACHE" bash experiments/crosscoders/crosscoder_olmo/scripts/run_olmo_pilot.sh
   uv run python experiments/crosscoders/crosscoder_olmo/scripts/eval_decision_rules.py \
       --crosscoder /workspace/results/olmo_cross_family/pilot/cc_olmo27b_dsae16384_seed0.pt \
       --wu-cache "$WU_CACHE" --stage pilot \
       --out /workspace/results/olmo_cross_family/pilot/verdict_seed0.json
   ```
3. Train the production dictionary (d_sae=32,768, 300 epochs, batch 1024, head-parallel over four GPUs through `scripts/train_distributed.py`):
   ```bash
   WU_CACHE="$WU_CACHE" bash experiments/crosscoders/crosscoder_olmo/scripts/run_olmo_production.sh
   ```
4. Compute the OLMo-2-7B lifecycle metrics with `feature_lifecycle_trajectories` (see its README).

## Inputs

- `allenai/OLMo-2-1124-7B` stage-1 revisions from the Hugging Face Hub (one fp16 download of about 14 GB per checkpoint; the extractor clears the Hugging Face cache between checkpoints).
- `$WU_CACHE`: directory for the extracted snapshots, about 50 GB for 32 checkpoints. The launchers default to `/workspace/wu_cache_olmo`.

## Outputs

- `$WU_CACHE/allenai_OLMo-2-1124-7B_step<N>_wu.pt`: fp32 `W_U` snapshots (`extract_wu_olmo.py`).
- `$OUT_DIR/cc_olmo27b_dsae16384_seed0.pt` and `verdict_seed0.json` for the pilot, `$OUT_DIR/cc_olmo27b_dsae32768_seed0.pt` and `train_seed0.log` for production. `OUT_DIR` defaults to `/workspace/results/olmo_cross_family/{pilot,production}`.

## Notes

- Settings of record: [`configs/runs/olmo-2-7b_wu_d32768_seed0.yaml`](../../../configs/runs/olmo-2-7b_wu_d32768_seed0.yaml). `tab:repro-dictionary-hparams` lists SparseAdam for OLMo-2-7B, while `run_olmo_production.sh` passes `--optimizer adam` and the config records `adam`; the config's provenance note explains the uncertainty.
- Compute (`tab:compute-resources`): four A100/H100-class GPUs with 40 to 80 GB each, 4 to 8 hours, excluding extraction.
- [`SCHEDULE.md`](SCHEDULE.md) records the checkpoint schedule and the pilot and production gates that `eval_decision_rules.py` evaluates. The gates and the Gaussian-snapshot control (`scripts/run_null_control.sh`) are not reported in the paper.
- The EV, L0, and activation-rate files under `crosscoder_main/derived/appendix_validation/large_evals/` are curated inputs; no script in this release regenerates them.
