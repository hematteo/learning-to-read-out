# crosscoder_main

Do the trajectory crosscoders reconstruct the readout faithfully, and how do
they compare with dense and per-snapshot baselines? This experiment computes
the dictionary inventory, per-checkpoint EV and L0, next-token fidelity of the
reconstructed readout, and the PCA baselines for the Pythia dictionaries. It
supports the dictionary validation appendix (`sec:app-instrument-validation`)
and the validation summary cited in `sec:readout-lifecycle`.

## Paper results

Outputs are under `experiments/crosscoders/crosscoder_main/derived/appendix_validation/` unless the path says otherwise.

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:main-multimodel-validation` | The selected dictionaries reach aggregate EV 0.808 to 0.920 with 1.17 to 2.26% of features active per row (A); per-checkpoint EV (B) and active density (C), comparing the Pythia-160M d=8,192 dictionary with its 32 per-snapshot SAEs. | A: `scripts/appendix_validation/recompute_quality.py`. B, C: `scripts/appendix_validation/eval_persnap_fidelity.py` (Pythia-160M), `scripts/main_1b/eval_persnap_1b.py` (Pythia-1B). Pythia-6.9B and OLMo-2-7B: no script (see Notes) | `full_inventory.csv`; `persnap_fidelity.csv`; `../main_1b/persnap_1b_d24576_seed0.csv`; `large_evals/*.json`, `large_evals/*_per_snap.csv` |
| `fig:app-fidelity-curves` | Matrix EV, logit R^2, KL to the native readout, and top-1 agreement of the reconstructed readout at each checkpoint; the text reports delta NLL and the token-stratified check. | `scripts/appendix_validation/readout_functional_fidelity.py`, `scripts/appendix_validation/readout_functional_fidelity_strata.py` | `readout_functional_fidelity/<model>_d<d_sae>_seed0/readout_functional_fidelity.csv`; `readout_functional_fidelity/pythia-1b_d24576_seed0/strata/strata_summary.csv` |
| `fig:app-pca-static` | Per-snapshot PCA of centered `W_U` needs many dense components to approach the dictionaries' EV. | `scripts/appendix_validation/baselines/eval_pca_static.py` | `figures/crosscoder_main/appendix_validation/pca_static.{csv,pt}` |
| `fig:app-concat-pca` | One PCA subspace fitted to the concatenated trajectories explains far less variance than the dictionaries at ranks 16 to 256. | `scripts/appendix_validation/baselines/eval_concat_pca.py` | `figures/crosscoder_main/appendix_validation/concat_pca.{csv,pt}` |
| `fig:app-lambda-sweep` | Raising lambda_1 from 0.4 to 1.8 lowers mean L0 from about 160 to about 23 and EV from 0.75 to 0.53 for the Pythia-160M d=8,192 dictionary. | `scripts/appendix_validation/recompute_quality.py` (`--filter lambda-sweep` for the sweep fits) | `lambda_sweep.csv` (sweep fits); `full_inventory.csv` (lambda_1 = 0.3 anchor fits, architecture variants, SAE baselines) |
| `tab:full-inventory` | EV, mean L0, active share, and dead features of every dictionary and baseline. | `scripts/appendix_validation/recompute_quality.py` | `full_inventory.csv` |
| `tab:repro-dictionary-hparams` | Training hyperparameters of the main dictionaries. | none; the settings of record are in `configs/runs/*.yaml` | none |

## Run

1. Recompute EV, mean L0, and dead rate of every released dictionary and SAE baseline (the per-snapshot SAE rows are read by step 2):
   ```bash
   uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/recompute_quality.py \
       --ckpt-root "$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders" --snap-dir "$UM_SSD_ROOT/snapshots" \
       --out-csv experiments/crosscoders/crosscoder_main/derived/appendix_validation/full_inventory.csv
   uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/recompute_quality.py \
       --ckpt-root "$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders" --snap-dir "$UM_SSD_ROOT/snapshots" \
       --filter lambda-sweep --out-csv experiments/crosscoders/crosscoder_main/derived/appendix_validation/lambda_sweep.csv
   ```
2. Compute per-checkpoint EV and L0 for the Pythia-160M dictionaries and SAE baselines:
   ```bash
   uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/eval_persnap_fidelity.py
   ```
3. Compute per-checkpoint FVU, L0, and dead rate for the Pythia-1B width sweep (d=8,192, 16,384, 24,576):
   ```bash
   uv run python experiments/crosscoders/crosscoder_main/scripts/main_1b/eval_persnap_1b.py
   ```
4. Build pre-readout hidden-state caches for the fidelity check (fp32 for Pythia-160M and 1B; the bf16 default for Pythia-6.9B), one call per model, for example:
   ```bash
   uv run python scripts/extract/build_hln_cache.py --model pythia-1b --steps all \
       --eval-tokens "$UM_SSD_ROOT/hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt" \
       --out-dir "$UM_SSD_ROOT/derived/readout_edit_timing_pythia1b" --save-dtype fp32
   ```
5. Score next-token fidelity per model, then the token-stratified check on Pythia-1B:
   ```bash
   F=experiments/crosscoders/crosscoder_main/scripts/appendix_validation
   uv run python $F/readout_functional_fidelity.py --model pythia-160m --d-sae 8192 --seed 0 --steps all \
       --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing
   uv run python $F/readout_functional_fidelity.py --model pythia-1b --d-sae 24576 --seed 0 --steps all \
       --snapshot-dtype bf16 --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing_pythia1b
   uv run python $F/readout_functional_fidelity.py --model pythia-6.9b --d-sae 32768 --seed 0 --steps all \
       --snapshot-dtype bf16 --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing_pythia69b \
       --ckpt ${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0-sparse.safetensors
   uv run python $F/readout_functional_fidelity_strata.py --model pythia-1b --d-sae 24576 --seed 0 --steps all \
       --max-seqs 64 --snapshot-dtype bf16 --hln-dir ${UM_SSD_ROOT}/derived/readout_edit_timing_pythia1b
   ```
6. Compute the dense PCA baselines. Both scripts default to `--models pythia-160m`; the figures cover `pythia-160m pythia-1b pythia-6.9b olmo-2-7b`:
   ```bash
   uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/baselines/eval_pca_static.py --models pythia-160m pythia-1b pythia-6.9b olmo-2-7b
   uv run python experiments/crosscoders/crosscoder_main/scripts/appendix_validation/baselines/eval_concat_pca.py --models pythia-160m pythia-1b pythia-6.9b olmo-2-7b
   ```

## Inputs

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/`: released dictionaries and SAE baselines, including `pythia-160m/W_U/{cross-snapshot-32,cross-snapshot-16,lambda-sweep,architecture-comparison,final-snapshot-saes}/`, `pythia-1b/W_U/cross-snapshot-32/d*/seed0.safetensors`, and `pythia-6.9b/W_U/cross-snapshot-32/d32768/seed0{,-sparse}.safetensors`.
- `${UM_SSD_ROOT}/snapshots/<model>/<slug>_step<N>_wu.pt`: extracted `W_U` snapshots.
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt`: held-out evaluation corpus.
- `${UM_SSD_ROOT}/derived/readout_edit_timing*/hLN_step*.pt`: hidden-state caches from step 4.
- `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/`: Pythia-6.9B and OLMo-2-7B summaries (see Notes).

## Outputs

Under `experiments/crosscoders/crosscoder_main/derived/` unless stated:

- `appendix_validation/full_inventory.csv`, `appendix_validation/lambda_sweep.csv` (+ `.provenance.json`): canonical and training-formula EV, MSE, L0, and dead rate per checkpoint file.
- `appendix_validation/persnap_fidelity.csv`: `family, step, ev, fvu, l0` for the Pythia-160M d=8,192 and d=24,576 dictionaries, the 16-checkpoint dictionary, the per-snapshot SAEs, and the final-snapshot SAE evaluated at every step.
- `main_1b/persnap_1b_d<d_sae>_seed0.csv`: per-checkpoint FVU, L0, and dead rate for Pythia-1B.
- `appendix_validation/readout_functional_fidelity/<model>_d<d_sae>_seed0/`: `readout_functional_fidelity.csv`, per-step `shards/`, metadata; `strata/strata_summary.csv` for the stratified check.
- `appendix_validation/baselines/raw/pca_static_spectra.pt`: per-snapshot singular spectra from `eval_pca_static.py`.
- `figures/crosscoder_main/appendix_validation/{pca_static,concat_pca}.{csv,pt}` (from the repository root): PCA EV by rank.

## Notes

- The `large_evals/` files (Pythia-6.9B and OLMo-2-7B EV, L0, per-checkpoint values, and activation rates) are curated inputs; no script in this release regenerates them. `feature_lifecycle_trajectories` and `olmo_matched_checkpoint_window` also read them.
- The Pythia-160M fidelity curve uses the d=8,192 seed-0 dictionary, not the selected d=24,576 one.
- The strata check reads 64 sequences (`--max-seqs 64`), so its Pythia-1B caches must hold at least 64 sequences (`build_hln_cache.py --max-seqs 64`); the other fidelity runs read the first 32 (16,352 predictions).
- Memory: the Pythia-1B and 6.9B fidelity runs hold all 32 snapshots and the dictionary in memory (about 25 GB for Pythia-1B). The Pythia-160M run takes about 15 minutes on a laptop CPU.
- PCA baselines discover checkpoints from the snapshot directory unless `--steps` is given. The concatenated PCA in the paper uses 45 Pythia-160M, 32 Pythia-1B and OLMo-2-7B, and 15 Pythia-6.9B checkpoints.
- Also released, with no paper figure or table mapped to their outputs: `scripts/aggregates/build_analysis_table.py`, `scripts/appendix_validation/eval_per_snap.py`, and `scripts/appendix_validation/baselines/eval_endpoint_linear.py`.
