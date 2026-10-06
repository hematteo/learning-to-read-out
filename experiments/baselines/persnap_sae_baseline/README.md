# persnap_sae_baseline

Can independent per-snapshot SAEs give one sparse feature index across
training? This experiment trains 32 per-snapshot JumpReLU SAEs on Pythia-160M
`W_U` (d_sae=8,192, one per checkpoint of the 32-step schedule) and compares
their per-checkpoint EV and L0 with the checkpoint-spanning dictionaries. It
supports the SAE baseline comparison in `sec:app-instrument-validation`.

## Paper results

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:app-snapshot-fidelity` | On Pythia-160M, the checkpoint-spanning dictionaries (d=8,192 and 24,576) reconstruct early and late checkpoints with one feature index, while per-snapshot SAEs need far higher L0 and the final-snapshot SAE fails on early checkpoints. | SAEs: `scripts/train_persnap_saes.py`. Per-checkpoint EV and L0 of every curve: `experiments/crosscoders/crosscoder_main/scripts/appendix_validation/eval_persnap_fidelity.py` | `experiments/crosscoders/crosscoder_main/derived/appendix_validation/persnap_fidelity.csv` |

## Run

1. Train the per-snapshot SAEs (defaults: `EleutherAI/pythia-160m`, d_sae=8,192, lambda_1=0.3, 100 epochs):
   ```bash
   uv run python experiments/baselines/persnap_sae_baseline/scripts/train_persnap_saes.py
   ```
2. Compute per-checkpoint EV and L0 for the figure with `crosscoder_main` steps 1 and 2 (`recompute_quality.py`, then `eval_persnap_fidelity.py`); see that README.
3. Optionally compare the SAEs with the crosscoders checkpoint by checkpoint:
   ```bash
   uv run python experiments/baselines/persnap_sae_baseline/scripts/persnap_vs_crosscoder.py
   uv run python experiments/baselines/persnap_sae_baseline/scripts/build_persnap_comparison.py
   uv run python experiments/baselines/persnap_sae_baseline/scripts/compute_cc_vs_persnap_gap.py
   ```

## Inputs

- `${UM_SSD_ROOT}/snapshots/pythia-160m/EleutherAI_pythia-160m_step*_wu.pt`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d{8192,24576}/seed0.safetensors`
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/per-snapshot-saes/d8192/`: released per-snapshot SAEs (default of `compute_cc_vs_persnap_gap.py`).
- `${UM_SSD_ROOT}/derived/rates/wu-d8192-persnap/train.log`: per-snapshot SAE training log that `build_persnap_comparison.py` parses for EV and L0 (`--sae-log`).

## Outputs

- `${UM_SSD_ROOT}/wu_crosscoder/per_snap_sae/wu_sae_dsae8192_step<step>.pt`: trained SAEs (`train_persnap_saes.py`).
- stdout: per-snapshot recovery `R = 1 - MSE/Var` and mean L0 for the d=8,192 crosscoder and the SAEs (`persnap_vs_crosscoder.py`).
- `figures/persnap_sae_baseline/crosscoder_vs_persnap.json`: per-snapshot SAE EV and L0 and per-snapshot EV of the d=24,576 seed-0 crosscoder, stored under the key `crosscoder_t46_d24576_seed0` (`build_persnap_comparison.py`).
- `figures/persnap_sae_baseline/crosscoder_vs_persnap_gap.txt`: per-snapshot crosscoder EV, SAE EV, and their gap, with mean, max, and min (`compute_cc_vs_persnap_gap.py`, `--out-dir`).
