# olmo_matched_checkpoint_window

Is the near-absence of early-decaying features in OLMo-2-7B explained by its
release having no checkpoints before step 150? This control trains a
Pythia-1B `W_U` crosscoder on a 32-checkpoint schedule that starts at step
256 and compares its lifecycle profiles, peak steps, and activation rates with
the selected Pythia-1B and OLMo-2-7B dictionaries. It supports
`sec:app-olmo-checkpoint-window-control` ("Late-Start Control for OLMo").

## Paper results

Outputs are under `figures/olmo_matched_checkpoint_window/`.

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:app-olmo-window-population` | The late-start Pythia-1B dictionary is dominated by first-checkpoint norm peaks, whereas OLMo-2-7B accumulates decoder-norm mass and peaks later. | `scripts/compute_late_start_vs_olmo.py` | `olmo_matched_checkpoint_window_population_lifecycle_diagnostics_{peak_counts,norm_mass}.csv`, `olmo_matched_checkpoint_window_population_lifecycle_diagnostics.pt` |
| `sec:app-olmo-checkpoint-window-control` (text) | Early-decaying share: 0.02% in the late-start dictionary, 24.1% in the selected Pythia-1B dictionary, 25.4% in the selected dictionary restricted to steps from 256 without refitting. | `scripts/compute_late_start_vs_olmo.py` | `late_start_profile_fractions.csv` |
| `sec:app-olmo-checkpoint-window-control` (text) | Activation-rate decayer shares of 0.012% (late start) and 0.015% (OLMo-2-7B); early-dominant shares of 0.057% and 0.595%. | `scripts/compute_late_start_vs_olmo.py` | `comparison_summary.csv` |

## Run

1. Train the late-start dictionary, or skip this step and use the released
   `pythia-1b/W_U/cross-snapshot-32-matched-window/d24576/seed0.safetensors`
   (see [`docs/DATA.md`](../../../docs/DATA.md)). The script defaults match
   [`configs/runs/pythia-1b_wu_d24576_seed0_olmo-matched-window.yaml`](../../../configs/runs/pythia-1b_wu_d24576_seed0_olmo-matched-window.yaml):
   ```bash
   uv run python experiments/ablations/olmo_matched_checkpoint_window/scripts/train_late_start.py \
       --d-sae 24576 --n-epochs 300 --device cuda
   ```
2. Compute the per-checkpoint firing rates of the late-start dictionary and save them as `results/experiments/olmo_matched_checkpoint_window/pythia1b_wu_late_start32_dsae24576_seed0_rates.pt` (see Inputs for the expected keys).
3. Run `experiments/lifecycle/feature_lifecycle_trajectories/scripts/compute_normalized_trajectories.py`, which writes the decoder-norm caches read here.
4. Compute the comparison:
   ```bash
   uv run python experiments/ablations/olmo_matched_checkpoint_window/scripts/compute_late_start_vs_olmo.py
   ```

The late-start schedule is

```text
256, 512, 1000, 2000, 3000, 4000, 5000, 6000,
7000, 8000, 9000, 10000, 13000, 14000, 21000, 23000,
27000, 33000, 34000, 43000, 47000, 53000, 61000, 73000,
83000, 89000, 93000, 102000, 113000, 123000, 130000, 143000
```

## Inputs

- `${UM_SSD_ROOT}/snapshots/pythia-1b/EleutherAI_pythia-1b_step*_wu.pt`: `W_U` snapshots at the late-start steps (training only).
- `results/experiments/olmo_matched_checkpoint_window/pythia1b_wu_late_start32_dsae24576_seed0_rates.pt`: firing rates of the late-start dictionary, a dict with `rates` `(K, D)`, optional `norms`, and `steps`.
- `experiments/crosscoders/crosscoder_main/derived/appendix_validation/large_evals/olmo27b_d32768_seed0.pt`: OLMo-2-7B firing rates (curated; no script in this release regenerates it).
- `figures/feature_lifecycle_trajectories/section52_lifecycle/cache/{olmo2_7b_d32768,pythia1b_d24576}_decoder_norms.npy`: decoder-norm caches from step 3.

## Outputs

- `figures/olmo_matched_checkpoint_window/`: the files in the table above, plus other sidecars that the paper does not use.
- `results/experiments/olmo_matched_checkpoint_window/pythia1b_wu_late_start32_dsae24576_seed0.pt`: the trained late-start dictionary (default `--output` of step 1).

## Notes

- `scripts/extract/extract_rates.py` computes the rates for step 2 but saves them under `rates_per_seed[<checkpoint stem>]`; `compute_late_start_vs_olmo.py` reads a top-level `rates` key, so the tensor has to be stored under `rates` together with `steps`.
- On a machine without CUDA, use `--device mps` and try a smaller pilot first, for example `--model EleutherAI/pythia-160m`.
- The control removes the earliest Pythia checkpoints but not the Pythia and OLMo differences in architecture, tokenizer, data, optimizer, and training length.
