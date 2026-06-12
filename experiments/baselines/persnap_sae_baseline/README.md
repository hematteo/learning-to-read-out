# persnap_sae_baseline

Per-snapshot SAE baseline: trains an independent SAE on each Pythia snapshot and
compares it against the cross-snapshot crosscoder. This experiment computes metrics only — it
computes and persists the per-snapshot recovery / explained-variance / L0 numbers
that back the appendix figures, but ships no figure-rendering code. The thesis
LaTeX tree renders the figures from these artifacts.

## Figures produced

| Thesis label | Metric file | Producing script |
|---|---|---|
| `fig:app-persnap-sae-trajectory` | `figures/persnap_sae_baseline/crosscoder_vs_persnap.json` | `experiments/baselines/persnap_sae_baseline/scripts/build_persnap_comparison.py` |
| `fig:app-snapshot-fidelity` | `figures/persnap_sae_baseline/crosscoder_vs_persnap.json` | `experiments/baselines/persnap_sae_baseline/scripts/build_persnap_comparison.py` |
| `fig:app-1b-baseline-suite` | `figures/persnap_sae_baseline/crosscoder_vs_persnap.json` | `experiments/baselines/persnap_sae_baseline/scripts/build_persnap_comparison.py` |
| `fig:app-160m-baselines` | `figures/persnap_sae_baseline/crosscoder_vs_persnap.json` | `experiments/baselines/persnap_sae_baseline/scripts/build_persnap_comparison.py` |
| `fig:app-parameter-budget` | `figures/persnap_sae_baseline/crosscoder_vs_persnap.json` | `experiments/baselines/persnap_sae_baseline/scripts/build_persnap_comparison.py` |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim

Establishes that the cross-snapshot crosscoder is on par with per-snap SAEs while
admitting trajectory analysis. Each appendix figure is rendered downstream from the
single comparison JSON, which records per-snapshot EV/L0 for the per-snap SAEs vs
the T4.6 d=24576 seed-0 crosscoder.

## Reproduce (in order; scripts bootstrap the repo onto `sys.path`)

1. **Train per-snapshot SAEs:** `uv run python experiments/baselines/persnap_sae_baseline/scripts/train_persnap_saes.py --model {pythia-160m,pythia-1b}`
2. **Compare vs crosscoder:** `uv run python experiments/baselines/persnap_sae_baseline/scripts/persnap_vs_crosscoder.py`
3. **Persist comparison metrics:** `uv run python experiments/baselines/persnap_sae_baseline/scripts/build_persnap_comparison.py` and `uv run python experiments/baselines/persnap_sae_baseline/scripts/plot_cc_vs_persnap_gap.py`

## Inputs (SSD canonical paths)

- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-{160m,1b}/W_U/cross-snapshot-32/d*/seed0.safetensors`
- Per-snapshot SAEs (trained by `train_persnap_saes.py`)

## Outputs

- `figures/persnap_sae_baseline/crosscoder_vs_persnap.json` — per-snapshot EV
  (and per-snap-SAE L0) for the per-snap SAEs vs the T4.6 d=24576 seed-0 crosscoder
  (written by `build_persnap_comparison.py`).
- `crosscoder_vs_persnap_gap.txt` — per-snapshot crosscoder EV, per-snap-SAE EV, and
  their gap, plus mean/max/min gap summary (written by `plot_cc_vs_persnap_gap.py`,
  which despite its name persists a text table, not a figure; `--out-dir` controls
  the location).
- Per-snapshot recovery (`R = 1 - MSE/Var`) and mean L0 for both the crosscoder and
  the per-snap SAEs are reported to stdout by `persnap_vs_crosscoder.py`.
- Per-snapshot SAE checkpoints `wu_sae_dsae8192_step{step}.pt` (written by
  `train_persnap_saes.py`).

## Layout

| path | role |
|---|---|
| `scripts/` | per-snapshot SAE baseline code: `train_persnap_saes` / `build_persnap_comparison` plus the crosscoder-vs-per-snap comparison (`persnap_vs_crosscoder`, `plot_cc_vs_persnap_gap`). `plot_cc_vs_persnap_gap` persists a metrics text table only — no figures are rendered in this repo. |
