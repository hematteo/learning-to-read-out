# dense_readout_diagnostics

Dense W_U diagnostics that serve as paper-facing controls for sparse readout
formation: a spectral-capacity diagnostic, an adjacent-subspace reorganization
timing diagnostic, and a mean-direction / centering audit. The model set is
Pythia-1B (primary), Pythia-160M (companion), Pythia-6.9B (scale stress test),
and OLMo-2-7B (cross-family check). Together they show that centered W_U remains
high-dimensional across families, motivating sparse trajectory dictionaries over
simple SVD/PCA summaries.

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| `fig:app-spectral-capacity` | `results/experiments/dense_readout_diagnostics/spectral_metrics.csv` | `experiments/baselines/dense_readout_diagnostics/scripts/build_spectral_capacity.py` |
| `fig:app-dense-reorg-pythia` | `results/experiments/dense_readout_diagnostics/dense_reorganization/subspace_metrics.csv` | `experiments/baselines/dense_readout_diagnostics/scripts/build_dense_reorganization_timing.py` |
| `fig:app-dense-reorg-olmo` | `figures/dense_readout_diagnostics/dense_reorganization_timing_olmo.csv` | `experiments/baselines/dense_readout_diagnostics/scripts/build_dense_reorganization_timing.py` |
| `fig:app-mean-direction-preprocessing` | `results/experiments/dense_readout_diagnostics/mean_direction_metrics.csv` | `experiments/baselines/dense_readout_diagnostics/scripts/build_mean_direction_audit.py` |
| `fig:app-mean-direction-spectral-gap` | `results/experiments/dense_readout_diagnostics/mean_direction_metrics.csv` | `experiments/baselines/dense_readout_diagnostics/scripts/build_mean_direction_audit.py` |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim

Dense W_U diagnostics act as appendix controls for sparse readout formation. The
spectral-capacity diagnostic and the adjacent-subspace timing diagnostic show
that centered W_U stays high-dimensional across Pythia and OLMo, and the
mean-direction audit confirms the spectral-gap and stable-rank preprocessing
choices. This experiment computes and persists the metrics behind these
diagnostics; it renders no figures. Paper figures are rendered in the separate
paper LaTeX tree from these regenerated metrics (gitignored — not shipped in
this code-only release).

## Reproduce

```bash
uv run python experiments/baselines/dense_readout_diagnostics/scripts/build_spectral_capacity.py
uv run python experiments/baselines/dense_readout_diagnostics/scripts/plot_spectral_capacity.py
uv run python experiments/baselines/dense_readout_diagnostics/scripts/build_dense_reorganization_timing.py
uv run python experiments/baselines/dense_readout_diagnostics/scripts/build_mean_direction_audit.py
```

`plot_spectral_capacity.py` renders no figures: it reads the spectral metrics
and writes plot-ready numeric rows (CSV/.pt) for the paper LaTeX tree to render.

The default run uses the 32 cross-snapshot checkpoint schedule and computes the
top centered W_U right-singular directions with randomized PCA. OLMo uses its own
32 stage-1 checkpoint schedule and is interpreted as cross-family evidence, not
controlled Pythia scaling.

## Inputs

- `${UM_SSD_ROOT}/snapshots/pythia-{160m,1b,6.9b}/EleutherAI_pythia-*_step*_wu.pt`
- `${UM_SSD_ROOT}/snapshots/OLMo-2-1124-7B/allenai_OLMo-2-1124-7B_step*_wu.pt`

Use `UM_SSD_ROOT` to override the SSD root.

The diagnostics persist three families of metrics:

- Dense spectral-capacity: effective-rank and stable-rank fractions, cumulative
  explained variance, and normalized spectra (`spectral_*` and
  `spectral_capacity_multimodel.*`).
- Dense adjacent-subspace timing on the controlled Pythia models and the
  OLMo-2-7B cross-family check (`subspace_metrics.csv`, `sv_stability.csv`, plus
  the OLMo timing CSV).
- Mean-direction / centering audit: spectral-gap and stable-rank preprocessing
  checks (`mean_direction_*`).

## Outputs

All outputs are persisted metrics (CSV/.pt/.json); no figures are written.

```text
results/experiments/dense_readout_diagnostics/
  spectral_metrics.csv
  spectral_cumulative.csv
  spectral_spectra.pt
  spectral_provenance.json
  mean_direction_metrics.csv
  mean_direction_metrics.pt
  mean_direction_provenance.json

results/experiments/dense_readout_diagnostics/dense_reorganization/
  subspace_metrics.csv
  sv_stability.csv
  top_svd_vectors.pt
  provenance.json

figures/dense_readout_diagnostics/
  spectral_capacity_multimodel.{csv,pt}
  dense_reorganization_timing.{csv,pt}
  dense_reorganization_timing_olmo.csv
```

The `figures/` path above holds plot-ready numeric tables (`.csv`/`.pt`)
consumed by the paper LaTeX tree, not rendered images.
