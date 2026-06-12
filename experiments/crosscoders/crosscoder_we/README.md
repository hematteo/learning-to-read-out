# crosscoder_we

W_E (input embedding) variant of the crosscoder, multi-seed pilot. Companion to the
W_U (output) crosscoder; it characterises read-write asymmetry by comparing when input
embeddings reorganize versus when output unembeddings do. No figures are rendered here:
the experiment computes and persists the metrics behind the WE appendix read/write
asymmetry panels and feature-card grids, and the thesis LaTeX tree renders them from the
committed CSV/`.pt` artifacts. Each metric base is written with both a `.csv` and a `.pt`
cache.

## Figures produced

Backs the 11 `fig:app-we-*` read-write-asymmetry figures under `sec:app-we-crosscoders`.

| Thesis label | Metric file | Producing script |
|---|---|---|
| `fig:app-we-read-write-asymmetry` | `read_write_asymmetry.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_appendix_plots.py` |
| `fig:app-we-quality-pareto` | `we_quality_pareto.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_appendix_plots.py` |
| `fig:app-we-lead-lag-family` | `lead_lag_family_heatmap.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_appendix_extended_plots.py` |
| `fig:app-we-token-overlap` | `token_overlap_jaccard.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_appendix_extended_plots.py` |
| `fig:app-we-geometry` | `wu_we_geometry_160m.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_appendix_extended_plots.py` |
| `fig:app-we-seed-correspondence` | `multiseed_hungarian.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_appendix_extended_plots.py` |
| `fig:app-we-capacity-note` | `capacity_note.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_appendix_extended_plots.py` |
| `fig:app-we-d24576-rate-timing` | `d24576_rate_timing.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_wu_top3_experiments.py` |
| `fig:app-we-matched-control-lead-lag` | `matched_control_lead_lag.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_wu_top3_experiments.py` |
| `fig:app-we-procrustes` | `procrustes_alignment.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_wu_top3_experiments.py` |
| `fig:app-we-feature-cards-representative` | `we_feature_cards_pythia160m_d8192_representative.{csv,pt}` | `experiments/crosscoders/crosscoder_we/scripts/build_we_feature_cards.py` |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim
W_E (input embedding) variant of the crosscoder, multi-seed pilot. Companion to
the W_U (output) crosscoder; characterises read-write asymmetry by comparing
when input embeddings reorganize vs. when output unembeddings do.

## Reproduce
1. `uv run python experiments/crosscoders/crosscoder_we/scripts/build_we_appendix_plots.py` — core read/write asymmetry and quality-pareto metrics
2. `uv run python experiments/crosscoders/crosscoder_we/scripts/build_we_appendix_extended_plots.py` — extended family, overlap, geometry, seed, and capacity metrics
3. `uv run python experiments/crosscoders/crosscoder_we/scripts/build_we_wu_top3_experiments.py` — matched-control, capacity-timing, and Procrustes metrics
4. `uv run python experiments/crosscoders/crosscoder_we/scripts/build_we_feature_cards.py` — WE feature-card metric payloads

## Inputs (SSD canonical paths)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-*/W_E/cross-snapshot-32/d*/seed*.safetensors`
- W_E base snapshots: `${UM_SSD_ROOT}/snapshots/pythia-*/EleutherAI_pythia-*_step*_we.pt`

## Outputs
The metrics behind the WE appendix read/write asymmetry panels and feature-card grids.
Each metric base is written with both a `.csv` and a `.pt` cache.

- `read_write_asymmetry.{csv,pt}`
- `we_quality_pareto.{csv,pt}`
- `lead_lag_family_heatmap.{csv,pt}`
- `token_overlap_jaccard.{csv,pt}`
- `wu_we_geometry_160m.{csv,pt}`
- `multiseed_hungarian.{csv,pt}`
- `capacity_note.{csv,pt}`
- `d24576_rate_timing.{csv,pt}`
- `matched_control_lead_lag.{csv,pt}`
- `procrustes_alignment.{csv,pt}`
- `token_family_masks.{csv,pt}`, `token_family_masks_top3.{csv,pt}`
- WE feature-card payloads: `we_feature_cards_pythia160m_d8192_representative.{csv,pt}`
  and `we_feature_cards_pythia160m_d8192_ambiguous.{csv,pt}` (audit only)

Bases are written under `figures/crosscoder_we/` and
`results/experiments/crosscoder_we/`.

## Layout
| path | role |
|---|---|
| `scripts/build_we_appendix_plots.py` | core W_E appendix read/write and quality metrics |
| `scripts/build_we_appendix_extended_plots.py` | extended W_E appendix audit metrics |
| `scripts/build_we_wu_top3_experiments.py` | top follow-up W_E/W_U metrics |
| `scripts/build_we_feature_cards.py` | W_E feature-card metric payloads |
