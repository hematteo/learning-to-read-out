# concept_evolution_validation

Pythia-160M (and Pythia-1B) WordNet lexicographer-file probe audit for when
independently specified vocabulary-family distinctions become linearly available
in `W_U`. This is paper-facing corroborating timing evidence, not a fully
controlled semantic-probe study and not evidence that WordNet supersenses map
one-to-one to sparse features.

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| `fig:app-wordnet-supersense-probes-160m-inventory` | `derived/wordnet_supersense_160m/wordnet_supersense_probe_trajectory_160m.csv` | `experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py` |
| `fig:main-wordnet-matched-controls` | `derived/wordnet_matched_controls_160m/wordnet_matched_control_summary_160m.csv` | `experiments/probes/concept_evolution_validation/scripts/run_wordnet_matched_controls.py` |
| `fig:app-wordnet-supersense-probes-1b` | `derived/wordnet_supersense_1b/wordnet_supersense_probe_trajectory_1b.csv` | `experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py` |
| `tab:app-wordnet-supersense-1b` | `derived/wordnet_supersense_1b/wordnet_supersense_probe_pos_summary_1b.csv` | `experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py` |

See [`docs/REPRODUCE.md`](../../../docs/REPRODUCE.md) for the full figure → metric map.

## Claim
Independently specified WordNet lexicographer-file (supersense) distinctions
become linearly available in `W_U` over snapshots, and the matched controls show
this timing is not explained by frequency or string-overlap confounds. The
result corroborates the readout-emergence timing in the paper.

This experiment computes metrics only (no figure rendering): the scripts persist the metrics below as the
reproducible artifacts. Paper figures are rendered in the separate thesis LaTeX
tree from these metrics; no figure-rendering code ships here. The probe scripts
read WordNet 3.0 through NLTK (`../../../src/readout/probes/concept_gazetteer.py`); before
the first run, download the corpus once with
`python -c "import nltk; nltk.download('wordnet')"` (one-time, into `~/nltk_data/`).

## Reproduce
1. `uv run python experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py` (run per `--model`, `160m` and `1b`)
2. `uv run python experiments/probes/concept_evolution_validation/scripts/run_wordnet_matched_controls.py`
3. `uv run python experiments/probes/concept_evolution_validation/scripts/build_gazetteer.py` (37-concept gazetteer audit)

## Inputs (SSD canonical paths)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `experiments/probes/concept_evolution_validation/derived/wordnet_supersense_160m/wordnet_supersense_probe_*_160m.{csv,json}`

## Outputs

`run_wordnet_supersense_probe.py` (per `--model`, `160m` and `1b`) writes under
`derived/wordnet_supersense_{160m,1b}/`:
- `wordnet_supersense_audit_{suffix}.csv`
- `wordnet_supersense_probe_trajectory_{suffix}.{csv,json}`
- `wordnet_supersense_probe_summary_{suffix}.csv`
- `wordnet_supersense_probe_pos_summary_{suffix}.csv`
- `wordnet_supersense_probe_metadata_{suffix}.json`

`run_wordnet_matched_controls.py` writes under
`derived/wordnet_matched_controls_160m/`:
- `wordnet_matched_control_summary_160m.csv`
- `wordnet_matched_control_null_samples_160m.csv`
- `wordnet_matched_control_match_quality_160m.csv`
- `wordnet_matched_control_audit_160m.csv`
- `wordnet_matched_control_metadata_160m.json`

`build_gazetteer.py` writes the 37-concept gazetteer audit:
- `configs/preregistration/concepts_v1.json`
- `configs/preregistration/concepts_v1_audit.json`

## Layout
| path | role |
|---|---|
| `derived/` | emergence tables / probe trajectories written here on run (gitignored; not shipped in this code-only release) |
