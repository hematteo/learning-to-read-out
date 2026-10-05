# concept_evolution_validation

Pythia-160M (and Pythia-1B) WordNet lexicographer-file probe audit for when
independently specified vocabulary-family distinctions become linearly available
in `W_U`. This is paper-facing corroborating timing evidence, not a fully
controlled semantic-probe study and not evidence that WordNet supersenses map
one-to-one to sparse features.

## Figures produced

| Paper label | Metric file | Producing script |
|---|---|---|
| `fig:main-wordnet-matched-controls` (Fig. 3, §4.2) | `derived/wordnet_matched_controls_160m/wordnet_matched_control_summary_160m.csv` | `experiments/probes/concept_evolution_validation/scripts/run_wordnet_matched_controls.py` |
| `fig:app-wordnet-supersense-probes-160m-inventory` (Fig. F.1) | `derived/wordnet_supersense_160m/wordnet_supersense_probe_trajectory_160m.csv` | `experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py` |
| `fig:app-wordnet-supersense-probes-1b` (Fig. F.2) | `derived/wordnet_supersense_1b/wordnet_supersense_probe_trajectory_1b.csv` | `experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py` |
| `tab:app-wordnet-supersense-1b` (Table F.1) | `derived/wordnet_supersense_1b/wordnet_supersense_probe_pos_summary_1b.csv` | `experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py` |
| App. F.6 lemma-grouped splits (`sec:app-wordnet-lemma-split`, prose only) | `derived/wordnet_supersense_{160m,1b}_lemma/wordnet_supersense_probe_{trajectory,summary}_{160m,1b}.csv` against the row-split files above; leakage counts in `wordnet_supersense_split_stats_{160m,1b}.csv` of both runs | `experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py --split lemma` |

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
1. `uv run python experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py --model pythia-160m` (and `--model pythia-1b`)
2. `uv run python experiments/probes/concept_evolution_validation/scripts/run_wordnet_matched_controls.py`
3. `uv run python experiments/probes/concept_evolution_validation/scripts/build_gazetteer.py` (37-concept gazetteer audit)
4. Lemma-grouped splits (App. F.6): same probe, only the fold builder changes
   (`StratifiedGroupKFold` over `decode(id).strip().lower()` for the outer
   2/3–1/3 split and the inner C-selection CV). The default output dir gains a
   `_lemma` suffix (and `_seed<N>` for a non-zero seed), so row-split outputs are
   never overwritten:
   ```bash
   P=experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py
   uv run python $P --model pythia-160m --split lemma --device cpu
   uv run python $P --model pythia-1b   --split lemma --device cpu
   # split-noise check (Pythia-160M, seeds 1-4, six checkpoints, both splits)
   for s in 1 2 3 4; do for sp in row lemma; do
     uv run python $P --model pythia-160m --split $sp --seed $s --steps 0,256,512,1000,2000,143000 --device cpu
   done; done
   ```
   The App. F.6 numbers compare `final_acc` / `emergence_step` / `max_gain_pair`
   in `wordnet_supersense_probe_summary_<suffix>.csv` between the `_lemma` and
   row-split dirs; the row split's `wordnet_supersense_split_stats_<suffix>.csv`
   gives the leakage count (sum of `n_test_pos_with_train_sibling` over sum of
   `n_test_positive`). The matched-control nulls are the row-split ones of step 2.
   NLTK reads WordNet from `~/nltk_data` or from the directory in `NLTK_DATA`.

## Inputs (SSD canonical paths)
- `${UM_SSD_ROOT}/snapshots/pythia-{160m,1b}/EleutherAI_pythia-*_step*_wu.pt` (the W_U rows every probe reads)
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/pythia-160m/W_U/cross-snapshot-32/d24576/seed0.safetensors`
- `experiments/probes/concept_evolution_validation/derived/wordnet_supersense_160m/wordnet_supersense_probe_*_160m.{csv,json}`

## Outputs

`run_wordnet_supersense_probe.py` (per `--model`, `160m` and `1b`) writes under
`derived/wordnet_supersense_{160m,1b}/`:
- `wordnet_supersense_audit_{suffix}.csv`
- `wordnet_supersense_probe_trajectory_{suffix}.{csv,json}`
- `wordnet_supersense_probe_summary_{suffix}.csv`
- `wordnet_supersense_probe_pos_summary_{suffix}.csv`
- `wordnet_supersense_probe_metadata_{suffix}.json` (records `split`, `seed`)
- `wordnet_supersense_split_stats_{suffix}.csv` (per-category test support and same-lemma train/test leakage)

With `--split lemma` the same files go to `derived/wordnet_supersense_{160m,1b}_lemma/`.

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
