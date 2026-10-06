# concept_evolution_validation

When do independently specified lexical categories become linearly separable
in the readout rows? WordNet supersense (lexicographer file) categories are
probed one-vs-rest on the centered `W_U` rows of Pythia-160M and Pythia-1B at
every checkpoint, with a metadata-only baseline and matched-random and
label-shuffle nulls on Pythia-160M, and a rerun with splits grouped by lemma.
It supports `sec:vocabulary-family-readout` ("Lexical Organization Develops
Alongside Structural Change") and `sec:app-vocabulary-family-probes`.

## Paper results

Outputs are under `experiments/probes/concept_evolution_validation/derived/`.

| Paper label | What it shows | Script | Output |
|---|---|---|---|
| `fig:main-wordnet-matched-controls` | Pythia-160M readout rows come to separate WordNet categories beyond matched controls: median held-out balanced accuracy over the 35 categories rises from 0.499 at initialization to 0.874 at the final checkpoint, with the largest gains between steps 128 and 1000. | `scripts/run_wordnet_matched_controls.py` | `wordnet_matched_controls_160m/wordnet_matched_control_summary_160m.csv` |
| `fig:app-wordnet-supersense-probes-160m-inventory` | All 35 categories reach sustained separability between steps 128 and 1000, and 33 have their largest adjacent-checkpoint gain in that interval. | `scripts/run_wordnet_supersense_probe.py --model pythia-160m` | `wordnet_supersense_160m/wordnet_supersense_probe_trajectory_160m.csv` |
| `fig:app-wordnet-supersense-probes-1b` | The same 35 categories on Pythia-1B, without matched controls. | `scripts/run_wordnet_supersense_probe.py --model pythia-1b` | `wordnet_supersense_1b/wordnet_supersense_probe_trajectory_1b.csv` |
| `tab:app-wordnet-supersense-1b` | Pythia-1B onset steps and largest-gain intervals by part-of-speech group. | `scripts/run_wordnet_supersense_probe.py --model pythia-1b` | `wordnet_supersense_1b/wordnet_supersense_probe_pos_summary_1b.csv` |
| `sec:app-wordnet-lemma-split` (text) | Grouping the split by lemma leaves the early rise unchanged and lowers late accuracy (Pythia-160M median 0.841 at the final checkpoint against 0.874 under the row split). | `scripts/run_wordnet_supersense_probe.py --split lemma` | `wordnet_supersense_{160m,1b}_lemma/wordnet_supersense_probe_{trajectory,summary}_{160m,1b}.csv`, compared with the row-split files; leakage counts in `wordnet_supersense_split_stats_{160m,1b}.csv` |

## Run

1. Download WordNet 3.0 for NLTK once (into `~/nltk_data/`, or the directory in `NLTK_DATA`):
   ```bash
   uv run python -c "import nltk; nltk.download('wordnet')"
   ```
2. Probe every category at every checkpoint (row split):
   ```bash
   uv run python experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py --model pythia-160m
   uv run python experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py --model pythia-1b
   ```
3. Run the matched controls on Pythia-160M:
   ```bash
   uv run python experiments/probes/concept_evolution_validation/scripts/run_wordnet_matched_controls.py
   ```
4. Repeat the probe with lemma-grouped splits (`StratifiedGroupKFold` over `decode(id).strip().lower()` for the outer split and the inner selection of C); output directories gain a `_lemma` suffix, and `_seed<N>` for a non-zero seed:
   ```bash
   P=experiments/probes/concept_evolution_validation/scripts/run_wordnet_supersense_probe.py
   uv run python $P --model pythia-160m --split lemma
   uv run python $P --model pythia-1b --split lemma
   ```
5. Check split noise on Pythia-160M (seeds 1 to 4, six checkpoints, both splits):
   ```bash
   for s in 1 2 3 4; do for sp in row lemma; do
     uv run python $P --model pythia-160m --split $sp --seed $s --steps 0,256,512,1000,2000,143000
   done; done
   ```

## Inputs

- `${UM_SSD_ROOT}/snapshots/pythia-{160m,1b}/EleutherAI_pythia-*_step*_wu.pt`: the `W_U` rows every probe reads.
- `${UM_SSD_ROOT}/hf_release/parameter-trajectory-crosscoders/evaluation/eval-corpus/eval_tokens.pt`: token counts for the metadata baseline and the matched nulls (`--eval-tokens`).
- WordNet 3.0 through NLTK.

## Outputs

`run_wordnet_supersense_probe.py`, per model, under `derived/wordnet_supersense_{160m,1b}/` (or `..._lemma/`):

- `wordnet_supersense_audit_<suffix>.csv`
- `wordnet_supersense_probe_trajectory_<suffix>.{csv,json}`
- `wordnet_supersense_probe_summary_<suffix>.csv`
- `wordnet_supersense_probe_pos_summary_<suffix>.csv`
- `wordnet_supersense_probe_metadata_<suffix>.json` (records `split` and `seed`)
- `wordnet_supersense_split_stats_<suffix>.csv` (per-category test support and same-lemma train/test leakage)

`run_wordnet_matched_controls.py`, under `derived/wordnet_matched_controls_160m/`:

- `wordnet_matched_control_summary_160m.csv`
- `wordnet_matched_control_null_samples_160m.csv`
- `wordnet_matched_control_match_quality_160m.csv`
- `wordnet_matched_control_audit_160m.csv`
- `wordnet_matched_control_metadata_160m.json`

## Notes

- The lemma-split numbers compare `final_acc`, `emergence_step`, and `max_gain_pair` in `wordnet_supersense_probe_summary_<suffix>.csv` between the `_lemma` and row-split directories. The leakage count is the sum of `n_test_pos_with_train_sibling` over the sum of `n_test_positive` in the row split's `wordnet_supersense_split_stats_<suffix>.csv`. The matched-control nulls were not rerun under the lemma split.
- `--device cpu` runs the probes on CPU.
- `scripts/build_gazetteer.py` builds a 37-concept token gazetteer for Pythia-1B (`configs/preregistration/concepts_v1*.json`); no paper result uses it.
