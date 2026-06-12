# configs/

Configuration records for the learning-to-read-out pipeline.

## `runs/` — crosscoder training settings-of-record

Each `runs/<id>.yaml` is the committed, version-controlled record of the exact
training hyperparameters for one parameter-trajectory crosscoder used in the
paper's results. Before this directory existed, those settings lived only in CLI
flags and inside the saved checkpoints; capturing them here means reproducibility
no longer depends on a surviving checkpoint or shell-history fragment.

The field names mirror the `scripts/train/train_crosscoder.py` CLI flags (e.g.
`expansion_factor`, `batch_size`, `lr`, `n_epochs`, `seed`, `input_preprocess`,
`amp_dtype`, `l1_coefficient`, `tanh_stretch`, `init_threshold`). Every file
carries a `source:` field citing where each value came from (the Makefile / CLI
docstring, `experiments.yaml`, a `run_*.sh` launcher, or the regenerated
`full_inventory.csv` / `large_evals/*.json` sidecars — which are not shipped in
this code-only release).

These files are **documentation/record, not a runtime loader**: nothing in the
codebase auto-loads them. To retrain, pass the corresponding flags to
`scripts/train/train_crosscoder.py` (or
`experiments/crosscoders/crosscoder_olmo/scripts/train_distributed.py` for the
multi-GPU OLMo/6.9B runs). `reported:` blocks list the published EV / L0 so a
retrain can be cross-checked.

Values that are not re-derivable from files shipped in this repo (per-model
`lr` for the 1B/6.9B Pythia runs, the selected sparse 6.9B run's
`l1_coefficient`/`init_threshold`) were recovered from the released checkpoints'
embedded training dicts and `.config.json` sidecars and are recorded as exact,
with the `source:` field naming the artifact each value came from.

## `preregistration/` — preregistered concept sets

`preregistration/concepts_v1.json` holds the preregistered concept / token-family
definitions used by the probe and causal experiments.
