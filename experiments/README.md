# experiments/

Research code grouped by topic. Each experiment lives in
`<topic>/<id>/` with a `README.md` and a `scripts/` directory. The
machine-readable map from experiments to paper labels is
[`../experiments.yaml`](../experiments.yaml), and the figure-by-figure
reproduction index is [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

| Topic | Experiments | What they cover |
|---|---|---|
| `crosscoders/` | `crosscoder_main`, `crosscoder_olmo` | Dictionary validation for the Pythia models (inventory, next-token fidelity, PCA baselines) and training of the OLMo-2-7B dictionary. |
| `lifecycle/` | `feature_lifecycle_trajectories` | Decoder-norm trajectories, lifecycle profiles, the reorganization window, and stability across dictionary fits. |
| `probes/` | `concept_evolution_validation`, `contrastive_readout_swap` | WordNet supersense probes on `W_U` rows, and hidden-state availability versus native readout expression on contrastive tasks. |
| `causal/` | `temporal_localization_patching`, `language_stratified_readout_swap`, `sparse_feature_causal_tests`, `contrastive_task_feature_rescue` | Readout swaps on held-out text, and ablations of sparse features in the reconstructed readout. |
| `baselines/` | `dense_readout_diagnostics`, `persnap_sae_baseline` | Dense `W_U` diagnostics and per-snapshot SAE baselines. |
| `ablations/` | `heldout_checkpoints`, `olmo_matched_checkpoint_window`, `pretraining_recipe_control` | Held-out checkpoint reconstruction, the late-start control for OLMo-2-7B, and the controlled 31M pretraining runs. |

Each experiment README has the same sections: the question it answers and
the paper section it supports, a **Paper results** table (paper label, what
it shows, producing script, output file), **Run** steps in order,
**Inputs**, **Outputs**, and, where a reproducer needs them, **Notes**.

The scripts compute and save metrics (CSV, JSON, `.pt`); they do not draw
figures. Outputs are regenerated on run and are not shipped. External inputs
under `${UM_SSD_ROOT}` (snapshots, released dictionaries, aggregates) are
described in [`../docs/DATA.md`](../docs/DATA.md). Run every command from the
repository root.
