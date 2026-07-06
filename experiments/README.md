# experiments/

Research code grouped by theme; each experiment is
`<topic>/<id>/{README.md,scripts/}`. The machine-readable index mapping every
experiment to the thesis figures it backs is [`../experiments.yaml`](../experiments.yaml);
the figure-by-figure reproduction map is [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

| Topic | What its experiments establish |
|---|---|
| `crosscoders/` | The instrument itself: training and validating the trajectory crosscoders (Pythia W_U `crosscoder_main`, OLMo-2 `crosscoder_olmo`, input-embedding `crosscoder_we`). The shared analysis library they import lives in `src/readout/dynamics/`. |
| `lifecycle/` | How individual sparse features form, reorganize, and persist across pretraining (trajectory/profile/wishbone metrics). |
| `causal/` | Whether the readout structure is load-bearing: activation patching, sparse-feature interventions, contrastive-task feature rescue. |
| `probes/` | Correlational evidence: concept probes over checkpoints and the contrastive readout-swap screen (whose causal companion lives under `causal/`). |
| `baselines/` | Alternatives the crosscoder is compared against: per-snapshot SAEs and dense readout diagnostics. |
| `ablations/` | Robustness controls: held-out checkpoints, matched OLMo checkpoint windows, pretraining-recipe control. |
| `capacity/` | Instrument quality: the EV-vs-L0 Pareto frontier behind the chosen `(d_sae, lambda)` operating point. |

Each leaf README opens with the claim the experiment backs, its "Figures
produced" ledger, and a Reproduce section. Bucket membership describes the
method class (probing vs. intervention), not the paper section: e.g.
`probes/contrastive_readout_swap` is the screening probe whose intervention
follow-up is `causal/contrastive_task_feature_rescue`.

Naming note: scripts named `plot_*` / `build_*_plots` / `build_*_figures`
compute and persist the *metrics behind* the figure they are named after
(CSV/JSON/`.pt` sidecars); no figure-rendering code ships in this repo — the
thesis LaTeX tree renders figures from those metrics.
