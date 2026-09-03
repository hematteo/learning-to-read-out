# notebooks/

Colab-ready guided tours of the paper's two core analyses. Each runs
end-to-end on a free Colab runtime (GPU optional), is fully seeded, and
downloads only public Hugging Face checkpoints — no release artifacts or SSD
layout required.

| Notebook | What it shows | Runtime / downloads |
|---|---|---|
| [`01_availability_expression_lag.ipynb`](01_availability_expression_lag.ipynb) | The **availability–expression lag**: contrastive readout-swap grids over Pythia-160M pretraining checkpoints — task signal is decodable from early hidden states before the native readout expresses it (live miniature of `experiments/probes/contrastive_readout_swap/`, paper figure `fig:app-contrastive-readout-lag`). | ~10–15 min CPU; ~2.3 GB (6 checkpoints, cached) |
| [`02_wu_trajectory_crosscoders.ipynb`](02_wu_trajectory_crosscoders.ipynb) | The **instrument**: trains a $W_U$ trajectory crosscoder — first on a toy with planted structure (recovers it), then on real Pythia-160M snapshots — and reads out feature lifecycles, formation timing, and vocabulary families (miniature of `experiments/crosscoders/` + `experiments/lifecycle/`). | ~12 min T4 (GPU recommended; CPU auto-falls back to a rougher 30-epoch run); ~3 GB (8 checkpoints, cached) |

Both notebooks cache everything they extract or train (readouts, hidden
states, the demo crosscoder) under the repo's gitignored `local_snapshots/`,
so re-runs and kernel restarts skip downloads, forward passes, and training.

Suggested order: 01 (the phenomenon) → 02 (the instrument that explains it).
Both notebooks end with a map from their sections to the full experiment
pipelines; the figure-by-figure reproduction map is
[`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

Run locally from the repo root (the notebooks detect the checkout and skip
cloning):

```bash
make install
uv run --with matplotlib --with jupyterlab jupyter lab notebooks/
```

Notebook 02 additionally needs the vendored SAE library, which `make install`
already builds; on Colab its setup cell installs everything itself.
