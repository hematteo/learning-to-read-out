# Learning to Read Out

[![ci](https://github.com/hematteo/learning-to-read-out/actions/workflows/ci.yml/badge.svg)](https://github.com/hematteo/learning-to-read-out/actions/workflows/ci.yml)

Code release for the paper **Learning to Read Out: Unembedding Dynamics in
Language Model Pretraining**.

A token logit is a dot product between a hidden state and one row of the
unembedding matrix `W_U`, the model's learned readout, and both factors move
during pretraining. This repository measures the readout factor directly with
*trajectory crosscoding*: across a fixed checkpoint schedule we extract `W_U`
snapshots from public Hugging Face models, fit one sparse dictionary to the
trajectory that each vocabulary row traces across checkpoints, and use the
resulting features to date when the readout reorganizes, to test which readout
directions carry a task margin, and to measure how far hidden state availability
runs ahead of native readout expression.

> **Chasing a figure from the paper?** Open [`docs/REPRODUCE.md`](docs/REPRODUCE.md),
> find its label in the index, and follow it to the experiment and the script
> that produces its metrics. Want to *reuse* the method instead? See
> [Reuse the library](#reuse-the-library).

## Quickstart

CPU-only; no GPU, data, or model weights needed for the test suite. The
contrastive-task tests fetch one small tokenizer (Pythia-160m) on the first
networked run — cached afterwards, and skipped when offline.

```bash
make install   # uv sync --extra dev (also builds the vendored SAE lib)
make test      # uv run --extra dev pytest -q
```

`make help` lists the full pipeline (`install -> test -> audit -> extract ->
train -> analyze`). This is a **code-only release**: it ships the source, the test suite,
and the manifest, but **no data and no metrics** — neither model checkpoints and
trained dictionaries nor the per-figure metric sidecars (CSV/JSON/`.pt`). Running
an experiment computes and persists the metrics behind its figures locally (to
gitignored `derived/`/`results/` trees); the repo ships no figure-rendering code,
and paper figures are rendered in the separate paper LaTeX tree from metrics
produced that way. Recomputing the metrics end-to-end needs external assets; see
the documentation map below.

## Reuse the library

The trajectory-crosscoder and probe code is importable directly:

```python
from readout.crosscoder import build_crosscoder, train, quick_quality
from readout.core.data import center_and_project
```

[`examples/minimal_crosscoder.py`](examples/minimal_crosscoder.py) is a small,
CPU-only, seeded demo: it synthesises a tiny `W_U` snapshot stack, fits a
trajectory crosscoder, and prints its explained variance and L0 — no data, GPU,
or downloads.

```bash
uv run python examples/minimal_crosscoder.py
```

## Guided tours (notebooks)

[`notebooks/`](notebooks/) holds two Colab-ready walkthroughs of the core
analyses, using public Pythia checkpoints only (no release artifacts needed):
the availability–expression lag via readout swaps, and the trajectory-crosscoder
instrument from a verifiable toy to real `W_U` snapshots, lifecycles, and
vocabulary families. See [`notebooks/README.md`](notebooks/README.md).

## Requirements

- **Python** 3.11 or 3.12; environment managed with [`uv`](https://docs.astral.sh/uv/).
- **Test suite** (`make install && make test`) runs CPU-only on **Linux, macOS,
  or Windows** — all three OSes are CI-tested (`.github/workflows/ci.yml`). No
  GPU, data, or model weights needed; the contrastive-task tests fetch one small
  tokenizer on the first networked run (cached; skipped offline).
- **GPU training** beyond 160M targets **Linux + CUDA 11.8** (torch is pinned
  to cu118 wheels on Linux for broad driver compatibility; see
  `pyproject.toml`). On other platforms torch installs from PyPI. Note the
  cu118 index also caps the torch version on Linux: the lockfile resolves
  torch 2.7.1+cu118 there vs. a newer PyPI torch on macOS/Windows.
- **Device selection is automatic** at runtime — CUDA > MPS > CPU — so small
  models also run on Apple Silicon (MPS) or on CPU anywhere.

## Repository layout

| Tier | What it holds |
|---|---|
| `src/readout/` | The installable `readout` package (`uv sync` installs it editable): `core/` (paths, model specs, data), `crosscoder/` (trajectory training), `dynamics/` (run discovery + lifecycle metrics), `probes/`, `baselines/`. |
| `experiments/` | Research code grouped by theme (`crosscoders/`, `lifecycle/`, `causal/`, `probes/`, `baselines/`, `ablations/`, `capacity/`); each experiment is `<topic>/<id>/{README,scripts}`, mapped to the paper figures it backs in [`experiments.yaml`](experiments.yaml). |
| `scripts/` | Command-line entry points grouped by verb: `extract/`, `train/`, `eval/`, `audit/`. |
| `examples/` | Small, CPU-only runnable demos of the library — no data, GPU, or downloads. |
| `notebooks/` | Colab-ready guided tours of the two core analyses (readout-swap lag, trajectory crosscoders); public checkpoints only. |
| `tests/` | Pytest suite — runs CPU-only with no external data. |
| `configs/` | Run configs and preregistration. |
| `docs/` | Companion guides: [`REPRODUCE.md`](docs/REPRODUCE.md), [`DATA.md`](docs/DATA.md), [`THIRD_PARTY.md`](docs/THIRD_PARTY.md). |
| `lib/` | Vendored OpenMOSS *Language-Model-SAEs* (`llamascopium`); see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md). |

`experiments.yaml` is the machine-readable manifest mapping each experiment to
the paper figures it backs and the metric files that underlie them (the metrics
are regenerated by running the scripts, not shipped; figures are rendered in the
paper LaTeX tree, not in this repo).

The layout is a strict contract, enforced in CI by `make audit`
(`scripts/audit/check_layout.py`): code lives in one of three tiers by reuse —
`src/readout/` and `scripts/<verb>/` (closed verb set: `train`, `extract`, `eval`,
`audit`) hold only code used by more than one experiment; anything single-use
lives under its experiment's `experiments/<topic>/<id>/scripts/`. Every
experiment directory carries a `README.md` and a matching `experiments.yaml`
entry (orphans on either side fail the audit), and no figure-rendering code
ships — scripts persist metrics (CSV/JSON/`.pt`) only.

## Documentation

Recommended reading order:

| Document | Purpose |
|---|---|
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | **Start here to chase a figure.** Figure → experiment → metric-producing script map (companion to `experiments.yaml`). |
| [`docs/DATA.md`](docs/DATA.md) | External data and checkpoints; the `UM_SSD_ROOT` storage layout. |
| [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) | The vendored `llamascopium` library and what was patched. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Dev workflow: the tier rule, how to add an experiment, the audit contract. |
| [`CITATION.cff`](CITATION.cff) | Citation metadata. |

## Notation

Shorthand recurring in module, experiment, and figure names:

| Token | Meaning |
|---|---|
| `W_U` | Output unembedding matrix |
| `W_E` | Input embedding matrix |
| `wu` | `W_U` (unembedding) |
| `we` | `W_E` (embedding) |
| `hln` | `h_LN`, the pre-readout (post-final-LayerNorm) hidden state |
| `sva` | Subject-verb agreement |
| `persnap` | Per-snapshot SAE (one SAE per checkpoint, a baseline) |
| `cc` | Crosscoder |
| `ev` | Explained variance (reconstruction quality, higher is better) |
| `l0` | Mean number of active dictionary features per row |

## Related repository

[`sparse-readout-prism`](https://github.com/hematteo/sparse-readout-prism) is the
companion release for *Sparse Readout Prism: Explaining Logit-Lens Scores in
Features Instead of Tokens* ([arXiv:2609.01936](https://arxiv.org/abs/2609.01936)).
It factorizes a model's *final* `W_U` into a sparse feature basis for logit lens
readout analysis, where this repo studies how that readout *forms over
pretraining* (trajectory crosscoders across checkpoints). The readout prism
appendix of the Learning to Read Out paper is a scoped preview of that work.

## Citation

If you use this code, please cite the paper (machine-readable metadata in
[`CITATION.cff`](CITATION.cff); the arXiv link will be added once the preprint
is up):

```bibtex
@misc{he2026learningtoreadout,
  title  = {Learning to Read Out: Unembedding Dynamics in Language Model Pretraining},
  author = {He, Matteo and Shen, William F. and Iacob, Alex and Jovanovic, Andrej
            and Qiu, Xinchi and Lane, Nicholas D.},
  year   = {2026},
  note   = {Under review. Code: https://github.com/hematteo/learning-to-read-out},
}
```

## License

MIT. See [`LICENSE`](LICENSE). The vendored library under `lib/` is also MIT
(© 2024 OpenMOSS); see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
