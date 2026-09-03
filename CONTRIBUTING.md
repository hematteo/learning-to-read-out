# Contributing / development workflow

This is the code release for a paper; the bar for changes is that a
future researcher can still reproduce and extend the published analyses.

## Setup and gates

```bash
make install   # uv sync --extra dev (installs the readout package editable,
               # builds the vendored SAE lib)
make test      # pytest — CPU-only, no data (~20 s); first networked run fetches
               # one small tokenizer (cached; those tests skip offline)
make lint      # ruff, same check as CI
make audit     # layout contract (scripts/audit/check_layout.py)
```

CI runs all four on Python 3.11 and 3.12 plus `examples/minimal_crosscoder.py`
as a quickstart smoke test. Every PR must keep them green.

## Where code lives (the tier rule)

Code is placed by **reuse**, and `make audit` enforces the skeleton:

| Tier | Rule |
|---|---|
| `src/readout/` | The installable library. Anything imported by more than one experiment lives here (loaders, metrics, inference math, path/env resolution). No `__main__` blocks. |
| `scripts/<verb>/` | CLI entry points, closed verb set `train / extract / eval / audit`. May import only `readout`, never `experiments`. |
| `experiments/<topic>/<id>/scripts/` | Single-experiment research code. May import `readout` and sibling modules in its *own* `scripts/` dir — never another experiment's tree (the audit fails on cross-experiment imports and on repo-root `sys.path` edits; the package is installed, so `import readout` always works). |

Outputs are metrics only (CSV/JSON/`.pt`), written under the experiment's
gitignored `derived/` (or `results/`); no figure-rendering code ships —
paper figures render in the separate LaTeX tree from these metrics.

## Adding an experiment

1. Create `experiments/<topic>/<id>/` with a `README.md` and a `scripts/` dir.
   Follow an existing README's shape: what it shows, **Figures produced**
   (label → metric file → producing script), **Claim**, **Reproduce**,
   **Inputs**, **Outputs**, **Layout**.
2. Add a matching entry in `experiments.yaml` — same `id` as the directory,
   with `description`, `scripts_dir`, `results_dir`, `inputs`, `paper_labels`
   (the schema is documented at the top of the file). The audit fails on
   orphans on either side.
3. Runnable commands in the README must be complete: a reader chasing a
   figure copy-pastes them (include required flags and output paths).
   The figure-index table in `docs/REPRODUCE.md` is generated from the
   manifest — run `uv run python scripts/audit/gen_reproduce_index.py` after
   editing `paper_labels`/`paper_figures` (CI checks for drift).
4. Seed everything (`readout.core.repro.seed_everything`); stamp provenance
   (`log_run_provenance` / `git_commit`) on anything that writes artifacts.

## Conventions

- Python ≥3.11, `uv` for everything; don't `pip install` into the venv.
- Reproducibility beats convenience: sorted globs, explicit seeds, atomic
  writes for anything resumable (`readout.core.resume`).
- Storage resolves through `readout.core.paths` (`UM_SSD_ROOT` and friends —
  see the environment-variable table in `docs/DATA.md`); never hardcode
  machine paths.
- Comment non-obvious tensor shapes: `# (batch, seq, d_model)`.
- The vendored `lib/Language-Model-SAEs` is third-party (see
  `docs/THIRD_PARTY.md`); don't edit it except to port an upstream fix, and
  record any local patch in that doc.
