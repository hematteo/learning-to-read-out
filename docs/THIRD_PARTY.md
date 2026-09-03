# Third-party code

This repository contains first-party code **and** one vendored third-party
library. This file records what is third-party, where it lives, and under what
licence, so the boundary between first-party code and reused software is
explicit.

## Vendored: `lib/Language-Model-SAEs/` (llamascopium)

- **What:** the OpenMOSS *Language-Model-SAEs* framework, distributed on PyPI as
  `llamascopium`. Upstream: <https://github.com/OpenMOSS/Language-Model-SAEs>.
- **Licence:** MIT, © 2024 OpenMOSS. See `lib/Language-Model-SAEs/LICENSE`.
- **Why it is vendored rather than pulled from PyPI:** the production training path
  (`src/readout/crosscoder/wu_adapter.py`) uses this library's `Crosscoder` /
  `CrosscoderConfig` model and `SparseAdam` optimizer. The vendored copy carries
  **one local patch** — a DTensor-safe `init_encoder_with_decoder_transpose`
  that avoids a deadlock observed during 4-GPU head-parallel training. The
  published PyPI wheel (`2.0.0b34`) lacks that patch, so it is not a drop-in
  substitute for reproducing the multi-GPU (6.9B, OLMo-2-7B) runs. Vendoring
  also lets the repository install from a clean clone with `uv sync` alone.
- **What was changed locally:** only the single function noted
  above (`lib/Language-Model-SAEs/src/llamascopium/models/crosscoder.py`, encoder-init path) plus the
  package's own dependency pins. Everything else under `lib/` is upstream code.
- **What was trimmed from the upstream tree:** to keep the vendored copy to its
  import surface, the following non-essential components were removed (none are
  imported by any first-party code — verified by grep across `src/`, `scripts/`,
  `experiments/`, `tests/`):
  - `server/` — the FastAPI web UI (16 files; optional `plotly`/`kaleido`/
    `uvicorn`/`fastapi` deps this release never installs);
  - `tests/` — the library's own unit/integration/distributed test suite
    (24 files);
  - `scripts/gen_ref_pages.py` — a docs-site generation helper (1 file);
  - `shim/lm-saes/` — the deprecated `lm-saes` backward-compatibility shim
    package, superseded by the `llamascopium` rename (3 files);
  - `Makefile`, `.pre-commit-config.yaml`, `.gitignore` — upstream development
    config not used by this release.

  What is **kept** is exactly the import surface plus build metadata:
  `src/llamascopium/` (the package, including the patched `Crosscoder` and
  `SparseAdam` the trainer imports), `pyproject.toml`, `LICENSE`, and `README.md`.
  The package's `uv_build` backend only discovers packages under
  `src/llamascopium/`, so dropping the above does not affect `uv sync` or the
  editable wheel; the `llamascopium` / `lm-saes` console-script entry points in
  `pyproject.toml` still resolve to `llamascopium.cli`, which is kept.

- **Provenance and verification:** the vendored tree corresponds to the PyPI
  release `llamascopium==2.0.0b34` plus the local patch documented above; the
  upstream git commit SHA was **not** recorded at vendoring time. To verify the
  correspondence, fetch the upstream source with
  `pip download llamascopium==2.0.0b34 --no-deps` and diff it against
  `lib/Language-Model-SAEs/`, excluding the trimmed components and the patched
  encoder-init function listed above. The dependency-pin changes in the vendored
  `pyproject.toml` were not individually recorded; diffing that file against the
  downloaded release's `pyproject.toml` recovers them.

This library is cited in the paper as the crosscoder architecture the trajectory
dictionaries follow (`\citet{ge2026crosscoder}`).

## First-party crosscoder code

The production trajectory dictionaries are trained by first-party glue code
in `src/readout/crosscoder/` (`wu_adapter.py`, `training.py`) wrapping the
vendored llamascopium `Crosscoder`. An earlier, independent `TiedTopKCrossCoder`
implementation (a per-checkpoint approach that was abandoned because it failed to
match the trajectory) is **not** on the result-producing path and has been
removed from this release to avoid confusion with the production code.

## Standard dependencies

All other dependencies (PyTorch, Transformers, NumPy, SciPy, scikit-learn,
pandas, datasets, etc.) are used as published under their own
open-source licences and are resolved by `uv` from `pyproject.toml` / `uv.lock`.
