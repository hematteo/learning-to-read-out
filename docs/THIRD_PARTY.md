# Third-party code

## llamascopium (OpenMOSS Language-Model-SAEs)

- **What:** the crosscoder implementation of
  [Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs)
  (Ge et al., 2026), installed from PyPI as `llamascopium==2.0.0b34`. The
  trainer in `src/readout/crosscoder/wu_adapter.py` uses its `Crosscoder`,
  `CrosscoderConfig`, and `SparseAdam`.
- **License:** MIT, © 2024 OpenMOSS.
- **Local patch:** `src/readout/crosscoder/_llamascopium_patch.py` replaces
  `Crosscoder.init_encoder_with_decoder_transpose` with a DTensor-safe version.
  The upstream method rearranges the full sharded `W_D`, which deadlocked during
  4-GPU head-parallel training (Pythia-6.9B, OLMo-2-7B). The patch transposes each
  local shard instead; single-device behavior is identical
  (`tests/test_llamascopium_patch.py`). `wu_adapter` applies it on import.
- **Torch pin:** llamascopium 2.0.0b34 declares `torch>=2.11`, which has no CUDA
  11.8 build. `pyproject.toml` overrides that requirement (`[tool.uv]
  override-dependencies`) so Linux keeps torch 2.7.1+cu118, the version the
  paper's runs used.

## Other dependencies

PyTorch, Transformers, NumPy, SciPy, scikit-learn, pandas, datasets, and the
other packages in `pyproject.toml` are used as published under their own
licenses; `uv.lock` pins their versions.
