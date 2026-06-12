"""Shared helpers for the developmental-dynamics analysis pipeline.

Analysis scripts under experiments/crosscoders/crosscoder_main/ import from this package:
    - discovery: locate available crosscoder runs on disk
    - metrics:   lifecycle, adjacent rotation, CUSUM, active-feature mask

Run loading lives in `src.crosscoder.checkpoint_loaders` (`load_run`,
`_resolve_wu_cache_dir`, `_load_npy_or_pt`).
"""
