"""Shared helpers for the developmental-dynamics analysis pipeline.

The analysis scripts under reproduce/ (validation, lifecycle) and the test
suite import from this package:
    - discovery: locate available crosscoder runs on disk
    - metrics:   lifecycle, adjacent rotation, CUSUM, active-feature mask
    - lifecycle: the paper's rule-based profile taxonomy
                 (tab:lifecycle-profile-rules)

Run loading lives in `readout.crosscoder.checkpoint_loaders` (`load_run`,
`_resolve_wu_cache_dir`, `_load_npy_or_pt`).
"""
