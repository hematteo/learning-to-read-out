"""W_U trajectory-crosscoder training, checkpoint loaders, and snapshot utilities.

Curated public API re-exported from ``wu_adapter`` for convenience; every submodule
remains importable directly, e.g. ``from readout.crosscoder.wu_adapter import build_crosscoder``.

Re-exports are lazy (PEP 562) so that lightweight imports such as
``readout.crosscoder.checkpoints`` do not pay the transformers/llamascopium
import cost of ``wu_adapter``.
"""

__all__ = [
    "build_crosscoder",
    "load_snapshots",
    "load_wu_snapshot",
    "preprocess_snapshots",
    "quick_quality",
    "train",
]


def __getattr__(name):
    if name in __all__:
        from readout.crosscoder import wu_adapter

        return getattr(wu_adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
