"""Resolve Hugging Face checkpoint revisions per model family.

Pythia publishes checkpoints as ``step{N}`` branches. OLMo-2 encodes
tokens-seen in the branch name (``stage1-step{N}-tokens{T}B``), so the exact
branch must be looked up via the HF API; the per-model branch list is cached
for the life of the process (one API hit per model).
"""

from __future__ import annotations

import re

_OLMO_REV_CACHE: dict[str, list[str]] = {}


def resolve_revision(model_name: str, step: int) -> str:
    """Return the HF branch name for ``step`` of ``model_name``."""
    if "olmo" in model_name.lower():
        if model_name not in _OLMO_REV_CACHE:
            import requests

            r = requests.get(f"https://huggingface.co/api/models/{model_name}/refs", timeout=30)
            r.raise_for_status()
            _OLMO_REV_CACHE[model_name] = [b["name"] for b in r.json().get("branches", [])]
        pat = re.compile(rf"^stage1-step{step}(-tokens.*)?$")
        for rev in _OLMO_REV_CACHE[model_name]:
            if pat.match(rev):
                return rev
        raise ValueError(f"no OLMo revision matches stage1-step{step} for {model_name}")
    return f"step{step}"
