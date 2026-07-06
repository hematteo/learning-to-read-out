"""Provenance helpers for the analysis table.

Computes file content hashes (sha256), the current git sha, and exposes
the locked metric_version constant.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from readout.core import paths as _paths

# Bumped when the operational definitions in plan §3 change. Analysis scripts
# treat (metric_version, git_sha) as the cache key for derived artifacts:
# bumping this invalidates everything downstream.
METRIC_VERSION = "0.1.0"

_HASH_CHUNK = 1 << 20  # 1 MiB
_GIT_SHA_CACHE: str | None = None


def file_sha256(path: Path | str | None) -> str | None:
    """Streaming sha256 of a file. Returns None when path is None or missing.

    Skipped (returns None) for directories — T1.3 per-snap SAEs are stored as
    a directory of 32 checkpoints; hashing a directory tree is out of scope
    for the per-row provenance field.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo_root: Path | str | None = None) -> str | None:
    """Short git sha of the working tree. Cached; HEAD is read once per process."""
    global _GIT_SHA_CACHE
    if _GIT_SHA_CACHE is not None:
        return _GIT_SHA_CACHE
    cwd = Path(repo_root) if repo_root is not None else _paths.repo_root()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        )
        _GIT_SHA_CACHE = out.decode().strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        _GIT_SHA_CACHE = None
    return _GIT_SHA_CACHE
