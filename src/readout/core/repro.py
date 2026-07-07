"""Reproducibility helpers.

`seed_everything` sets all RNG sources (Python `random`, NumPy, PyTorch CPU
and CUDA) from a single integer seed.

`log_run_provenance` prints (and optionally returns) the current git
commit hash, dirty state, Python version, torch version, and CUDA device
info. Call at the top of every entry-point script so a downstream
reviewer can reconstruct the environment the run was executed in.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from .paths import repo_root


def git_commit(repo: Path | None = None) -> str | None:
    """Full HEAD commit hash, or ``None`` if git is unavailable.

    Used by entry-point scripts to stamp result sidecars with provenance.
    (``log_run_provenance`` uses the short hash + dirty flag for stdout.)
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo) if repo is not None else str(repo_root()),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError):
        # RuntimeError: repo_root() found no repo (packaged install).
        return None


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python `random`, NumPy, and PyTorch (CPU + all CUDA devices).

    If ``deterministic=True``, also force cuDNN deterministic mode and call
    ``torch.use_deterministic_algorithms(True, warn_only=True)``. This can
    slow training but eliminates remaining nondeterminism for matched-seed
    reproducibility comparisons.
    """
    random.seed(seed)
    # NOTE: setting PYTHONHASHSEED here cannot affect the already-running
    # interpreter (the str/bytes hash salt is fixed at startup); it only
    # covers subprocesses inheriting this environment. No in-repo logic may
    # depend on str hash order.
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _git_hash(repo: Path | None = None) -> tuple[str, bool]:
    """Return (short_hash, is_dirty). ('unknown', False) if not in a git repo.

    Anchors git at the repo containing this package, never the process cwd
    (which may be an unrelated git repo, e.g. a dotfiles-managed $HOME).
    """
    if repo is None:
        try:
            repo = repo_root()
        except RuntimeError:
            # Packaged install: no repo root to anchor at; skip git entirely
            # rather than record whatever repo the process was launched from.
            repo = None
    if repo is not None:
        try:
            h = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short=12", "HEAD"],
                    cwd=str(repo),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            dirty = (
                subprocess.call(
                    ["git", "diff-index", "--quiet", "HEAD", "--"],
                    cwd=str(repo),
                    stderr=subprocess.DEVNULL,
                )
                != 0
            )
            return h, dirty
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    # No git available (e.g. a packaged tarball with no .git). Fall back to
    # an explicit commit record so run provenance stays meaningful: the
    # READOUT_COMMIT env var, then a COMMIT_HASH file at the repo root.
    env_h = os.environ.get("READOUT_COMMIT")
    if env_h:
        return env_h.strip(), False
    try:
        commit_file = repo_root() / "COMMIT_HASH"
        if commit_file.is_file():
            return (commit_file.read_text().strip() or "unknown"), False
    except (OSError, RuntimeError):
        pass
    return "unknown", False


def log_run_provenance(seed: int | None = None) -> dict[str, Any]:
    """Print run provenance to stdout and return it as a dict.

    Pair with ``seed_everything(seed)`` at the start of each entry-point
    script.
    """
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    h, dirty = _git_hash()
    info["git_hash"] = h
    info["git_dirty"] = dirty
    if seed is not None:
        info["seed"] = seed
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    print("[provenance] " + ", ".join(f"{k}={v}" for k, v in info.items()), flush=True)
    return info
