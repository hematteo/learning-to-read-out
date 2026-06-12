"""Validate / check experiments.yaml.

Modes:
    --scan    : enumerate experiments/<id>/ dirs and emit a stub manifest entry
                for any id missing from experiments.yaml.
    --check   : verify every id in experiments.yaml has matching dirs in
                experiments/, figures/ (flat <id>/),
                results/experiments/. Verify the inverse too (no orphan dirs).

Run from repo root:
    uv run python scripts/audit/validate_experiments.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "experiments.yaml"
EXP_DIR = REPO / "experiments"
FIG_DIR = REPO / "figures"  # flat at <id>/
RES_DIR = REPO / "results" / "experiments"


# Underscore-prefix dirs are conventional skip
def _flat_dir_ids(root: Path) -> set[str]:
    """Top-level dir names under root (flat layout used by figures/, results/experiments/)."""
    if not root.exists():
        return set()
    return {
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and not p.name.startswith(("_", "."))
        and p.name not in {"__pycache__"}
    }


def _experiment_ids(root: Path) -> set[str]:
    """Leaf experiment ids in the 2-level experiments/ layout.

    A leaf is a directory containing README.md at depth 2 (under a topic dir).
    """
    if not root.exists():
        return set()
    ids: set[str] = set()
    for top in root.iterdir():
        if (
            not top.is_dir()
            or top.name.startswith(("_", "."))
            or top.name == "__pycache__"
        ):
            continue
        for child in top.iterdir():
            if (
                not child.is_dir()
                or child.name.startswith(("_", "."))
                or child.name == "__pycache__"
            ):
                continue
            if (child / "README.md").exists():
                ids.add(child.name)
    return ids


def _load_manifest() -> dict:
    with MANIFEST.open() as f:
        return yaml.safe_load(f) or {"experiments": []}


def _manifest_ids(manifest: dict) -> set[str]:
    return {e["id"] for e in (manifest.get("experiments") or [])}


def cmd_scan() -> int:
    in_repo = _experiment_ids(EXP_DIR)
    in_manifest = _manifest_ids(_load_manifest())
    missing = sorted(in_repo - in_manifest)
    if not missing:
        print("manifest covers all experiment dirs")
        return 0
    print(f"# {len(missing)} experiment dir(s) not in experiments.yaml:")
    for eid in missing:
        print(f"- id: {eid}\n  status: draft\n  description: TODO")
    return 1


def cmd_check() -> int:
    repo_ids = _experiment_ids(EXP_DIR)
    fig_ids = _flat_dir_ids(FIG_DIR)
    res_ids = _flat_dir_ids(RES_DIR)
    manifest_ids = _manifest_ids(_load_manifest())

    issues: list[str] = []
    for eid in manifest_ids - repo_ids:
        issues.append(f"manifest id '{eid}' has no experiments/{eid}/ dir")
    # figures/<id>/ (flat) and results/experiments/<id>/ are optional
    # (an experiment may produce no figures, or no separate results dir)
    for eid in (fig_ids | res_ids) - repo_ids:
        orphan_paths = []
        if eid in fig_ids:
            orphan_paths.append(f"figures/{eid}/")
        if eid in res_ids:
            orphan_paths.append(f"results/experiments/{eid}/")
        issues.append(
            f"orphan: {' and '.join(orphan_paths)} has no experiments/{eid}/ dir"
        )

    if not issues:
        print(
            f"OK: {len(manifest_ids)} manifest entries, {len(repo_ids)} experiment dirs"
        )
        return 0
    for i in issues:
        print(f"  - {i}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.scan:
        return cmd_scan()
    if args.check:
        return cmd_check()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
