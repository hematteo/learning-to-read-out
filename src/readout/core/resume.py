"""Resumability primitives for cluster jobs.

Every script submitted to a remote cluster must produce identical final output
across arbitrary kill+restart cycles. This module provides the building blocks:
atomic writes, per-item exists-check, append-only manifest, restart banner.

Usage pattern:

    from readout.core.resume import (
        atomic_write_json, atomic_write_torch,
        iter_undone, append_manifest, aggregate_json_shards,
    )

    out_dir = Path("results/.../shards")
    out_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(target_steps)

    def shard_path(step):
        return out_dir / f"step{step}.json"

    for step in iter_undone(items, shard_path, label="step"):
        payload = compute(step)
        atomic_write_json(shard_path(step), payload)
        append_manifest(out_dir / "manifest.csv", {"step": step, **payload})

    aggregate_json_shards(out_dir, out_dir.parent / "results.csv", key="step")
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch


def _atomic_replace(path: Path, write_to_tmp: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_str)
    try:
        write_to_tmp(tmp)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def atomic_write_json(path: Path, payload: dict | list) -> None:
    def _write(tmp: Path) -> None:
        tmp.write_text(json.dumps(payload, indent=2, default=str))

    _atomic_replace(path, _write)


def atomic_write_torch(path: Path, payload: object) -> None:
    def _write(tmp: Path) -> None:
        torch.save(payload, tmp)

    _atomic_replace(path, _write)


def iter_undone(
    items: Iterable,
    shard_path_fn: Callable[[object], Path],
    *,
    label: str = "item",
) -> Iterator:
    items_list = list(items)
    done = [it for it in items_list if shard_path_fn(it).exists()]
    todo = [it for it in items_list if not shard_path_fn(it).exists()]
    if done:
        first_todo = todo[0] if todo else None
        print(
            f"[resume] {len(done)}/{len(items_list)} {label}s already done; starting at {label}={first_todo}",
            flush=True,
        )
    else:
        print(f"[resume] starting fresh, {len(items_list)} {label}s todo", flush=True)
    yield from todo


def append_manifest(manifest_path: Path, row: dict) -> None:
    """Append-only CSV manifest. Writes header on first append.

    Schema-safe across rows with differing keys: rows are always written
    under the existing header (missing fields as ""); a row that introduces
    new keys triggers an atomic rewrite with the union header, so columns
    never silently misalign.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    header: list[str] = []
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        with manifest_path.open("r", newline="") as f:
            header = next(csv.reader(f), []) or []
    new_keys = [k for k in row.keys() if k not in header]
    fieldnames = header + new_keys

    if header and new_keys:
        with manifest_path.open("r", newline="") as f:
            old_rows = list(csv.DictReader(f))

        def _rewrite(tmp: Path) -> None:
            with tmp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
                writer.writeheader()
                for r in old_rows:
                    writer.writerow({k: r.get(k) or "" for k in fieldnames})
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        _atomic_replace(manifest_path, _rewrite)
        return

    with manifest_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        if not header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def aggregate_json_shards(
    shard_dir: Path,
    output_csv: Path,
    *,
    pattern: str = "*.json",
    key: str | None = None,
) -> int:
    """Collect per-shard JSON dicts into a single CSV. Returns row count.

    Caller is responsible for ensuring all expected shards are present before
    aggregation; this function aggregates whatever is on disk.
    """
    shards = sorted(shard_dir.glob(pattern))
    rows: list[dict] = []
    for shard in shards:
        payload = json.loads(shard.read_text())
        if isinstance(payload, list):
            rows.extend(payload)
        else:
            rows.append(payload)
    if key is not None:
        rows.sort(key=lambda r: r.get(key, 0))
    if not rows:
        return 0
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    return len(rows)
