"""Append/update rows in the crosscoder sweep manifest CSV.

The manifest header is:
    kind,model,matrix,d_sae,d_model,vocab,seed,n_steps,epochs,batch,lr,l1,
    jumprelu_lr_factor,tanh_stretch,input_preprocess,optimizer,amp_dtype,
    EV,L0,dead_rate,CUSUM_p,rate_rotation_r,path,notes

Idempotent: rows are keyed by absolute resolved checkpoint path.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .checkpoints import unwrap_ckpt

MANIFEST_HEADER = [
    "kind",
    "model",
    "matrix",
    "d_sae",
    "d_model",
    "vocab",
    "seed",
    "n_steps",
    "epochs",
    "batch",
    "lr",
    "l1",
    "jumprelu_lr_factor",
    "tanh_stretch",
    "input_preprocess",
    "optimizer",
    "amp_dtype",
    "EV",
    "L0",
    "dead_rate",
    "CUSUM_p",
    "rate_rotation_r",
    "path",
    "notes",
]

DEFAULT_MANIFEST = Path("results/crosscoder_sweep_manifest.csv")


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return repr(v)
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return str(v)


def _ensure_manifest_dir(manifest_csv: Path, strict: bool) -> None:
    parent = manifest_csv.parent.resolve()
    if not parent.exists():
        msg = (
            f"Manifest parent directory does not exist: {parent}. "
            f"If results/ is a symlink to ${{UM_SSD_ROOT}}/, mount the SSD first."
        )
        if strict:
            raise FileNotFoundError(msg)
        parent.mkdir(parents=True, exist_ok=True)


def _read_existing(manifest_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not manifest_csv.exists():
        return list(MANIFEST_HEADER), []
    with manifest_csv.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return list(MANIFEST_HEADER), []
    header = rows[0]
    data = [dict(zip(header, r)) for r in rows[1:] if any(c.strip() for c in r)]
    return header, data


def _atomic_write(manifest_csv: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    fd, tmp_path = tempfile.mkstemp(prefix=".manifest_", suffix=".csv", dir=str(manifest_csv.parent))
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in rows:
                writer.writerow([r.get(k, "") for k in header])
        os.replace(tmp_path, manifest_csv)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def infer_row_from_checkpoint(ckpt_path: Path, notes: str = "") -> dict[str, str]:
    """Read a saved crosscoder checkpoint and synthesise a manifest row."""
    ckpt_path = Path(ckpt_path)
    # Normalise new-format ("wrapper") checkpoints whose steps/training/model_name
    # are nested under "config" to the flat layout this function reads.
    ck = unwrap_ckpt(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    cfg = ck.get("config", {}) or {}
    training = ck.get("training", {}) or {}
    quality = ck.get("quality", {}) or {}
    sd = ck.get("state_dict", {}) or {}

    # d_sae / d_model: prefer state-dict shapes (authoritative), fall back to cfg/training.
    d_sae = None
    d_model = None
    if "W_E" in sd:
        we = sd["W_E"]
        # (n_heads, d_model, d_sae)
        if we.ndim == 3:
            d_model = int(we.shape[1])
            d_sae = int(we.shape[2])
    if d_sae is None:
        d_sae = training.get("d_sae") or (
            int(cfg.get("d_model", 0) * cfg.get("expansion_factor", 0)) if cfg.get("d_model") else None
        )
    if d_model is None:
        d_model = cfg.get("d_model")

    steps = ck.get("steps", []) or []
    matrix = "WU"  # wu_adapter is W_U-specific; downstream callers can override via notes

    dead_rate = quality.get("dead_rate")
    if dead_rate is None:
        dead_rate = quality.get("dead_feature_rate")

    row = {
        "kind": cfg.get("sae_type", "crosscoder"),
        "model": ck.get("model_name", ""),
        "matrix": matrix,
        "d_sae": d_sae,
        "d_model": d_model,
        "vocab": quality.get("n_rows"),
        "seed": ck.get("seed"),
        "n_steps": len(steps) if steps else "",
        "epochs": training.get("n_epochs"),
        "batch": training.get("batch_size"),
        "lr": training.get("lr"),
        "l1": training.get("l1_coefficient"),
        "jumprelu_lr_factor": training.get("jumprelu_lr_factor"),
        "tanh_stretch": training.get("tanh_stretch_coefficient"),
        "input_preprocess": training.get("input_preprocess"),
        "optimizer": training.get("optimizer"),
        "amp_dtype": training.get("amp_dtype"),
        "EV": quality.get("explained_variance"),
        "L0": quality.get("mean_l0"),
        "dead_rate": dead_rate,
        "CUSUM_p": quality.get("CUSUM_p"),
        "rate_rotation_r": quality.get("rate_rotation_r"),
        "path": str(ckpt_path.resolve()),
        "notes": notes,
    }
    return {k: _fmt(v) for k, v in row.items()}


def update_or_insert(
    rows: list[dict[str, str]], new_row: dict[str, str], key: str = "path"
) -> tuple[list[dict[str, str]], str]:
    """Replace an existing row sharing ``new_row[key]`` or append it. Returns (rows, action)."""
    target = new_row.get(key, "")
    for i, r in enumerate(rows):
        if r.get(key, "") == target and target != "":
            rows[i] = {**r, **new_row}
            return rows, "updated"
    rows.append(new_row)
    return rows, "inserted"


def _merge_header(header: list[str]) -> list[str]:
    """MANIFEST_HEADER extended with any unknown columns already in the file."""
    merged = list(MANIFEST_HEADER)
    for c in header:
        if c not in merged:
            merged.append(c)
    return merged


def upsert_manifest_rows(
    manifest_csv: Path,
    new_rows: list[dict[str, Any]],
    key: str = "path",
    strict: bool = True,
) -> list[str]:
    """Merge rows into the manifest CSV keyed by ``key``; atomic and idempotent.

    Values are formatted with the manifest conventions (lists join with ';',
    None/NaN become empty). Unknown columns already in the file are preserved.
    Returns one action per row ("updated" or "inserted").
    """
    manifest_csv = Path(manifest_csv)
    _ensure_manifest_dir(manifest_csv, strict=strict)
    header, rows = _read_existing(manifest_csv)
    header = _merge_header(header)
    actions: list[str] = []
    for r in new_rows:
        rows, action = update_or_insert(rows, {k: _fmt(v) for k, v in r.items()}, key=key)
        actions.append(action)
    _atomic_write(manifest_csv, header, rows)
    return actions


def append_manifest_row(
    ckpt_path: Path,
    manifest_csv: Path = DEFAULT_MANIFEST,
    notes: str = "",
    strict: bool = True,
) -> dict[str, str]:
    """Append (or idempotently update) one manifest row for ``ckpt_path``.

    Raises ``FileNotFoundError`` if the manifest's parent directory does not
    exist (e.g. external SSD not mounted) when ``strict=True``.
    Returns the row that was written.
    """
    manifest_csv = Path(manifest_csv)
    # Fail fast on a missing parent dir before deserializing the checkpoint.
    _ensure_manifest_dir(manifest_csv, strict=strict)
    new_row = infer_row_from_checkpoint(ckpt_path, notes=notes)
    upsert_manifest_rows(manifest_csv, [new_row], key="path", strict=strict)
    return new_row
