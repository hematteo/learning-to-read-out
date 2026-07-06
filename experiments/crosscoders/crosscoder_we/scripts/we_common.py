"""Shared helpers for the W_E appendix metric scripts in this directory.

Not an entry point. Holds the loaders, token-family machinery, and CSV/cache
writers that `build_we_appendix_plots.py`, `build_we_appendix_extended_plots.py`,
`build_we_feature_cards.py`, and `build_we_wu_top3_experiments.py` all share, so
the entry scripts never import each other.
"""

from __future__ import annotations

import csv
import json
import string
import unicodedata
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from readout.core.model_specs import DEFAULT_STEPS_32 as STEPS_32
from readout.core.paths import release_path

MODEL_NAME = "EleutherAI/pythia-160m"
MODEL_SHORT = "pythia-160m"
TOKENIZER_VOCAB = 50254


def _load_we_rates_and_norms(cache_root: Path) -> tuple[np.ndarray, np.ndarray]:
    we_dir = cache_root / "derived" / "rates" / "we-d8192-multiseed"
    rates = []
    norms = []
    for seed in range(5):
        rate_path = we_dir / f"we_rates_dsae8192_seed{seed}.pt"
        norm_path = we_dir / f"we_cc_dsae8192_seed{seed}_norms.npy"
        payload = torch.load(rate_path, map_location="cpu", weights_only=False)
        if payload.get("steps") != STEPS_32:
            raise ValueError(f"unexpected W_E steps in {rate_path}")
        per_seed = payload["rates_per_seed"]
        if len(per_seed) != 1:
            raise ValueError(f"expected one rate tensor in {rate_path}, got {per_seed.keys()}")
        rates.append(next(iter(per_seed.values())).float().numpy())
        norms.append(np.load(norm_path).astype(np.float32))
    return np.stack(rates, axis=0), np.stack(norms, axis=0)


def _load_wu_rates_and_norms(cache_root: Path) -> tuple[np.ndarray, np.ndarray]:
    run3 = cache_root / "derived" / "rates" / "wu-d8192-multiseed"
    rates = np.load(run3 / "firing_rates_all_seeds.npy").astype(np.float32)
    norms = np.load(run3 / "decoder_norms_all_seeds.npy").astype(np.float32)
    if rates.shape != (5, 32, 8192):
        raise ValueError(f"unexpected W_U rates shape {rates.shape}")
    if norms.shape != (5, 32, 8192):
        raise ValueError(f"unexpected W_U norms shape {norms.shape}")
    return rates, norms


def _quality_rows() -> list[dict]:
    rows: list[dict] = []
    specs = [
        ("W_E", 8192, range(5), "W_E d8192"),
        ("W_E", 24576, range(1), "W_E d24576"),
        ("W_U", 8192, range(5), "W_U d8192"),
        ("W_U", 24576, range(3), "W_U d24576"),
    ]
    for kind, dim, seeds, label in specs:
        for seed in seeds:
            path = release_path(MODEL_SHORT, kind, dim=dim, seed=seed)
            cfg_path = path.with_suffix(".config.json")
            if not cfg_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            q = cfg["quality"]
            rows.append(
                {
                    "matrix": kind,
                    "d_sae": dim,
                    "seed": seed,
                    "label": label,
                    "explained_variance": float(q["explained_variance"]),
                    "mean_l0": float(q["mean_l0"]),
                    "pct_active": float(100.0 * q["mean_l0"] / dim),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _figure_bases(name: str, fig_dir: Path) -> list[Path]:
    return [fig_dir / name]


def _write_csv_flexible(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_and_cache(
    bases: list[Path],
    rows: list[dict],
    payload: dict | list | tuple,
) -> None:
    for base in bases:
        _write_csv_flexible(base.with_suffix(".csv"), rows)
        torch.save(payload, base.with_suffix(".pt"))


def _terminal_decoder(path: Path) -> torch.Tensor:
    return load_safetensors(str(path), device="cpu")["W_D"][-1].float()


def _top_tokens_per_feature(
    decoder: torch.Tensor,
    terminal_wu: torch.Tensor,
    *,
    top_k: int,
    chunk: int = 512,
) -> np.ndarray:
    """Return top-k token ids per feature via terminal W_U projection."""
    decoder = decoder.float()
    terminal_wu = terminal_wu.float()
    out = []
    for start in range(0, decoder.shape[0], chunk):
        end = min(decoder.shape[0], start + chunk)
        logits = terminal_wu @ decoder[start:end].T
        idx = torch.topk(logits, k=top_k, dim=0).indices.T.contiguous()
        out.append(idx.cpu())
        del logits, idx
    return torch.cat(out, dim=0).numpy().astype(np.int32)


def _decoded_token_texts() -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    texts = []
    for token_id in range(tokenizer.vocab_size):
        texts.append(tokenizer.decode([token_id], clean_up_tokenization_spaces=False))
    return texts


def _contains_range(text: str, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in ranges)


def _is_punctuationish(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return all(unicodedata.category(ch)[0] in {"P", "S"} for ch in stripped)


def _is_latin_alpha(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 2 and all(ch in string.ascii_letters for ch in stripped)


def _token_family_masks(vocab_rows: int) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Build decoded-token masks, padded to the snapshot row count."""
    texts = _decoded_token_texts()
    family_defs = {
        "latin alpha": lambda t, i: _is_latin_alpha(t),
        "space-prefixed": lambda t, i: t.startswith(" "),
        "digit-bearing": lambda t, i: any(ch.isdigit() for ch in t),
        "punctuation": lambda t, i: _is_punctuationish(t),
        "Cyrillic": lambda t, i: _contains_range(t, [(0x0400, 0x04FF)]),
        "Greek": lambda t, i: _contains_range(t, [(0x0370, 0x03FF)]),
        "CJK": lambda t, i: _contains_range(
            t,
            [
                (0x3400, 0x4DBF),
                (0x4E00, 0x9FFF),
                (0x3040, 0x30FF),
                (0xAC00, 0xD7AF),
            ],
        ),
        "Arabic": lambda t, i: _contains_range(t, [(0x0600, 0x06FF)]),
        "Devanagari": lambda t, i: _contains_range(t, [(0x0900, 0x097F)]),
    }
    masks: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for name, pred in family_defs.items():
        mask = np.zeros(vocab_rows, dtype=bool)
        for token_id, text in enumerate(texts):
            if pred(text, token_id):
                mask[token_id] = True
        masks[name] = mask
        rows.append(
            {
                "family": name,
                "n_tokenizer_tokens": int(mask[: len(texts)].sum()),
                "n_snapshot_rows": int(mask.sum()),
                "definition": "decoded tokenizer string predicate",
            }
        )
    return masks, rows


def _feature_family_fraction(top_tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = top_tokens < mask.shape[0]
    return (mask[top_tokens] & valid).mean(axis=1)


def _family_selected_features(
    top_tokens: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    threshold: float,
    min_features: int = 24,
) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {}
    for family, mask in masks.items():
        frac = _feature_family_fraction(top_tokens, mask)
        idx = np.flatnonzero(frac >= threshold)
        if idx.size < min_features:
            idx = np.argsort(frac)[-min_features:]
        selected[family] = idx.astype(np.int64)
    return selected
