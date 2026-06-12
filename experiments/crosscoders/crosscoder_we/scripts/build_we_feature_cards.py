"""Build static W_E feature-card grids for the Appendix.

These grids mirror the W_U atlas-card format used in Appendix feature
interpretation figures: each card shows a normalized decoder-norm trajectory
and early / peak / final nearest-token blocks. W_E has no atlas browser data,
so token blocks are computed directly by projecting W_E decoder directions
through the dense input-embedding snapshots.

Outputs:
  figures/crosscoder_we/we_feature_cards_pythia160m_d8192_*.{csv,pt}
  results/experiments/crosscoder_we/we_feature_cards_pythia160m_d8192_*.{csv,pt}
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from experiments.crosscoders.crosscoder_we.scripts.we_common import (  # noqa: E402
    MODEL_SHORT,
    STEPS_32,
    _load_we_rates_and_norms,
)
from src.core.paths import release_path, ssd_root  # noqa: E402
from src.crosscoder.snapshots import load_snapshot  # noqa: E402

MODEL_NAME = "EleutherAI/pythia-160m"
TOKENIZER = "EleutherAI/pythia-160m"
DIM = 8192
SEED = 0
EARLY_STEP = 16
FINAL_STEP = 143000
TOKENS_PER_BLOCK = 12

FIG_OUT = REPO / "figures" / "crosscoder_we"
RESULT_OUT = REPO / "results" / "experiments" / "crosscoder_we"


@dataclass(frozen=True)
class CardSpec:
    feature_id: int
    label: str
    note: str


REPRESENTATIVE: tuple[CardSpec, ...] = (
    CardSpec(3567, "numeric codes", "manual-readable"),
    CardSpec(4512, "quote punctuation", "manual-readable"),
    CardSpec(2070, "nested closers", "manual-readable"),
    CardSpec(4407, "Cyrillic morphemes", "manual-readable"),
    CardSpec(3795, "Greek accents", "manual-readable"),
    CardSpec(5779, "CJK common tokens", "manual-readable"),
    CardSpec(1410, "English plural nouns", "manual-readable"),
    CardSpec(5928, "title-case topics", "manual-readable"),
)


AMBIGUOUS: tuple[CardSpec, ...] = (
    CardSpec(1922, "mixed stems", "high-norm early-decay"),
    CardSpec(6511, "mixed fragments", "high-norm early-decay"),
    CardSpec(1213, "mixed scripts", "high-norm early-decay"),
    CardSpec(3189, "mixed lexical pieces", "high-norm early-decay"),
    CardSpec(6905, "short fragments", "high-norm early-decay"),
    CardSpec(6236, "mixed numerals/scripts", "high-norm early-decay"),
    CardSpec(5023, "mixed common stems", "high-norm early-decay"),
    CardSpec(218, "mixed technical pieces", "high-norm early-decay"),
)


def _escape_text(value: str) -> str:
    parts = []
    for ch in value:
        code = ord(ch)
        if ch == "\n":
            parts.append("\\n")
        elif ch == "\t":
            parts.append("\\t")
        elif ch == "$":
            parts.append("\\$")
        elif 32 <= code < 127:
            parts.append(ch)
        elif code <= 0xFFFF:
            parts.append(f"\\u{code:04x}")
        else:
            parts.append(f"\\U{code:08x}")
    return "".join(parts)


def _clean_token(token: str) -> str:
    if token.startswith(" "):
        token = "<sp>" + token[1:]
    token = _escape_text(token)
    if len(token) > 28:
        token = token[:25] + "..."
    return token


def _decode(token_ids: list[int], tokenizer) -> str:
    vocab_size = len(tokenizer)
    decoded = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id >= vocab_size:
            decoded.append(f"<row{token_id}>")
        else:
            decoded.append(_clean_token(tokenizer.decode([token_id], clean_up_tokenization_spaces=False)))
    return ", ".join(decoded)


def _first_active_step(rate_traj: np.ndarray) -> int:
    active = np.flatnonzero(rate_traj > 0)
    if active.size == 0:
        return -1
    return int(STEPS_32[int(active[0])])


def _lifecycle_from_norm(norm_traj: np.ndarray) -> str:
    peak_idx = int(np.argmax(norm_traj))
    peak_step = int(STEPS_32[peak_idx])
    peak = max(float(norm_traj[peak_idx]), 1e-12)
    terminal_ratio = float(norm_traj[-1]) / peak
    if peak_step <= 16 and terminal_ratio < 0.3:
        return "init"
    if peak_step <= 2000 and terminal_ratio < 0.3:
        return "early-decay"
    if terminal_ratio >= 0.75:
        return "persistent"
    if peak_step <= 1000:
        return "step1000-emergent"
    if peak_step <= 47000:
        return "mid-emergent"
    return "late-emergent"


def _trajectory_group(lifecycle: str) -> str:
    if lifecycle in {"init", "early-decay"}:
        return "decaying"
    if lifecycle == "persistent":
        return "persistent"
    return "emerging"


def _collect_cards(specs: tuple[CardSpec, ...], rates: np.ndarray, norms: np.ndarray) -> list[dict]:
    cards = []
    for spec in specs:
        norm_traj = norms[:, spec.feature_id].astype(np.float32)
        rate_traj = rates[:, spec.feature_id].astype(np.float32)
        norm_peak_idx = int(np.argmax(norm_traj))
        rate_peak_idx = int(np.argmax(rate_traj))
        lifecycle = _lifecycle_from_norm(norm_traj)
        cards.append(
            {
                "model": MODEL_SHORT,
                "matrix": "W_E",
                "dim": DIM,
                "seed": SEED,
                "feature_id": spec.feature_id,
                "label": spec.label,
                "note": spec.note,
                "lifecycle": lifecycle,
                "trajectory_group": _trajectory_group(lifecycle),
                "peak_step": int(STEPS_32[norm_peak_idx]),
                "rate_peak_step": int(STEPS_32[rate_peak_idx]),
                "first_active_step": _first_active_step(rate_traj),
                "norm_peak": float(norm_traj[norm_peak_idx]),
                "norm_terminal": float(norm_traj[-1]),
                "rate_peak": float(rate_traj[rate_peak_idx]),
                "rate_terminal": float(rate_traj[-1]),
                "norm_traj": norm_traj,
                "rate_traj": rate_traj,
                "early_tokens": "",
                "peak_tokens": "",
                "final_tokens": "",
                "early_token_ids": [],
                "peak_token_ids": [],
                "final_token_ids": [],
                "early_scores": [],
                "peak_scores": [],
                "final_scores": [],
            }
        )
    return cards


def _attach_exact_tokens(cards: list[dict]) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    ckpt = release_path(MODEL_SHORT, "W_E", dim=DIM, seed=SEED)
    by_step: dict[int, list[tuple[dict, str]]] = {}
    for card in cards:
        by_step.setdefault(EARLY_STEP, []).append((card, "early"))
        by_step.setdefault(int(card["peak_step"]), []).append((card, "peak"))
        by_step.setdefault(FINAL_STEP, []).append((card, "final"))

    with safe_open(str(ckpt), framework="pt", device="cpu") as f:
        w_d = f.get_slice("W_D")
        for step, jobs in sorted(by_step.items()):
            step_idx = STEPS_32.index(step)
            snapshot = load_snapshot(MODEL_NAME, step, kind="we").to(torch.float32)
            rows = torch.stack([w_d[step_idx, int(card["feature_id"])].to(torch.float32) for card, _key in jobs])
            logits = snapshot @ rows.T
            vals, ids = torch.topk(logits, k=TOKENS_PER_BLOCK, dim=0)
            for job_idx, (card, key) in enumerate(jobs):
                token_ids = ids[:, job_idx].tolist()
                scores = vals[:, job_idx].tolist()
                card[f"{key}_token_ids"] = [int(x) for x in token_ids]
                card[f"{key}_scores"] = [float(x) for x in scores]
                card[f"{key}_tokens"] = _decode(token_ids, tokenizer)
            del snapshot, rows, logits, vals, ids
            gc.collect()


def _csv_rows(cards: list[dict]) -> list[dict]:
    rows = []
    for card in cards:
        rows.append(
            {
                "model": card["model"],
                "matrix": card["matrix"],
                "dim": card["dim"],
                "seed": card["seed"],
                "feature_id": card["feature_id"],
                "label": card["label"],
                "note": card["note"],
                "lifecycle": card["lifecycle"],
                "trajectory_group": card["trajectory_group"],
                "peak_step": card["peak_step"],
                "rate_peak_step": card["rate_peak_step"],
                "first_active_step": card["first_active_step"],
                "norm_peak": card["norm_peak"],
                "norm_terminal": card["norm_terminal"],
                "rate_peak": card["rate_peak"],
                "rate_terminal": card["rate_terminal"],
                "early_token_ids": " ".join(map(str, card["early_token_ids"])),
                "peak_token_ids": " ".join(map(str, card["peak_token_ids"])),
                "final_token_ids": " ".join(map(str, card["final_token_ids"])),
                "early_tokens": card["early_tokens"],
                "peak_tokens": card["peak_tokens"],
                "final_tokens": card["final_tokens"],
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


def _write_sidecars(name: str, cards: list[dict], bases: list[Path]) -> None:
    rows = _csv_rows(cards)
    payload = {
        "model": MODEL_NAME,
        "matrix": "W_E",
        "dim": DIM,
        "seed": SEED,
        "checkpoint": str(release_path(MODEL_SHORT, "W_E", dim=DIM, seed=SEED)),
        "steps": STEPS_32,
        "early_step": EARLY_STEP,
        "final_step": FINAL_STEP,
        "tokens_per_block": TOKENS_PER_BLOCK,
        "cards": [
            {
                **{key: value for key, value in card.items() if key not in {"norm_traj", "rate_traj"}},
                "norm_traj": torch.as_tensor(card["norm_traj"]),
                "rate_traj": torch.as_tensor(card["rate_traj"]),
            }
            for card in cards
        ],
    }
    for base in bases:
        _write_csv(base.with_suffix(".csv"), rows)
        base.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, base.with_suffix(".pt"))
    print(f"wrote sidecars for {name}")


def _bases(name: str) -> list[Path]:
    return [FIG_OUT / name, RESULT_OUT / name]


def build() -> None:
    we_rates, we_norms = _load_we_rates_and_norms(ssd_root())
    rates = we_rates[SEED]
    norms = we_norms[SEED]

    jobs = [
        ("we_feature_cards_pythia160m_d8192_representative", REPRESENTATIVE),
        ("we_feature_cards_pythia160m_d8192_ambiguous", AMBIGUOUS),
    ]
    all_cards: list[dict] = []
    rendered: list[tuple[str, list[dict]]] = []
    for name, specs in jobs:
        cards = _collect_cards(specs, rates, norms)
        rendered.append((name, cards))
        all_cards.extend(cards)
    _attach_exact_tokens(all_cards)

    for name, cards in rendered:
        _write_sidecars(name, cards, _bases(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()
