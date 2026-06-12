"""Run leakage-controlled hidden-state probes for readout-swap tasks.

The original probe trajectory deliberately used direct labels for each task:
SVA number, IOI recipient identity, and relational EU/non-EU. Those probes are
useful as a loose upper bound, but they are easy to solve from prompt surface
features. This script keeps the cached-hidden-state workflow and adds controls:

  - SVA uses held-out noun lemmas, so train/test cannot share key/keys, etc.
  - IOI uses held-out unordered name pairs.
  - Relational facts reports balanced accuracy because the dataset is tiny and
    class-imbalanced.
  - Prompt-token binary/count probes and Gaussian random features are evaluated
    with the exact same splits.

Outputs a long CSV and a torch sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from src.core.model_specs import DEFAULT_STEPS_32 as PYTHIA_STEPS_32  # noqa: E402

DEFAULT_HIDDEN_DIR = REPO / "results/experiments/probes/contrastive_readout_swap/hidden_p1b"
DEFAULT_DATASETS_DIR = REPO / "results/experiments/probes/contrastive_readout_swap/datasets_p1b"
DEFAULT_OUT_DIR = REPO / "results/experiments/probes/contrastive_readout_swap/controlled_hidden_probes"
DEFAULT_STEM = "main_claim_controlled_hidden_probe_trajectory"

SVA_LEMMAS = {
    "keys": "key",
    "key": "key",
    "books": "book",
    "book": "book",
    "doors": "door",
    "door": "door",
    "cars": "car",
    "car": "car",
    "students": "student",
    "student": "student",
    "teachers": "teacher",
    "teacher": "teacher",
    "papers": "paper",
    "paper": "paper",
    "tables": "table",
    "table": "table",
    "windows": "window",
    "window": "window",
    "boxes": "box",
    "box": "box",
    "ships": "ship",
    "ship": "ship",
    "roads": "road",
    "road": "road",
    "leaves": "leaf",
    "leaf": "leaf",
    "letters": "letter",
    "letter": "letter",
    "bells": "bell",
    "bell": "bell",
    "clocks": "clock",
    "clock": "clock",
    "rooms": "room",
    "room": "room",
    "cups": "cup",
    "cup": "cup",
    "lamps": "lamp",
    "lamp": "lamp",
    "shoes": "shoe",
    "shoe": "shoe",
}

EU_COUNTRIES = {
    "France",
    "Germany",
    "Italy",
    "Spain",
    "Greece",
    "Poland",
    "Portugal",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Austria",
    "Hungary",
    "Belgium",
    "Ireland",
}

SOURCE_LABELS = {
    "hidden": "hidden state",
    "prompt_binary": "prompt tokens (binary)",
    "prompt_count": "prompt tokens (count)",
    "random_gaussian": "random Gaussian",
}


@dataclass(frozen=True)
class ProbeSpec:
    family: str
    probe_name: str
    target: str
    split_name: str
    y: np.ndarray
    groups: np.ndarray | None
    multiclass: bool
    notes: str


def load_examples(datasets_dir: Path, family: str) -> list[dict]:
    path = datasets_dir / f"{family}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f]


def noun_from_sva_prompt(prompt: str) -> str:
    parts = prompt.split()
    if len(parts) < 2 or parts[0] != "The":
        raise ValueError(f"unexpected SVA prompt form: {prompt!r}")
    return parts[1]


def build_probe_spec(family: str, examples: list[dict]) -> ProbeSpec:
    if family == "sva":
        labels = np.array([1 if ex["meta"]["pos"] == "are" else 0 for ex in examples])
        nouns = [noun_from_sva_prompt(ex["prompt"]) for ex in examples]
        missing = sorted({noun for noun in nouns if noun not in SVA_LEMMAS})
        if missing:
            raise ValueError(f"missing SVA lemma mapping for {missing}")
        groups = np.array([SVA_LEMMAS[noun] for noun in nouns])
        return ProbeSpec(
            family=family,
            probe_name="sva_number_lemma_grouped",
            target="plural vs singular",
            split_name="GroupKFold(noun lemma)",
            y=labels,
            groups=groups,
            multiclass=False,
            notes="Holds out each singular/plural noun lemma pair together.",
        )

    if family == "ioi_role_balanced":
        labels = np.array([1 if ex["meta"]["recipient_position"] == "second" else 0 for ex in examples])
        groups = np.array([ex["meta"]["unordered_pair"] for ex in examples])
        return ProbeSpec(
            family=family,
            probe_name="ioi_role_position_pair_grouped",
            target="recipient mention position (first vs second)",
            split_name="GroupKFold(unordered name pair)",
            y=labels,
            groups=groups,
            multiclass=False,
            notes=(
                "Balanced IOI target: predicts whether the indirect object is the "
                "first or second listed name, with name pairs held out."
            ),
        )

    if family == "ioi":
        recipients = sorted({ex["meta"]["recipient"] for ex in examples})
        recipient_to_id = {name: i for i, name in enumerate(recipients)}
        labels = np.array([recipient_to_id[ex["meta"]["recipient"]] for ex in examples])
        groups = np.array(["::".join(sorted((ex["meta"]["subj"], ex["meta"]["recipient"]))) for ex in examples])
        return ProbeSpec(
            family=family,
            probe_name="ioi_recipient_pair_grouped",
            target=f"recipient identity ({len(recipients)}-way)",
            split_name="GroupKFold(unordered name pair)",
            y=labels,
            groups=groups,
            multiclass=True,
            notes=(
                "Holds out both directions of each unordered name pair. This is still "
                "not a pure IOI capability test because recipient names are visible in "
                "the prompt; prompt-count controls estimate that leakage."
            ),
        )

    if family == "relational_facts":
        labels = np.array([1 if ex["meta"]["country"] in EU_COUNTRIES else 0 for ex in examples])
        return ProbeSpec(
            family=family,
            probe_name="relational_eu_stratified",
            target="EU vs non-EU country",
            split_name="StratifiedKFold",
            y=labels,
            groups=None,
            multiclass=False,
            notes=(
                "Tiny exploratory task. Balanced accuracy is the safer headline metric "
                "because the class split is imbalanced."
            ),
        )

    raise ValueError(f"unsupported family {family!r}")


def prompt_token_matrix(examples: list[dict], *, mode: str) -> tuple[np.ndarray, list[int]]:
    vocab = sorted({int(tok) for ex in examples for tok in ex["prompt_ids"]})
    tok_to_col = {tok: i for i, tok in enumerate(vocab)}
    X = np.zeros((len(examples), len(vocab)), dtype=np.float32)
    for row, ex in enumerate(examples):
        for tok in ex["prompt_ids"]:
            col = tok_to_col[int(tok)]
            if mode == "binary":
                X[row, col] = 1.0
            elif mode == "count":
                X[row, col] += 1.0
            else:
                raise ValueError(f"unknown prompt-token mode {mode!r}")
    return X, vocab


def split_indices(spec: ProbeSpec, *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if spec.groups is not None:
        n_group = len(set(spec.groups.tolist()))
        n_splits = min(n_splits, n_group)
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(len(spec.y)), spec.y, groups=spec.groups))

    class_counts = np.bincount(spec.y)
    min_class_count = int(class_counts[class_counts > 0].min())
    n_splits = min(n_splits, min_class_count)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(spec.y)), spec.y))


def probe_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=10000, solver="lbfgs"),
    )


def evaluate_splits(
    X: np.ndarray,
    y: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    fold_acc: list[float] = []
    fold_bal: list[float] = []
    for train_idx, test_idx in splits:
        clf = probe_model()
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        fold_acc.append(float(accuracy_score(y[test_idx], pred)))
        fold_bal.append(float(balanced_accuracy_score(y[test_idx], pred)))

    return {
        "acc": float(np.mean(fold_acc)),
        "acc_sd": float(np.std(fold_acc)),
        "balanced_acc": float(np.mean(fold_bal)),
        "balanced_acc_sd": float(np.std(fold_bal)),
        "fold_acc": fold_acc,
        "fold_balanced_acc": fold_bal,
    }


def class_summary(y: np.ndarray) -> dict[str, float | int]:
    values, counts = np.unique(y, return_counts=True)
    return {
        "n_classes": int(len(values)),
        "chance_acc": float(1.0 / len(values)),
        "majority_acc": float(counts.max() / counts.sum()),
    }


def result_row(
    *,
    spec: ProbeSpec,
    h_step: int,
    source: str,
    n_features: int,
    split_count: int,
    metrics: dict[str, object],
) -> dict[str, object]:
    summary = class_summary(spec.y)
    row = {
        "family": spec.family,
        "probe_name": spec.probe_name,
        "target": spec.target,
        "h_step": int(h_step),
        "feature_source": source,
        "feature_source_label": SOURCE_LABELS[source],
        "splitter": spec.split_name,
        "n_examples": int(len(spec.y)),
        "n_features": int(n_features),
        "n_splits": int(split_count),
        "multiclass": bool(spec.multiclass),
        "notes": spec.notes,
        **summary,
    }
    for key, value in metrics.items():
        if key.startswith("fold_"):
            row[key] = json.dumps(value)
        else:
            row[key] = value
    return row


def hidden_path(hidden_dir: Path, family: str, step: int) -> Path:
    return hidden_dir / f"{family}_h{step}.pt"


def load_hidden(hidden_dir: Path, family: str, step: int) -> np.ndarray:
    path = hidden_path(hidden_dir, family, step)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False).float().numpy()


def run_spec(
    *,
    spec: ProbeSpec,
    examples: list[dict],
    h_steps: list[int],
    hidden_dir: Path,
    n_splits: int,
    seed: int,
) -> list[dict[str, object]]:
    splits = split_indices(spec, n_splits=n_splits, seed=seed)
    prompt_binary, _ = prompt_token_matrix(examples, mode="binary")
    prompt_count, _ = prompt_token_matrix(examples, mode="count")

    first_hidden = load_hidden(hidden_dir, spec.family, h_steps[0])
    rng = np.random.default_rng(seed + 17)
    random_gaussian = rng.standard_normal(first_hidden.shape).astype(np.float32)

    baseline_features = {
        "prompt_binary": prompt_binary,
        "prompt_count": prompt_count,
        "random_gaussian": random_gaussian,
    }
    baseline_metrics = {source: evaluate_splits(X, spec.y, splits) for source, X in baseline_features.items()}

    rows: list[dict[str, object]] = []
    for step in h_steps:
        X_hidden = first_hidden if step == h_steps[0] else load_hidden(hidden_dir, spec.family, step)
        hidden_metrics = evaluate_splits(X_hidden, spec.y, splits)
        rows.append(
            result_row(
                spec=spec,
                h_step=step,
                source="hidden",
                n_features=X_hidden.shape[1],
                split_count=len(splits),
                metrics=hidden_metrics,
            )
        )
        for source, X in baseline_features.items():
            rows.append(
                result_row(
                    spec=spec,
                    h_step=step,
                    source=source,
                    n_features=X.shape[1],
                    split_count=len(splits),
                    metrics=baseline_metrics[source],
                )
            )
        print(
            f"{spec.probe_name:30s} step={step:>6} "
            f"hidden_bal={hidden_metrics['balanced_acc']:.3f} "
            f"prompt_count_bal={baseline_metrics['prompt_count']['balanced_acc']:.3f}",
            flush=True,
        )

    return rows


def save_outputs(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    torch.save(
        {
            "rows": df.to_dict("records"),
            "description": "Leakage-controlled hidden probes with prompt-token and random-feature controls.",
            "source_script": "experiments/probes/contrastive_readout_swap/scripts/run_controlled_hidden_probes.py",
        },
        out_dir / f"{stem}.pt",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-dir", type=Path, default=DEFAULT_HIDDEN_DIR)
    ap.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--families", nargs="+", default=["sva", "ioi", "relational_facts"])
    ap.add_argument("--h-steps", nargs="+", type=int, default=PYTHIA_STEPS_32)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_rows: list[dict[str, object]] = []
    for family in args.families:
        examples = load_examples(args.datasets_dir, family)
        spec = build_probe_spec(family, examples)
        print(
            f"\n=== {spec.probe_name}: {spec.target}, n={len(examples)}, split={spec.split_name} ===",
            flush=True,
        )
        all_rows.extend(
            run_spec(
                spec=spec,
                examples=examples,
                h_steps=args.h_steps,
                hidden_dir=args.hidden_dir,
                n_splits=args.n_splits,
                seed=args.seed,
            )
        )

    df = pd.DataFrame(all_rows).sort_values(["family", "probe_name", "h_step", "feature_source"])
    save_outputs(df, args.out_dir, args.stem)
    print(f"\n[done] wrote {args.out_dir / (args.stem + '.csv')}", flush=True)


if __name__ == "__main__":
    main()
