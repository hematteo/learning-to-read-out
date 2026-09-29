"""Linear availability probe with label-shuffle and random-label nulls.

``availability_probe(h, y)`` asks whether an integer target ``y`` is linearly
present in hidden states ``h`` ``(N, d)``: a standardized logistic-regression
probe scored by stratified k-fold accuracy, next to two nulls (labels shuffled
across examples; labels drawn uniformly at random) and a bootstrap CI over the
folds. The recipe-control expression lag analysis compares this "availability"
against what the checkpoint's own readout expresses.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def probe_pipeline() -> object:
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(C=1.0, max_iter=10000, solver="lbfgs"),
    )


def fit_eval_folds(
    X: np.ndarray,
    y: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[float], np.ndarray]:
    """Per-fold held-out accuracy and the out-of-fold predictions."""
    fold_acc = []
    preds = np.zeros_like(y)
    for tr, te in splits:
        clf = probe_pipeline()
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        preds[te] = p
        fold_acc.append(float((p == y[te]).mean()))
    return fold_acc, preds


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """2.5 / 97.5 percentile bootstrap of the mean of ``values`` (NaN if < 2 values)."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    samples = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def availability_probe(
    h: np.ndarray,  # (N, d)
    y: np.ndarray,  # (N,) integer class labels
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Probe accuracy plus label-shuffle and random-label nulls.

    Cells whose smallest class has fewer than ``n_splits`` members (StratifiedKFold
    infeasible) or fewer than two classes return NaN metrics and a ``skip_reason``.
    """
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return {
            "n": n,
            "n_classes": int(len(classes)),
            "probe_acc": float("nan"),
            "probe_acc_label_shuffle": float("nan"),
            "probe_acc_random_label": float("nan"),
            "skip_reason": "fewer than 2 distinct classes",
        }
    if int(counts.min()) < n_splits:
        return {
            "n": n,
            "n_classes": int(len(classes)),
            "probe_acc": float("nan"),
            "probe_acc_label_shuffle": float("nan"),
            "probe_acc_random_label": float("nan"),
            "skip_reason": (
                f"smallest class has {int(counts.min())} members < n_splits={n_splits} (StratifiedKFold infeasible)"
            ),
        }
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(n), y))

    real_folds, _ = fit_eval_folds(h, y, splits)

    rng = np.random.default_rng(seed)
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    shuf_folds, _ = fit_eval_folds(h, y_shuffled, splits)

    # Uniform random labels, mapped back into the original label space so the
    # stratified splits stay valid.
    y_random = classes[rng.integers(0, len(classes), size=n)]
    rand_folds, _ = fit_eval_folds(h, y_random, splits)

    lo, hi = bootstrap_ci(real_folds, seed=seed)
    return {
        "n": n,
        "n_classes": int(len(classes)),
        "probe_acc": float(np.mean(real_folds)),
        "probe_acc_sd": float(np.std(real_folds)),
        "probe_acc_ci_lo": lo,
        "probe_acc_ci_hi": hi,
        "probe_acc_label_shuffle": float(np.mean(shuf_folds)),
        "probe_acc_random_label": float(np.mean(rand_folds)),
        "skip_reason": "",
        "fold_acc": real_folds,
        "fold_acc_label_shuffle": shuf_folds,
        "fold_acc_random_label": rand_folds,
    }
