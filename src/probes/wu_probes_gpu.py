"""Batched GPU logistic regression for vocabulary concept probes on W_U.

Replaces the sklearn-per-concept loop with one batched call that fits all
N concepts' probes in parallel on the GPU. Matches sklearn's
``LogisticRegression(solver='liblinear', class_weight='balanced')`` to
within a few thousandths of balanced accuracy.

Procedure mirrors `wu_probes.probe_balanced_accuracy`: a single stratified
2/3–1/3 train/test split per concept, inner n_folds stratified CV on the
train split to pick best C, refit best-C model on full train, report
held-out test balanced accuracy. The selection bias that would come from
picking C on the same folds used to report accuracy is avoided.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def _build_splits(
    y_np: np.ndarray, n_folds: int, seed: int, test_size: float = 1.0 / 3.0
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return (train_idx, test_idx, inner_cv_splits) for one concept.

    Inner splits index into the train split (not the full vocabulary).
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    ((train_idx, test_idx),) = sss.split(np.zeros_like(y_np), y_np)
    y_tr = y_np[train_idx]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    inner = [(train_idx[tr], train_idx[va]) for tr, va in skf.split(y_tr, y_tr)]
    return train_idx, test_idx, inner


def _pack_concept_folds(
    V: int,
    concepts: dict[str, set[int]],
    n_folds: int,
    seed: int,
    test_size: float = 1.0 / 3.0,
) -> tuple[
    np.ndarray,  # inner_train_mask (N, n_folds, V)
    np.ndarray,  # inner_val_mask   (N, n_folds, V)
    np.ndarray,  # inner_train_sw   (N, n_folds, V)
    np.ndarray,  # outer_train_mask (N, V)
    np.ndarray,  # outer_test_mask  (N, V)
    np.ndarray,  # outer_train_sw   (N, V)
    np.ndarray,  # y_mat            (N, V)
    np.ndarray,  # valid_mask       (N,)
]:
    """Precompute fold assignments for every concept.

    Outer: single stratified 2/3–1/3 train/test split.
    Inner: n_folds stratified CV on the train split, used only for C selection.

    Sample weights implement sklearn's ``class_weight='balanced'`` on each
    training subset; test / val subsets use raw counts (balanced accuracy is
    computed from recall ratios, not weighted sums).
    """
    n_concepts = len(concepts)
    inner_train_mask = np.zeros((n_concepts, n_folds, V), dtype=np.float32)
    inner_val_mask = np.zeros((n_concepts, n_folds, V), dtype=np.float32)
    inner_train_sw = np.zeros((n_concepts, n_folds, V), dtype=np.float32)
    outer_train_mask = np.zeros((n_concepts, V), dtype=np.float32)
    outer_test_mask = np.zeros((n_concepts, V), dtype=np.float32)
    outer_train_sw = np.zeros((n_concepts, V), dtype=np.float32)
    y_mat = np.zeros((n_concepts, V), dtype=np.float32)
    valid = np.zeros(n_concepts, dtype=bool)

    min_per_class = int(np.ceil(n_folds / (1.0 - test_size)))

    for ci, (_name, toks) in enumerate(concepts.items()):
        idx = np.fromiter(toks, dtype=np.int64)
        y_mat[ci, idx] = 1.0
        y_c = y_mat[ci]
        n_pos = int(y_c.sum())
        n_neg = V - n_pos
        if n_pos < min_per_class or n_neg < min_per_class:
            continue
        train_idx, test_idx, inner = _build_splits(
            y_c.astype(np.int64), n_folds, seed, test_size=test_size
        )
        outer_train_mask[ci, train_idx] = 1.0
        outer_test_mask[ci, test_idx] = 1.0
        # Outer balanced sample weights on the train split.
        y_tr = y_c[train_idx]
        n_tr = len(train_idx)
        n_tr_pos = int(y_tr.sum())
        n_tr_neg = n_tr - n_tr_pos
        if n_tr_pos == 0 or n_tr_neg == 0:
            continue
        w_pos = n_tr / (2.0 * n_tr_pos)
        w_neg = n_tr / (2.0 * n_tr_neg)
        outer_train_sw[ci, train_idx] = np.where(y_tr > 0, w_pos, w_neg)
        # Inner balanced sample weights per fold (on sub-train only).
        for fi, (tr, va) in enumerate(inner):
            inner_train_mask[ci, fi, tr] = 1.0
            inner_val_mask[ci, fi, va] = 1.0
            y_fold_tr = y_c[tr]
            n_f = len(tr)
            n_f_pos = int(y_fold_tr.sum())
            n_f_neg = n_f - n_f_pos
            if n_f_pos == 0 or n_f_neg == 0:
                continue
            w_p = n_f / (2.0 * n_f_pos)
            w_n = n_f / (2.0 * n_f_neg)
            inner_train_sw[ci, fi, tr] = np.where(y_fold_tr > 0, w_p, w_n)
        valid[ci] = True

    return (
        inner_train_mask,
        inner_val_mask,
        inner_train_sw,
        outer_train_mask,
        outer_test_mask,
        outer_train_sw,
        y_mat,
        valid,
    )


def _fit_lr_batched(
    X: torch.Tensor,  # (V, d)
    Y: torch.Tensor,  # (N, V) binary
    train_mask: torch.Tensor,  # (N, V) {0,1}
    train_sw: torch.Tensor,  # (N, V) sample weights on train
    C: float,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit N independent L2-regularised logistic regressions jointly via LBFGS.

    Loss (matches sklearn's liblinear L2-LR with class_weight='balanced'):
        (1/(2C)) ||w||^2 + sum_i sw_i * bce(x_i @ w + b, y_i)
    """
    V, d = X.shape
    N = Y.shape[0]

    W = torch.zeros(N, d, device=X.device, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(N, device=X.device, dtype=X.dtype, requires_grad=True)

    opt = torch.optim.LBFGS(
        [W, b],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        logits = X @ W.T + b[None, :]  # (V, N)
        Y_T = Y.T
        mask_T = train_mask.T
        sw_T = train_sw.T
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, Y_T, reduction="none"
        )
        data_loss = (bce * mask_T * sw_T).sum()
        l2 = (W * W).sum()
        loss = data_loss + (1.0 / (2.0 * C)) * l2
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def _balanced_accuracy_batched(
    logits: torch.Tensor,  # (V, N)
    Y: torch.Tensor,  # (N, V)
    test_mask: torch.Tensor,  # (N, V)
) -> torch.Tensor:
    """Balanced accuracy per concept (macro-averaged recall over 2 classes)."""
    pred = (logits > 0).T.float()
    y = Y
    test = test_mask
    tp = (pred * y * test).sum(dim=1)
    tn = ((1 - pred) * (1 - y) * test).sum(dim=1)
    pos = (y * test).sum(dim=1).clamp(min=1.0)
    neg = ((1 - y) * test).sum(dim=1).clamp(min=1.0)
    return 0.5 * (tp / pos + tn / neg)


def probe_shuffle_null_batched(
    W_U: torch.Tensor,
    concepts: dict[str, set[int]],
    n_permutations: int,
    C_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    n_folds: int = 3,
    seed: int = 0,
    device: str | None = None,
    chunk_size: int = 400,
    max_iter: int = 100,
) -> dict[str, np.ndarray]:
    """Batched shuffle-null: for each concept, draw `n_permutations` size-matched
    random token subsets from V\\concept_tokens and fit the identical probe
    pipeline on each. Returns the empirical null distribution per concept.

    Synthetic subsets are created up front, then fed into
    `probe_balanced_accuracy_batched` in chunks of `chunk_size` to keep
    fold-mask memory in bounds (O(chunk_size · V) floats per chunk). Chunks
    are shuffled so the 100 perms of a given concept span multiple chunks,
    which in turn span multiple split seeds — matches the per-perm split
    variance of the CPU `probe_shuffle_null`.
    """
    V = W_U.shape[0]
    rng = np.random.default_rng(seed)
    all_idx = np.arange(V, dtype=np.int64)

    synth: dict[str, set[int]] = {}
    meta: list[tuple[str, int, str]] = []
    for name, toks in concepts.items():
        n = len(toks)
        excl = np.zeros(V, dtype=bool)
        excl[np.fromiter(toks, dtype=np.int64)] = True
        pool = all_idx[~excl]
        if len(pool) < n:
            # Degenerate: can't draw a disjoint size-n subset.
            for m in range(n_permutations):
                meta.append((name, m, f"{name}__perm{m:04d}"))
            continue
        for m in range(n_permutations):
            key = f"{name}__perm{m:04d}"
            draw = rng.choice(pool, size=n, replace=False)
            synth[key] = {int(x) for x in draw}
            meta.append((name, m, key))

    order = np.arange(len(meta))
    rng.shuffle(order)

    null_by_concept: dict[str, np.ndarray] = {
        name: np.full(n_permutations, np.nan, dtype=np.float32) for name in concepts
    }

    for chunk_start in range(0, len(order), chunk_size):
        chunk_ids = order[chunk_start : chunk_start + chunk_size]
        chunk_concepts = {}
        for i in chunk_ids:
            _, _, key = meta[i]
            if key in synth:
                chunk_concepts[key] = synth[key]
        if not chunk_concepts:
            continue
        chunk_seed = int(rng.integers(1 << 30))
        res = probe_balanced_accuracy_batched(
            W_U,
            chunk_concepts,
            C_grid=C_grid,
            n_folds=n_folds,
            seed=chunk_seed,
            device=device,
            max_iter=max_iter,
        )
        for i in chunk_ids:
            name, m, key = meta[i]
            if key in res:
                null_by_concept[name][m] = res[key]["balanced_accuracy"]

    return null_by_concept


def probe_balanced_accuracy_batched(
    W_U: torch.Tensor,
    concepts: dict[str, set[int]],
    C_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    n_folds: int = 3,
    seed: int = 0,
    device: str | None = None,
    max_iter: int = 100,
) -> dict[str, dict]:
    """Fit concept probes for ALL concepts in one batched GPU op.

    For each concept the best C is picked by mean inner-CV balanced accuracy
    on the train split; held-out test balanced accuracy is reported at the
    selected C.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    V = W_U.shape[0]
    concept_names = list(concepts.keys())
    N = len(concept_names)

    (
        inner_train_mask_np,
        inner_val_mask_np,
        inner_train_sw_np,
        outer_train_mask_np,
        outer_test_mask_np,
        outer_train_sw_np,
        y_np,
        valid_np,
    ) = _pack_concept_folds(V, concepts, n_folds, seed)

    X = W_U.detach().to(device=device, dtype=torch.float32).contiguous()
    if not torch.isfinite(X).all():
        return {
            name: {
                "balanced_accuracy": float("nan"),
                "best_C": float("nan"),
                "n_positive": int(y_np[i].sum()),
            }
            for i, name in enumerate(concept_names)
        }
    Y = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)
    outer_train_mask = torch.from_numpy(outer_train_mask_np).to(
        device=device, dtype=torch.float32
    )
    outer_test_mask = torch.from_numpy(outer_test_mask_np).to(
        device=device, dtype=torch.float32
    )
    outer_train_sw = torch.from_numpy(outer_train_sw_np).to(
        device=device, dtype=torch.float32
    )

    # Inner-CV C selection: pick best C per concept by mean val balanced acc.
    best_inner = torch.full((N,), -1.0, device=device)
    best_C = torch.full((N,), float("nan"), device=device)

    for C in C_grid:
        fold_accs = torch.zeros(n_folds, N, device=device)
        for fi in range(n_folds):
            tm = torch.from_numpy(inner_train_mask_np[:, fi, :]).to(
                device=device, dtype=torch.float32
            )
            vm = torch.from_numpy(inner_val_mask_np[:, fi, :]).to(
                device=device, dtype=torch.float32
            )
            sw = torch.from_numpy(inner_train_sw_np[:, fi, :]).to(
                device=device, dtype=torch.float32
            )
            W, b = _fit_lr_batched(X, Y, tm, sw, C, max_iter=max_iter)
            with torch.no_grad():
                logits = X @ W.T + b[None, :]
                fold_accs[fi] = _balanced_accuracy_batched(logits, Y, vm)
        mean_acc = fold_accs.mean(dim=0)
        improved = mean_acc > best_inner
        best_inner = torch.where(improved, mean_acc, best_inner)
        best_C = torch.where(improved, torch.full_like(best_C, float(C)), best_C)

    # Held-out test evaluation. Refit is per distinct C value present in best_C.
    best_C_cpu = best_C.cpu().numpy()
    test_acc = torch.full((N,), float("nan"), device=device)
    unique_Cs = [c for c in C_grid if np.any(best_C_cpu == float(c))]
    for C in unique_Cs:
        active = torch.from_numpy((best_C_cpu == float(C)).astype(np.float32)).to(
            device=device
        )  # (N,)
        if active.sum() == 0:
            continue
        # Zero out inactive concepts' loss contribution by zeroing their masks.
        tm = outer_train_mask * active[:, None]
        sw = outer_train_sw * active[:, None]
        W, b = _fit_lr_batched(X, Y, tm, sw, float(C), max_iter=max_iter)
        with torch.no_grad():
            logits = X @ W.T + b[None, :]
            acc = _balanced_accuracy_batched(logits, Y, outer_test_mask)
        test_acc = torch.where(active.bool(), acc, test_acc)

    out: dict[str, dict] = {}
    test_acc_cpu = test_acc.cpu().numpy()
    for i, name in enumerate(concept_names):
        n_pos = int(y_np[i].sum())
        if not valid_np[i]:
            out[name] = {
                "balanced_accuracy": float("nan"),
                "best_C": float("nan"),
                "n_positive": n_pos,
            }
        else:
            out[name] = {
                "balanced_accuracy": float(test_acc_cpu[i]),
                "best_C": float(best_C_cpu[i]),
                "n_positive": n_pos,
            }
    return out
