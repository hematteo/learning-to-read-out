"""Availability vs expression metrics for one (family, checkpoint) cell.

Availability (A) is whether the task feature is linearly decodable from the
checkpoint's final-position hidden state; expression (E) is whether the
checkpoint's own readout already separates the two answer tokens:

  - ``probe_acc_1d`` (A1, headline): standardized shrinkage-LDA, a single
    direction, so it is rank-matched to the rank-1 contrast readout E.
  - ``probe_acc`` (A_full): standardized L2 logistic regression, the full-rank ceiling.
  - nulls on the same folds: label-shuffled (both probes), uniform random labels.
  - ``native_readout_acc`` (E): accuracy of ``LN_t(z_t) . (W_U^t[y+] - W_U^t[y-]) > 0``.
  - ``best_swept_readout_acc`` (B): best other checkpoint's readout ``(LN_s, W_U^s)``
    on the same ``z_t`` (terminal step excluded; in-sample argmax, optimistic), and
    ``best_swept_acc_heldout``: s picked on a random half, scored on the other half.
  - full-vocabulary rank of ``y+`` under E and B (mean rank, rank-1 rate).

The probe label is the task's registered feature (``feature_label``: head number,
IOI recipient position, numeric answer position / truth label, BLiMP number or
antecedent gender), never the answer token. Folds are grouped by a lexical key
(``derive_group_key``) so train and test never share, e.g., a noun lemma.

GroupKFold note: sklearn orders groups by size with ``np.argsort``, and the order
of equal-size groups (hence the folds) depends on numpy's sort backend: x86-64
builds use x86-simd-sort, macOS/arm64 builds an introsort. The paper's probes ran
on x86. ``group_kfold_splits`` reimplements sklearn's non-shuffled assignment on top
of ``x86_simd_argsort``, an emulation of that backend, so the published folds are
reproduced on any platform.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TERMINAL_STEP = 143000

# Plain-SVA surface noun -> number-invariant lemma (group key).
SVA_LEMMAS = {
    **{n: n for n in ["key", "book", "door", "car", "student", "teacher", "paper", "table", "window", "box"]},
    **{n: n for n in ["ship", "road", "leaf", "letter", "bell", "clock", "room", "cup", "lamp", "shoe"]},
    "keys": "key",
    "books": "book",
    "doors": "door",
    "cars": "car",
    "students": "student",
    "teachers": "teacher",
    "papers": "paper",
    "tables": "table",
    "windows": "window",
    "boxes": "box",
    "ships": "ship",
    "roads": "road",
    "leaves": "leaf",
    "letters": "letter",
    "bells": "bell",
    "clocks": "clock",
    "rooms": "room",
    "cups": "cup",
    "lamps": "lamp",
    "shoes": "shoe",
}
_ANAPHOR_GENDER = {"herself": "fem", "himself": "masc"}


# ---------------------------------------------------------------------------
# Labels and group keys
# ---------------------------------------------------------------------------
def _sva_group_key(ex: dict) -> str | None:
    meta = ex.get("meta", {}) or {}
    if meta.get("head_lemma"):
        return str(meta["head_lemma"])
    noun = meta.get("noun")
    if noun is None:
        parts = str(ex.get("prompt", "")).split()
        if len(parts) >= 2 and parts[0] == "The":
            noun = parts[1]
    if noun is None:
        return None
    return SVA_LEMMAS.get(str(noun), str(noun))


def derive_group_key(ex: dict) -> str | None:
    """Held-out group key by family prefix; None if unavailable.

    sva* -> head lemma; numeric* -> unordered number pair; ioi* -> unordered name
    pair; fv_* -> query input; blimp det-noun -> singular noun, anaphor -> antecedent
    name, other blimp -> paradigm.
    """
    family = str(ex.get("family", ""))
    meta = ex.get("meta", {}) or {}
    if family.startswith("sva"):
        return _sva_group_key(ex)
    if family.startswith("numeric"):
        big, small = meta.get("big"), meta.get("small")
        if big is not None and small is not None:
            return "num::" + "::".join(sorted((str(big), str(small))))
        return None
    if family.startswith("ioi"):
        if meta.get("unordered_pair") is not None:
            return str(meta["unordered_pair"])
        subj, recip = meta.get("subj"), meta.get("recipient")
        if subj is not None and recip is not None:
            return "::".join(sorted((str(subj), str(recip))))
        return None
    if family.startswith("fv_"):
        return str(meta["input"]) if meta.get("input") is not None else None
    if family.startswith("blimp"):
        if "det_noun" in family or "determiner_noun" in family:
            good = str(meta.get("good_token", "")).strip()
            bad = str(meta.get("bad_token", "")).strip()
            sg = good if not good.endswith("s") else bad
            return f"detnoun::{sg.lower()}" if sg else None
        if "anaphor" in family:
            sent = str(meta.get("sentence_good", "")) or str(ex.get("prompt", ""))
            for w in sent.split():
                w2 = w.strip(".,'\"")
                if w2[:1].isupper() and w2.lower() not in ("the", "a", "an"):
                    return f"anaphor::{w2.lower()}"
            return f"anaphor::prefix::{sent[:40]}"
        return str(meta["paradigm"]) if meta.get("paradigm") is not None else None
    return None


def derive_groups(examples: list[dict]) -> np.ndarray | None:
    """Group-key vector, or None if any example lacks a key (caller falls back to stratified folds)."""
    keys = [derive_group_key(ex) for ex in examples]
    if any(k is None for k in keys):
        return None
    return np.array(keys, dtype=object)


def _blimp_feature(family: str, meta: dict) -> str | None:
    good = str(meta.get("good_token", "")).strip()
    bad = str(meta.get("bad_token", "")).strip()
    if "det_noun" in family or "determiner_noun" in family:
        if good.endswith("s") and not bad.endswith("s"):
            return "plural"
        if bad.endswith("s") and not good.endswith("s"):
            return "singular"
        return "plural" if len(good) > len(bad) else "singular"  # irregular plural is longer
    if "anaphor" in family:
        return _ANAPHOR_GENDER.get(good)  # themselves / itself -> unlabeled
    return None  # npi: good token is always "not", no contrasting class


def feature_label(ex: dict) -> str | None:
    """Probe class for an example; None for open-vocabulary families (no probe)."""
    family = str(ex.get("family", ""))
    meta = ex.get("meta", {}) or {}
    if family.startswith("sva"):
        v = meta.get("head_number") or meta.get("number")
    elif family.startswith("ioi"):
        v = meta.get("recipient_position")
    elif family.startswith("numeric_tf"):
        v = meta.get("label")
    elif family.startswith("numeric_gt") or family.startswith("numeric_lt"):
        v = meta.get("answer_position")
    elif family.startswith("blimp"):
        return _blimp_feature(family, meta)
    else:
        return None  # fv_*, induction, relational_*, copy_control, fixed_token_control
    return str(v) if v else None


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def probe_pipeline_full() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=10000, solver="lbfgs"))


def probe_pipeline_1d() -> object:
    # Binary LDA projects onto one direction; Ledoit-Wolf shrinkage handles n < d.
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def fit_eval_folds(X: np.ndarray, y: np.ndarray, splits: Iterable, pipeline_fn=probe_pipeline_full) -> list[float]:
    fold_acc = []
    for tr, te in splits:
        clf = pipeline_fn()
        clf.fit(X[tr], y[tr])
        fold_acc.append(float((clf.predict(X[te]) == y[te]).mean()))
    return fold_acc


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    samples = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


# ---------------------------------------------------------------------------
# x86-simd-sort argsort (numpy 2.x on x86-64, int64 keys), for GroupKFold ties
# ---------------------------------------------------------------------------
_I64_MAX = np.iinfo(np.int64).max
_LANES = 8  # AVX-512 register of int64; the AVX2 path (4 lanes) gives the same order on the panel's inputs


def _lane_swap(k: int) -> np.ndarray:
    return np.arange(_LANES) ^ (k // 2)


def _lane_reverse(k: int) -> np.ndarray:
    i = np.arange(_LANES)
    return (i // k) * k + (k - 1 - i % k)


# (partner lane, bitmask of lanes that keep the max) per compare-exchange stage.
_SORT_REG = [
    (_lane_reverse(2), 0xAA),
    (_lane_reverse(4), 0xCC),
    (_lane_swap(2), 0xAA),
    (_lane_reverse(8), 0xF0),
    (_lane_swap(4), 0xCC),
    (_lane_swap(2), 0xAA),
]
_MERGE_REG = [(_lane_swap(8), 0xF0), (_lane_swap(4), 0xCC), (_lane_swap(2), 0xAA)]


def _reg_network(key: np.ndarray, idx: np.ndarray, stages) -> tuple[np.ndarray, np.ndarray]:
    for partner, mask in stages:
        keep_max = ((mask >> np.arange(_LANES)) & 1).astype(bool)
        k2, i2 = key[partner], idx[partner]
        new = np.where(keep_max, np.maximum(k2, key), np.minimum(k2, key))
        idx = np.where(new == key, idx, i2)  # equal keys never trade indices
        key = new
    return key, idx


def _coex(kr: list, ir: list, a: int, b: int) -> None:
    lo, hi = np.minimum(kr[a], kr[b]), np.maximum(kr[a], kr[b])
    keep = lo == kr[a]
    ir[a], ir[b] = np.where(keep, ir[a], ir[b]), np.where(keep, ir[b], ir[a])
    kr[a], kr[b] = lo, hi


def _merge_vecs(kr: list, ir: list, lo: int, n: int) -> None:
    for i in range(n // 2):  # reverse the upper half, compare-exchange mirrored registers
        j = lo + n - i - 1
        kr[j], ir[j] = kr[j][::-1], ir[j][::-1]
        _coex(kr, ir, lo + i, j)
        kr[j], ir[j] = kr[j][::-1], ir[j][::-1]
    num = n // 2
    while num >= 2:
        for j in range(0, n, num):
            for i in range(num // 2):
                _coex(kr, ir, lo + i + j, lo + i + j + num // 2)
        num //= 2
    for i in range(lo, lo + n):
        kr[i], ir[i] = _reg_network(kr[i], ir[i], _MERGE_REG)


def x86_simd_argsort(keys: np.ndarray) -> np.ndarray:
    """``np.argsort(keys)`` as numpy 2.x computes it on x86-64 for int64 keys.

    Emulates x86-simd-sort (the version vendored in numpy 2.4): the identity for
    already-sorted input, otherwise, up to 256 keys, its bitonic key-index network.
    The tie order differs from a stable sort. Longer unsorted inputs (the quicksort
    partition path) are not emulated and fall back to a stable sort.
    """
    keys = np.asarray(keys, dtype=np.int64)
    n = len(keys)
    if n <= 1 or bool(np.all(keys[:-1] <= keys[1:])):
        return np.arange(n)
    if n > 256:
        return np.argsort(keys, kind="stable")
    num_vecs = 256 // _LANES
    while num_vecs > 1 and n * 2 <= num_vecs * _LANES:
        num_vecs //= 2
    kr, ir = [], []
    for v in range(num_vecs):
        idx = np.full(_LANES, _I64_MAX, dtype=np.int64)  # padding lanes hold max keys
        take = min(max(0, n - v * _LANES), _LANES)
        idx[:take] = np.arange(v * _LANES, v * _LANES + take)
        key = np.full(_LANES, _I64_MAX, dtype=np.int64)
        key[:take] = keys[idx[:take]]
        k, i = _reg_network(key, idx, _SORT_REG)
        kr.append(k)
        ir.append(i)
    per = 2
    while per <= num_vecs:
        for i in range(num_vecs // per):
            _merge_vecs(kr, ir, i * per, per)
        per *= 2
    return np.concatenate(ir)[:n]


def group_kfold_splits(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """sklearn ``GroupKFold(n_splits)`` (non-shuffled) with x86 numpy's group order.

    Groups are placed largest first into the currently lightest fold; the order of
    equal-size groups is ``x86_simd_argsort``'s, reversed.
    """
    unique_groups, group_idx = np.unique(groups, return_inverse=True)
    if n_splits > len(unique_groups):
        raise ValueError(f"n_splits={n_splits} > number of groups {len(unique_groups)}")
    sizes = np.bincount(group_idx)
    order = x86_simd_argsort(sizes)[::-1]
    fold_weight = np.zeros(n_splits)
    group_to_fold = np.zeros(len(unique_groups))
    for g in order:
        f = np.argmin(fold_weight)
        fold_weight[f] += sizes[g]
        group_to_fold[g] = f
    fold_of = group_to_fold[group_idx]
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(n_splits)]


def build_splits(y: np.ndarray, *, n_splits: int, seed: int, groups: np.ndarray | None):
    """``(splits, splitter_name, skip_reason)``; empty splits with a reason when infeasible."""
    if len(np.unique(y)) < 2:
        return [], "", "fewer than 2 distinct y_plus classes"
    if groups is not None:
        n_group = len(set(groups.tolist()))
        if n_group < 2:
            return [], "GroupKFold", f"only {n_group} group(s) (GroupKFold infeasible)"
        k = min(n_splits, n_group)
        return group_kfold_splits(groups, k), f"GroupKFold(k={k})", ""
    min_class = int(np.unique(y, return_counts=True)[1].min())
    if min_class < 2:
        return [], "StratifiedKFold", f"smallest class has {min_class} member(s) (StratifiedKFold infeasible)"
    k = min(n_splits, min_class)
    splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return splits, f"StratifiedKFold(k={k})", ""


_PROBE_KEYS = (
    "probe_acc",
    "probe_acc_sd",
    "probe_acc_ci_lo",
    "probe_acc_ci_hi",
    "probe_acc_1d",
    "probe_acc_1d_sd",
    "probe_acc_1d_ci_lo",
    "probe_acc_1d_ci_hi",
    "probe_acc_1d_label_shuffle",
    "probe_acc_label_shuffle",
    "probe_acc_random_label",
    "probe_acc_randinit",
)
_FOLD_KEYS = (
    "fold_acc",
    "fold_acc_1d",
    "fold_acc_1d_label_shuffle",
    "fold_acc_label_shuffle",
    "fold_acc_random_label",
    "fold_acc_randinit",
)


def empty_probe(n: int, skip_reason: str) -> dict:
    """Probe record for a family with no probeable feature (same keys as a fitted cell)."""
    out = {"n": n, "n_classes": 0, **{k: float("nan") for k in _PROBE_KEYS}}
    out.update({"probe_splitter": "", "skip_reason": skip_reason, **{k: None for k in _FOLD_KEYS}})
    return out


def availability_probe(
    h: np.ndarray,  # (N, d)
    y: np.ndarray,  # (N,) integer feature classes
    *,
    n_splits: int = 5,
    seed: int = 0,
    groups: np.ndarray | None = None,
    h_randinit: np.ndarray | None = None,  # (N, d) untrained-model hiddens (optional floor)
) -> dict:
    """A1 / A_full probe accuracies (fold means), nulls on the same folds, bootstrap CIs."""
    n = len(y)
    classes = np.unique(y)
    splits, splitter, skip_reason = build_splits(y, n_splits=n_splits, seed=seed, groups=groups)
    if not splits:
        nan = float("nan")
        return {
            "n": n,
            "n_classes": int(len(classes)),
            "probe_acc": nan,
            "probe_acc_1d": nan,
            "probe_acc_1d_label_shuffle": nan,
            "probe_acc_label_shuffle": nan,
            "probe_acc_random_label": nan,
            "probe_acc_randinit": nan,
            "probe_splitter": splitter,
            "skip_reason": skip_reason,
        }

    real = fit_eval_folds(h, y, splits)
    real_1d = fit_eval_folds(h, y, splits, probe_pipeline_1d)
    rng = np.random.default_rng(seed)
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    shuf = fit_eval_folds(h, y_shuffled, splits)
    shuf_1d = fit_eval_folds(h, y_shuffled, splits, probe_pipeline_1d)
    y_random = classes[rng.integers(0, len(classes), size=n)]
    rand = fit_eval_folds(h, y_random, splits)
    randinit = None
    if h_randinit is not None and h_randinit.shape[0] == n:
        randinit = fit_eval_folds(h_randinit, y, splits)

    lo, hi = bootstrap_ci(real, seed=seed)
    lo_1d, hi_1d = bootstrap_ci(real_1d, seed=seed)
    return {
        "n": n,
        "n_classes": int(len(classes)),
        "probe_acc": float(np.mean(real)),
        "probe_acc_sd": float(np.std(real)),
        "probe_acc_ci_lo": lo,
        "probe_acc_ci_hi": hi,
        "probe_acc_1d": float(np.mean(real_1d)),
        "probe_acc_1d_sd": float(np.std(real_1d)),
        "probe_acc_1d_ci_lo": lo_1d,
        "probe_acc_1d_ci_hi": hi_1d,
        "probe_acc_1d_label_shuffle": float(np.mean(shuf_1d)),
        "probe_acc_label_shuffle": float(np.mean(shuf)),
        "probe_acc_random_label": float(np.mean(rand)),
        "probe_acc_randinit": float(np.mean(randinit)) if randinit is not None else float("nan"),
        "probe_splitter": splitter,
        "skip_reason": "",
        "fold_acc": real,
        "fold_acc_1d": real_1d,
        "fold_acc_1d_label_shuffle": shuf_1d,
        "fold_acc_label_shuffle": shuf,
        "fold_acc_random_label": rand,
        "fold_acc_randinit": randinit,
    }


# ---------------------------------------------------------------------------
# Readouts (gauge-clean: logits_s = LN_s(z) @ W_U^s.T)
# ---------------------------------------------------------------------------
def layernorm(z: torch.Tensor, ln: dict) -> torch.Tensor:
    bias = ln.get("bias")
    return torch.nn.functional.layer_norm(
        z.float(),
        (z.shape[-1],),
        weight=ln["weight"].float(),
        bias=None if bias is None else bias.float(),
        eps=ln["eps"],
    )


def contrast_correct(z: np.ndarray, W: torch.Tensor, ln: dict | None, yp_ids, ym_ids) -> np.ndarray:
    """(N,) bool: does ``LN_s(z) . W_U^s`` score y+ above y-? Computed on CPU."""
    zt = torch.from_numpy(np.asarray(z)).float()  # (N, d)
    h = layernorm(zt, ln) if ln is not None else zt
    Wp = W[torch.from_numpy(np.asarray(yp_ids)).long()].float()  # (N, d)
    Wm = W[torch.from_numpy(np.asarray(ym_ids)).long()].float()
    margin = (h * Wp).sum(-1) - (h * Wm).sum(-1)
    return (margin > 0).cpu().numpy()


def full_vocab_rank_rates(
    z: np.ndarray, W: torch.Tensor, ln: dict | None, correct_ids, device: str = "cpu"
) -> tuple[float, float]:
    """Mean 1-based full-vocab rank of the gold token (strictly-higher count + 1) and the rank-1 rate."""
    dev = torch.device(device)
    zt = torch.from_numpy(np.asarray(z)).float().to(dev)  # (N, d)
    hN = layernorm(zt, {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ln.items()}) if ln else zt
    logits = hN @ W.to(dev).t()  # (N, V)
    gold = logits[torch.arange(logits.shape[0], device=dev), torch.as_tensor(np.asarray(correct_ids)).long().to(dev)]
    ranks = (logits > gold.unsqueeze(1)).sum(dim=1).cpu().numpy() + 1
    return float(ranks.mean()), float((ranks == 1).mean())


def native_and_best_swept(
    z: np.ndarray | None,  # (N, d) pre-LN residual at the h_step checkpoint
    correct_ids: np.ndarray,  # (N,) y_plus ids
    y_minus: np.ndarray,  # (N,) y_minus ids
    h_step: int,
    snapshots,  # object with .steps (sorted list[int]), .wu(step), .lnf(step)
    *,
    seed: int = 0,
    device: str = "cpu",
    exclude_s: tuple[int, ...] = (TERMINAL_STEP,),
) -> dict:
    """Native (E) and best-swept (B) contrast accuracies plus full-vocab rank diagnostics."""
    out = {
        "native_readout_acc": float("nan"),
        "best_swept_readout_acc": float("nan"),
        "best_swept_s_step": -1,
        "best_swept_in_sample_biased": True,
        "delta_best_minus_native": float("nan"),
        "best_swept_acc_heldout": float("nan"),
        "best_swept_s_step_heldout": -1,
        "native_full_vocab_mean_rank": float("nan"),
        "native_full_vocab_rank1_rate": float("nan"),
        "best_swept_full_vocab_mean_rank": float("nan"),
        "best_swept_full_vocab_rank1_rate": float("nan"),
        "full_vocab_rank_skip_reason": "",
    }
    if z is None or snapshots is None:
        # Never substitute post-LN h: LN_s(h) = LN(LN(z)) is a gauge no checkpoint uses.
        out["full_vocab_rank_skip_reason"] = "contrast/rank inputs not provided"
        return out
    if not snapshots.steps:
        out["full_vocab_rank_skip_reason"] = "no W_U+LN snapshots"
        return out

    W_t, ln_t = snapshots.wu(h_step), snapshots.lnf(h_step)
    if W_t is not None and ln_t is not None:
        out["native_readout_acc"] = float(contrast_correct(z, W_t, ln_t, correct_ids, y_minus).mean())
        mr, r1 = full_vocab_rank_rates(z, W_t, ln_t, correct_ids, device)
        out["native_full_vocab_mean_rank"], out["native_full_vocab_rank1_rate"] = mr, r1
    else:
        out["full_vocab_rank_skip_reason"] = f"missing W_U/LN snapshot for native step {h_step}"

    correct_by_s: dict[int, np.ndarray] = {}
    for s in snapshots.steps:
        if s in exclude_s:
            continue
        W_s, ln_s = snapshots.wu(s), snapshots.lnf(s)
        if W_s is not None and ln_s is not None:
            correct_by_s[s] = contrast_correct(z, W_s, ln_s, correct_ids, y_minus)

    best_acc, best_s = float("nan"), -1
    for s, c in correct_by_s.items():
        a = float(c.mean())
        if np.isnan(best_acc) or a > best_acc:
            best_acc, best_s = a, s
    out["best_swept_readout_acc"], out["best_swept_s_step"] = best_acc, best_s
    if not np.isnan(out["native_readout_acc"]) and not np.isnan(best_acc):
        out["delta_best_minus_native"] = best_acc - out["native_readout_acc"]

    if best_s >= 0:
        if best_s == h_step and not np.isnan(out["native_full_vocab_mean_rank"]):
            out["best_swept_full_vocab_mean_rank"] = out["native_full_vocab_mean_rank"]
            out["best_swept_full_vocab_rank1_rate"] = out["native_full_vocab_rank1_rate"]
        else:
            mr, r1 = full_vocab_rank_rates(z, snapshots.wu(best_s), snapshots.lnf(best_s), correct_ids, device)
            out["best_swept_full_vocab_mean_rank"], out["best_swept_full_vocab_rank1_rate"] = mr, r1

    if correct_by_s:
        n = len(next(iter(correct_by_s.values())))
        perm = np.random.default_rng(seed).permutation(n)
        idx_a, idx_b = perm[: n // 2], perm[n // 2 :]
        if len(idx_a) > 0 and len(idx_b) > 0:
            best_sel, s_star = -1.0, -1
            for s, c in correct_by_s.items():
                a = float(c[idx_a].mean())
                if a > best_sel:
                    best_sel, s_star = a, s
            out["best_swept_acc_heldout"] = float(correct_by_s[s_star][idx_b].mean())
            out["best_swept_s_step_heldout"] = int(s_star)
    return out
