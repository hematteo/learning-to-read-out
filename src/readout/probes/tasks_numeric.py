"""Head-to-head numeric comparison families (``numeric_gt_v2``, ``numeric_lt_v2``, ``numeric_tf_v2``).

Both numbers appear in the prompt, so the contrast is between the two candidates
rather than one of many valid continuations (the failure mode of
``contrastive_tasks.build_numeric_greater_than``'s open-ended "17 is greater than"):

  - gt: "Which is greater, A or B? Answer:"  -> larger (y_plus) vs smaller (y_minus)
  - lt: "Which is smaller, A or B? Answer:"  -> smaller vs larger
  - tf: "A is greater than B. True or False? Answer:" -> " True"/" False"

Numbers are drawn from 2..99 (single-token, leading-space). Each unordered pair
is used once and emitted in both presentation orders (gt/lt: answer first and
second; tf: true and false claim), so the datasets are position/label balanced.
The ``family`` field is ``numeric_gt`` / ``numeric_lt`` / ``numeric_tf``; the
registry key (and file name) carries the ``_v2`` suffix.
"""

from __future__ import annotations

import numpy as np

from readout.probes.contrastive_tasks import Example, _filter_pair, _row_norm, _single_token_id

NORM_TOL = 0.2


def _number_pool(tokenizer) -> list[tuple[int, int]]:
    pool = []
    for n in range(2, 100):
        tid = _single_token_id(tokenizer, str(n), leading_space=True)
        if tid is not None:
            pool.append((n, tid))
    return pool


def _norm_mismatch(W_U, a_id: int, b_id: int) -> bool:
    if W_U is None:
        return False
    r_a, r_b = _row_norm(W_U, a_id), _row_norm(W_U, b_id)
    return r_a > 0 and r_b > 0 and abs(np.log(r_a / r_b)) > NORM_TOL


def _build_head_to_head(tokenizer, *, which: str, n_max: int, rng_seed: int, W_U_for_norm_match) -> list[Example]:
    rng = np.random.default_rng(rng_seed)
    pool = _number_pool(tokenizer)
    if len(pool) < 4:
        return []
    out: list[Example] = []
    seen: set[tuple[int, int]] = set()
    max_attempts = max(8 * n_max, 4 * (len(pool) * (len(pool) - 1) // 2))
    attempts = 0
    while len(out) < n_max and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(len(pool), size=2, replace=False)
        a_n, a_id = pool[i]
        b_n, b_id = pool[j]
        if a_n == b_n:
            continue
        key = (min(a_n, b_n), max(a_n, b_n))
        if key in seen:
            continue
        seen.add(key)
        small_n, small_id = (a_n, a_id) if a_n < b_n else (b_n, b_id)
        big_n, big_id = (b_n, b_id) if a_n < b_n else (a_n, a_id)
        if _norm_mismatch(W_U_for_norm_match, small_id, big_id):
            continue
        if which == "gt":
            word, ans_n, y_plus, y_minus = "greater", big_n, big_id, small_id
        else:
            word, ans_n, y_plus, y_minus = "smaller", small_n, small_id, big_id
        for first_n, second_n in [(small_n, big_n), (big_n, small_n)]:
            prompt = f"Which is {word}, {first_n} or {second_n}? Answer:"
            out.append(
                Example(
                    family=f"numeric_{which}",
                    prompt=prompt,
                    prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
                    y_plus=y_plus,
                    y_minus=y_minus,
                    meta={
                        "big": big_n,
                        "small": small_n,
                        "first": first_n,
                        "second": second_n,
                        "answer_position": "first" if first_n == ans_n else "second",
                        "format": f"which_is_{word}",
                    },
                )
            )
            if len(out) >= n_max:
                return out
    return out


def build_numeric_greater_than_v2(
    tokenizer, *, n_max: int = 1000, rng_seed: int = 0, W_U_for_norm_match=None
) -> list[Example]:
    return _build_head_to_head(
        tokenizer, which="gt", n_max=n_max, rng_seed=rng_seed, W_U_for_norm_match=W_U_for_norm_match
    )


def build_numeric_less_than_v2(
    tokenizer, *, n_max: int = 1000, rng_seed: int = 0, W_U_for_norm_match=None
) -> list[Example]:
    return _build_head_to_head(
        tokenizer, which="lt", n_max=n_max, rng_seed=rng_seed, W_U_for_norm_match=W_U_for_norm_match
    )


def build_numeric_truefalse_v2(
    tokenizer, *, n_max: int = 1000, rng_seed: int = 0, W_U_for_norm_match=None
) -> list[Example]:
    pair = _filter_pair(tokenizer, "True", "False", W_U_for_norm_match=W_U_for_norm_match, norm_tol=0.5)
    if pair is None:
        return []
    true_id, false_id = pair
    rng = np.random.default_rng(rng_seed)
    pool = _number_pool(tokenizer)
    if len(pool) < 4:
        return []
    out: list[Example] = []
    seen: set[tuple[int, int]] = set()
    max_attempts = max(8 * n_max, 4 * len(pool) * (len(pool) - 1))
    attempts = 0
    while len(out) < n_max and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(len(pool), size=2, replace=False)
        a_n, b_n = pool[i][0], pool[j][0]
        if a_n == b_n:
            continue
        key = (min(a_n, b_n), max(a_n, b_n))
        if key in seen:
            continue
        seen.add(key)
        small_n, big_n = key
        for prompt, y_plus, y_minus, label in [
            (f"{big_n} is greater than {small_n}. True or False? Answer:", true_id, false_id, "true"),
            (f"{small_n} is greater than {big_n}. True or False? Answer:", false_id, true_id, "false"),
        ]:
            out.append(
                Example(
                    family="numeric_tf",
                    prompt=prompt,
                    prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
                    y_plus=y_plus,
                    y_minus=y_minus,
                    meta={"big": big_n, "small": small_n, "label": label, "format": "true_false"},
                )
            )
            if len(out) >= n_max:
                return out
    return out
