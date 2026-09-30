"""Zero-point control families for the availability/expression panel.

Both are built so that any availability-minus-expression gap they show would be
an artefact of the measurement, not of the model:

  - ``copy_control``: "<wA><wA><wB><wB><wC>" -> y_plus = wC (a verbatim copy of
    the adjacent token), y_minus = a pool word absent from the prompt.
  - ``fixed_token_control``: the answer is the same token (" the" vs " ,") for
    every prompt; prompts are 2-5 random pool words carrying no information.

Words come from ``contrastive_tasks._induction_token_pool`` (single-token,
leading-space). Neither family has a probeable feature, so the availability probe
skips them; they enter the panel through the readout metrics only.
"""

from __future__ import annotations

import numpy as np

from readout.probes.contrastive_tasks import (
    Example,
    _filter_pair,
    _induction_token_pool,
    _row_norm,
    _single_token_id,
)

NORM_TOL = 0.2


def build_copy_control(tokenizer, *, n_max: int = 2000, rng_seed: int = 0, W_U_for_norm_match=None) -> list[Example]:
    pool = _induction_token_pool(tokenizer, k=200)
    if len(pool) < 4:
        return []
    rng = np.random.default_rng(rng_seed)
    out: list[Example] = []
    seen: set[tuple[int, int, int, int]] = set()
    max_attempts = max(8 * n_max, 4 * len(pool) * (len(pool) - 1))
    attempts = 0
    while len(out) < n_max and attempts < max_attempts:
        attempts += 1
        a, b, c, d = (int(x) for x in rng.choice(pool, size=4, replace=False))
        if (a, b, c, d) in seen:
            continue
        seen.add((a, b, c, d))
        a_str, b_str, c_str = tokenizer.decode([a]), tokenizer.decode([b]), tokenizer.decode([c])
        prompt_text = f"{a_str}{a_str}{b_str}{b_str}{c_str}"
        ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(ids) < 5:
            continue
        if W_U_for_norm_match is not None:
            r_c, r_d = _row_norm(W_U_for_norm_match, c), _row_norm(W_U_for_norm_match, d)
            if r_c > 0 and r_d > 0 and abs(np.log(r_c / r_d)) > NORM_TOL:
                continue
        out.append(
            Example(
                family="copy_control",
                prompt=prompt_text,
                prompt_ids=ids,
                y_plus=c,
                y_minus=d,
                meta={
                    "construction": "no_lag_verbatim_copy",
                    "pattern": "wA wA wB wB wC -> wC",
                    "copy_token": c,
                    "doubled_tokens": [a, b],
                    "distractor": d,
                    "distractor_in_prompt": False,
                    "expected_margin": "zero",
                    "role": "falsification_zero_point",
                },
            )
        )
    return out


def build_fixed_token_control(
    tokenizer, *, n_max: int = 2000, rng_seed: int = 0, W_U_for_norm_match=None
) -> list[Example]:
    rng = np.random.default_rng(rng_seed)
    # Fixed (y_plus, y_minus) candidates in preference order; the first pair that is
    # single-token (and norm-matched when W_U is given) is used for every item.
    candidate_pairs = [("the", ","), ("the", "and"), ("of", "the"), ("and", "the"), (",", ".")]
    fixed_pair = chosen = None
    for pos, neg in candidate_pairs:
        pair = _filter_pair(tokenizer, pos, neg, W_U_for_norm_match=W_U_for_norm_match, norm_tol=0.5)
        if pair is not None:
            fixed_pair, chosen = pair, (pos, neg)
            break
    if fixed_pair is None:
        for pos, neg in candidate_pairs:
            pid = _single_token_id(tokenizer, pos, leading_space=True)
            nid = _single_token_id(tokenizer, neg, leading_space=True)
            if pid is not None and nid is not None and pid != nid:
                fixed_pair, chosen = (pid, nid), (pos, neg)
                break
    if fixed_pair is None:
        return []
    y_plus, y_minus = fixed_pair

    pool = _induction_token_pool(tokenizer, k=200)
    if len(pool) < 3:
        return []
    out: list[Example] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = max(8 * n_max, 1000)
    attempts = 0
    while len(out) < n_max and attempts < max_attempts:
        attempts += 1
        span = int(rng.integers(2, 6))
        words = [int(x) for x in rng.choice(pool, size=span, replace=False)]
        if tuple(words) in seen:
            continue
        seen.add(tuple(words))
        prompt_text = "".join(tokenizer.decode([w]) for w in words)
        ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(ids) < 1:
            continue
        out.append(
            Example(
                family="fixed_token_control",
                prompt=prompt_text,
                prompt_ids=ids,
                y_plus=y_plus,
                y_minus=y_minus,
                meta={
                    "construction": "constant_fixed_answer",
                    "fixed_pos": chosen[0],
                    "fixed_neg": chosen[1],
                    "fixed_y_plus_id": int(y_plus),
                    "fixed_y_minus_id": int(y_minus),
                    "prompt_words": words,
                    "answer_depends_on_context": False,
                    "expected_margin": "zero",
                    "role": "falsification_zero_point",
                },
            )
        )
    return out
