"""BLiMP minimal-pair families (``nyu-mll/blimp``) as single-token contrasts.

A good/bad sentence pair is kept only if the two token sequences differ at exactly
one aligned position with identical prefix and suffix; the prompt is the shared
prefix, ``y_plus`` the good token and ``y_minus`` the bad token (both re-derived as
leading-space single tokens). Families:

  - ``blimp_det_noun_agreement``: ``determiner_noun_agreement_1``
  - ``blimp_anaphor_agreement``: ``anaphor_gender_agreement`` then ``anaphor_number_agreement``
  - ``blimp_npi``: ``sentential_negation_npi_licensor_present``

The Example ``family`` field is ``blimp_<config>``; the registry key names the file.
Requires the ``datasets`` package and network access (or a warm HF cache) on first use.
"""

from __future__ import annotations

from readout.probes.contrastive_tasks import Example, _filter_pair, _load_hf_split

BLIMP_REPO = "nyu-mll/blimp"
NORM_TOL = 0.2


def _first_diff_pos(good_ids: list[int], bad_ids: list[int]) -> int | None:
    for i in range(min(len(good_ids), len(bad_ids))):
        if good_ids[i] != bad_ids[i]:
            return i
    return None


def reduce_pair_to_contrast(tokenizer, good: str, bad: str, *, W_U_for_norm_match=None):
    """``(prefix_ids, pos, y_plus, y_minus)`` for a clean single-token substitution, else None."""
    good_ids = tokenizer.encode(good, add_special_tokens=False)
    bad_ids = tokenizer.encode(bad, add_special_tokens=False)
    pos = _first_diff_pos(good_ids, bad_ids)
    if pos is None or good_ids[pos + 1 :] != bad_ids[pos + 1 :]:
        return None
    good_str = tokenizer.decode([good_ids[pos]])
    bad_str = tokenizer.decode([bad_ids[pos]])
    if not good_str.strip() or not bad_str.strip():
        return None
    pair = _filter_pair(
        tokenizer,
        good_str.strip(),
        bad_str.strip(),
        leading_space=True,
        W_U_for_norm_match=W_U_for_norm_match,
        norm_tol=NORM_TOL,
    )
    if pair is None:
        return None
    return good_ids[:pos], pos, pair[0], pair[1]


def build_blimp_paradigm(tokenizer, config_name: str, *, n_max: int = 2000, W_U_for_norm_match=None) -> list[Example]:
    ds = _load_hf_split(BLIMP_REPO, config_name, split="train")
    if ds is None:
        raise RuntimeError(f"could not load {BLIMP_REPO}/{config_name} (see the message above)")
    out: list[Example] = []
    for r in ds:
        if len(out) >= n_max:
            break
        good, bad = r.get("sentence_good"), r.get("sentence_bad")
        if not isinstance(good, str) or not isinstance(bad, str):
            continue
        reduced = reduce_pair_to_contrast(tokenizer, good, bad, W_U_for_norm_match=W_U_for_norm_match)
        if reduced is None:
            continue
        prefix_ids, pos, y_plus, y_minus = reduced
        out.append(
            Example(
                family=f"blimp_{config_name}",
                prompt=tokenizer.decode(prefix_ids),
                prompt_ids=prefix_ids,
                y_plus=y_plus,
                y_minus=y_minus,
                meta={
                    "config": config_name,
                    "paradigm": r.get("linguistics_term", config_name),
                    "sentence_good": good,
                    "sentence_bad": bad,
                    "diff_pos": pos,
                    "good_token": tokenizer.decode([y_plus]),
                    "bad_token": tokenizer.decode([y_minus]),
                },
            )
        )
    return out


def build_blimp_det_noun_agreement(
    tokenizer, *, n_max: int = 2000, rng_seed: int = 0, W_U_for_norm_match=None
) -> list[Example]:
    return build_blimp_paradigm(
        tokenizer, "determiner_noun_agreement_1", n_max=n_max, W_U_for_norm_match=W_U_for_norm_match
    )


def build_blimp_anaphor_agreement(
    tokenizer, *, n_max: int = 2000, rng_seed: int = 0, W_U_for_norm_match=None
) -> list[Example]:
    out: list[Example] = []
    for config in ("anaphor_gender_agreement", "anaphor_number_agreement"):
        if len(out) >= n_max:
            break
        out.extend(
            build_blimp_paradigm(tokenizer, config, n_max=n_max - len(out), W_U_for_norm_match=W_U_for_norm_match)
        )
    return out[:n_max]


def build_blimp_npi(tokenizer, *, n_max: int = 2000, rng_seed: int = 0, W_U_for_norm_match=None) -> list[Example]:
    return build_blimp_paradigm(
        tokenizer, "sentential_negation_npi_licensor_present", n_max=n_max, W_U_for_norm_match=W_U_for_norm_match
    )
