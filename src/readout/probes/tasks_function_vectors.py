"""Function-vector ICL families (Todd et al., 2024) as single-token contrasts.

Each task is an input->output map (antonym, English->French, present->past,
country->capital) shown as a 10-shot prompt ``"<in>:<out>"`` per line followed by
the query ``"<in>:"``. ``y_plus`` is the query output (must be a single
leading-space token); ``y_minus`` is another item's output, the first candidate in
a seeded permutation that is single-token and differs from the gold output.

Data: the four JSON files from github.com/ericwtodd/function_vectors
(``dataset_files/abstractive/{antonym,english-french,present-past,country-capital}.json``,
MIT licence, ~530 KB together). They are not shipped here; the directory is
``$FV_DATA_DIR`` if set, else ``<repo>/data/function_vectors``. A missing file
raises ``FileNotFoundError``; omit the ``fv_*`` families to build without them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from readout.core.paths import repo_root
from readout.probes.contrastive_tasks import Example, _filter_pair, _single_token_id

NORM_TOL = 0.2
FV_TASK_FILES: dict[str, str] = {
    "fv_antonym": "antonym.json",
    "fv_en_fr": "english-french.json",
    "fv_present_past": "present-past.json",
    "fv_country_capital": "country-capital.json",
}
FV_SOURCE = "https://raw.githubusercontent.com/ericwtodd/function_vectors/main/dataset_files/abstractive/"


def fv_data_dir() -> Path:
    env = os.environ.get("FV_DATA_DIR")
    return Path(env).expanduser() if env else repo_root() / "data" / "function_vectors"


def load_fv_pairs(task_name: str) -> list[tuple[str, str]]:
    """``(input, output)`` string pairs; list outputs take their first entry."""
    path = fv_data_dir() / FV_TASK_FILES[task_name]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download {FV_SOURCE}{FV_TASK_FILES[task_name]} into that directory "
            "(or set FV_DATA_DIR), or drop the fv_* families."
        )
    pairs: list[tuple[str, str]] = []
    for row in json.loads(path.read_text()):
        if not isinstance(row, dict):
            continue
        inp, outp = row.get("input"), row.get("output")
        if isinstance(outp, list):
            outp = outp[0] if outp else None
        if not isinstance(inp, str) or not isinstance(outp, str):
            continue
        inp, outp = inp.strip(), outp.strip()
        if inp and outp:
            pairs.append((inp, outp))
    return pairs


def build_fv_task(
    tokenizer,
    task_name: str,
    *,
    n_max: int = 2000,
    rng_seed: int = 0,
    n_shots: int = 10,
    W_U_for_norm_match=None,
) -> list[Example]:
    pairs = load_fv_pairs(task_name)
    if len(pairs) < n_shots + 2:
        return []
    rng = np.random.default_rng(rng_seed)
    all_idx = np.arange(len(pairs))
    out: list[Example] = []
    for q_idx, (q_in, q_out) in enumerate(pairs):
        if len(out) >= n_max:
            break
        if _single_token_id(tokenizer, q_out, leading_space=True) is None:
            continue
        # A fresh permutation per query is drawn even when no distractor is found,
        # so the rng stream (and every later item) depends on this exact order.
        ym = d_idx = d_out = None
        for cand in rng.permutation(all_idx):
            cand = int(cand)
            if cand == q_idx:
                continue
            cand_out = pairs[cand][1]
            if cand_out.strip().lower() == q_out.strip().lower():
                continue
            pair = _filter_pair(
                tokenizer, q_out, cand_out, leading_space=True, W_U_for_norm_match=W_U_for_norm_match, norm_tol=NORM_TOL
            )
            if pair is None:
                continue
            yp, ym = pair
            d_idx, d_out = cand, cand_out
            break
        if ym is None:
            continue
        pool = all_idx[all_idx != q_idx]
        if pool.size < n_shots:
            continue
        demo_idx = rng.choice(pool, size=n_shots, replace=False)
        prompt = "\n".join(f"{pairs[int(i)][0]}:{pairs[int(i)][1]}" for i in demo_idx) + f"\n{q_in}:"
        out.append(
            Example(
                family=task_name,
                prompt=prompt,
                prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
                y_plus=yp,
                y_minus=ym,
                meta={
                    "task": task_name,
                    "input": q_in,
                    "output": q_out,
                    "n_shots": n_shots,
                    "demo_indices": [int(i) for i in demo_idx],
                    "query_index": q_idx,
                    "distractor_index": d_idx,
                    "distractor_output": d_out,
                },
            )
        )
    return out


def _fv_builder(task_name: str):
    def build(tokenizer, *, n_max: int = 2000, rng_seed: int = 0, W_U_for_norm_match=None) -> list[Example]:
        return build_fv_task(
            tokenizer, task_name, n_max=n_max, rng_seed=rng_seed, W_U_for_norm_match=W_U_for_norm_match
        )

    build.__name__ = f"build_{task_name}"
    return build


FV_TASK_BUILDERS = {name: _fv_builder(name) for name in FV_TASK_FILES}
