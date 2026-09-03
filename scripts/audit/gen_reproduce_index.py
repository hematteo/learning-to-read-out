"""Generate the docs/REPRODUCE.md figure index from experiments.yaml.

experiments.yaml is the single source of truth for the label -> experiment ->
source mapping; the human-readable index table in docs/REPRODUCE.md is
generated from it between the BEGIN/END markers. CI runs --check so the two
can never drift.

Run:
    uv run python scripts/audit/gen_reproduce_index.py           # rewrite the table
    uv run python scripts/audit/gen_reproduce_index.py --check   # exit 1 on drift
"""

from __future__ import annotations

import argparse
import sys

import yaml

from readout.core.paths import repo_root

REPO = repo_root()
EXPERIMENTS_YAML = REPO / "experiments.yaml"
REPRODUCE_MD = REPO / "docs" / "REPRODUCE.md"
BEGIN = "<!-- BEGIN GENERATED FIGURE INDEX (scripts/audit/gen_reproduce_index.py) -->"
END = "<!-- END GENERATED FIGURE INDEX -->"


def render_table() -> str:
    data = yaml.safe_load(EXPERIMENTS_YAML.read_text())
    exps = data.get("experiments") or []
    paper_only = set(data.get("paper_only_labels") or [])

    by_label_exp: dict[str, set[str]] = {}
    by_label_src: dict[str, set[str]] = {}
    for e in exps:
        eid = e["id"]
        for lbl in e.get("paper_labels") or []:
            by_label_exp.setdefault(lbl, set()).add(eid)
        for pf in e.get("paper_figures") or []:
            if "->" not in pf:
                continue
            src, lbl = pf.rsplit("->", 1)
            by_label_src.setdefault(lbl.strip(), set()).add(src.strip())

    lines = [
        "| Label | Experiment | Rendered figure (paper LaTeX tree) |",
        "|---|---|---|",
    ]
    for lbl in sorted(set(by_label_exp) | paper_only):
        exps_col = ", ".join(sorted(by_label_exp.get(lbl, ()))) or "(paper LaTeX tree only)"
        srcs = sorted(by_label_src.get(lbl, ()))
        if srcs:
            src_col = "; ".join(srcs)
        elif lbl in paper_only and lbl not in by_label_exp:
            src_col = "(paper LaTeX tree only)"
        else:
            src_col = "(produced; see experiments.yaml / tables below)"
        lines.append(f'| <a id="{lbl}"></a>`{lbl}` | {exps_col} | {src_col} |')
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the table is stale instead of rewriting")
    args = ap.parse_args()

    if not REPRODUCE_MD.exists():
        print(
            f"{REPRODUCE_MD} not found; restore docs/REPRODUCE.md (with the "
            f"{BEGIN!r} ... {END!r} markers) before regenerating the figure index",
            file=sys.stderr,
        )
        return 1
    doc = REPRODUCE_MD.read_text()
    if BEGIN not in doc or END not in doc:
        print(f"markers missing in {REPRODUCE_MD}; expected {BEGIN!r} ... {END!r}", file=sys.stderr)
        return 1
    head, rest = doc.split(BEGIN, 1)
    stale, tail = rest.split(END, 1)
    fresh = "\n" + render_table() + "\n"
    if stale == fresh:
        print("figure index up to date")
        return 0
    if args.check:
        print(
            "docs/REPRODUCE.md figure index is stale relative to experiments.yaml;\n"
            "regenerate with: uv run python scripts/audit/gen_reproduce_index.py",
            file=sys.stderr,
        )
        return 1
    REPRODUCE_MD.write_text(head + BEGIN + fresh + END + tail)
    print("figure index rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
