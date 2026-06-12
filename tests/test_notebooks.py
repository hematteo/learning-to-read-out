"""Structural rot-guards for the Colab notebooks (no execution, no downloads).

Executing the notebooks needs checkpoint downloads and minutes of training, so
CI only checks what can rot silently: that every code cell still parses (with
IPython magics masked), and that the shipped notebooks stay output-free (no
stored outputs, execution counts, or local absolute paths).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_expected_notebooks_present():
    names = [p.name for p in NOTEBOOKS]
    assert "01_availability_expression_lag.ipynb" in names
    assert "02_wu_trajectory_crosscoders.ipynb" in names


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_parse(path: Path):
    nb = _load(path)
    assert nb["nbformat"] == 4
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        # Mask IPython magic/shell lines (%pip, !git ...) preserving indentation,
        # so conditional magics don't leave empty blocks behind.
        lines = []
        for line in "".join(cell["source"]).splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!")):
                lines.append(line[: len(line) - len(stripped)] + "pass")
            else:
                lines.append(line)
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as exc:  # pragma: no cover - failure reporting only
            raise AssertionError(f"{path.name} cell {i} does not parse: {exc}") from exc


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_ship_clean(path: Path):
    nb = _load(path)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs") == [], f"{path.name} cell {i} has stored outputs"
        assert cell.get("execution_count") is None, f"{path.name} cell {i} has an execution count"
    blob = json.dumps(nb)
    assert "/Users/" not in blob and "/home/" not in blob, f"{path.name} embeds a local absolute path"
