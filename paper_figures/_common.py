"""Paths shared by the paper-figure scripts; importing this also selects the Agg backend."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"

# The two lifecycle scripts import the profile rules from readout.dynamics.lifecycle (numpy only);
# this lets them run without installing the package.
SRC = ROOT.parent / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))


def out_path(name: str) -> Path:
    OUT.mkdir(exist_ok=True)
    return OUT / f"{name}.pdf"
