"""Shared style of the camera-ready figures: palette, type sizes and the save call.

One palette with fixed roles: readout (native, readout rows) blue, swapped or swept readout orange, hidden-state
probe (availability) teal, controls and nulls gray, the reorganization window a pale sand band, and the lifecycle
profiles wine / ochre / indigo. Models take the categorical slots in a fixed order (160M teal, 1B blue, 6.9B
orange, OLMo plum); ordered conditions (readout learning rate) take a one-hue blue ramp. Figures are drawn at the
5.5 in text width with 7.5 pt ticks and 8 pt labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt

TEXT_WIDTH = 5.5

BLUE = "#2863AB"
ORANGE = "#D46A34"
TEAL = "#0F9485"
PLUM = "#92417A"
INDIGO = "#57429C"
WINE = "#A14174"
OCHRE = "#BB881A"
NAVY = "#173B78"
BLUE_RAMP = ["#7BA4D9", "#346AAC", "#0C335F"]  # light -> dark, ordinal

READOUT = BLUE
SWAP = ORANGE
PROBE = TEAL
CONTROL = "#8A8A8A"
CONTROL_LIGHT = "#DCDCDC"
CHANCE = "#777777"
WINDOW = "#F4EAD9"
WINDOW_EDGE = "#C9B48E"
MODEL = {"160m": TEAL, "1b": BLUE, "6.9b": ORANGE, "olmo": PLUM}
PROFILE = {
    "early_decay": WINE,
    "persistent": OCHRE,
    "late_emerge": INDIGO,
    "transitional": "#DCDCDC",
    "mixed": "#DCDCDC",
}
PROFILE_LABEL = {
    "early_decay": "early-decaying",
    "persistent": "persistent",
    "late_emerge": "late-emerging",
    "transitional": "other",
    "mixed": "other",
}
# Selected reorganization windows (Section 4.1).
PYTHIA_WINDOW = (512, 1621)
OLMO_WINDOW = (9000, 28463)

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
}


def apply():
    plt.rcParams.update(RC)


def panel_title(ax, letter, text, **kw):
    ax.set_title(rf"$\bf{{({letter})}}$ {text}", loc="left", pad=kw.pop("pad", 4), **kw)


def save(fig, path):
    path = Path(path)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {path}")
