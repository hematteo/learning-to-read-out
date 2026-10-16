"""Lifecycle profiles, trajectories and reorganization scores (fig:main-selected-normalized-trajectories).

One column per dictionary, three rows on a shared log step axis:
  top: share of active features in each lifecycle profile (readout.dynamics.lifecycle.classify_profiles_refined);
  middle: the decoder-norm trajectory of every active feature (maximum decoder norm above 0.01), divided by its own
    maximum and colored by profile, with each profile's median on top (data/lifecycle_<run>.npz);
  bottom: the reorganization score as a z-score against circular-shift nulls per half-decade window; filled markers
    have p <= 0.05 (data/reorganization_window_metrics_selected.csv).
The sand band is the selected window (Pythia 512-1621; OLMo-2-7B 9k-28k, hatched, p = 0.12).
"""

import _style as style
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import DATA, out_path
from lifecycle_profile_composition import profile_quantiles
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

from readout.dynamics.lifecycle import PROFILE_ORDER, classify_profiles_refined

RUNS = [
    ("pythia160m_d24576", "Pythia-160M"),
    ("pythia1b_d24576", "Pythia-1B"),
    ("pythia69b_d32768", "Pythia-6.9B"),
    ("olmo2_7b_d32768", "OLMo-2-7B"),
]
FIG_HEIGHT = 1.75
SHARE_ORDER = ["early_decay", "persistent", "late_emerge", "other"]


def column(fig, gs, col, key, label, windows):
    z = np.load(DATA / f"lifecycle_{key}.npz")
    steps = z["steps"].astype(np.int64)
    traj = z["traj"].astype(np.float32)  # (32, n_active), float16 values as classified in the paper
    profile = classify_profiles_refined(traj, steps)["profile"]
    olmo = key.startswith("olmo")
    window = style.OLMO_WINDOW if olmo else style.PYTHIA_WINDOW

    share = {p: float((profile == p).mean()) for p in PROFILE_ORDER}
    share["other"] = share.pop("mixed") + share.pop("transitional")

    ax_s = fig.add_subplot(gs[0, col])
    left = 0.0
    for p in SHARE_ORDER:
        v = share[p]
        ax_s.barh(
            0, v, left=left, height=1.0, color=style.PROFILE["mixed" if p == "other" else p], edgecolor="white", lw=0.4
        )
        if v >= 0.12:
            ax_s.text(
                left + v / 2,
                0,
                f"{100 * v:.0f}%",
                ha="center",
                va="center",
                fontsize=6.5,
                color="black" if p == "persistent" else "white",
            )
        left += v
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(-0.5, 0.5)
    ax_s.axis("off")
    ax_s.set_title(label, loc="left", pad=2.5)

    ax_t = fig.add_subplot(gs[1, col])
    mask = steps > 0
    x = steps[mask].astype(np.float64)
    # Largest profile first so the minority profiles stay visible on top.
    for p in sorted(PROFILE_ORDER, key=lambda q: -(profile == q).sum()):
        idx = np.where(profile == p)[0]
        if idx.size == 0:
            continue
        segments = [np.column_stack([x, traj[mask, j]]) for j in idx]
        ax_t.add_collection(
            LineCollection(segments, colors=style.PROFILE[p], linewidths=0.3, alpha=0.035, rasterized=True)
        )
    for p in ["early_decay", "persistent", "late_emerge"]:
        m = profile == p
        if m.mean() < 0.01:
            continue
        median, _, _ = profile_quantiles(traj, m)
        ax_t.plot(
            x,
            median[mask],
            color=style.PROFILE[p],
            lw=1.4,
            path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()],
        )

    ax_z = fig.add_subplot(gs[2, col], sharex=ax_t)
    w = windows[(windows.model == key) & (windows.window_center_step >= x.min())]
    sig = w.structural_score_null_p <= 0.05
    ax_z.axhline(0, color=style.CONTROL, lw=0.6)
    ax_z.plot(w.window_center_step, w.structural_score_z, color="0.25", lw=0.9)
    ax_z.scatter(w.window_center_step[sig], w.structural_score_z[sig], s=9, color="0.15", zorder=3)
    ax_z.scatter(
        w.window_center_step[~sig],
        w.structural_score_z[~sig],
        s=9,
        facecolor="white",
        edgecolor="0.35",
        lw=0.6,
        zorder=3,
    )

    for ax in (ax_t, ax_z):
        if olmo:
            ax.axvspan(*window, facecolor="none", edgecolor=style.WINDOW_EDGE, hatch="////", lw=0, zorder=0)
        else:
            ax.axvspan(*window, facecolor=style.WINDOW, lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.minorticks_off()
    for edge in window:
        ax_t.axvline(edge, color=style.WINDOW_EDGE, lw=0.7, ls=(0, (2, 1.5)), zorder=4)
    ax_t.set_ylim(-0.02, 1.05)
    ax_t.set_yticks([0, 0.5, 1], ["0", ".5", "1"])
    ax_t.tick_params(labelbottom=False)
    ax_z.set_ylim(-1.8, 8.2)
    ax_z.set_yticks([0, 4, 8])
    if olmo:
        ax_z.set_xticks([1000, 14000, 928000], ["1k", "14k", "928k"])
    else:
        ax_z.set_xticks([10, 1000, 100000], ["10", "1k", "100k"])
    if col == 0:
        ax_t.set_ylabel("norm / peak", fontsize=7, labelpad=2)
        ax_z.set_ylabel("z", fontsize=7, labelpad=2)
    else:
        ax_t.tick_params(labelleft=False)
        ax_z.tick_params(labelleft=False)


def main():
    style.apply()
    windows = pd.read_csv(DATA / "reorganization_window_metrics_selected.csv")
    fig = plt.figure(figsize=(style.TEXT_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[0.14, 1.0, 0.56],
        hspace=0.17,
        wspace=0.13,
        left=0.085,
        right=0.995,
        top=0.83,
        bottom=0.165,
    )
    for col, (key, label) in enumerate(RUNS):
        column(fig, gs, col, key, label, windows)
    fig.supxlabel("training step", fontsize=8, y=-0.01)
    fig.legend(
        handles=[
            Patch(color=style.PROFILE[p], label=style.PROFILE_LABEL[p])
            for p in ["early_decay", "persistent", "late_emerge", "mixed"]
        ]
        + [Patch(color=style.WINDOW, label="reorganization window")],
        loc="upper center",
        bbox_to_anchor=(0.54, 1.005),
        ncol=5,
        frameon=False,
        fontsize=7,
        handlelength=1.0,
        handleheight=0.7,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    style.save(fig, out_path("lifecycle_trajectories"))


if __name__ == "__main__":
    main()
