"""Decoder-norm trajectories of every active feature (fig:main-selected-normalized-trajectories).

Active features have maximum decoder norm above 0.01; each trajectory is divided by its own maximum and colored by
the log step of its peak. Reads data/lifecycle_<run>.npz (see data/README.md).
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import DATA, out_path
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

RUNS = [
    ("pythia160m_d24576", "Pythia-160M"),
    ("pythia1b_d24576", "Pythia-1B"),
    ("pythia69b_d32768", "Pythia-6.9B"),
    ("olmo2_7b_d32768", "OLMo-2-7B"),
]
LW, ALPHA, CMAP = 0.3, 0.04, "viridis"


def load(key):
    z = np.load(DATA / f"lifecycle_{key}.npz")
    # traj: (steps, active features), each feature divided by its own peak; peak_idx: (active features,)
    return z["steps"], z["traj"].astype(np.float32), z["peak_idx"].astype(np.int64)


def panel(ax, traj, steps, peak_idx, title, color_norm):
    peak_step = steps[peak_idx]
    mask = steps > 0
    x = steps[mask].astype(np.float64)
    segments = [np.column_stack([x, yj]) for yj in traj[mask].T]
    colors = plt.get_cmap(CMAP)(color_norm(np.log10(peak_step + 1)))
    order = np.argsort(peak_step)
    ax.add_collection(
        LineCollection(
            [segments[i] for i in order],
            colors=colors[order],
            linewidths=LW,
            alpha=ALPHA,
            rasterized=True,
        )
    )
    ax.set_xscale("log")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.2)


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    runs = {key: load(key) for key, _ in RUNS}
    olmo_last = runs["olmo2_7b_d32768"][0][-1]
    color_norm = Normalize(vmin=0.0, vmax=float(np.log10(olmo_last + 1)))
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.65), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.885, bottom=0.25, top=0.86, wspace=0.14)
    for ax, (key, title) in zip(axes, RUNS):
        steps, traj, peak_idx = runs[key]
        panel(ax, traj, steps, peak_idx, title, color_norm)
        if key.startswith("olmo"):
            ax.set_xticks([1000, 14000, 928000], ["1k", "14k", "928k"])
        else:
            ax.set_xticks([10, 1000, 100000], ["10", "1k", "100k"])
        ax.minorticks_off()
        ax.tick_params(axis="both", pad=1.5)
    axes[0].set_ylabel("norm / feature peak")
    fig.supxlabel("training step", fontsize=7.5, y=0.07)
    cax = fig.add_axes([0.905, 0.25, 0.014, 0.61])
    cbar = fig.colorbar(ScalarMappable(cmap=plt.get_cmap(CMAP), norm=color_norm), cax=cax)
    ticks = [0, 1000, 143000, 928000]
    cbar.set_ticks([np.log10(t + 1) for t in ticks])
    cbar.set_ticklabels(["0", "1k", "143k", "928k"])
    cbar.set_label("peak step", labelpad=2)
    cbar.ax.tick_params(labelsize=7, pad=1)
    out = out_path("lifecycle_trajectories")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
