"""Lifecycle profile composition and median profile trajectories (fig:app-lifecycle-profile-composition).

Profiles are assigned by the paper's refined ruleset (readout.dynamics.lifecycle.classify_profiles_refined,
tab:lifecycle-profile-rules) from the peak-normalized float16 trajectories in data/lifecycle_<run>.npz.
"""

import _style as style
import matplotlib.pyplot as plt
import numpy as np
from _common import DATA, out_path

from readout.dynamics.lifecycle import PROFILE_LABELS, PROFILE_ORDER, classify_profiles_refined

RUNS = [
    ("pythia160m_d24576", "Pythia-160M", (1, 1000, 14000, 143000)),
    ("pythia1b_d24576", "Pythia-1B", (1, 1000, 14000, 143000)),
    ("pythia69b_d32768", "Pythia-6.9B", (1, 1000, 14000, 143000)),
    ("olmo2_7b_d32768", "OLMo-2-7B", (150, 1000, 14000, 928000)),
]
PROFILE_COLORS = {**style.PROFILE, "transitional": style.TEAL, "mixed": style.CONTROL}


def fmt_step(step: int) -> str:
    if step < 1000:
        return str(step)
    if step % 1000 == 0:
        return f"{step // 1000}k"
    return f"{step / 1000:.0f}k"


def profile_quantiles(traj: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = traj[:, mask]
    if selected.shape[1] == 0:
        nan = np.full(traj.shape[0], np.nan, dtype=np.float64)
        return nan, nan, nan
    return (
        np.nanmedian(selected, axis=1),
        np.nanpercentile(selected, 25, axis=1),
        np.nanpercentile(selected, 75, axis=1),
    )


def main() -> None:
    source = {}
    for key, _, _ in RUNS:
        z = np.load(DATA / f"lifecycle_{key}.npz")
        source[key] = {"traj": z["traj"].astype(np.float32), "steps": z["steps"].astype(np.int64)}  # traj: (32, n)
    profiles = {
        key: classify_profiles_refined(source[key]["traj"], source[key]["steps"])["profile"] for key, _, _ in RUNS
    }
    frac = {(key, name): float((profiles[key] == name).mean()) for key, _, _ in RUNS for name in PROFILE_ORDER}
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
            "legend.fontsize": 7.5,
        }
    )
    fig = plt.figure(figsize=(5.5, 4.7))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=(0.62, 1.0, 1.0),
        hspace=0.72,
        wspace=0.3,
        left=0.1,
        right=0.98,
        top=0.9,
        bottom=0.08,
    )
    ax_bar = fig.add_subplot(gs[0, :])
    line_axes = [fig.add_subplot(gs[i, j]) for i in (1, 2) for j in (0, 1)]
    bar_pos = ax_bar.get_position()
    ax_bar.set_position([0.155, bar_pos.y0, 0.98 - 0.155, bar_pos.height])

    y = np.arange(len(RUNS))
    left = np.zeros(len(RUNS))
    for name in PROFILE_ORDER:
        values = np.array([frac[(key, name)] for key, _, _ in RUNS])
        ax_bar.barh(
            y,
            values,
            left=left,
            color=PROFILE_COLORS[name],
            edgecolor="white",
            linewidth=0.5,
            label=PROFILE_LABELS[name],
        )
        left += values
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([label for _, label, _ in RUNS])
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("fraction of active features", labelpad=2)
    ax_bar.grid(axis="x", alpha=0.2)
    ax_bar.legend(
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.45, 1.02),
        columnspacing=1.0,
        handlelength=1.2,
        handletextpad=0.4,
    )

    for ax, (key, label, xticks) in zip(line_axes, RUNS):
        traj = source[key]["traj"]
        x = source[key]["steps"].astype(np.float64) + 1.0
        for name in PROFILE_ORDER:
            mask = profiles[key] == name
            if mask.sum() == 0:
                continue
            median, q25, q75 = profile_quantiles(traj, mask)
            color = PROFILE_COLORS[name]
            ax.plot(x, median, color=color, lw=1.2)
            ax.fill_between(x, q25, q75, color=color, alpha=0.10, lw=0)
        ax.set_title(label, loc="left", pad=3)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xticks([t + 1 for t in xticks])
        ax.set_xticklabels([fmt_step(t) for t in xticks])
        ax.grid(alpha=0.18)
        ax.set_ylabel("normalized decoder norm")
        ax.set_xlabel("training step", labelpad=2)
        ax.tick_params(axis="both", pad=1.5)

    out = out_path("lifecycle_profile_composition")
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
