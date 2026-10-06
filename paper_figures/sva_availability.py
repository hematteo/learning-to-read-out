"""Availability versus expression of subject-verb agreement in Pythia-6.9B (fig:main-sva-availability-expression).

Left: SVA contrast accuracy during training for a linear probe on the hidden state, the native readout and the
best swept readout. The probe series is run output (data/sva_hidden_probe_trajectory_69b.csv); the native and
swept series are values transcribed from the original figure, whose run output is unavailable
(data/sva_native_swept_transcribed_69b.csv). Right: availability A_1 and native expression E at step 143k for
the four agreement families (data/availability_probe_summary_69b.csv).
"""

import matplotlib.pyplot as plt
import pandas as pd
from _common import DATA, out_path
from matplotlib.lines import Line2D

COLORS = {"probe": "#2a9d55", "readout": "#2f5aa8", "best": "#d8832f", "grid": "#bdbdbd"}
# Tick positions of steps 0, 1, 10, 100, 1k, 10k and 143k in the x_display coordinates of the transcribed series.
TICKS = [0.0, 16.0592, 69.4069, 122.7545, 176.1021, 229.4497, 291.0842]
TICK_LABELS = ["0", "1", "10", "100", "1k", "10k", "143k"]
FAMILIES = [("Simple", "sva"), ("Across-PP", "sva_across_pp"), ("Object-RC", "sva_object_rc"),
            ("Subject-RC", "sva_subject_rc")]  # fmt: skip


def load_timeline():
    swept = pd.read_csv(DATA / "sva_native_swept_transcribed_69b.csv", comment="#")
    probe = pd.read_csv(DATA / "sva_hidden_probe_trajectory_69b.csv")[["h_step", "acc"]]
    t = swept.merge(probe.rename(columns={"h_step": "step", "acc": "probe_acc"}), on="step", validate="1:1")
    assert len(t) == len(swept) == len(probe)
    return t.sort_values("step")


def load_ladder():
    panel = pd.read_csv(DATA / "availability_probe_summary_69b.csv")
    panel = panel[(panel.h_step == 143000) & (panel.alignment == "none")].set_index("family")
    return [
        (name, float(panel.loc[fam, "probe_acc_1d"]), float(panel.loc[fam, "native_readout_acc"]))
        for name, fam in FAMILIES
    ]


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.65)
    ax.tick_params(width=0.65, length=2.8, pad=2.0)


def plot_timeline(ax, t):
    x, probe, same, best = (
        list(t.x_display),
        list(t.probe_acc),
        list(t.native_readout_acc),
        list(t.best_swept_readout_acc),
    )
    ax.axhline(0.5, color="#777777", lw=0.7, ls=":", zorder=0)
    ax.fill_between(
        x, same, probe, where=[p >= s for p, s in zip(probe, same)], color=COLORS["probe"], alpha=0.08, linewidth=0
    )
    ax.plot(x, probe, color=COLORS["probe"], marker="o", ms=2.8, lw=1.25)
    ax.plot(x, same, color=COLORS["readout"], marker="s", ms=2.6, lw=1.25)
    ax.plot(x, best, color=COLORS["best"], marker="^", ms=2.7, lw=1.1, ls="--")
    ax.set_title("Agreement during training (Pythia-6.9B)")
    ax.set_xticks(TICKS)
    ax.set_xticklabels(TICK_LABELS)
    ax.set_xlim(-14.6, 305.6)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel("training step")
    ax.set_ylabel("contrast accuracy")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45, alpha=0.38)
    style_axis(ax)
    ax.text(300, 0.51, "chance", ha="right", va="bottom", fontsize=7, color="#666666")
    handles = [
        Line2D([0], [0], color=COLORS["probe"], marker="o", lw=1.25, ms=2.8, label="linear probe"),
        Line2D([0], [0], color=COLORS["readout"], marker="s", lw=1.25, ms=2.6, label="native readout"),
        Line2D([0], [0], color=COLORS["best"], marker="^", lw=1.1, ls="--", ms=2.7, label="best swept readout"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.35,
    )


def plot_ladder(ax, ladder):
    ys = list(range(len(ladder) - 1, -1, -1))
    for y, (_, a1, e) in zip(ys, ladder):
        ax.plot([e, a1], [y, y], color=COLORS["grid"], lw=2.4, solid_capstyle="round", zorder=1)
        ax.scatter([e], [y], s=28, marker="s", color=COLORS["readout"], zorder=3)
        ax.scatter([a1], [y], s=30, marker="o", color=COLORS["probe"], zorder=3)
        if a1 - e >= 0.005:
            ax.text(
                (e + a1) / 2,
                y + 0.14,
                f"{round(a1 - e + 1e-9, 2):.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444444",
            )
    ax.axvline(0.5, color="#777777", lw=0.7, ls=":", zorder=0)
    ax.text(0.515, -0.55, "chance", ha="left", va="bottom", fontsize=7, color="#666666")
    ax.set_title("Agreement at the end of training")
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _ in ladder])
    ax.set_ylim(-0.6, len(ladder) - 0.3)
    ax.set_xlim(0.4, 1.06)
    ax.set_xlabel("contrast accuracy")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.45, alpha=0.38)
    style_axis(ax)
    handles = [
        ax.scatter([], [], s=30, marker="o", color=COLORS["probe"], label=r"availability $A_1$"),
        ax.scatter([], [], s=28, marker="s", color=COLORS["readout"], label=r"native expression $E$"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, handletextpad=0.3)


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.65,
        }
    )
    fig = plt.figure(figsize=(5.5, 2.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.42)
    plot_timeline(fig.add_subplot(gs[0, 0]), load_timeline())
    plot_ladder(fig.add_subplot(gs[0, 1]), load_ladder())
    fig.subplots_adjust(left=0.09, right=0.995, top=0.89, bottom=0.19)
    out = out_path("sva_availability")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
