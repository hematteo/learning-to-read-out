"""Availability versus expression of subject-verb agreement in Pythia-6.9B (fig:main-sva-availability-expression).

(a) SVA contrast accuracy during training for a linear probe on the hidden state, the native readout and the best
swept readout. The probe series is run output (data/sva_hidden_probe_trajectory_69b.csv); the native and swept
series are values transcribed from the original figure, whose run output is unavailable
(data/sva_native_swept_transcribed_69b.csv). The arrow marks the gap at step 512 that the text cites.
(b) Availability A_1 and native expression E at step 143k for the four agreement families
(data/availability_probe_summary_69b.csv); error bars are 95% Wilson intervals for native expression over the n
items of each set.
"""

import math

import _style as style
import matplotlib.pyplot as plt
import pandas as pd
from _common import DATA, out_path
from matplotlib.lines import Line2D

FAMILIES = [("Simple", "sva"), ("Across-PP", "sva_across_pp"), ("Object-RC", "sva_object_rc"),
            ("Subject-RC", "sva_subject_rc")]  # fmt: skip


def load_timeline():
    swept = pd.read_csv(DATA / "sva_native_swept_transcribed_69b.csv", comment="#")
    probe = pd.read_csv(DATA / "sva_hidden_probe_trajectory_69b.csv")[["h_step", "acc"]]
    t = swept.merge(probe.rename(columns={"h_step": "step", "acc": "probe_acc"}), on="step", validate="1:1")
    assert len(t) == len(swept) == len(probe)
    return t.sort_values("step").reset_index(drop=True)


def load_ladder():
    panel = pd.read_csv(DATA / "availability_probe_summary_69b.csv")
    panel = panel[(panel.h_step == 143000) & (panel.alignment == "none")].set_index("family")
    return [
        (
            name,
            float(panel.loc[fam, "probe_acc_1d"]),
            float(panel.loc[fam, "native_readout_acc"]),
            int(panel.loc[fam, "n"]),
        )
        for name, fam in FAMILIES
    ]


def disp(step):
    # x_display of the recovered axis: step 0 at 0, then 53.348 units per decade from 16.06 at step 1.
    return 16.0592 + 53.3477 * math.log10(step)


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def plot_timeline(ax, t):
    x, probe, same, best = (
        list(t.x_display),
        list(t.probe_acc),
        list(t.native_readout_acc),
        list(t.best_swept_readout_acc),
    )
    ax.axhline(0.5, color=style.CHANCE, lw=0.7, ls=":", zorder=0)
    ax.fill_between(x, same, probe, where=[p >= s for p, s in zip(probe, same)], color=style.PROBE, alpha=0.08, lw=0)
    ax.plot(x, probe, color=style.PROBE, marker="o", ms=2.8, lw=1.25)
    ax.plot(x, same, color=style.READOUT, marker="s", ms=2.6, lw=1.25)
    ax.plot(x, best, color=style.SWAP, marker="^", ms=2.7, lw=1.1, ls="--")
    (i,) = t.index[t.step == 512]
    ax.annotate(
        "",
        (x[i], probe[i] - 0.012),
        (x[i], same[i] + 0.012),
        arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.2", shrinkA=0, shrinkB=0),
    )
    ax.text(
        x[i] + 4,
        0.445,
        f"step 512: probe {probe[i]:.2f}, native {same[i]:.2f}",
        ha="right",
        va="center",
        fontsize=7,
        color="0.2",
    )
    ticks = [10, 100, 1000, 10000, 143000]
    ax.set_xticks([0.0] + [disp(s) for s in ticks], ["0", "10", "100", "1k", "10k", "143k"])
    ax.set_xlim(-14.6, 305.6)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel("training step")
    ax.set_ylabel("contrast accuracy")
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.text(300, 0.51, "chance", ha="right", va="bottom", fontsize=7, color="0.45")
    handles = [
        Line2D([0], [0], color=style.PROBE, marker="o", lw=1.25, ms=2.8, label="linear probe"),
        Line2D([0], [0], color=style.READOUT, marker="s", lw=1.25, ms=2.6, label="native readout"),
        Line2D([0], [0], color=style.SWAP, marker="^", lw=1.1, ls="--", ms=2.7, label="best swept readout"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.5,
        labelspacing=0.3,
        borderaxespad=0.1,
    )
    style.panel_title(ax, "a", "Agreement during training, Pythia-6.9B")


def plot_ladder(ax, ladder):
    ys = list(range(len(ladder) - 1, -1, -1))
    for y, (_, a1, e, n) in zip(ys, ladder):
        lo, hi = wilson(e, n)
        ax.plot([e, a1], [y, y], color=style.CONTROL_LIGHT, lw=2.4, solid_capstyle="round", zorder=1)
        # Clip at zero: at e = 1 the upper Wilson bound rounds to just below 1.
        ax.errorbar(
            [e],
            [y],
            xerr=[[max(e - lo, 0.0)], [max(hi - e, 0.0)]],
            fmt="none",
            ecolor=style.READOUT,
            elinewidth=0.8,
            capsize=1.8,
            zorder=2,
        )
        ax.scatter([e], [y], s=26, marker="s", color=style.READOUT, zorder=3)
        ax.scatter([a1], [y], s=28, marker="o", color=style.PROBE, zorder=3)
        gap = round(a1 - e + 1e-9, 2)
        ax.text(
            (e + a1) / 2 if gap >= 0.005 else a1 - 0.012,
            y + 0.16,
            f"{gap:.2f}",
            ha="center" if gap >= 0.005 else "right",
            va="bottom",
            fontsize=7,
            color="0.3",
        )
    ax.axvline(0.5, color=style.CHANCE, lw=0.7, ls=":", zorder=0)
    ax.text(0.515, -0.55, "chance", ha="left", va="bottom", fontsize=7, color="0.45")
    ax.set_yticks(ys, [name for name, _, _, _ in ladder])
    ax.set_ylim(-0.6, len(ladder) - 0.3)
    ax.set_xlim(0.4, 1.06)
    ax.set_xlabel("contrast accuracy")
    ax.grid(axis="x", color="0.9", lw=0.6)
    handles = [
        ax.scatter([], [], s=28, marker="o", color=style.PROBE, label=r"availability $A_1$"),
        ax.scatter([], [], s=26, marker="s", color=style.READOUT, label=r"native expression $E$"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.1)
    style.panel_title(ax, "b", "At the end of training")


def main():
    style.apply()
    fig = plt.figure(figsize=(style.TEXT_WIDTH, 1.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.42)
    plot_timeline(fig.add_subplot(gs[0, 0]), load_timeline())
    plot_ladder(fig.add_subplot(gs[0, 1]), load_ladder())
    fig.subplots_adjust(left=0.09, right=0.995, top=0.89, bottom=0.19)
    style.save(fig, out_path("sva_availability"))


if __name__ == "__main__":
    main()
