"""WordNet supersense probes on Pythia-160M readout rows against matched controls (fig:main-wordnet-matched-controls).

Reads the per-category, per-checkpoint summary (data/wordnet_matched_controls_160m.csv). The sand band is the Pythia
reorganization window of Section 4.1 (as in Figure 2), the bracket the interval of the largest per-category gains
(steps 128 to 1000), and controls are gray.
"""

import _style as style
import matplotlib.pyplot as plt
import pandas as pd
from _common import DATA, out_path

TICKS = [0, 128, 1000, 14000, 143000]
TICK_LABELS = ["0", "128", "1k", "14k", "143k"]


def summarize(df):
    rows = []
    for step, g in df.groupby("step", sort=True):
        above_p95 = (g.observed_acc > g.matched_random_p95) & (g.observed_acc > g.label_shuffle_p95)
        above_all = (g.matched_random_p_ge_observed <= 0.05) & (g.label_shuffle_p_ge_observed <= 0.05)
        above_meta = g.observed_acc > g.metadata_acc
        nulls = pd.concat([g.matched_random_p95, g.label_shuffle_p95], axis=1)
        rows.append(
            {
                "step": int(step),
                "median_observed_acc": g.observed_acc.median(),
                "median_metadata_acc": g.metadata_acc.median(),
                "median_null_p95_low": nulls.min(axis=1).median(),
                "median_null_p95_high": nulls.max(axis=1).median(),
                "n_clear_null_p95": int(above_p95.sum()),
                "n_clear_all_null_draws": int(above_all.sum()),
                "n_clear_null_p95_and_metadata": int((above_p95 & above_meta).sum()),
            }
        )
    return pd.DataFrame(rows)


def style_axis(ax):
    ax.set_xscale("symlog", linthresh=128)
    ax.set_xlim(-4, 320000)
    ax.set_xticks(TICKS, TICK_LABELS)
    ax.minorticks_off()
    ax.axvspan(*style.PYTHIA_WINDOW, color=style.WINDOW, lw=0, zorder=0)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_xlabel("training step")


def main():
    style.apply()
    s = summarize(pd.read_csv(DATA / "wordnet_matched_controls_160m.csv"))
    x = s.step.to_numpy()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(style.TEXT_WIDTH, 1.65))

    style_axis(ax_a)
    ax_a.axhline(0.5, color=style.CHANCE, ls=":", lw=0.7)
    ax_a.text(300000, 0.497, "chance", fontsize=7, color="0.45", va="top", ha="right")
    ax_a.fill_between(x, s.median_null_p95_low, s.median_null_p95_high, color=style.CONTROL_LIGHT, lw=0, zorder=1)
    ax_a.plot(x, s.median_null_p95_high, color=style.CONTROL, lw=0.6, zorder=2)
    ax_a.plot(x, s.median_metadata_acc, color="0.25", lw=1.1, ls=(0, (2.4, 1.6)), zorder=3)
    ax_a.plot(x, s.median_observed_acc, color=style.READOUT, lw=1.8, zorder=4)
    for col, text, color in [
        ("median_observed_acc", "readout rows", style.READOUT),
        ("median_metadata_acc", "metadata only", "0.25"),
    ]:
        ax_a.annotate(
            text,
            (float(x[-1]), float(s[col].iloc[-1])),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color=color,
        )
    ax_a.text(4, 0.535, "null p95", fontsize=7, color="0.45", va="bottom")
    # Interval of the largest per-category gains (33 of 35 categories, Section 4.2).
    ax_a.annotate(
        "", (128, 0.915), (1000, 0.915), arrowprops=dict(arrowstyle="|-|", lw=0.7, color="0.35", mutation_scale=2.5)
    )
    ax_a.text(358, 0.928, "largest gains", fontsize=7, color="0.35", ha="center", va="bottom")
    ax_a.set_ylim(0.45, 0.97)
    ax_a.set_ylabel("median balanced accuracy")
    style.panel_title(ax_a, "a", "Pythia-160M readout rows vs controls")

    style_axis(ax_b)
    ax_b.axhline(35, color=style.CHANCE, ls=":", lw=0.7)
    series = [
        ("n_clear_null_p95", "above both null p95", style.READOUT, "-", 1.8),
        ("n_clear_all_null_draws", "p ≤ 0.05 vs both nulls", style.NAVY, (0, (3, 1.6)), 1.1),
        ("n_clear_null_p95_and_metadata", "above nulls and metadata", "0.25", "-", 1.1),
    ]
    for col, _, color, ls, lw in series:
        ax_b.plot(x, s[col], color=color, ls=ls, lw=lw)
    ax_b.annotate(
        "window",
        (style.PYTHIA_WINDOW[0], 2),
        xytext=(2, 0),
        textcoords="offset points",
        fontsize=7,
        color="0.45",
        ha="left",
        va="bottom",
    )
    # Direct labels in the empty lower right, each with a short line sample.
    for i, (_, label, color, ls, lw) in enumerate(series):
        y = 19 - 5.2 * i
        ax_b.plot([5000, 11000], [y, y], color=color, ls=ls, lw=lw)
        ax_b.text(13500, y, label, fontsize=7, color=color, va="center")
    ax_b.set_ylim(0, 37)
    ax_b.set_yticks([0, 10, 20, 30, 35])
    ax_b.set_ylabel("categories (of 35)")
    style.panel_title(ax_b, "b", "Categories clearing controls")

    fig.subplots_adjust(wspace=0.3, left=0.09, right=0.99)
    style.save(fig, out_path("wordnet_controls"))


if __name__ == "__main__":
    main()
