"""WordNet supersense probes on Pythia-160M readout rows against matched controls (fig:main-wordnet-matched-controls).

Reads the per-category, per-checkpoint summary (data/wordnet_matched_controls_160m.csv).
"""

import matplotlib.pyplot as plt
import pandas as pd
from _common import DATA, out_path

COLORS = {
    "observed": "#0072B2",
    "metadata": "#6F6F6F",
    "null": "#D55E00",
    "strict": "#004488",
    "joint": "#009E73",
    "phase": "#0072B2",
}
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
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.90", linewidth=0.7)
    ax.tick_params(axis="both", length=3.0, width=0.8, color="0.25")
    ax.set_xscale("symlog", linthresh=128)
    ax.set_xlim(-4, 320000)
    ax.set_xticks(TICKS, TICK_LABELS)
    ax.minorticks_off()
    ax.axvspan(128, 1000, color=COLORS["phase"], alpha=0.07, linewidth=0)
    ax.set_xlabel("training step")


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
        }
    )
    s = summarize(pd.read_csv(DATA / "wordnet_matched_controls_160m.csv"))
    x = s.step.to_numpy()
    fig, (ax_acc, ax_count) = plt.subplots(1, 2, figsize=(5.5, 1.85), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02, wspace=0.08)

    ax_acc.axhline(0.5, color="0.70", linestyle=(0, (1.4, 1.8)), linewidth=0.8)
    ax_acc.fill_between(
        x, s.median_null_p95_low, s.median_null_p95_high, color=COLORS["null"], alpha=0.18, linewidth=0, zorder=1
    )
    for col in ("median_null_p95_low", "median_null_p95_high"):
        ax_acc.plot(x, s[col], color=COLORS["null"], linestyle=(0, (4.0, 2.0)), lw=1.0, alpha=0.8, zorder=2)
    ax_acc.plot(x, s.median_observed_acc, color=COLORS["observed"], lw=2.0, zorder=4)
    ax_acc.plot(x, s.median_metadata_acc, color=COLORS["metadata"], lw=1.25, linestyle=(0, (2.4, 1.8)), zorder=3)
    style_axis(ax_acc)
    ax_acc.set_ylim(0.45, 0.94)
    ax_acc.set_ylabel("median balanced accuracy")
    ax_acc.set_title("A. Readout rows against controls", loc="left", pad=4)
    ax_acc.text(0, 0.495, "chance", fontsize=7.5, color="0.45", va="top")
    for col, text, color, dy in [
        ("median_observed_acc", "readout probe", COLORS["observed"], 4),
        ("median_metadata_acc", "metadata only", COLORS["metadata"], -3),
        ("median_null_p95_high", "null p95", COLORS["null"], 3),
    ]:
        ax_acc.annotate(
            text,
            xy=(float(x[-1]), float(s[col].iloc[-1])),
            xytext=(-2, dy),
            textcoords="offset points",
            ha="right",
            va="bottom" if dy > 0 else "top",
            fontsize=7.5,
            color=color,
        )

    ax_count.axhline(35, color="0.70", linestyle=(0, (1.4, 1.8)), linewidth=0.8)
    ax_count.plot(x, s.n_clear_null_p95, color=COLORS["observed"], lw=2.0, label="above both null p95")
    ax_count.plot(
        x,
        s.n_clear_all_null_draws,
        color=COLORS["strict"],
        lw=1.2,
        linestyle=(0, (3.0, 1.8)),
        label="p ≤ 0.05 vs both nulls",
    )
    ax_count.plot(x, s.n_clear_null_p95_and_metadata, color=COLORS["joint"], lw=1.35, label="above nulls and metadata")
    style_axis(ax_count)
    ax_count.set_ylim(0, 37)
    ax_count.set_yticks([0, 10, 20, 30, 35])
    ax_count.set_ylabel("categories (of 35)")
    ax_count.set_title("B. Categories clearing controls", loc="left", pad=4)
    ax_count.legend(frameon=False, loc="lower right", handlelength=2.2, borderpad=0.2, labelspacing=0.3)

    out = out_path("wordnet_controls")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
