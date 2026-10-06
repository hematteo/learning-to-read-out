"""Aligned readout swaps in Pythia-1B (fig:app-aligned-readout-swap).

NLL of hidden states from one checkpoint read out through W_U from every other checkpoint, under five alignments of
the swapped-in readout (data/aligned_swap_summary_1b.csv). "none" (raw swap) equals "mean" by construction and is
drawn dashed underneath it.
"""

import matplotlib.pyplot as plt
import pandas as pd
from _common import DATA, out_path

H_EVALS = [256, 512, 1000, 2000, 143000]
ALIGNMENTS = ["none", "mean", "scale", "row_norm", "procrustes"]
COLORS = {
    "none": "#444444",
    "mean": "#1f77b4",
    "scale": "#ff7f0e",
    "row_norm": "#2ca02c",
    "procrustes": "#d62728",
}
TICKS = [0, 10, 1000, 100000]
TICKLABELS = ["0", "10", "1k", "100k"]
MINOR_TICKS = [1, 100, 10000]
TRIM_PT = 21  # blank top margin, removed by the paper's \includegraphics trim=0 0 0 21


def main() -> None:
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
    d = pd.read_csv(DATA / "aligned_swap_summary_1b.csv")
    width, body = 5.5, 3.45
    height = body + TRIM_PT / 72
    fig, axes = plt.subplots(2, 3, figsize=(width, height))
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.115 * body / height,
        top=body / height - 0.055,
        wspace=0.3,
        hspace=0.42,
    )
    for ax, h in zip(axes.flat, H_EVALS):
        for a in ALIGNMENTS:
            q = d[(d.alignment == a) & (d.h_eval_step == h)].sort_values("snapshot_step")
            dashed = a == "none"
            ax.plot(
                q.snapshot_step,
                q.nll,
                color=COLORS[a],
                label=a,
                ls=(0, (2.5, 1.5)) if dashed else "-",
                lw=1.5 if dashed else 0.9,
                marker=None if dashed else "o",
                ms=1.8,
                zorder=1.5 if dashed else 2,
            )
        ax.set_xscale("symlog", linthresh=1, linscale=0.4)
        ax.set_xticks(TICKS, TICKLABELS)
        ax.set_xticks(MINOR_TICKS, minor=True)
        ax.set_xlim(-0.15, 2.2e5)
        ax.axvline(1000, color="0.85", linewidth=0.7, zorder=0)  # step 1000 readout
        ax.set_title(f"hidden step = {h}", pad=3)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(axis="both", pad=1.5)
    leg_ax = axes.flat[-1]
    leg_ax.axis("off")
    leg_ax.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="center",
        title="alignment",
        title_fontsize=7.5,
        frameon=False,
        handlelength=2.4,
    )
    fig.supxlabel("swap-in W_U snapshot step", fontsize=7.5, y=0.01)
    fig.supylabel("NLL (nats / token)", fontsize=7.5, x=0.01, y=0.5 * body / height)
    out = out_path("aligned_swap_grid")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
