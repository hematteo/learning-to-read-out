"""Readout learning rate shifts readout development (fig:main-recipe-dose-response).

Left: median feature peak step and peak norm-turnover step of the 31M recipe-control trajectory
crosscoders, three fits (seeds 0-2) per condition. Right: validation NLL of the four training runs.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import DATA, out_path

COLORS = ["#3977b7", "#444444", "#d8712b", "#8c56a2"]
CONDITIONS = ["wu_lr_0p25", "baseline", "wu_lr_4x", "warmup_short"]
LABELS = ["Readout LR 0.25×", "Baseline (1×)", "Readout LR 4×", "Short warmup (1×)"]


def main():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fits = pd.read_csv(DATA / "recipe_lifecycle_multiseed.csv")
    val = pd.read_csv(DATA / "recipe_val_loss.csv")
    # Drawn at its printed width (0.94 of the 5.5 in text width) so that text prints at 7.5-8 pt.
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 1.8), gridspec_kw={"width_ratios": [0.9, 1.1]})
    x = [0.25, 1, 4]
    for key, label, marker, color, ls in [
        ("median_peak_step", "Median feature peak step", "o", "#222222", "-"),
        ("reorg_peak_step", "Peak turnover step", "s", "#8a8a8a", "--"),
    ]:
        per_fit = [fits[fits.condition == c][key].to_numpy() for c in CONDITIONS]
        lo = [v.min() for v in per_fit[:3]]
        hi = [v.max() for v in per_fit[:3]]
        mid = [np.median(v) for v in per_fit[:3]]
        axs[0].plot(x, mid, color=color, lw=1.4, ls=ls, zorder=2)
        axs[0].vlines(x, lo, hi, color=color, lw=2.6, zorder=2)
        axs[0].scatter(x, mid, marker=marker, color=color, s=16, zorder=3, label=label)
        axs[0].scatter(
            [1], [np.median(per_fit[3])], marker="o", s=70, facecolor="none", edgecolor="#8c56a2", lw=1.3, zorder=4
        )
        for xx, a, b in zip(x, lo, hi):
            axs[0].annotate(
                f"{int(a)}" if a == b else f"{int(a)}–{int(b)}",
                (xx, b),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
            )
    axs[0].scatter([], [], marker="o", s=40, facecolor="none", edgecolor="#8c56a2", lw=1.3, label="Short warmup (1×)")
    axs[0].set(
        xscale="log",
        yscale="log",
        xlim=(0.17, 5.9),
        ylim=(12, 4000),
        xticks=x,
        xticklabels=["0.25×", "1×", "4×"],
    )
    axs[0].set_xlabel("Readout learning rate multiplier", fontsize=8)
    axs[0].set_ylabel("Training step", fontsize=8)
    axs[0].set_title("Readout timing, three fits per run", fontsize=8.5)
    axs[0].set_yticks([32, 128, 512, 2048], ["32", "128", "512", "2048"])
    axs[0].minorticks_off()
    axs[0].tick_params(labelsize=7.5)
    axs[0].annotate(
        "peak turnover", (4, 512), xytext=(0, -11), textcoords="offset points", ha="right", fontsize=7, color="#6f6f6f"
    )
    axs[0].annotate(
        "median feature peak",
        (4, 32),
        xytext=(0, -11),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color="#222222",
    )
    for c, label, col in zip(CONDITIONS, LABELS, COLORS):
        d = val[val.condition == c]
        axs[1].plot(d.tokens / 1e9, d.val_loss, color=col, label=label, lw=1.4, ls="--" if c == "warmup_short" else "-")
    axs[1].set(
        xscale="log",
        xlim=(0.05, 10.3),
        ylim=(3, 11),
        xticks=[0.1, 1, 10],
        xticklabels=["0.1", "1", "10"],
    )
    axs[1].set_xlabel("Training tokens (billions)", fontsize=8)
    axs[1].set_ylabel("Validation NLL (nats)", fontsize=8)
    axs[1].set_title("Early gains, similar final loss", fontsize=8.5)
    axs[1].tick_params(labelsize=7.5)
    axs[1].legend(loc="upper right", frameon=False, fontsize=7, handlelength=1.8)
    fig.subplots_adjust(wspace=0.38, bottom=0.2, top=0.88, left=0.1, right=0.99)
    out = out_path("recipe_dose_response")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
