"""Reorganization window of the 31M recipe-control readouts (fig:app-recipe-control-reorg).

Norm turnover (mean |delta rho_f|) and decoder rotation (mean 1 - cos) between adjacent checkpoints of each
condition's seed-0 trajectory crosscoder (data/recipe_reorg_window_seed0.csv), at the geometric midpoint of each pair.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from _common import DATA, out_path

COND_ORDER = ["wu_lr_0p25", "baseline", "wu_lr_4x", "warmup_short"]
COND_SHORT = {
    "wu_lr_0p25": "wu_lr 0.25x",
    "baseline": "baseline (1x)",
    "wu_lr_4x": "wu_lr 4x",
    "warmup_short": "warmup_short (1x)",
}
COND_COLOR = {
    "wu_lr_0p25": "#1f77b4",
    "baseline": "#2ca02c",
    "wu_lr_4x": "#d62728",
    "warmup_short": "#9467bd",
}
STEP_TICKS = [1, 16, 256, 4096]
STEP_TICKLABELS = ["1", "16", "256", "4096"]


def step_x(steps):
    """Plot positions for a log step axis; step 0 -> 0.7 (just left of 1)."""
    x = np.asarray(steps, dtype=float).copy()
    x[x <= 0] = 0.7
    return x


def set_step_axis(ax, label):
    ax.set_xscale("log")
    ax.set_xticks(STEP_TICKS)
    ax.set_xticklabels(STEP_TICKLABELS)
    ax.set_xlabel(label)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
        }
    )
    reorg = pd.read_csv(DATA / "recipe_reorg_window_seed0.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 2.35), sharex=True)
    for cond in COND_ORDER:
        r = reorg[reorg.condition == cond].sort_values("pair_idx")
        pair_x = np.sqrt(step_x(r.step_lo) * step_x(r.step_hi))  # geometric midpoint
        style = dict(color=COND_COLOR[cond], lw=1.2, marker="o", ms=2.2)
        axes[0].plot(pair_x, r.norm_turnover, label=COND_SHORT[cond], **style)
        axes[1].plot(pair_x, r.decoder_rotation, **style)
    axes[0].set_ylabel(r"norm turnover $\overline{|\Delta\rho_f|}$")
    axes[1].set_ylabel(r"decoder rotation $\overline{1-\cos}$")
    axes[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    for ax in axes:
        set_step_axis(ax, "training step (pair midpoint)")
        ax.tick_params(axis="both", pad=1.5)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87], w_pad=1.0)
    out = out_path("recipe_reorganization_window")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
