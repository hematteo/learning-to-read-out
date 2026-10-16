"""Readout learning rate shifts readout development (fig:main-recipe-dose-response).

(a) Median feature peak step and peak norm-turnover step of the 31M recipe-control trajectory crosscoders, three
fits (seeds 0-2) per condition (data/recipe_lifecycle_multiseed.csv). Conditions are a blue ramp in readout LR; the
short-warmup run (1x LR) is the dashed open circle.
(b) Validation NLL minus the 1x baseline run's (data/recipe_val_loss.csv), with the baseline interpolated in log
tokens onto each run's evaluation grid (one mid-run evaluation differs by at most 15 steps).
"""

import _style as style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import DATA, out_path

CONDITIONS = ["wu_lr_0p25", "baseline", "wu_lr_4x"]


def main():
    style.apply()
    fits = pd.read_csv(DATA / "recipe_lifecycle_multiseed.csv")
    val = pd.read_csv(DATA / "recipe_val_loss.csv")
    ramp = dict(zip(CONDITIONS, style.BLUE_RAMP))
    x = [0.25, 1, 4]
    fig, (ax_t, ax_l) = plt.subplots(1, 2, figsize=(style.TEXT_WIDTH, 1.72), gridspec_kw={"width_ratios": [0.85, 1.15]})
    for key, label, marker, ls, label_xy in [
        ("reorg_peak_step", "peak turnover", "s", "--", (4, 512)),
        ("median_peak_step", "median feature peak", "o", "-", (4, 32)),
    ]:
        per_fit = [fits[fits.condition == c][key].to_numpy() for c in CONDITIONS]
        lo, hi = [v.min() for v in per_fit], [v.max() for v in per_fit]
        mid = [np.median(v) for v in per_fit]
        ax_t.plot(x, mid, color="0.35", lw=1.0, ls=ls, zorder=1)
        ax_t.vlines(x, lo, hi, color=[ramp[c] for c in CONDITIONS], lw=2.6, zorder=2)
        ax_t.scatter(x, mid, marker=marker, s=18, c=[ramp[c] for c in CONDITIONS], zorder=3)
        warm = np.median(fits[fits.condition == "warmup_short"][key])
        ax_t.scatter(
            [1], [warm], marker="o", s=70, facecolor="none", edgecolor=ramp["baseline"], lw=1.0, ls=(0, (2, 1.2)),
            zorder=4,
        )  # fmt: skip
        for xx, a, b in zip(x, lo, hi):
            ax_t.annotate(
                f"{int(a)}" if a == b else f"{int(a)}–{int(b)}",
                (xx, b),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax_t.annotate(label, label_xy, xytext=(0, -10), textcoords="offset points", ha="right", fontsize=7, color="0.3")
    ax_t.set(xscale="log", yscale="log", xlim=(0.17, 5.9), ylim=(12, 4500))
    ax_t.set_xticks(x, ["0.25×", "1×", "4×"])
    ax_t.set_yticks([32, 128, 512, 2048], ["32", "128", "512", "2048"])
    ax_t.minorticks_off()
    ax_t.set_xlabel("readout learning rate multiplier")
    ax_t.set_ylabel("training step")
    style.panel_title(ax_t, "a", "Readout timing, three fits per run")

    base = val[val.condition == "baseline"]
    lb, vb = np.log(base.tokens.to_numpy(float)), base.val_loss.to_numpy()
    ax_l.axhline(0, color=ramp["baseline"], lw=1.2)
    # (run, label, color, linestyle, label anchor index or extremum, offset in points, alignment)
    for cond, label, color, ls, at, off, ha in [
        ("wu_lr_0p25", "0.25×\nfinal", ramp["wu_lr_0p25"], "-", "max", (5, 0), "left"),
        ("wu_lr_4x", "4×, final", ramp["wu_lr_4x"], "-", "min", (0, -5), "center"),
        ("warmup_short", "1×, short warmup\nfinal", ramp["baseline"], (0, (3, 1.5)), 10, (0, -5), "center"),
    ]:
        d = val[val.condition == cond]
        t = d.tokens.to_numpy(float)
        delta = d.val_loss.to_numpy() - np.interp(np.log(t), lb, vb)
        ax_l.plot(t / 1e9, delta, color=color, lw=1.4, ls=ls, marker="o", ms=2.2)
        k = {"max": int(np.argmax(delta)), "min": int(np.argmin(delta))}.get(at, at)
        final = f"{delta[-1]:+.2f}".replace("-", "−")
        ax_l.annotate(
            f"{label} {final}",
            (t[k] / 1e9, delta[k]),
            xytext=off,
            textcoords="offset points",
            ha=ha,
            va="center" if off[1] == 0 else "top",
            fontsize=7,
            color=color,
            linespacing=1.0,
        )
    ax_l.set(xscale="log", xlim=(0.03, 10.3), ylim=(-1.75, 1.6))
    ax_l.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax_l.minorticks_off()
    ax_l.set_xlabel("training tokens (billions)")
    ax_l.set_ylabel("Δ NLL vs 1× baseline (nats)")
    ax_l.grid(axis="y", color="0.9", lw=0.6)
    style.panel_title(ax_l, "b", "Validation loss minus baseline")
    fig.subplots_adjust(wspace=0.36, left=0.1, right=0.99)
    style.save(fig, out_path("recipe_dose_response"))


if __name__ == "__main__":
    main()
