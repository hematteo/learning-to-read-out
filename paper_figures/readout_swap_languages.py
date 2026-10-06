"""Language-stratified readout swap, Pythia-1B (fig:app-readout-swap-english-aggregate, fig:app-readout-swap-languages).

Draws two figures: the token-weighted aggregate next to English only (readout_swap_aggregate.pdf), and one
panel per language (readout_swap_languages.pdf).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import DATA, out_path

LANG_ORDER = ["en", "ru", "zh", "ja", "th", "ar", "hi", "ko", "bn"]
LANG_NAMES = ["English", "Russian", "Chinese", "Japanese", "Thai", "Arabic", "Hindi", "Korean", "Bengali"]
H_STEPS = [256, 512, 1000]
COLORS = ["#3977b7", "#d8712b", "#389269"]
STYLES = [("o", "-"), ("s", "--"), ("^", ":")]
YLIM = (-2.5, 2.8)


def swap_panel(ax, d, lang, title):
    # Gain is native NLL minus swapped NLL, so positive values favor the swapped readout; bands are the 95%
    # block-bootstrap intervals; open circles mark each curve's best readout.
    for h, col, (mk, ls) in zip(H_STEPS, COLORS, STYLES):
        q = d[(d.language == lang) & (d.h_eval_step == h) & (d.snapshot_step >= 128)].sort_values("snapshot_step")
        s = q.snapshot_step.to_numpy()
        g = -q.delta_nll.to_numpy()
        ax.plot(s, g, color=col, lw=1.3, ls=ls, marker=mk, ms=2.2, label=f"$t={h}$")
        ax.fill_between(s, -q.delta_nll_ci_high.to_numpy(), -q.delta_nll_ci_low.to_numpy(), color=col, alpha=0.25, lw=0)
        k = int(np.argmax(g))
        ax.plot(s[k], g[k], marker="o", ms=5, mfc="white", mec=col, mew=1.3, zorder=5)
    ax.axhline(0, color="0.45", ls=":", lw=0.8)
    ax.set(
        xscale="log",
        ylim=YLIM,
        xticks=[128, 512, 2000, 14000, 143000],
        xticklabels=["128", "512", "2k", "14k", "143k"],
    )
    ax.set_title(title, fontsize=8)
    ax.minorticks_off()
    ax.tick_params(labelsize=7)


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
    d = pd.read_csv(DATA / "language_swap_summary.csv")

    fig, axes = plt.subplots(3, 3, figsize=(5.5, 4.55), sharex=True, sharey=True)
    for ax, lang, name in zip(axes.flat, LANG_ORDER, LANG_NAMES):
        swap_panel(ax, d, lang, f"{name} ({int(d[d.language == lang].n_tokens.iloc[0]):,} targets)")
    fig.supylabel("NLL gain over native (nats)", fontsize=7.5, x=0.015)
    for ax in axes[-1]:
        ax.set_xlabel("readout checkpoint $s$", fontsize=7.5)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.5,
        title="hidden state checkpoint",
        title_fontsize=7.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.15, hspace=0.45, wspace=0.12)
    out = out_path("readout_swap_languages")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharex=True, sharey=True)
    for ax, lang, title in zip(axes, ["all", "en"], ["Token-weighted aggregate, nine languages", "English only"]):
        swap_panel(ax, d, lang, title)
        ax.set_xlabel("readout checkpoint $s$", fontsize=7.5)
    axes[0].set_ylabel("NLL gain over native (nats)", fontsize=7.5)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.5,
        title="hidden state checkpoint",
        title_fontsize=7.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.42, wspace=0.1)
    out = out_path("readout_swap_aggregate")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
