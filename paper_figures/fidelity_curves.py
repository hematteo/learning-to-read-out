"""Readout functional fidelity of the reconstructed W_U per checkpoint (fig:app-fidelity-curves).

Matrix EV, logit R^2, KL(native || reconstructed) and top-1 agreement on held-out hidden states for the selected
Pythia dictionaries (data/fidelity_curves.csv).
"""

import _style as style
import matplotlib.pyplot as plt
import pandas as pd
from _common import DATA, out_path
from matplotlib.ticker import MultipleLocator

MODELS = [
    ("160m", "Pythia-160M d=8192", style.MODEL["160m"]),
    ("1b", "Pythia-1B d=24576", style.MODEL["1b"]),
    ("69b", "Pythia-6.9B d=32768", style.MODEL["6.9b"]),
]
PANELS = [
    ("matrix_ev", "matrix EV"),
    ("logit_r2", r"logit R$^2$"),
    ("kl_native_to_recon", "KL(native ‖ recon)"),
    ("top1_agreement", "top-1 agreement"),
]


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
    curves = pd.read_csv(DATA / "fidelity_curves.csv")
    dfs = {key: curves[curves.run == key].sort_values("step") for key, _, _ in MODELS}
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.95))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.1, wspace=0.25, hspace=0.42)
    for ax, (col, title) in zip(axes.flat, PANELS):
        for key, label, color in MODELS:
            d = dfs[key]
            ax.plot(d.step, d[col], marker="o", ms=2.2, lw=1.1, color=color, label=label)
        ax.set_xscale("symlog", linthresh=10, linscale=1)
        ax.set_title(title, pad=3)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", pad=1.5)
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(0.025))
    axes[0, 1].yaxis.set_major_locator(MultipleLocator(0.05))
    fig.supxlabel("checkpoint step", fontsize=7.5, y=0.01)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.53, 1.0),
        handlelength=2.0,
        columnspacing=1.6,
    )
    out = out_path("fidelity_curves")
    # 0.12 in padding so the paper's trim=7pt 7pt 6pt 6pt removes only white space.
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
