"""Population lifecycle diagnostics (fig:app-selected-population-lifecycle-diagnostics).

Row A: fraction of active features peaking at each checkpoint (unweighted). Row B: total decoder-norm mass of the
active features over training, divided by its maximum. Reads peak_idx and mass from data/lifecycle_<run>.npz.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import DATA, out_path

RUNS = [
    ("pythia160m_d24576", "Pythia-160M"),
    ("pythia1b_d24576", "Pythia-1B"),
    ("pythia69b_d32768", "Pythia-6.9B"),
    ("olmo2_7b_d32768", "OLMo-2-7B"),
]


def summarize(key):
    z = np.load(DATA / f"lifecycle_{key}.npz")
    steps = z["steps"].astype(np.int64)
    peak_step = steps[z["peak_idx"].astype(np.int64)]  # (active features,)
    unique_steps = np.unique(peak_step)
    count_fraction = np.array([(peak_step == step).mean() for step in unique_steps], dtype=np.float32)
    total_mass = z["mass"]  # (steps,) summed decoder norm of the active features
    mass_normalized = total_mass / np.clip(total_mass.max(), 1e-12, None)
    return steps, unique_steps, count_fraction, mass_normalized


def set_log_ticks(ax, key):
    if key.startswith("olmo"):
        ticks = [1000, 14000, 110000, 928000]
        labels = ["1k", "14k", "110k", "928k"]
        ax.set_xlim(120, 1_150_000)
    else:
        ticks = [1, 1000, 14000, 143000]
        labels = ["1", "1k", "14k", "143k"]
        ax.set_xlim(0.8, 180_000)
    ax.set_xticks([t + 1 for t in ticks])
    ax.set_xticklabels(labels)


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
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(5.38, 2.8), sharey="row")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.9, hspace=0.5, wspace=0.18)
    for col, (key, label) in enumerate(RUNS):
        steps, unique_steps, count_fraction, mass_normalized = summarize(key)

        ax_peak = axes[0, col]
        x = unique_steps.astype(np.float64) + 1.0
        ax_peak.bar(x, count_fraction, width=x * 0.22, align="center", color="#2a9d8f", alpha=0.78, linewidth=0)
        ax_peak.set_xscale("log")
        ax_peak.set_ylim(0, 0.42)
        ax_peak.set_title(label, loc="left", pad=3)
        ax_peak.grid(axis="y", alpha=0.2)
        set_log_ticks(ax_peak, key)

        ax_mass = axes[1, col]
        mass_x = steps.astype(np.float64) + 1.0
        ax_mass.plot(mass_x, mass_normalized, color="#264653", lw=1.2)
        ax_mass.fill_between(mass_x, 0, mass_normalized, color="#264653", alpha=0.12, linewidth=0)
        ax_mass.set_xscale("log")
        ax_mass.set_ylim(0, 1.05)
        ax_mass.grid(axis="y", alpha=0.2)
        set_log_ticks(ax_mass, key)
        ax_peak.set_yticks(np.arange(0, 0.41, 0.1))
        ax_mass.set_yticks(np.arange(0, 1.01, 0.25))
        ax_mass.yaxis.set_major_formatter("{x:.2f}")
        for ax in (ax_peak, ax_mass):
            ax.tick_params(axis="both", pad=1.5)
            # 35 degree labels: upright 14k/143k and 110k/928k overlap at print width
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    for ax, letter in ((axes[0, 0], "A"), (axes[1, 0], "B")):
        ax.text(-0.42, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
    axes[0, 0].set_ylabel("peak-step\nfeature fraction")
    axes[1, 0].set_ylabel("total norm mass\n/ model max")
    fig.supxlabel("training step", y=0.045, fontsize=7.5)
    out = out_path("lifecycle_population_diagnostics")
    fig.savefig(out, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
