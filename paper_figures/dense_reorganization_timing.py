"""Dense reorganization timing of the Pythia readouts (fig:app-dense-reorg-pythia).

Left: chordal Grassmann distance between the top-10 right-singular subspaces of centered W_U at adjacent
checkpoints. Middle: the same for k = 5, 20, 50 up to step 9k. Right: mean best |overlap| of the top-20 singular
vectors between adjacent checkpoints. Inputs: data/dense_subspace_metrics.csv, data/dense_sv_stability.csv.
"""

import matplotlib.pyplot as plt
import pandas as pd
from _common import DATA, out_path

ORDER = ["pythia-160m", "pythia-1b", "pythia-6.9b"]
MODEL_COLORS = {"pythia-160m": "#2ca02c", "pythia-1b": "#1f77b4", "pythia-6.9b": "#d62728"}
MODEL_LINESTYLES = {"pythia-160m": "-", "pythia-1b": "--", "pythia-6.9b": "-."}
MODEL_LABELS = {"pythia-160m": "Pythia-160M", "pythia-1b": "Pythia-1B", "pythia-6.9b": "Pythia-6.9B"}


def format_step_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(["1", "10", "100", "1K", "10K", "100K"])
    ax.grid(alpha=0.25, linewidth=0.6)


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
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    subspace = pd.read_csv(DATA / "dense_subspace_metrics.csv")
    sv = pd.read_csv(DATA / "dense_sv_stability.csv")

    fig, axes = plt.subplots(1, 3, figsize=(5.41, 2.15), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.04)

    ax = axes[0]
    for model in ORDER:
        grp = subspace[(subspace["model"] == model) & (subspace["k"] == 10)].sort_values("to_step")
        ax.plot(
            grp["to_step"],
            grp["grassmann_chordal_norm"],
            linestyle=MODEL_LINESTYLES[model],
            lw=1.3,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_title("Pythia top-10 subspace drift")
    ax.set_ylabel("Chordal distance / sqrt(k)")
    ax.set_xlabel("Training step")
    ax.set_ylim(bottom=0)
    format_step_axis(ax)

    ax = axes[1]
    early = subspace[(subspace["to_step"] <= 9000) & (subspace["k"].isin([5, 20, 50]))]
    linestyles = {5: ":", 20: "-", 50: "--"}
    for model in ORDER:
        for k in [5, 20, 50]:
            grp = early[(early["model"] == model) & (early["k"] == k)].sort_values("to_step")
            ax.plot(
                grp["to_step"],
                grp["grassmann_chordal_norm"],
                linestyle=linestyles[k],
                lw=1.05,
                color=MODEL_COLORS[model],
                alpha=0.9 if k == 20 else 0.65,
            )
    for x in [64, 256, 1000]:
        ax.axvline(x, color="0.75", lw=0.7, zorder=0)
    ax.set_xlabel("Training step")
    ax.set_ylim(bottom=0)
    format_step_axis(ax)

    ax = axes[2]
    summary = (
        sv[sv["sv_index"] <= 20]
        .groupby(["model", "to_step"], as_index=False)["best_abs_overlap"]
        .mean()
        .sort_values("to_step")
    )
    for model in ORDER:
        grp = summary[summary["model"] == model].sort_values("to_step")
        ax.plot(
            grp["to_step"],
            grp["best_abs_overlap"],
            linestyle=MODEL_LINESTYLES[model],
            lw=1.3,
            color=MODEL_COLORS[model],
        )
    ax.set_title("Pythia top-20 vector stability")
    ax.set_ylabel("Mean best |overlap|")
    ax.set_xlabel("Training step")
    ax.set_ylim(0, 1.02)
    format_step_axis(ax)
    for ax in axes:
        ax.tick_params(axis="both", pad=1.5)

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.8,
    )
    out = out_path("dense_reorganization_timing")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
