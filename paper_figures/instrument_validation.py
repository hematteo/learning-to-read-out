"""Instrument validation summary across models (fig:main-multimodel-validation).

A: aggregate explained variance against the share of active features for the selected dictionaries and reference
fits. B, C: per-checkpoint explained variance and active-feature share. Inputs: data/instrument_inventory.csv,
data/persnap_fidelity_160m.csv, data/persnap_fidelity_1b_d24576.csv, data/large_dictionary_summary.csv and
data/large_dictionary_per_snapshot.csv.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from _common import DATA, out_path
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

C_160M = "#8ecae6"
C_1B = "#2f5aa8"
C_69B = "#d04a3a"
C_OLMO = "#348b42"
C_DENSE = "#a04d44"
C_BASE = "#9b9b9b"
C_CROSS = "#3366cc"
C_SAE = "#ff5a4f"
PANEL_ASPECT = 3 / 4


def _read_csv(name: str) -> list[dict]:
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def _read_inventory() -> list[dict]:
    rows: list[dict] = []
    for r in _read_csv("instrument_inventory.csv"):
        if not r["EV"] or not r["mean_L0"] or not r["d_sae"]:
            continue
        d = int(r["d_sae"])
        ev = float(r["EV"])
        l0 = float(r["mean_L0"])
        rows.append(
            {
                "model": r["model"],
                "matrix": r["matrix"],
                "family": r["family"],
                "d_sae": d,
                "seed": r["seed"],
                "ev": ev,
                "mean_l0": l0,
                "pct_active": 100.0 * l0 / d,
                "dead_rate": float(r["dead_rate"]),
            }
        )
    return rows


def _mean_point(rows: list[dict], *, model: str, matrix: str, family: str, d_sae: int) -> dict:
    group = [
        r
        for r in rows
        if r["model"] == model and r["matrix"] == matrix and r["family"] == family and r["d_sae"] == d_sae
    ]
    if not group:
        raise ValueError((model, matrix, family, d_sae))
    return {
        "model": model,
        "family": family,
        "d_sae": d_sae,
        "ev": float(np.mean([r["ev"] for r in group])),
        "mean_l0": float(np.mean([r["mean_l0"] for r in group])),
        "pct_active": float(np.mean([r["pct_active"] for r in group])),
        "dead_rate": float(np.mean([r["dead_rate"] for r in group])),
    }


def _large_point(stem: str, *, model: str, label: str, selected: bool) -> dict:
    (j,) = [r for r in _read_csv("large_dictionary_summary.csv") if r["stem"] == stem]
    d = int(j["D"])
    return {
        "model": model,
        "label": label,
        "d_sae": d,
        "ev": float(j["canon_ev"]),
        "mean_l0": float(j["canon_l0"]),
        "pct_active": 100.0 * float(j["canon_l0"]) / d,
        "dead_rate": float(j["canon_dead_rate"]),
        "selected": selected,
    }


def _read_1b_persnap() -> dict:
    steps, ev, l0 = [], [], []
    for r in _read_csv("persnap_fidelity_1b_d24576.csv"):
        steps.append(int(r["step"]))
        ev.append(float(r["ev"]))
        l0.append(float(r["l0"]))
    return {
        "key": "pythia_1b",
        "label": "Pythia-1B selected",
        "steps": steps,
        "ev": ev,
        "l0": l0,
        "d_sae": 24576,
        "color": C_1B,
        "marker": "o",
        "alpha": 0.28,
        "lw": 0.55,
    }


def _large_persnap(stem: str, *, key: str, label: str, color: str) -> dict:
    (summary,) = [r for r in _read_csv("large_dictionary_summary.csv") if r["stem"] == stem]
    rows = [r for r in _read_csv("large_dictionary_per_snapshot.csv") if r["stem"] == stem]
    return {
        "key": key,
        "label": label,
        "steps": [int(r["step"]) for r in rows],
        "ev": [float(r["ev"]) for r in rows],
        "l0": [float(r["l0"]) for r in rows],
        "d_sae": int(summary["D"]),
        "color": color,
        "marker": "o",
        "alpha": 0.28,
        "lw": 0.55,
    }


def _read_160m_comparison() -> list[dict]:
    rows_by_family: dict[str, list[dict]] = {}
    for r in _read_csv("persnap_fidelity_160m.csv"):
        rows_by_family.setdefault(r["family"], []).append(
            {"step": int(r["step"]), "ev": float(r["ev"]), "l0": float(r["l0"])}
        )
    series = []
    specs = [
        ("cross-snap-32 d=8192 (W_U)", "160M cross-snap CC", C_CROSS, "o"),
        ("per-snap d=8192", "160M per-snap SAE", C_SAE, "s"),
    ]
    for family, label, color, marker in specs:
        rows = sorted(rows_by_family[family], key=lambda r: r["step"])
        series.append(
            {
                "key": family,
                "label": label,
                "steps": [r["step"] for r in rows],
                "ev": [r["ev"] for r in rows],
                "l0": [r["l0"] for r in rows],
                "d_sae": 8192,
                "color": color,
                "marker": marker,
                "alpha": 0.98,
                "lw": 0.95,
            }
        )
    return series


def _read_trajectory_series() -> list[dict]:
    s_69b = _large_persnap("p69b_sparse_d32768_seed0", key="p69b_sparse", label="Pythia-6.9B selected", color=C_69B)
    s_olmo = _large_persnap("olmo27b_d32768_seed0", key="olmo27b", label="OLMo-2-7B selected", color=C_OLMO)
    return _read_160m_comparison() + [_read_1b_persnap(), s_69b, s_olmo]


def build_points() -> tuple[list[dict], list[dict]]:
    rows = _read_inventory()
    selected = [
        {
            **_mean_point(rows, model="pythia-160m", matrix="W_U", family="cross-snap-32", d_sae=24576),
            "label": "Pythia-160M",
            "selected": True,
        },
        {
            **_mean_point(rows, model="pythia-1b", matrix="W_U", family="cross-snap-32", d_sae=24576),
            "label": "Pythia-1B",
            "selected": True,
        },
        _large_point("p69b_sparse_d32768_seed0", model="pythia-6.9b", label="Pythia-6.9B", selected=True),
        _large_point("olmo27b_d32768_seed0", model="OLMo-2-7B", label="OLMo-2-7B", selected=True),
    ]

    reference: list[dict] = []
    for d in (8192, 16384, 24576):
        p = _mean_point(rows, model="pythia-1b", matrix="W_U", family="cross-snap-32", d_sae=d)
        reference.append({**p, "label": f"1B width sweep d={d:,}", "kind": "1b_width"})
    reference.append(
        {
            **_mean_point(rows, model="pythia-160m", matrix="W_U", family="cross-snap-32", d_sae=8192),
            "label": "160M cross-snap CC d=8192",
            "kind": "160m_cc_8192",
        }
    )
    for r in rows:
        if r["model"] == "pythia-160m" and r["matrix"] == "W_U" and r["family"] in {"final-snap-SAE", "per-snap-SAE"}:
            reference.append({**r, "label": "160M single-snapshot SAE", "kind": "sae"})
    reference.append(
        {
            **_large_point(
                "p69b_normal_d32768_seed0", model="pythia-6.9b", label="6.9B denser capacity point", selected=False
            ),
            "kind": "dense_69b",
        }
    )
    return selected, reference


def _plot_aggregate(ax, selected: list[dict], reference: list[dict]) -> None:
    sae = [r for r in reference if r.get("kind") == "sae"]
    if sae:
        ax.scatter(
            [r["pct_active"] for r in sae],
            [r["ev"] for r in sae],
            s=13,
            marker="o",
            facecolor="none",
            edgecolor=C_BASE,
            lw=0.65,
            alpha=0.62,
            zorder=1,
        )

    sweep = sorted([r for r in reference if r.get("kind") == "1b_width"], key=lambda r: r["d_sae"])
    if sweep:
        ax.plot([r["pct_active"] for r in sweep], [r["ev"] for r in sweep], color=C_1B, lw=0.7, alpha=0.45, zorder=2)
        ax.scatter(
            [r["pct_active"] for r in sweep],
            [r["ev"] for r in sweep],
            s=22,
            marker="s",
            facecolor="white",
            edgecolor=C_1B,
            lw=0.65,
            alpha=0.9,
            zorder=3,
        )

    dense = [r for r in reference if r.get("kind") == "dense_69b"]
    if dense:
        ax.scatter(
            [dense[0]["pct_active"]],
            [dense[0]["ev"]],
            s=35,
            marker="^",
            facecolor="white",
            edgecolor=C_DENSE,
            lw=0.75,
            zorder=4,
        )

    cc_160m = [r for r in reference if r.get("kind") == "160m_cc_8192"]
    if cc_160m:
        p = cc_160m[0]
        ax.scatter(
            [p["pct_active"]], [p["ev"]], s=34, marker="o", facecolor=C_CROSS, edgecolor="black", lw=0.55, zorder=6
        )

    ps_sae = [r for r in sae if r.get("family") == "per-snap-SAE" and r.get("d_sae") == 8192]
    if ps_sae:
        x = float(np.median([r["pct_active"] for r in ps_sae]))
        y = float(np.median([r["ev"] for r in ps_sae]))
        ax.scatter([x], [y], s=26, marker="s", facecolor=C_SAE, edgecolor="white", lw=0.5, zorder=6)

    markers = {
        "pythia-160m": ("o", C_160M, "160M"),
        "pythia-1b": ("s", C_1B, "1B"),
        "pythia-6.9b": ("^", C_69B, "6.9B"),
        "OLMo-2-7B": ("p", C_OLMO, "OLMo"),
    }
    offsets = {
        "pythia-160m": (0, 6, "center"),
        "pythia-1b": (5, 4, "left"),
        "pythia-6.9b": (5, -1, "left"),
        "OLMo-2-7B": (-5, -7, "right"),
    }
    for p in selected:
        marker, color, short = markers[p["model"]]
        ax.scatter(
            [p["pct_active"]], [p["ev"]], s=46, marker=marker, facecolor=color, edgecolor="black", lw=0.65, zorder=5
        )
        dx, dy, ha = offsets[p["model"]]
        ax.annotate(
            short,
            (p["pct_active"], p["ev"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            color=color,
            ha=ha,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Active features (% of $d_{\\mathrm{SAE}}$)")
    ax.set_ylabel("Aggregate EV")
    ax.grid(color="#d7d7d7", lw=0.35, alpha=0.65)
    ax.set_xlim(0.55, 16)
    ax.set_ylim(0.56, 0.98)
    ax.set_box_aspect(PANEL_ASPECT)
    _clean_axes(ax)


def _plot_trajectory_comparison(ax, series: list[dict], *, metric: str) -> None:
    for s in series:
        x = np.maximum(np.array(s["steps"]), 1)
        y = np.array(s["ev"]) if metric == "ev" else 100.0 * np.array(s["l0"]) / s["d_sae"]
        is_selected_context = "selected" in s["label"]
        ax.plot(
            x,
            y,
            color=s["color"],
            lw=s.get("lw", 0.85),
            ls="--" if is_selected_context else "-",
            marker=s["marker"],
            ms=1.65 if is_selected_context else 2.0,
            markeredgewidth=0.0,
            alpha=s.get("alpha", 0.9),
            label=s["label"],
            zorder=2 if is_selected_context else 4,
        )
    ax.grid(color="#d7d7d7", lw=0.35, alpha=0.65)
    if metric == "ev":
        ax.set_ylabel("Per-snapshot EV")
    else:
        ax.set_ylabel("Active features (% of $d_{\\mathrm{SAE}}$)")
    ax.set_xlabel("Training step")
    ax.set_xscale("log")
    ax.set_xlim(0.85, 180000)
    ax.set_xticks([1, 32, 1000, 14000, 143000])
    ax.set_xticklabels(["0", "32", "1k", "14k", "143k"])
    ax.set_box_aspect(PANEL_ASPECT)
    _clean_axes(ax)


def _clean_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.55)
    ax.tick_params(colors="#222222", labelcolor="#222222", pad=2)


def _add_panel_labels(axes) -> None:
    for label, ax in zip(("A", "B", "C"), axes):
        ax.text(
            0.5,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="#111111",
        )


def _legend_handles() -> list[Line2D]:
    def selected(color, label):
        return Line2D([0], [0], color=color, marker="o", lw=0.55, ls="--", ms=2.1, alpha=0.35, label=label)

    return [
        Line2D([0], [0], color=C_CROSS, marker="o", lw=0.85, ms=2.4, label="160M cross-snap CC"),
        Line2D([0], [0], color=C_SAE, marker="s", lw=0.85, ms=2.4, label="160M per-snap SAE"),
        selected(C_1B, "Pythia-1B selected"),
        selected(C_69B, "Pythia-6.9B selected"),
        selected(C_OLMO, "OLMo-2-7B selected"),
        Line2D(
            [0], [0], color=C_BASE, marker="o", lw=0, ms=2.8, mfc="white", label="160M single-snapshot SAE baselines"
        ),
        Line2D([0], [0], color=C_1B, marker="s", lw=0.7, ms=2.8, mfc="white", label="1B width sweep"),
        Line2D([0], [0], color=C_DENSE, marker="^", lw=0, ms=3.3, mfc="white", label="6.9B denser capacity point"),
    ]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.minor.size": 1.7,
            "ytick.minor.size": 1.7,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            # 0.14 in padding so the paper's trim=7pt 9pt 7pt 8pt removes only white space.
            "savefig.pad_inches": 0.14,
        }
    )
    selected, reference = build_points()
    series = _read_trajectory_series()

    fig = plt.figure(figsize=(5.4, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.34, wspace=0.42)
    gs = fig.add_gridspec(1, 3)
    ax_agg = fig.add_subplot(gs[0, 0])
    ax_ev = fig.add_subplot(gs[0, 1])
    ax_pct = fig.add_subplot(gs[0, 2], sharex=ax_ev)

    _plot_aggregate(ax_agg, selected, reference)
    _plot_trajectory_comparison(ax_ev, series, metric="ev")
    _plot_trajectory_comparison(ax_pct, series, metric="pct")
    ax_ev.set_ylim(0.70, 1.00)
    ax_pct.set_ylim(0.0, 15.0)
    ax_agg.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_ev.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_pct.yaxis.set_major_locator(MultipleLocator(2.5))
    ax_ev.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_pct.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    _add_panel_labels((ax_agg, ax_ev, ax_pct))

    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.8,
    )
    out = out_path("instrument_validation")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
