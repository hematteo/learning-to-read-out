"""Two single-direction availability probes on Pythia-6.9B hidden states (fig:app-ae-capacity).

A_1 is the shrinkage LDA direction and A_full the logistic regression probe; both define one direction for a binary
contrast. Values at the checkpoint nearest step 1000 (data/availability_probe_summary_69b.csv).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import DATA, out_path

FAMILIES = [
    ("sva", "SVA simple"),
    ("sva_across_pp", "SVA across-PP"),
    ("sva_subject_rc", "SVA subject-RC"),
    ("sva_object_rc", "SVA object-RC"),
    ("ioi_role_balanced", "IOI"),
    ("numeric_gt_v2", "numeric >"),
    ("numeric_lt_v2", "numeric <"),
    ("numeric_tf_v2", "numeric T/F"),
]
STEP = 1000


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
        }
    )
    d = pd.read_csv(DATA / "availability_probe_summary_69b.csv")
    a1, afull, labels = [], [], []
    for key, label in FAMILIES:
        g = d[d.family == key]
        row = g.iloc[(g.h_step - STEP).abs().argmin()]
        a1.append(row.probe_acc_1d)
        afull.append(row.probe_acc)
        labels.append(label)
    x = np.arange(len(FAMILIES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 2.2), constrained_layout=True)
    ax.bar(x - w / 2, a1, w, color="#1f77b4", label=r"$A_1$ (shrinkage LDA direction)")
    ax.bar(x + w / 2, afull, w, color="#9ecae1", label=r"$A_{\mathrm{full}}$ (logistic regression)")
    ax.axhline(0.5, color="0.3", lw=0.7, ls=":")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0.4, 1.04)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
    out = out_path("probe_fits")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    for label, u, v in zip(labels, a1, afull):
        print(f"{label:16s} A1 {u:.3f}  Afull {v:.3f}")


if __name__ == "__main__":
    main()
