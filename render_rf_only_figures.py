"""RF-only Varianten der Figures für die Hausarbeit.

Liest die vorhandenen CSVs aus results/tables/ und schreibt zwei kompakte
RF-only Abbildungen direkt in 04_Ausarbeitung/LaTeX/Abbildungen/.
Kein Experiment-Rerun nötig.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
TABLES = HERE / "results" / "tables"
OUT = HERE.parent / "04_Ausarbeitung" / "LaTeX" / "Abbildungen"
OUT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def fig_fidelity_rf() -> None:
    df = pd.read_csv(TABLES / "fidelity_by_k.csv")
    df = df[df["Model"] == "RF"].sort_values("k")

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.plot(df["k"], df["SHAP_mean"], "-o", color="#1f77b4", linewidth=2, markersize=7, label="SHAP (TreeSHAP)")
    ax.plot(df["k"], df["LIME_mean"], "-s", color="#d62728", linewidth=2, markersize=7, label="LIME")
    ax.set_xlabel(r"$k$ (Anzahl ersetzter Top-Features)")
    ax.set_ylabel(r"Mittleres $|\Delta p_{\mathrm{malware}}|$")
    ax.set_xticks(df["k"].tolist())
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_fidelity_rf.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_jaccard_rf() -> None:
    # Die Jaccard-Verteilungswerte sind nicht als Rohdaten pro Instanz gespeichert,
    # wir nutzen jaccard_by_k.csv (aggregiert) und stellen k=3,5,10 als Balken dar.
    df = pd.read_csv(TABLES / "jaccard_by_k.csv")
    df = df[df["Model"] == "RF"].sort_values("k")

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    x = range(len(df))
    ax.bar([xi - 0.2 for xi in x], df["Mean Jaccard"], width=0.4, color="#1f77b4", label="Mittelwert")
    ax.bar([xi + 0.2 for xi in x], df["Median Jaccard"], width=0.4, color="#7fbfd6", label="Median")
    ax.errorbar(list(x), df["Mean Jaccard"], yerr=df["Std"], fmt="none", ecolor="black", capsize=4, linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"k = {k}" for k in df["k"]])
    ax.set_ylabel("Jaccard-Index (SHAP vs. LIME)")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_jaccard_rf.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_fidelity_rf()
    fig_jaccard_rf()
    print(f"RF-only Figures geschrieben nach {OUT}")
