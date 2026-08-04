"""Histogramm-Plot der per-instance Jaccard-Werte (RF, k=5).

Rechnet SHAP und LIME frisch fuer die 1000 Instanzen des XAI-Subsets,
berechnet den Jaccard-Index pro Instanz und speichert ein sauberes
Histogramm in ../04_Ausarbeitung/LaTeX/Abbildungen/fig4_jaccard_rf.pdf.

Dauert ca. 1 Minute (LIME auf 1000 Instanzen).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import shap

RANDOM_STATE = 42
XAI_SUBSET_PER_CLASS = 500
K = 5

HERE = Path(__file__).parent
DATA = HERE / "data" / "MalMem2022.csv"
TABLES = HERE / "results" / "tables"
OUT_FIG = HERE.parent / "04_Ausarbeitung" / "LaTeX" / "Abbildungen"
OUT_FIG.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA)
    non_feature_cols = ["Category", "Class"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols].select_dtypes(include="number").copy()
    y = (df["Class"] == "Malware").astype(int).values
    return X, y


def top_k_from_shap(shap_row, k):
    return set(np.argsort(-np.abs(shap_row))[:k])


def top_k_from_lime(explanation, feature_names, k):
    weights = np.zeros(len(feature_names))
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for name, w in explanation.as_list():
        for feat, idx in name_to_idx.items():
            if feat in name:
                weights[idx] += abs(w)
                break
    return set(np.argsort(-weights)[:k])


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    print("[1/5] Lade Datensatz")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      Train={len(X_train)}, Test={len(X_test)}")

    print("[2/5] Trainiere Random Forest")
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)

    print(f"[3/5] Waehle XAI-Subset (je {XAI_SUBSET_PER_CLASS}/Klasse)")
    rng = np.random.default_rng(RANDOM_STATE)
    idx_ben = np.where(y_test == 0)[0]
    idx_mal = np.where(y_test == 1)[0]
    sel = np.concatenate(
        [
            rng.choice(idx_ben, XAI_SUBSET_PER_CLASS, replace=False),
            rng.choice(idx_mal, XAI_SUBSET_PER_CLASS, replace=False),
        ]
    )
    X_sub = X_test.iloc[sel]

    print("[4/5] Berechne SHAP + LIME")
    explainer_shap = shap.TreeExplainer(clf)
    shap_vals_raw = explainer_shap.shap_values(X_sub)
    if isinstance(shap_vals_raw, list):
        shap_matrix = np.abs(shap_vals_raw[1])
    else:
        arr = np.asarray(shap_vals_raw)
        shap_matrix = np.abs(arr[..., 1]) if arr.ndim == 3 else np.abs(arr)

    explainer_lime = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X.columns),
        class_names=["benign", "malware"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    feature_names = list(X.columns)
    jaccards = np.zeros(len(X_sub))
    for i in range(len(X_sub)):
        if i % 100 == 0:
            print(f"      LIME instance {i}/{len(X_sub)}")
        exp = explainer_lime.explain_instance(
            X_sub.iloc[i].values, clf.predict_proba, num_features=len(feature_names), num_samples=2000
        )
        shap_top = top_k_from_shap(shap_matrix[i], K)
        lime_top = top_k_from_lime(exp, feature_names, K)
        jaccards[i] = jaccard(shap_top, lime_top)

    mean_j = float(np.mean(jaccards))
    print(f"      Jaccard mean={mean_j:.4f}, median={np.median(jaccards):.4f}")

    pd.DataFrame({"jaccard_top5": jaccards}).to_csv(
        TABLES / "jaccard_rf_per_instance.csv", index=False
    )

    print("[5/5] Rendere Histogramm")
    # Referenzwert aus dem konsolidierten experiment.py-Lauf (tab4_summary.csv)
    canonical_mean = float(pd.read_csv(TABLES / "jaccard_by_k.csv").query("Model=='RF' and k==5")["Mean Jaccard"].iloc[0])
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.hist(jaccards, bins=np.linspace(0, 1, 21), color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.axvline(canonical_mean, color="#d62728", linestyle="--", linewidth=2,
               label=f"Mittelwert = {canonical_mean:.3f}")
    ax.set_xlabel("Jaccard(SHAP, LIME) — Top-5")
    ax.set_ylabel("Anzahl Instanzen")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"fig4_jaccard_rf.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Geschrieben nach {OUT_FIG}")


if __name__ == "__main__":
    main()
