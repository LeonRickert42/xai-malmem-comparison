"""Vergleich der Erklärungsverfahren SHAP und LIME für zwei
baumbasierte Klassifikatoren (Random Forest, XGBoost) auf memory-basierten
Malware-Features (CIC-MalMem-2022).

Autor: Leon Rickert (HAW Hamburg, FW1, 2026)
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Feature-Namen-Warnungen aus LIME/sklearn unterdrücken (kein Effekt auf Ergebnisse)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=DataConversionWarning)

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42
DATA_PATH = Path("data/MalMem2022.csv")
FIG_DIR = Path("results/figures")
TAB_DIR = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

XAI_SUBSET_PER_CLASS = 500          # ergibt 1000 Instanzen für die XAI-Analyse
TOP_K_VALUES = [3, 5, 10]
FREQUENCY_K = 5                     # Fenster für die qualitative Feature-Häufigkeit
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 2000
ROBUSTNESS_N_RUNS = 5
ROBUSTNESS_SUBSET_SIZE = 30

# Farbschema für Grafiken
COLORS = {
    "RF": {"SHAP": "#003CA0", "LIME": "#0096D2"},
    "XGB": {"SHAP": "#B00020", "LIME": "#E76F51"},
}

# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #


def train_and_evaluate(clf, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return {
        "clf": clf,
        "train_time_s": train_time,
        "metrics": {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def _normalize_shap_row(sv):
    if isinstance(sv, list):
        row = sv[1]
        return row.squeeze(axis=0) if row.ndim > 1 else row
    if sv.ndim == 3:
        return sv[0, :, 1]
    return sv[0] if sv.ndim > 1 else sv


def compute_shap(clf, X_xai):
    explainer = shap.TreeExplainer(clf)
    times, rows = [], []
    for i in range(len(X_xai)):
        t0 = time.perf_counter()
        sv = explainer.shap_values(X_xai.iloc[[i]])
        times.append(time.perf_counter() - t0)
        rows.append(_normalize_shap_row(sv))
    return np.vstack(rows), times


def _new_lime_explainer(X_train, feature_names, seed=RANDOM_STATE):
    return LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=["Benign", "Malware"],
        discretize_continuous=True,
        random_state=seed,
        mode="classification",
    )


def compute_lime(clf, X_train, X_xai, feature_names):
    explainer = _new_lime_explainer(X_train, feature_names)
    times, weights = [], []
    for i in range(len(X_xai)):
        t0 = time.perf_counter()
        exp = explainer.explain_instance(
            X_xai.iloc[i].values,
            clf.predict_proba,
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
            labels=(1,),
        )
        times.append(time.perf_counter() - t0)
        weights.append(dict(exp.as_map()[1]))
    return weights, times


def top_k_from_shap(shap_row, k):
    return np.argsort(-np.abs(shap_row))[:k]


def top_k_from_lime(weights, k):
    items = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
    return np.array([idx for idx, _ in items[:k]])


def compute_fidelity(clf, X_xai, shap_vals, lime_weights, top_k_values, median_row):
    proba_baseline = clf.predict_proba(X_xai.values)[:, 1]
    per_k = {"SHAP": {}, "LIME": {}}
    for k in top_k_values:
        pert_s = X_xai.values.copy()
        pert_l = X_xai.values.copy()
        for i in range(len(X_xai)):
            top_s = top_k_from_shap(shap_vals[i], k)
            top_l = top_k_from_lime(lime_weights[i], k)
            pert_s[i, top_s] = median_row[top_s]
            pert_l[i, top_l] = median_row[top_l]
        per_k["SHAP"][k] = np.abs(proba_baseline - clf.predict_proba(pert_s)[:, 1])
        per_k["LIME"][k] = np.abs(proba_baseline - clf.predict_proba(pert_l)[:, 1])
    return per_k


def compute_lime_robustness(clf, X_train, X_xai, feature_names, subset_idx, k=5,
                            n_runs=ROBUSTNESS_N_RUNS):
    results = []
    for i in subset_idx:
        top_sets = []
        for r in range(n_runs):
            exp = _new_lime_explainer(
                X_train, feature_names, seed=RANDOM_STATE + r,
            ).explain_instance(
                X_xai.iloc[i].values,
                clf.predict_proba,
                num_features=k,
                num_samples=LIME_NUM_SAMPLES,
                labels=(1,),
            )
            top_sets.append({idx for idx, _ in exp.as_map()[1]})
        js = []
        for a in range(len(top_sets)):
            for b in range(a + 1, len(top_sets)):
                union = top_sets[a] | top_sets[b]
                inter = top_sets[a] & top_sets[b]
                js.append(len(inter) / len(union) if union else 1.0)
        results.append(float(np.mean(js)))
    return results


def compute_jaccard(shap_vals, lime_weights, top_k_values):
    per_k = {k: [] for k in top_k_values}
    for i in range(len(shap_vals)):
        for k in top_k_values:
            s = set(top_k_from_shap(shap_vals[i], k).tolist())
            l = set(top_k_from_lime(lime_weights[i], k).tolist())
            union = s | l
            per_k[k].append(len(s & l) / len(union) if union else 0.0)
    return per_k


def compute_feature_frequency(shap_vals, lime_weights, feature_names, k=FREQUENCY_K):
    shap_counts = np.zeros(len(feature_names), dtype=int)
    lime_counts = np.zeros(len(feature_names), dtype=int)
    for i in range(len(shap_vals)):
        for idx in top_k_from_shap(shap_vals[i], k):
            shap_counts[idx] += 1
        for idx in top_k_from_lime(lime_weights[i], k):
            lime_counts[idx] += 1
    n = len(shap_vals)
    return pd.DataFrame({
        "Feature": feature_names,
        f"SHAP_top{k}_freq": shap_counts / n,
        f"LIME_top{k}_freq": lime_counts / n,
    })


# --------------------------------------------------------------------------- #
# 1. Daten laden
# --------------------------------------------------------------------------- #

print(f"[1/6] Lade Datensatz aus {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
non_feature_cols = ["Category", "Class"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]
X = df[feature_cols].select_dtypes(include="number").copy()
y = (df["Class"] == "Malware").astype(int)
feature_names = list(X.columns)
print(f"      Instanzen={len(df)}, Features={X.shape[1]}, "
      f"Klassen (benign / malware) = {(y == 0).sum()} / {(y == 1).sum()}")

# --------------------------------------------------------------------------- #
# 2. Split
# --------------------------------------------------------------------------- #

print(f"[2/6] Stratifizierter 80/20-Split (random_state={RANDOM_STATE})")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"      Training={len(X_train)}, Test={len(X_test)}")

# --------------------------------------------------------------------------- #
# 3. XAI-Subset auswählen (balanciert)
# --------------------------------------------------------------------------- #

print(f"[3/6] Wähle balanciertes XAI-Subset ({XAI_SUBSET_PER_CLASS}/Klasse)")
rng = np.random.default_rng(RANDOM_STATE)
X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)
benign_idx = rng.choice(np.where(y_test_reset == 0)[0], size=XAI_SUBSET_PER_CLASS, replace=False)
malware_idx = rng.choice(np.where(y_test_reset == 1)[0], size=XAI_SUBSET_PER_CLASS, replace=False)
xai_idx = np.concatenate([benign_idx, malware_idx])
X_xai = X_test_reset.iloc[xai_idx].reset_index(drop=True)
y_xai = y_test_reset.iloc[xai_idx].reset_index(drop=True)
print(f"      Insgesamt {len(X_xai)} Instanzen (je {XAI_SUBSET_PER_CLASS} pro Klasse)")

median_row = X_train.median().values
rob_subset_idx = rng.choice(len(X_xai), size=ROBUSTNESS_SUBSET_SIZE, replace=False)

# --------------------------------------------------------------------------- #
# 4. Klassifikatoren nacheinander auswerten
# --------------------------------------------------------------------------- #

# Für einen fairen Vergleich beide Ensembles mit demselben n_estimators
CLASSIFIERS = {
    "RF": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "XGB": XGBClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="logloss", tree_method="hist",
    ),
}

results = {}

for step, (name, clf) in enumerate(CLASSIFIERS.items(), start=1):
    print(f"[4/6][{step}/{len(CLASSIFIERS)}] Klassifikator '{name}'")
    r = train_and_evaluate(clf, X_train, y_train, X_test, y_test)
    print(f"      Training in {r['train_time_s']:.2f}s · "
          + " · ".join(f"{k}={v:.4f}" for k, v in r["metrics"].items()))

    print(f"      Berechne TreeSHAP-Erklärungen ({len(X_xai)} Instanzen)")
    shap_vals, shap_times = compute_shap(r["clf"], X_xai)

    print(f"      Berechne LIME-Erklärungen ({len(X_xai)} Instanzen)")
    lime_weights, lime_times = compute_lime(r["clf"], X_train, X_xai, feature_names)

    print(f"      Berechne Fidelity (top-k perturbation)")
    fidelity = compute_fidelity(r["clf"], X_xai, shap_vals, lime_weights,
                                TOP_K_VALUES, median_row)

    print(f"      Berechne LIME-Robustness ({ROBUSTNESS_SUBSET_SIZE} Instanzen × "
          f"{ROBUSTNESS_N_RUNS} Läufe)")
    lime_rob = compute_lime_robustness(
        r["clf"], X_train, X_xai, feature_names, rob_subset_idx,
    )

    print(f"      Berechne SHAP↔LIME-Agreement (Jaccard)")
    jaccard = compute_jaccard(shap_vals, lime_weights, TOP_K_VALUES)

    print(f"      Berechne Feature-Häufigkeit (top-{FREQUENCY_K})")
    feature_freq = compute_feature_frequency(
        shap_vals, lime_weights, feature_names, k=FREQUENCY_K,
    )

    results[name] = {
        "train_time_s": r["train_time_s"],
        "metrics": r["metrics"],
        "confusion_matrix": r["confusion_matrix"],
        "shap_times": shap_times,
        "lime_times": lime_times,
        "fidelity": fidelity,
        "lime_robustness": lime_rob,
        "jaccard": jaccard,
        "feature_freq": feature_freq,
    }

# --------------------------------------------------------------------------- #
# 5. Konsolidierte Tabellen schreiben
# --------------------------------------------------------------------------- #

print("[5/6] Schreibe Tabellen")

# --- Tab. 2: Klassifikationsmetriken (beide Modelle) --------------------- #
tab2_rows = []
for name, r in results.items():
    row = {"Model": name, **{k: f"{v:.4f}" for k, v in r["metrics"].items()}}
    cm = r["confusion_matrix"]
    row.update(TN=int(cm[0, 0]), FP=int(cm[0, 1]),
               FN=int(cm[1, 0]), TP=int(cm[1, 1]))
    tab2_rows.append(row)
pd.DataFrame(tab2_rows).to_csv(TAB_DIR / "tab2_classification_metrics.csv", index=False)

# --- Fidelity über k, beide Modelle -------------------------------------- #
fid_rows = []
for name, r in results.items():
    for k in TOP_K_VALUES:
        fid_rows.append({
            "Model": name, "k": k,
            "SHAP_mean": float(np.mean(r["fidelity"]["SHAP"][k])),
            "SHAP_median": float(np.median(r["fidelity"]["SHAP"][k])),
            "LIME_mean": float(np.mean(r["fidelity"]["LIME"][k])),
            "LIME_median": float(np.median(r["fidelity"]["LIME"][k])),
        })
pd.DataFrame(fid_rows).to_csv(TAB_DIR / "fidelity_by_k.csv", index=False)

# --- Tab. 3: Robustness (beide Modelle) ---------------------------------- #
rob_rows = []
for name, r in results.items():
    rob_rows.append({
        "Model": name, "Method": "SHAP (TreeSHAP)",
        "Mean pairwise Jaccard (top-5, N=5 runs)": 1.0,
        "Std": 0.0,
    })
    rob_rows.append({
        "Model": name, "Method": "LIME",
        "Mean pairwise Jaccard (top-5, N=5 runs)": float(np.mean(r["lime_robustness"])),
        "Std": float(np.std(r["lime_robustness"])),
    })
pd.DataFrame(rob_rows).to_csv(TAB_DIR / "tab3_robustness.csv", index=False)

# --- Jaccard über k, beide Modelle --------------------------------------- #
jacc_rows = []
for name, r in results.items():
    for k in TOP_K_VALUES:
        jacc_rows.append({
            "Model": name, "k": k,
            "Mean Jaccard": float(np.mean(r["jaccard"][k])),
            "Median Jaccard": float(np.median(r["jaccard"][k])),
            "Std": float(np.std(r["jaccard"][k])),
        })
pd.DataFrame(jacc_rows).to_csv(TAB_DIR / "jaccard_by_k.csv", index=False)

# --- Runtime (beide Modelle) --------------------------------------------- #
rt_rows = []
for name, r in results.items():
    rt_rows.append({
        "Model": name, "Method": "SHAP (TreeSHAP)",
        "Median time [s]": float(np.median(r["shap_times"])),
        "Mean time [s]": float(np.mean(r["shap_times"])),
        "95%-tile [s]": float(np.percentile(r["shap_times"], 95)),
    })
    rt_rows.append({
        "Model": name, "Method": "LIME",
        "Median time [s]": float(np.median(r["lime_times"])),
        "Mean time [s]": float(np.mean(r["lime_times"])),
        "95%-tile [s]": float(np.percentile(r["lime_times"], 95)),
    })
pd.DataFrame(rt_rows).to_csv(TAB_DIR / "runtime.csv", index=False)

# --- Tab. 4: Zusammenfassung --------------------------------------------- #
summary_rows = []
for name, r in results.items():
    fid_shap_5 = float(np.mean(r["fidelity"]["SHAP"][5]))
    fid_lime_5 = float(np.mean(r["fidelity"]["LIME"][5]))
    jacc_5 = float(np.mean(r["jaccard"][5]))
    summary_rows.append({
        "Model": name,
        "Fidelity SHAP (mean |Δp|, k=5)": f"{fid_shap_5:.4f}",
        "Fidelity LIME (mean |Δp|, k=5)": f"{fid_lime_5:.4f}",
        "Robustness LIME (mean Jaccard, top-5, N=5)":
            f"{float(np.mean(r['lime_robustness'])):.3f}",
        "SHAP↔LIME Jaccard (top-5)": f"{jacc_5:.3f}",
        "Runtime SHAP (median, ms)":
            f"{np.median(r['shap_times'])*1000:.1f}",
        "Runtime LIME (median, ms)":
            f"{np.median(r['lime_times'])*1000:.1f}",
    })
pd.DataFrame(summary_rows).to_csv(TAB_DIR / "tab4_summary.csv", index=False)

# --- Feature-Häufigkeit: Top-15 pro Verfahren, beide Modelle ------------- #
freq_frames = []
for name, r in results.items():
    df_ff = r["feature_freq"].copy()
    df_ff["Model"] = name
    freq_frames.append(df_ff)
feature_frequency_all = pd.concat(freq_frames, ignore_index=True)
feature_frequency_all.to_csv(TAB_DIR / "feature_frequency.csv", index=False)

# Aggregierte Top-15 nach SHAP für den RF (dient als Referenz-Ranking im Bericht)
top_shap_rf = (
    results["RF"]["feature_freq"]
    .sort_values(f"SHAP_top{FREQUENCY_K}_freq", ascending=False)
    .head(15)
    .reset_index(drop=True)
)
top_shap_rf.to_csv(TAB_DIR / "top15_features_shap_rf.csv", index=False)

# --------------------------------------------------------------------------- #
# 6. Figures
# --------------------------------------------------------------------------- #

print("[6/6] Erzeuge Figures")

# Fig. 3: Fidelity-Kurven, beide Modelle
fig, ax = plt.subplots(figsize=(6.5, 3.7))
for name, r in results.items():
    ax.plot(TOP_K_VALUES,
            [float(np.mean(r["fidelity"]["SHAP"][k])) for k in TOP_K_VALUES],
            marker="o", color=COLORS[name]["SHAP"],
            label=f"SHAP ({name})")
    ax.plot(TOP_K_VALUES,
            [float(np.mean(r["fidelity"]["LIME"][k])) for k in TOP_K_VALUES],
            marker="s", linestyle="--", color=COLORS[name]["LIME"],
            label=f"LIME ({name})")
ax.set_xlabel("Top-k Features perturbiert")
ax.set_ylabel(r"Mittelwert $|\Delta p_{\mathrm{malware}}|$")
ax.set_title("Fidelity: SHAP vs. LIME für zwei baumbasierte Klassifikatoren")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_fidelity.pdf")
fig.savefig(FIG_DIR / "fig3_fidelity.png", dpi=200)
plt.close(fig)

# Fig. 4: Jaccard-Verteilung, beide Modelle (nebeneinander)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
bins = np.linspace(0, 1, 21)
for ax, (name, r) in zip(axes, results.items()):
    ax.hist(r["jaccard"][5], bins=bins, color=COLORS[name]["SHAP"], alpha=0.8)
    ax.axvline(np.mean(r["jaccard"][5]), color="red", linestyle="--",
               label=f"mean={np.mean(r['jaccard'][5]):.2f}")
    ax.set_title(f"{name}")
    ax.set_xlabel("Jaccard(SHAP, LIME) — top-5")
    ax.legend()
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Anzahl Instanzen")
fig.suptitle("SHAP↔LIME-Agreement je Klassifikator", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_jaccard.pdf")
fig.savefig(FIG_DIR / "fig4_jaccard.png", dpi=200)
plt.close(fig)

# Fig. 5: Feature-Häufigkeit (Top-10 pro Explainer für RF) --------------- #
def _top_features(df_ff, col, n=10):
    return df_ff.sort_values(col, ascending=False).head(n).iloc[::-1]

rf_ff = results["RF"]["feature_freq"]
top_shap = _top_features(rf_ff, f"SHAP_top{FREQUENCY_K}_freq", n=10)
top_lime = _top_features(rf_ff, f"LIME_top{FREQUENCY_K}_freq", n=10)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].barh(top_shap["Feature"], top_shap[f"SHAP_top{FREQUENCY_K}_freq"],
             color=COLORS["RF"]["SHAP"])
axes[0].set_title(f"SHAP — häufigste Top-{FREQUENCY_K} Features (RF)")
axes[0].set_xlabel(f"Anteil der {len(X_xai)} Instanzen")
axes[1].barh(top_lime["Feature"], top_lime[f"LIME_top{FREQUENCY_K}_freq"],
             color=COLORS["RF"]["LIME"])
axes[1].set_title(f"LIME — häufigste Top-{FREQUENCY_K} Features (RF)")
axes[1].set_xlabel(f"Anteil der {len(X_xai)} Instanzen")
for ax in axes:
    ax.grid(True, alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_feature_frequency.pdf")
fig.savefig(FIG_DIR / "fig5_feature_frequency.png", dpi=200)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Metadaten
# --------------------------------------------------------------------------- #

meta = {
    "random_state": RANDOM_STATE,
    "dataset": {
        "path": str(DATA_PATH),
        "samples_total": int(len(df)),
        "samples_train": int(len(X_train)),
        "samples_test": int(len(X_test)),
        "features": int(X.shape[1]),
        "class_balance_test": {
            "benign": int((y_test == 0).sum()),
            "malware": int((y_test == 1).sum()),
        },
    },
    "xai_subset_size": int(len(X_xai)),
    "xai_subset_per_class": XAI_SUBSET_PER_CLASS,
    "top_k_values": TOP_K_VALUES,
    "frequency_k": FREQUENCY_K,
    "robustness": {
        "subset_size": ROBUSTNESS_SUBSET_SIZE,
        "n_runs": ROBUSTNESS_N_RUNS,
    },
    "classifiers": {},
}
for name, r in results.items():
    cm = r["confusion_matrix"]
    meta["classifiers"][name] = {
        "training_time_s": r["train_time_s"],
        "classification_metrics": {k: float(v) for k, v in r["metrics"].items()},
        "confusion_matrix": {
            "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
        },
        "shap_time_s": {
            "median": float(np.median(r["shap_times"])),
            "mean": float(np.mean(r["shap_times"])),
        },
        "lime_time_s": {
            "median": float(np.median(r["lime_times"])),
            "mean": float(np.mean(r["lime_times"])),
        },
        "fidelity_mean_by_k": {
            "SHAP": {k: float(np.mean(r["fidelity"]["SHAP"][k])) for k in TOP_K_VALUES},
            "LIME": {k: float(np.mean(r["fidelity"]["LIME"][k])) for k in TOP_K_VALUES},
        },
        "jaccard_shap_lime": {k: float(np.mean(r["jaccard"][k])) for k in TOP_K_VALUES},
        "robustness_lime_mean_pairwise_jaccard_top5": float(np.mean(r["lime_robustness"])),
    }
(TAB_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))

print()
print("=" * 72)
print("Fertig. Outputs geschrieben nach:")
print(f"  Figures: {FIG_DIR}")
print(f"  Tables:  {TAB_DIR}")
print("=" * 72)
