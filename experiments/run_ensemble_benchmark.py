"""
Focused GPE-Ensemble benchmark across three financial datasets.

Trains a RandomForestClassifier on each of:
    - Credit Risk
    - Credit Fraud
    - Lending Club
and compares GPE-Ensemble against TreeSHAP and LIME on 50 test
instances per dataset. Saves results to
experiments/results/ensemble_benchmark.csv|json.

Usage:
    cd GPE-Framework && python experiments/run_ensemble_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOCS = ROOT / "docs"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from gpe import GPEEnsemble  # noqa: E402

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lime.lime_tabular

    HAS_LIME = True
except ImportError:
    HAS_LIME = False


def load_dataset(name: str, max_samples: int = 100_000):
    if name == "Lending Club":
        fp = DOCS / "lending_club_2007_to_2018Q4.csv"
        df = pd.read_csv(fp, nrows=max_samples, low_memory=False)
        bad = ["Charged Off", "Default", "Late (31-120 days)"]
        df = df[df["loan_status"].isin(["Fully Paid", "Current"] + bad)]
        y = df["loan_status"].isin(bad).astype(int).values
        cols = [
            "loan_amnt",
            "int_rate",
            "annual_inc",
            "dti",
            "fico_range_low",
            "revol_util",
            "total_acc",
            "inq_last_6mths",
        ]
        X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        return X, y, cols, ["Paid", "Default"]

    if name == "Credit Fraud":
        fp = DOCS / "CreditCardFraudDetection.csv"
        df = pd.read_csv(fp, nrows=max_samples)
        y = df["Class"].values.astype(int)
        cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
        X = df[cols].values
        return X, y, cols, ["Normal", "Fraud"]

    if name == "Credit Risk":
        fp = DOCS / "credit_risk_dataset.csv"
        df = pd.read_csv(fp)
        y = df["loan_status"].values
        cat_cols = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file",
        ]
        for c in cat_cols:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        cols = [c for c in df.columns if c != "loan_status"]
        X = df[cols].fillna(df[cols].median()).values
        return X, y, cols, ["Good", "Default"]

    raise ValueError(name)


def run_one(dataset: str, n_explain: int = 50, n_repeats: int = 3) -> list[dict]:
    print(f"\n=== {dataset} on RandomForest ===")
    X, y, fn, cn = load_dataset(dataset)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    bg = Xtr[np.random.RandomState(42).choice(len(Xtr), size=100, replace=False)]

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)
    acc = model.score(Xte, yte)
    print(f"  Tree-RF accuracy: {acc:.4f}")
    instances = Xte[:n_explain]

    results = []

    # GPE-Ensemble
    print("  Running GPE-Ensemble...", end=" ", flush=True)
    explainer = GPEEnsemble(model, feature_names=fn, X_train=bg, aggregation="majority")
    times, comp, prec, cov = [], [], [], []
    for _ in range(n_repeats):
        for x in instances:
            t0 = time.perf_counter()
            try:
                exp = explainer.explain(x, X_background=bg)
                times.append((time.perf_counter() - t0) * 1000.0)
                comp.append(exp.complexity)
                prec.append(exp.precision)
                cov.append(exp.coverage)
            except Exception as e:  # noqa: BLE001
                print(f"  GPE-Ensemble failure: {e}")
                continue
    results.append(
        {
            "Database": dataset,
            "Model": "RandomForest",
            "Method": "GPE-Ensemble",
            "Time (ms)": float(np.mean(times)),
            "Time std": float(np.std(times)),
            "Complexity": float(np.mean(comp)),
            "Precision": float(np.mean(prec)) if prec else float("nan"),
            "Coverage": float(np.mean(cov)) if cov else float("nan"),
            "N samples": len(times),
        }
    )
    print(f"done ({np.mean(times):.2f} ± {np.std(times):.2f} ms)")

    # TreeSHAP
    if HAS_SHAP:
        print("  Running TreeSHAP...", end=" ", flush=True)
        tse = shap.TreeExplainer(model)
        times = []
        for _ in range(n_repeats):
            for x in instances:
                t0 = time.perf_counter()
                _ = tse.shap_values(x)
                times.append((time.perf_counter() - t0) * 1000.0)
        results.append(
            {
                "Database": dataset,
                "Model": "RandomForest",
                "Method": "TreeSHAP",
                "Time (ms)": float(np.mean(times)),
                "Time std": float(np.std(times)),
                "Complexity": float(len(fn)),
                "Precision": float("nan"),
                "Coverage": 1.0,
                "N samples": len(times),
            }
        )
        print(f"done ({np.mean(times):.2f} ms)")

    # LIME
    if HAS_LIME:
        print("  Running LIME...", end=" ", flush=True)
        lme = lime.lime_tabular.LimeTabularExplainer(
            Xtr[:1000], feature_names=fn, mode="classification", verbose=False
        )
        times = []
        # LIME is slow — use 20 instances and warm-up
        warm = instances[0]
        _ = lme.explain_instance(warm, model.predict_proba, num_features=5)
        for _ in range(n_repeats):
            for x in instances[:20]:
                t0 = time.perf_counter()
                _ = lme.explain_instance(x, model.predict_proba, num_features=5)
                times.append((time.perf_counter() - t0) * 1000.0)
        results.append(
            {
                "Database": dataset,
                "Model": "RandomForest",
                "Method": "LIME",
                "Time (ms)": float(np.mean(times)),
                "Time std": float(np.std(times)),
                "Complexity": 5.0,
                "Precision": float("nan"),
                "Coverage": 1.0,
                "N samples": len(times),
            }
        )
        print(f"done ({np.mean(times):.2f} ms)")

    return results


def main():
    print("GPE-Ensemble benchmark across three ансамбль-датасетів")
    all_results = []
    for ds in ["Credit Risk", "Credit Fraud", "Lending Club"]:
        all_results.extend(run_one(ds))

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS / "ensemble_benchmark.csv", index=False)
    with open(RESULTS / "ensemble_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved:\n  {RESULTS / 'ensemble_benchmark.csv'}\n  {RESULTS / 'ensemble_benchmark.json'}")


if __name__ == "__main__":
    main()
