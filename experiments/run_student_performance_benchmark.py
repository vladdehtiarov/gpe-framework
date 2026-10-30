"""
Run a GPE-vs-LIME benchmark on the UCI Student Performance dataset
and write results to experiments/results/student_performance_benchmark.csv.

Usage:
    cd GPE-Framework && python experiments/run_student_performance_benchmark.py

Dependencies: gpe (local), lime, scikit-learn, numpy, pandas.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, "datasets"))

import loaders
from gpe import GPEExplainer


def time_gpe(tree, X_train, feature_names, X_samples, n_repeats=3):
    """Return per-instance timings (ms), rule lengths, precisions."""
    times, lens, precs = [], [], []
    explainer = GPEExplainer(
        tree, feature_names=feature_names, X_train=X_train, min_precision=0.95
    )
    # Warm-up
    _ = explainer.explain(X_samples[0])

    for _ in range(n_repeats):
        for x in X_samples:
            t0 = time.perf_counter()
            exp = explainer.explain(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            lens.append(len(exp.rule.conditions))
            precs.append(exp.precision)
    return np.array(times), np.array(lens), np.array(precs)


def time_lime(tree, X_train, feature_names, class_names, X_samples, n_repeats=3):
    """Return per-instance timings (ms) and explanation sizes for LIME."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )

    times, lens = [], []
    # Warm-up
    _ = explainer.explain_instance(
        X_samples[0], tree.predict_proba, num_features=5, num_samples=500
    )

    for _ in range(n_repeats):
        for x in X_samples:
            t0 = time.perf_counter()
            exp = explainer.explain_instance(
                x, tree.predict_proba, num_features=5, num_samples=500
            )
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            lens.append(len(exp.as_list()))
    return np.array(times), np.array(lens)


def main():
    print("=" * 72)
    print("Student Performance benchmark: GPE vs LIME on a depth-7 decision tree")
    print("=" * 72)

    X_train, X_test, y_train, y_test, fn, cn = loaders.load_student_performance("merged")
    print(f"Dataset: merged (Math+Portuguese), train={X_train.shape}, test={X_test.shape}")
    print(f"Pass rate (train): {y_train.mean():.3f}")

    tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=20, random_state=42)
    tree.fit(X_train, y_train)
    acc = tree.score(X_test, y_test)
    print(f"Tree accuracy on test: {acc:.3f}")
    print(f"Tree depth: {tree.get_depth()}, leaves: {tree.get_n_leaves()}")

    # Sample 100 test instances
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(100, len(X_test)), replace=False)
    X_samples = X_test[idx]

    print("\nRunning GPE...")
    gpe_t, gpe_len, gpe_prec = time_gpe(tree, X_train, fn, X_samples, n_repeats=3)

    print("Running LIME...")
    lime_t, lime_len = time_lime(tree, X_train, fn, cn, X_samples, n_repeats=3)

    result = {
        "dataset": "student_performance_merged",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "tree_acc": float(acc),
        "tree_depth": int(tree.get_depth()),
        "tree_leaves": int(tree.get_n_leaves()),
        "n_instances_sampled": int(len(X_samples)),
        "n_repeats": 3,
        "gpe": {
            "time_ms_mean": float(gpe_t.mean()),
            "time_ms_std": float(gpe_t.std()),
            "rule_len_mean": float(gpe_len.mean()),
            "rule_len_std": float(gpe_len.std()),
            "precision_mean": float(gpe_prec.mean()),
            "precision_std": float(gpe_prec.std()),
        },
        "lime": {
            "time_ms_mean": float(lime_t.mean()),
            "time_ms_std": float(lime_t.std()),
            "rule_len_mean": float(lime_len.mean()),
            "rule_len_std": float(lime_len.std()),
        },
        "speedup_vs_lime": float(lime_t.mean() / gpe_t.mean()),
    }

    # Welch's t-test on timings
    try:
        from scipy.stats import ttest_ind

        t_stat, p_val = ttest_ind(gpe_t, lime_t, equal_var=False)
        # Cohen's d using pooled std
        pooled = np.sqrt(
            ((len(gpe_t) - 1) * gpe_t.var(ddof=1) + (len(lime_t) - 1) * lime_t.var(ddof=1))
            / (len(gpe_t) + len(lime_t) - 2)
        )
        cohen_d = (gpe_t.mean() - lime_t.mean()) / pooled
        result["stats"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohen_d": float(cohen_d),
        }
    except Exception as exc:
        result["stats"] = {"error": str(exc)}

    # Save
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "student_performance_benchmark.json")
    csv_path = os.path.join(out_dir, "student_performance_benchmark.csv")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    rows = [
        {
            "method": "GPE",
            "time_ms_mean": result["gpe"]["time_ms_mean"],
            "time_ms_std": result["gpe"]["time_ms_std"],
            "rule_len_mean": result["gpe"]["rule_len_mean"],
            "precision_mean": result["gpe"]["precision_mean"],
        },
        {
            "method": "LIME",
            "time_ms_mean": result["lime"]["time_ms_mean"],
            "time_ms_std": result["lime"]["time_ms_std"],
            "rule_len_mean": result["lime"]["rule_len_mean"],
            "precision_mean": float("nan"),
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 72)
    print("Results")
    print("=" * 72)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
