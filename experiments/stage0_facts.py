"""
Stage 0 facts about GPE 1.2.0 behaviour, measured on real runs.

(a) Is an explanation a function of the leaf only?
(b) How often is the precision of the full path below tau, against the model
    and against the true labels? How often is it computed on zero rows?
(c) How many rows does a float64 evaluation of a path rule put outside the leaf?

Usage:
    GPE_DOCS_DIR=/path/to/csvs python experiments/stage0_facts.py
Output: experiments/results_stage0/stage0_facts_v<version>.csv
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gpe import GPEExplainer, GPEInformationTheoretic, __version__  # noqa: E402
from gpe.tree_utils import get_decision_path, simplify_conditions  # noqa: E402

DOCS_DIR = Path(os.environ.get("GPE_DOCS_DIR", ROOT / "docs"))
OUT_DIR = ROOT / "experiments" / "results_stage0"
TAU = 0.95
N_EXPLAIN = 300
DEPTHS = (5, 8, 12)
SEEDS = (0, 1, 2)


def credit_risk(dtype):
    df = pd.read_csv(DOCS_DIR / "credit_risk_dataset.csv")
    y = df["loan_status"].values.astype(int)
    X_df = df.drop(columns=["loan_status"]).copy()
    for col in X_df.columns:
        if X_df[col].dtype == object or str(X_df[col].dtype) in ("str", "string"):
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        X_df[col] = X_df[col].fillna(X_df[col].median())
    return X_df.values.astype(dtype), y


def fraud():
    df = pd.read_csv(DOCS_DIR / "CreditCardFraudDetection.csv", nrows=100000)
    return df.drop(columns=["Class"]).values.astype(np.float64), df["Class"].values.astype(int)


def money_like(seed=0, n=40000):
    """float64 features with cent resolution around 1e6: the float32 gap is visible here."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.round(rng.normal(1e6, 3e5, n), 2), np.round(rng.normal(0, 1, n), 6)])
    y = (X[:, 0] + 2e5 * X[:, 1] + rng.normal(0, 2e5, n) > 1.0e6).astype(int)
    return X, y


def datasets():
    yield "breast_cancer", lambda: load_breast_cancer(return_X_y=True)
    yield "synthetic_32k", lambda: make_classification(
        n_samples=32000, n_features=20, n_informative=10, flip_y=0.05, random_state=0
    )
    yield "money_like_float64", money_like
    if (DOCS_DIR / "credit_risk_dataset.csv").exists():
        yield "credit_risk", lambda: credit_risk(np.float64)
    else:
        print("credit_risk: file not found, skipped")
    if (DOCS_DIR / "CreditCardFraudDetection.csv").exists():
        yield "fraud_100k", fraud
    else:
        print("fraud: file not found, skipped")


def key(rule):
    return tuple(sorted((c.feature_index, c.operator, float(c.threshold)) for c in rule.conditions))


def measure(name, X, y, depth, seed):
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_bg, X_te, y_bg, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
    tree = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(X_tr, y_tr)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_te), min(N_EXPLAIN, len(X_te)), replace=False)
    X_ex = X_te[idx]
    leaves = tree.apply(X_ex)
    bg_pred = tree.predict(X_bg)
    bg_leaf = tree.apply(X_bg)

    core = GPEExplainer(tree, X_train=X_bg, min_precision=TAU)
    gpe_it = GPEInformationTheoretic(tree, X_train=X_bg, min_precision=TAU)

    by_leaf = {}
    full_f, full_y, full_n, core_prec, core_vac, it_prec = [], [], [], [], [], []
    gap_rows, rule_rows, literals = 0, 0, []
    for x, leaf in zip(X_ex, leaves):
        exp = core.explain(x)
        by_leaf.setdefault(int(leaf), set()).add(key(exp.rule))
        core_prec.append(exp.precision)
        core_vac.append(int(exp.rule.evaluate_batch(X_bg).sum() == 0))

        path = get_decision_path(tree, x)
        path.conditions = simplify_conditions(path.conditions)
        literals.append(len(path.conditions))
        c = tree.predict(x.reshape(1, -1))[0]
        mask = path.evaluate_batch(X_bg)
        full_n.append(int(mask.sum()))
        if mask.sum():
            full_f.append((bg_pred[mask] == c).mean())
            full_y.append((y_bg[mask] == c).mean())
        gap_rows += int((mask & (bg_leaf != leaf)).sum())
        rule_rows += int(mask.sum())

        it_prec.append(gpe_it.explain(x).precision)

    full_f, full_y, full_n = np.array(full_f), np.array(full_y), np.array(full_n)
    return {
        "dataset": name,
        "depth": depth,
        "seed": seed,
        "n_explained": len(X_ex),
        "n_bg": len(X_bg),
        "tree_leaves": tree.get_n_leaves(),
        "mean_k": float(np.mean(literals)),
        # (a)
        "leaves_on_test": len(by_leaf),
        "unique_core_rules": len({r for rules in by_leaf.values() for r in rules}),
        "leaves_with_several_rules": sum(len(v) > 1 for v in by_leaf.values()),
        # (b)
        "full_path_n0_share": float((full_n == 0).mean()),
        "full_path_f_lt_tau_share": float((full_f < TAU).mean()) if len(full_f) else np.nan,
        "full_path_f_min": float(full_f.min()) if len(full_f) else np.nan,
        "full_path_labels_lt_tau_share": float((full_y < TAU).mean()) if len(full_y) else np.nan,
        "core_reported_lt_tau_share": float((np.array(core_prec) < TAU).mean()),
        "core_vacuous_n0_share": float(np.mean(core_vac)),
        "gpe_it_reported_lt_tau_share": float((np.array(it_prec) < TAU).mean()),
        "gpe_it_reported_min": float(np.min(it_prec)),
        # (c)
        "float64_rule_rows_outside_leaf": gap_rows,
        "float64_rule_rows_total": rule_rows,
    }


def main():
    rows = []
    for name, load in datasets():
        X, y = load()
        for depth in DEPTHS:
            for seed in SEEDS:
                rows.append(measure(name, X, y, depth, seed))
                r = rows[-1]
                print(
                    f"{name:20s} d={depth:2d} s={seed} leaves={r['leaves_on_test']:4d} "
                    f"rules={r['unique_core_rules']:4d} multi={r['leaves_with_several_rules']} "
                    f"full<tau(f)={r['full_path_f_lt_tau_share']:.3f} "
                    f"full<tau(y)={r['full_path_labels_lt_tau_share']:.3f} "
                    f"n0={r['full_path_n0_share']:.3f} it<tau={r['gpe_it_reported_lt_tau_share']:.3f} "
                    f"gap={r['float64_rule_rows_outside_leaf']}/{r['float64_rule_rows_total']}",
                    flush=True,
                )
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    path = OUT_DIR / f"stage0_facts_v{__version__}.csv"
    df.to_csv(path, index=False)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
