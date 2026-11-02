"""
Compare the rules of two releases of gpe-framework on the same data.

Run once under each release, then compare:
    python experiments/release_regression.py dump old.json
    python experiments/release_regression.py dump new.json
    python experiments/release_regression.py compare old.json new.json
"""

import json
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import gpe
from gpe import CertifiedGPE, GPEExplainer, GPEOptimal, GPEWeighted
from gpe.novel_methods import GPEInformationTheoretic

TAU = 0.95
DEPTHS = (3, 5, 8, 12)
SEEDS = (0, 1, 2)
N_EXPLAIN = 200


def datasets():
    yield "breast_cancer", *load_breast_cancer(return_X_y=True)
    yield "synthetic_easy", *make_classification(8000, 20, n_informative=6, class_sep=2.0, random_state=0)
    yield "synthetic_hard", *make_classification(8000, 20, n_informative=6, flip_y=0.2, class_sep=0.5, random_state=1)


METHODS = {
    "core": lambda tree, X: (GPEExplainer(tree, min_precision=TAU), X),
    "it": lambda tree, X: (GPEInformationTheoretic(tree, min_precision=TAU), X),
    "optimal": lambda tree, X: (GPEOptimal(tree, min_precision=TAU, max_conditions=12), X),
    "weighted": lambda tree, X: (GPEWeighted(tree, min_precision=TAU), X),
    "certified": lambda tree, X: (CertifiedGPE(tree, X, min_precision=TAU), None),
}


def key(condition):
    return [int(condition.feature_index), condition.operator, float(condition.threshold)]


def dump(path):
    out = {"version": gpe.__version__, "results": {}}
    for name, X, y in datasets():
        for depth in DEPTHS:
            for seed in SEEDS:
                X_tr, X_rest, y_tr, _ = train_test_split(X, y, test_size=0.4, random_state=seed)
                X_bg, X_te = train_test_split(X_rest, test_size=0.5, random_state=seed)
                tree = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(X_tr, y_tr)
                for method, make in METHODS.items():
                    explainer, background = make(tree, X_bg)
                    rows = []
                    for x in X_te[:N_EXPLAIN]:
                        exp = explainer.explain(x, background) if background is not None else explainer.explain(x)
                        rows.append({
                            "conditions": sorted(key(c) for c in exp.rule.conditions),
                            "prediction": int(exp.prediction),
                            "precision": float(exp.precision),
                            "coverage": float(exp.coverage),
                            "full_path_rows": int(exp.original_rule.evaluate_batch(X_bg).sum()),
                        })
                    out["results"][f"{name}|{depth}|{seed}|{method}"] = rows
    with open(path, "w") as handle:
        json.dump(out, handle)


def compare(old_path, new_path):
    old, new = json.load(open(old_path)), json.load(open(new_path))
    print(f"{old['version']} -> {new['version']}")
    totals = {}
    for run, rows_old in old["results"].items():
        method = run.split("|")[3]
        t = totals.setdefault(method, dict(n=0, rule_differs=0, rule_differs_with_evidence=0,
                                           rule_differs_full_path_has_rows=0,
                                           precision_differs=0, vacuous_old=0))
        for a, b in zip(rows_old, new["results"][run]):
            t["n"] += 1
            evidence = a["coverage"] > 0
            t["vacuous_old"] += not evidence
            if a["conditions"] != b["conditions"] or a["prediction"] != b["prediction"]:
                t["rule_differs"] += 1
                t["rule_differs_full_path_has_rows"] += a["full_path_rows"] > 0
                if evidence:
                    t["rule_differs_with_evidence"] += 1
            if evidence and not np.isclose(a["precision"], b["precision"]):
                t["precision_differs"] += 1
    for method, t in totals.items():
        print(method, t)
    return totals


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
