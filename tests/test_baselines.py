"""
The baseline wrappers must run on the installed versions and return rules true on x.

Their libraries are optional, so each test is skipped when the library is missing.
Benchmarks recompute precision and coverage with shared code; see anchor issue #82.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from baselines import AnchorsWrapper, LIMEWrapper, SHAPWrapper


@pytest.fixture(scope="module")
def setup():
    X, y = make_classification(800, 6, n_informative=4, random_state=0)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X[:600], y[:600])
    names = [f"f{i}" for i in range(6)]
    return tree, X[:600], X[600:], names


def test_lime(setup):
    pytest.importorskip("lime")
    tree, X_tr, X_te, names = setup
    wrapper = LIMEWrapper(tree, X_tr, feature_names=names, n_features=3)
    exp = wrapper.explain(X_te[0], num_samples=300)
    assert exp.method == "LIME"
    assert exp.prediction == tree.predict(X_te[:1])[0]
    assert len({c.feature_index for c in exp.rule.conditions}) == 3
    assert exp.rule.evaluate(X_te[0])


def test_shap(setup):
    pytest.importorskip("shap")
    tree, X_tr, X_te, names = setup
    exp = SHAPWrapper(tree, X_tr, feature_names=names, n_features=3).explain(X_te[0])
    assert exp.method.upper().startswith("SHAP") or "SHAP" in exp.method
    assert exp.prediction == tree.predict(X_te[:1])[0]
    assert 1 <= exp.complexity <= 3


def test_anchors(setup):
    pytest.importorskip("anchor")
    np.random.seed(0)
    tree, X_tr, X_te, names = setup
    exp = AnchorsWrapper(tree, X_tr, feature_names=names, threshold=0.9).explain(X_te[0])
    assert exp.method == "Anchors"
    assert exp.prediction == tree.predict(X_te[:1])[0]
    assert exp.rule.evaluate(X_te[0]), str(exp.rule)


def rows_for_hard_names():
    """Small-scale features and names that break string parsing."""
    X, y = make_classification(800, 4, n_informative=3, n_redundant=0, random_state=1)
    X = X * np.array([1e-4, 1e4, 1.0, 1e-3])
    names = ["capital-gain", "hours per week", "x(1)", "a<b"]
    tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X[:600], y[:600])
    return tree, X[:600], X[600:], names


def test_lime_keeps_the_whole_bin_and_is_true_on_x():
    pytest.importorskip("lime")
    tree, X_tr, X_te, names = rows_for_hard_names()
    wrapper = LIMEWrapper(tree, X_tr, feature_names=names, n_features=3)
    for x in X_te[:10]:
        exp = wrapper.explain(x, num_samples=300)
        assert exp.rule.evaluate(x), str(exp.rule)
        features = {c.feature_index for c in exp.rule.conditions}
        assert len(features) == 3
        assert exp.complexity >= 3  # a middle bin gives two conditions


def test_anchors_survive_hard_names_and_small_scales():
    pytest.importorskip("anchor")
    np.random.seed(0)
    tree, X_tr, X_te, names = rows_for_hard_names()
    wrapper = AnchorsWrapper(tree, X_tr, feature_names=names, threshold=0.9)
    non_empty = 0
    for x in X_te[:6]:
        exp = wrapper.explain(x)
        assert exp.rule.evaluate(x), str(exp.rule)
        assert len({c.feature_index for c in exp.rule.conditions}) == len(exp.metadata["anchor_names"])
        non_empty += exp.complexity > 0
    assert non_empty


def test_shap_selects_the_features_that_matter():
    """Only feature 2 moves the prediction, so it must come first."""
    pytest.importorskip("shap")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5))
    y = (X[:, 2] > 0).astype(int)
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    for x in X[:20]:
        exp = SHAPWrapper(tree, X, n_features=2).explain(x)
        assert exp.metadata["top_features"][0] == 2
        assert len(exp.metadata["shap_values"]) == 5
        assert exp.metadata["shap_values"][2] > 0  # supports the predicted class
        assert exp.complexity == 1  # features with no effect give no condition
