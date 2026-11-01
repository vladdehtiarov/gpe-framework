"""
Facts about GPE behaviour that the dissertation text relies on.

Author: Vladyslav Dehtiarov
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from gpe import GPEExplainer, GPEInformationTheoretic
from gpe.explanation import Condition, Rule
from gpe.tree_utils import extract_rules_from_tree, get_decision_path, simplify_conditions

TAU = 0.95


def rule_key(rule):
    return tuple(sorted((c.feature_index, c.operator, float(c.threshold)) for c in rule.conditions))


@pytest.fixture(params=["breast_cancer", "synthetic"])
def split(request):
    if request.param == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
    else:
        X, y = make_classification(
            n_samples=4000, n_features=15, n_informative=8, flip_y=0.05, random_state=0
        )
    return train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)


class TestLeafDependence:
    """(a) An explanation is a function of the leaf only."""

    @pytest.mark.parametrize("depth", [4, 8])
    @pytest.mark.parametrize("cls", [GPEExplainer, GPEInformationTheoretic])
    def test_one_rule_per_leaf(self, split, depth, cls):
        X_tr, X_te, y_tr, _ = split
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
        explainer = cls(tree, X_train=X_tr, min_precision=TAU)

        by_leaf = {}
        for x, leaf in zip(X_te[:300], tree.apply(X_te[:300])):
            by_leaf.setdefault(int(leaf), set()).add(rule_key(explainer.explain(x).rule))

        assert all(len(rules) == 1 for rules in by_leaf.values())
        unique = {rule for rules in by_leaf.values() for rule in rules}
        assert len(unique) <= len(by_leaf)


class TestFullPathPrecision:
    """(b) Against the model the full path has precision 1; against labels it need not."""

    def test_full_path_against_model_and_labels(self, split):
        X_tr, X_te, y_tr, y_te = split
        tree = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X_tr, y_tr)
        pred = tree.predict(X_te)

        against_labels = []
        for x in X_te[:200]:
            c = tree.predict(x.reshape(1, -1))[0]
            mask = get_decision_path(tree, x).evaluate_batch(X_te)
            assert (pred[mask] == c).all()
            against_labels.append((y_te[mask] == c).mean())

        assert min(against_labels) < 1.0

    def test_core_reports_at_least_tau(self, split):
        X_tr, X_te, y_tr, _ = split
        tree = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X_tr, y_tr)
        explainer = GPEExplainer(tree, X_train=X_tr, min_precision=TAU)
        assert all(exp.precision >= TAU for exp in explainer.explain_batch(X_te[:200]))

    def test_gpe_it_reaches_tau(self, split):
        X_tr, X_te, y_tr, _ = split
        tree = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X_tr, y_tr)
        explainer = GPEInformationTheoretic(tree, X_train=X_tr, min_precision=TAU)
        assert all(exp.precision >= TAU for exp in explainer.explain_batch(X_te[:200]))

    def test_vacuous_precision_is_visible(self):
        X = np.array([[0.0], [1.0]] * 20)
        tree = DecisionTreeClassifier(random_state=0).fit(X, [0, 1] * 20)
        explanation = GPEExplainer(tree).explain(np.array([0.0]), np.array([[1.0]] * 5))
        assert explanation.precision == 1.0
        assert explanation.metadata['n_background'] == 0


class TestFloat32:
    """(c) Path rules select the rows sklearn sends to the leaf, whatever the input dtype."""

    EPS = float(np.finfo(np.float32).eps)

    def fitted(self):
        X = np.array([[1.0], [1.0 + 4 * self.EPS]])
        return DecisionTreeClassifier(random_state=0).fit(X, [0, 1])

    def test_value_rounded_onto_threshold_goes_left(self):
        tree = self.fitted()
        threshold = float(tree.tree_.threshold[0])
        assert threshold == 1.0 + 2 * self.EPS

        # Above the threshold in float64, on it in float32: sklearn goes left.
        x = np.array([threshold + 0.25 * self.EPS])
        assert x[0] > threshold
        assert tree.apply(x.reshape(1, -1))[0] == tree.tree_.children_left[0]

        rule = get_decision_path(tree, x)
        assert rule.conditions[0].operator == '<='
        assert rule.evaluate(x)
        assert rule.evaluate_batch(x.reshape(1, -1))[0]

    def test_rules_partition_rows_like_the_tree(self):
        tree = self.fitted()
        threshold = float(tree.tree_.threshold[0])
        offsets = np.array([-2, -0.25, 0, 0.25, 0.75, 2]) * self.EPS
        rows = (threshold + offsets).reshape(-1, 1)

        for X in (rows, rows.astype(np.float32)):
            leaves = tree.apply(X)
            rules = extract_rules_from_tree(tree)
            left = next(r for r in rules if r.conditions[0].operator == '<=')
            assert (left.evaluate_batch(X) == (leaves == tree.tree_.children_left[0])).all()

    def test_float32_threshold_not_rounded(self):
        """A half-grid threshold must stay exact against a float32 array."""
        X = np.array([[1.0], [1.0 + 3 * self.EPS]])
        tree = DecisionTreeClassifier(random_state=0).fit(X, [0, 1])
        assert float(tree.tree_.threshold[0]) == 1.0 + 1.5 * self.EPS

        rows = (1.0 + np.arange(4) * self.EPS).reshape(-1, 1).astype(np.float32)
        leaves = tree.apply(rows)
        assert (leaves == tree.tree_.children_left[0]).sum() == 2
        for rule in extract_rules_from_tree(tree):
            went_left = rule.conditions[0].operator == '<='
            child = tree.tree_.children_left[0] if went_left else tree.tree_.children_right[0]
            assert (rule.evaluate_batch(rows) == (leaves == child)).all()

    def test_plain_conditions_keep_float64(self):
        value = 1.0 + 0.25 * self.EPS
        assert Condition(0, "x", ">", 1.0).evaluate(np.array([value]))
        assert not Condition(0, "x", ">", 1.0, float32_input=True).evaluate(np.array([value]))

    def test_full_path_matches_leaf_on_large_values(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.round(rng.normal(1e6, 3e5, 20000), 2), rng.normal(0, 1, 20000)])
        y = (X[:, 0] + 2e5 * X[:, 1] + rng.normal(0, 2e5, 20000) > 1e6).astype(int)
        tree = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X[:10000], y[:10000])
        probe = X[10000:]
        leaves = tree.apply(probe)
        for x in probe[:100]:
            rule = get_decision_path(tree, x)
            rule.conditions = simplify_conditions(rule.conditions)
            leaf = tree.apply(x.reshape(1, -1))[0]
            assert (rule.evaluate_batch(probe) == (leaves == leaf)).all()

    def test_serialization_keeps_the_flag(self):
        tree = self.fitted()
        rule = get_decision_path(tree, np.array([1.0]))
        restored = Rule.from_dict(rule.to_dict())
        assert restored.conditions[0].float32_input

    @pytest.mark.parametrize("operator", ["<=", ">", "<", ">="])
    def test_batch_agrees_with_single_evaluation(self, operator):
        rng = np.random.default_rng(0)
        thresholds = np.concatenate([
            1.0 + np.arange(-4, 5) * 0.5 * self.EPS,
            [0.0, -0.0, 1e-45, -1e-45, 1e40, -1e40, np.inf, -np.inf],
            rng.normal(0, 1e6, 50),
        ])
        values = np.concatenate([
            thresholds,
            thresholds * (1 + 0.25 * self.EPS),
            thresholds * (1 - 0.25 * self.EPS),
            rng.normal(0, 1e6, 100),
        ])
        values = values[np.isfinite(values)].reshape(-1, 1)
        for threshold in thresholds:
            condition = Condition(0, "x", operator, float(threshold), float32_input=True)
            with np.errstate(over="ignore"):  # 1e40 does not fit float32
                for X in (values, values.astype(np.float32)):
                    single = np.array([condition.evaluate(row) for row in X])
                    assert (Rule([condition]).evaluate_batch(X) == single).all()
