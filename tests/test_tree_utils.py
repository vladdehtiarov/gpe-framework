"""tree_utils: simplification against brute force, and the helpers checked against sklearn."""

import sys
import types

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from gpe import tree_utils
from gpe.explanation import Condition, Rule
from gpe.tree_utils import (
    extract_rules_from_tree,
    get_decision_path,
    get_feature_importance_path,
    get_leaf_id,
    get_leaf_samples,
    get_n_leaves,
    get_tree_depth,
    is_ensemble_tree,
    is_single_tree,
    is_tree_model,
    simplify_conditions,
)


def data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    return X, (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)


class TestSimplifyByBruteForce:
    GRID = [-1.0, 0.0, 0.1, float(np.float32(0.1)), 1.0, 2.0]
    ROWS = np.array(
        [[v] for v in [-2.0, -1.0, -0.5, 0.0, 0.05, 0.1, float(np.float32(0.1)), 0.5, 1.0, 1.5, 2.0, 3.0, np.nan]]
    )

    def same_rows(self, conditions):
        before = Rule(conditions=list(conditions)).evaluate_batch(self.ROWS)
        kept = simplify_conditions(conditions)
        after = Rule(conditions=list(kept)).evaluate_batch(self.ROWS)
        return np.array_equal(before, after), kept

    @pytest.mark.parametrize("float32", [False, True])
    @pytest.mark.parametrize("seed", range(4))
    def test_any_bounds_select_the_same_rows(self, seed, float32):
        rng = np.random.default_rng(seed)
        for _ in range(500):
            conditions = [
                Condition(0, "x", str(rng.choice(["<=", "<", ">", ">="])), float(rng.choice(self.GRID)),
                          float32_input=float32, matches_missing=bool(rng.random() < 0.5))
                for _ in range(rng.integers(1, 6))
            ]
            same, kept = self.same_rows(conditions)
            assert same, (conditions, kept)
            assert sum(c.operator in (">", ">=") for c in kept) <= 1
            assert sum(c.operator in ("<", "<=") for c in kept) <= 1

    def test_mixed_rounding_is_not_merged_into_a_wider_region(self):
        rng = np.random.default_rng(9)
        for _ in range(2000):
            conditions = [
                Condition(0, "x", str(rng.choice(["<=", ">"])), float(rng.choice(self.GRID)),
                          float32_input=bool(rng.random() < 0.5))
                for _ in range(rng.integers(2, 5))
            ]
            same, kept = self.same_rows(conditions)
            assert same, (conditions, kept)

    def test_membership_condition_is_kept(self):
        conditions = [Condition(0, "x", "in", [0.0, 1.0]), Condition(0, "x", ">", -1.0)]
        same, kept = self.same_rows(conditions)
        assert same and len(kept) == 2

    def test_tie_keeps_the_strict_bound(self):
        assert simplify_conditions([Condition(0, "x", ">", 1.0), Condition(0, "x", ">=", 1.0)]) == [Condition(0, "x", ">", 1.0)]
        assert simplify_conditions([Condition(0, "x", "<", 1.0), Condition(0, "x", "<=", 1.0)]) == [Condition(0, "x", "<", 1.0)]


class TestPathRouting:
    """The path has to end where predict goes, also on versions of sklearn whose decision_path does not."""

    @pytest.mark.parametrize("seed", range(3))
    def test_path_ends_in_the_leaf_of_apply(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(1500, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        X[rng.random(X.shape) < 0.2] = np.nan
        tree = DecisionTreeClassifier(max_depth=6, random_state=seed).fit(X, y)
        leaves = tree.apply(X)
        for i, x in enumerate(X[:400]):
            assert tree_utils._path_nodes(tree, x)[-1] == leaves[i]
            assert get_decision_path(tree, x).evaluate(x)

    def test_each_step_follows_a_child(self):
        X, y = data()
        tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
        nodes = tree_utils._path_nodes(tree, X[0])
        assert nodes[0] == 0 and nodes[-1] == tree.apply(X[:1])[0]
        for parent, child in zip(nodes, nodes[1:]):
            assert child in (tree.tree_.children_left[parent], tree.tree_.children_right[parent])


class TestRegressionTrees:
    def test_paths_and_rules_of_a_regressor(self):
        X, y = data()
        target = y + 0.5 * X[:, 3]
        tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, target)
        rule = get_decision_path(tree, X[0])
        assert rule.prediction == pytest.approx(tree.predict(X[:1])[0])
        assert rule.confidence is None

        rules = extract_rules_from_tree(tree)
        assert len(rules) == tree.get_n_leaves()
        membership = np.array([r.evaluate_batch(X) for r in rules])
        assert (membership.sum(axis=0) == 1).all()
        predicted = np.array([r.prediction for r in rules])[membership.argmax(axis=0)]
        assert predicted == pytest.approx(tree.predict(X))


class TestHelpers:
    @pytest.fixture
    def tree(self):
        X, y = data()
        return DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y), X

    def test_leaf_helpers_agree_with_apply(self, tree):
        model, X = tree
        leaf = get_leaf_id(model, X[3])
        assert leaf == model.apply(X[3:4])[0]
        samples = get_leaf_samples(model, X, leaf)
        assert 3 in samples and (model.apply(X[samples]) == leaf).all()
        assert len(samples) == (model.apply(X) == leaf).sum()

    def test_size_helpers_agree_with_sklearn(self, tree):
        model, _ = tree
        assert get_tree_depth(model) == model.get_depth()
        assert get_n_leaves(model) == model.get_n_leaves()

    def test_path_importance_on_a_hand_tree(self):
        # Root: 8 rows with gini 0.5, split into two pure children of 4, so the decrease is 0.5
        X = np.array([[0.0]] * 4 + [[1.0]] * 4)
        model = DecisionTreeClassifier(random_state=0).fit(X, [0] * 4 + [1] * 4)
        assert get_feature_importance_path(model, X[0]) == pytest.approx({0: 0.5})

    def test_path_importance_adds_up_over_a_repeated_feature(self):
        # The path splits on x twice. Decreases at the nodes: 0.375 - 0 and then 0.5 - 0.
        X = np.array([[0.0]] * 2 + [[1.0]] * 2 + [[2.0]] * 4)
        model = DecisionTreeClassifier(random_state=0).fit(X, [0, 0, 1, 1, 0, 0, 0, 0])
        node = model.tree_
        expected = 0.0
        for n in model.decision_path(X[2:3]).indices[:-1]:
            left, right = node.children_left[n], node.children_right[n]
            total = node.n_node_samples[n]
            expected += node.impurity[n] - (
                node.n_node_samples[left] * node.impurity[left] + node.n_node_samples[right] * node.impurity[right]
            ) / total
        assert get_feature_importance_path(model, X[2]) == pytest.approx({0: expected})

    def test_sklearn_without_missing_support_means_no_missing_branch(self):
        assert tree_utils._matches_missing(types.SimpleNamespace(), 0, True) is False


class TestModelKinds:
    def test_sklearn_kinds(self):
        X, y = data(100)
        single = DecisionTreeClassifier(max_depth=2).fit(X, y)
        forest = RandomForestClassifier(n_estimators=2, random_state=0).fit(X, y)
        boosted = GradientBoostingClassifier(n_estimators=2).fit(X, y)
        linear = LogisticRegression().fit(X, y)
        assert is_tree_model(single) and is_single_tree(single) and not is_ensemble_tree(single)
        assert is_tree_model(DecisionTreeRegressor()) and is_tree_model(RandomForestRegressor())
        for model in (forest, boosted):
            assert is_tree_model(model) and is_ensemble_tree(model) and not is_single_tree(model)
        assert not is_tree_model(linear) and not is_ensemble_tree(linear) and not is_single_tree(linear)

    @pytest.mark.parametrize("module, classifier, regressor", [
        ("xgboost", "XGBClassifier", "XGBRegressor"),
        ("lightgbm", "LGBMClassifier", "LGBMRegressor"),
    ])
    def test_optional_libraries_present_and_absent(self, monkeypatch, module, classifier, regressor):
        fake = types.ModuleType(module)
        Classifier = type(classifier, (), {})
        setattr(fake, classifier, Classifier)
        setattr(fake, regressor, type(regressor, (), {}))
        other = "lightgbm" if module == "xgboost" else "xgboost"
        monkeypatch.setitem(sys.modules, other, None)

        monkeypatch.setitem(sys.modules, module, fake)
        assert is_tree_model(Classifier()) and is_ensemble_tree(Classifier())

        monkeypatch.setitem(sys.modules, module, None)
        assert not is_tree_model(Classifier()) and not is_ensemble_tree(Classifier())
