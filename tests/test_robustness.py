"""
Properties every explainer must hold, on messy input as well as clean input.

The rule of thumb behind these tests: a rule must be true on its own row, must
select what the tree selects, and a number must not claim evidence it does not have.
"""

import itertools
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from gpe import (
    CertifiedGPE,
    Condition,
    Explanation,
    GPEEnsemble,
    GPEExplainer,
    GPEMonotonic,
    GPEOptimal,
    GPEWeighted,
    Rule,
    compile_explanations,
)
from gpe.metrics import aggregate_metrics, compare_explanations, precision_score
from gpe.novel_methods import (
    GPECounterfactual,
    GPEInformationTheoretic,
    GPEMultiResolution,
    GPEPareto,
    GPEStable,
)
from gpe.tree_utils import (
    extract_rules_from_tree,
    get_leaf_paths,
    representative_instance,
    simplify_conditions,
)

TAU = 0.95


def make_clean(n=3000, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = ((X[:, 0] + 0.5 * X[:, 1] - X[:, 2] * (X[:, 3] > 0)) + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def make_missing(n=3000, d=6, seed=0, share=0.2):
    """Data where being missing carries signal, so trees split on it."""
    X, y = make_clean(n, d, seed)
    rng = np.random.default_rng(seed + 1)
    holes = rng.random(X.shape) < share
    holes[:, 4:] = False
    y = np.where(holes[:, 0], 1 - y * (rng.random(n) < 0.1), y)
    X = X.copy()
    X[holes] = np.nan
    return X, y


@pytest.fixture(scope="module")
def clean():
    X, y = make_clean()
    tree = DecisionTreeClassifier(max_depth=7, random_state=0).fit(X[:2000], y[:2000])
    return tree, X[:2000], X[2000:]


@pytest.fixture(scope="module")
def missing():
    X, y = make_missing()
    tree = DecisionTreeClassifier(max_depth=7, random_state=0).fit(X[:2000], y[:2000])
    return tree, X[:2000], X[2000:]


EXPLAINERS = {
    "core": lambda t, X: GPEExplainer(t, X_train=X, min_precision=TAU),
    "it": lambda t, X: GPEInformationTheoretic(t, X_train=X, min_precision=TAU),
    "optimal": lambda t, X: GPEOptimal(t, X_train=X, min_precision=TAU),
    "weighted": lambda t, X: GPEWeighted(t, X_train=X, min_precision=TAU),
    "certified": lambda t, X: CertifiedGPE(t, X, min_precision=TAU),
    "chain": lambda t, X: CertifiedGPE(t, X, min_precision=TAU, alpha_scheme="chain"),
}


def rows_with_missing(X, count):
    return X[np.isnan(X).any(axis=1)][:count]


class TestMissingValues:
    def test_tree_really_routes_missing_values(self, missing):
        tree, _, X_te = missing
        assert np.isnan(X_te).any()
        assert tree.tree_.node_count > 3

    @pytest.mark.parametrize("name", sorted(EXPLAINERS))
    def test_rule_is_true_on_its_own_row(self, missing, name):
        tree, X_tr, X_te = missing
        explainer = EXPLAINERS[name](tree, X_tr)
        for x in rows_with_missing(X_te, 60):
            exp = explainer.explain(x)
            assert exp.rule.evaluate(x), str(exp.rule)
            assert exp.rule.evaluate_batch(x.reshape(1, -1))[0], str(exp.rule)
            assert exp.original_rule.evaluate(x)

    def test_full_path_selects_exactly_its_leaf(self, missing):
        tree, _, X_te = missing
        leaves = tree.apply(X_te)
        for leaf, path in get_leaf_paths(tree).items():
            mask = Rule(conditions=[c for _, c in path]).evaluate_batch(X_te)
            assert np.array_equal(mask, leaves == leaf)

    def test_simplifying_keeps_the_region(self, missing):
        tree, _, X_te = missing
        for leaf, path in get_leaf_paths(tree).items():
            raw = [c for _, c in path]
            region = Rule(conditions=raw).evaluate_batch(X_te)
            assert np.array_equal(Rule(conditions=simplify_conditions(raw)).evaluate_batch(X_te), region)

    @pytest.mark.parametrize("name", sorted(EXPLAINERS))
    def test_precision_is_recomputed_exactly(self, missing, name):
        tree, X_tr, X_te = missing
        explainer = EXPLAINERS[name](tree, X_tr)
        for x in X_te[:60]:
            exp = explainer.explain(x)
            mask = exp.rule.evaluate_batch(X_tr)
            if mask.any():
                assert exp.precision == pytest.approx((tree.predict(X_tr[mask]) == exp.prediction).mean())
            assert exp.coverage == pytest.approx(mask.mean())

    def test_gpe_reaches_tau_with_missing_values(self, missing):
        tree, X_tr, X_te = missing
        explainer = GPEExplainer(tree, X_train=X_tr, min_precision=TAU)
        for x in X_te[:200]:
            exp = explainer.explain(x)
            assert exp.metadata["n_background"] == 0 or exp.precision >= TAU

    def test_missingness_only_split(self):
        """A split that separates NaN from all values has threshold +inf."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(800, 2))
        y = (rng.random(800) < 0.5).astype(int)
        X[y == 1, 0] = np.nan
        tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
        assert np.isinf(tree.tree_.threshold[0])

        for x in X[:50]:
            exp = GPEExplainer(tree, X_train=X, min_precision=TAU).explain(x)
            assert exp.rule.evaluate(x)
            assert exp.prediction == (1 if np.isnan(x[0]) else 0)
        table = compile_explanations(GPEExplainer(tree, X_train=X, min_precision=TAU))
        assert len(table.entries) == tree.get_n_leaves()
        for x in X[:50]:
            assert table.explain(x).rule.evaluate(x)

    @pytest.mark.parametrize("name", ["core", "it", "certified"])
    def test_compiled_table_matches_online(self, missing, name):
        tree, X_tr, X_te = missing
        explainer = EXPLAINERS[name](tree, X_tr)
        table = compile_explanations(explainer)
        assert set(table.entries) == {int(i) for i in np.flatnonzero(tree.tree_.children_left == -1)}
        for x in np.vstack([rows_with_missing(X_te, 150), X_te[:50]]):
            online, compiled = explainer.explain(x), table.explain(x)
            assert compiled.rule.conditions == online.rule.conditions
            assert compiled.rule.evaluate(x)

    def test_representative_reaches_each_leaf(self, missing):
        tree, _, _ = missing
        for leaf, path in get_leaf_paths(tree).items():
            x = representative_instance(tree, path)
            assert int(tree.apply(x.reshape(1, -1))[0]) == leaf

    def test_text_mentions_missing_value(self, missing):
        tree, X_tr, X_te = missing
        explainer = GPEExplainer(tree, X_train=X_tr, min_precision=1.0)
        texts = [
            (exp, str(exp.rule))
            for exp in (explainer.explain(x) for x in rows_with_missing(X_te, 100))
        ]
        with_missing = [(exp, t) for exp, t in texts if any(c.missing_in_instance for c in exp.rule.conditions)]
        assert with_missing
        for exp, text in with_missing:
            assert "(or missing)" in text
            assert "is missing" in exp.to_natural_language()

    def test_clean_instance_text_stays_clean(self, clean):
        tree, X_tr, X_te = clean
        exp = GPEExplainer(tree, X_train=X_tr).explain(X_te[0])
        assert "missing" not in str(exp.rule)

    def test_evaluate_agrees_with_evaluate_batch(self, missing):
        tree, _, X_te = missing
        for path in list(get_leaf_paths(tree).values())[:30]:
            rule = Rule(conditions=[c for _, c in path])
            batch = rule.evaluate_batch(X_te[:200])
            assert [bool(rule.evaluate(x)) for x in X_te[:200]] == batch.tolist()


def all_explanations(tree, X_tr, x):
    forest = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0).fit(
        np.nan_to_num(X_tr), tree.predict(X_tr)
    )
    yield GPEExplainer(tree, X_train=X_tr).explain(x)
    yield GPEExplainer(tree).explain(x)
    yield GPEInformationTheoretic(tree, X_train=X_tr).explain(x)
    yield GPEOptimal(tree, X_train=X_tr).explain(x)
    yield GPEWeighted(tree, X_train=X_tr).explain(x)
    yield GPEMonotonic(tree, X_train=X_tr).explain(x)
    yield GPEStable(tree, X_train=X_tr[:200], n_perturbations=2).explain(x)
    yield GPEPareto(tree, X_train=X_tr, n_candidates=3).explain(x)
    yield GPEEnsemble(forest, X_train=np.nan_to_num(X_tr)).explain(np.nan_to_num(x))
    yield CertifiedGPE(tree, X_tr).explain(x)
    yield CertifiedGPE(tree, X_tr, min_lift=0.01).explain(x)
    yield compile_explanations(GPEExplainer(tree, X_train=X_tr)).explain(x)
    yield GPECounterfactual(tree, X_train=X_tr).explain_with_counterfactual(x).explanation
    yield GPEMultiResolution(tree, X_train=X_tr).explain_multi(x).levels[0]


class TestSerialization:
    @pytest.mark.parametrize("data_name", ["clean", "missing"])
    def test_strict_json_round_trip(self, request, data_name):
        tree, X_tr, X_te = request.getfixturevalue(data_name)
        x = rows_with_missing(X_te, 1)[0] if data_name == "missing" else X_te[0]
        for exp in all_explanations(tree, X_tr, x):
            text = json.dumps(exp.to_dict(), allow_nan=False)
            back = Explanation.from_dict(json.loads(text))
            assert back.prediction == exp.prediction
            assert back.rule.conditions == exp.rule.conditions
            assert back.precision == pytest.approx(exp.precision, nan_ok=True)
            assert np.array_equal(back.instance, exp.instance, equal_nan=True)
            assert back.certificate == exp.certificate
            if exp.original_rule is not None:
                assert back.original_rule.conditions == exp.original_rule.conditions

    def test_infinite_threshold_survives(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(400, 2))
        y = (rng.random(400) < 0.5).astype(int)
        X[y == 1, 0] = np.nan
        tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
        exp = GPEExplainer(tree, X_train=X).explain(X[y == 1][0])
        text = json.dumps(exp.to_dict(), allow_nan=False)
        back = Explanation.from_dict(json.loads(text))
        assert np.array_equal(back.rule.evaluate_batch(X), exp.rule.evaluate_batch(X))

    def test_string_classes(self, clean):
        tree, X_tr, X_te = clean
        labels = np.array(["no", "yes"])[tree.predict(X_tr)]
        named = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, labels)
        exp = GPEExplainer(named, X_train=X_tr).explain(X_te[0])
        assert json.loads(json.dumps(exp.to_dict()))["prediction"] in ("no", "yes")


class TestNoEvidence:
    def test_empty_region_is_nan_not_one(self):
        X = np.array([[0.0], [1.0]] * 20)
        tree = DecisionTreeClassifier(random_state=0).fit(X, [0, 1] * 20)
        exp = GPEExplainer(tree).explain(np.array([0.0]), np.array([[1.0]] * 5))
        assert np.isnan(exp.precision)
        assert exp.metadata["n_background"] == 0
        assert exp.metadata["meets_threshold"] is False
        assert "n/a" in str(exp)
        assert "unknown" in exp.to_natural_language()

    def test_no_background_reports_nothing_measured(self, clean):
        tree, _, X_te = clean
        exp = GPEExplainer(tree).explain(X_te[0])
        assert np.isnan(exp.precision) and np.isnan(exp.coverage)
        assert exp.metadata["n_background"] is None
        assert exp.complexity == exp.original_rule.complexity

    def test_empty_region_is_never_pruned_towards(self, clean):
        """A leaf missing from the background keeps its path until rows appear."""
        tree, X_tr, X_te = clean
        leaves = tree.apply(X_tr)
        for leaf in np.flatnonzero(tree.tree_.children_left == -1)[:10]:
            background = X_tr[leaves != leaf]
            row = X_tr[leaves == leaf]
            if len(row) == 0:
                continue
            exp = GPEExplainer(tree, min_precision=TAU).explain(row[0], background)
            n_rows = exp.rule.evaluate_batch(background).sum()
            assert (n_rows > 0 and exp.precision >= TAU) or (
                n_rows == 0 and exp.complexity == exp.original_rule.complexity
            )

    def test_metrics_skip_undefined_rows(self, clean):
        tree, X_tr, X_te = clean
        empty = Explanation(instance=X_te[0], rule=Rule([Condition(0, "a", ">", 1e9)]), prediction=1)
        assert np.isnan(precision_score(empty, tree, X_tr))
        normal = GPEExplainer(tree, X_train=X_tr).explain(X_te[0])
        agg = aggregate_metrics([empty, normal], tree, X_tr)
        assert agg["precision"]["n"] == 1
        assert agg["precision"]["mean"] == pytest.approx(normal.precision)
        cmp = compare_explanations([empty, normal, normal], [normal, normal, normal], tree, X_tr)
        assert cmp["precision"]["n_pairs"] == 2


class TestBadInput:
    def test_dataframe_background_gives_same_result(self, clean):
        tree, X_tr, X_te = clean
        names = [f"f{i}" for i in range(X_tr.shape[1])]
        df = pd.DataFrame(X_tr, columns=names)
        explainer = GPEExplainer(tree, feature_names=names, min_precision=TAU)
        for a, b in zip(
            explainer.explain_batch(X_te[:30], X_tr),
            explainer.explain_batch(pd.DataFrame(X_te[:30], columns=names), df),
        ):
            assert a.rule.conditions == b.rule.conditions
            assert a.precision == b.precision
        assert GPEExplainer(tree, X_train=df).explain(X_te[0]).complexity >= 1

    def test_tree_fitted_on_dataframe_checks_columns(self, clean):
        tree, X_tr, X_te = clean
        names = [f"f{i}" for i in range(X_tr.shape[1])]
        fitted = DecisionTreeClassifier(max_depth=4, random_state=0).fit(
            pd.DataFrame(X_tr, columns=names), tree.predict(X_tr)
        )
        explainer = GPEExplainer(fitted)
        assert explainer.feature_names == names
        with pytest.raises(ValueError, match="columns"):
            explainer.explain(X_te[0], pd.DataFrame(X_tr, columns=names[::-1]))

    @pytest.mark.parametrize("background, message", [
        (np.empty((0, 6)), "empty"),
        (np.zeros((5, 3)), "features"),
        (np.zeros(6), "2D"),
        ([["a"] * 6], "numbers"),
    ])
    def test_bad_background_is_explained(self, clean, background, message):
        tree, _, X_te = clean
        with pytest.raises(ValueError, match=message):
            GPEExplainer(tree).explain(X_te[0], background)

    def test_bad_instance_is_explained(self, clean):
        tree, X_tr, _ = clean
        with pytest.raises(ValueError, match="features"):
            GPEExplainer(tree, X_train=X_tr).explain(np.zeros(3))
        with pytest.raises(ValueError, match="numbers"):
            GPEExplainer(tree, X_train=X_tr).explain(["a"] * 6)

    def test_bad_feature_names(self, clean):
        tree, _, _ = clean
        with pytest.raises(ValueError, match="feature_names"):
            GPEExplainer(tree, feature_names=["a", "b"])

    @pytest.mark.parametrize("tau", [0, -0.1, 1.5, float("nan"), "high", None])
    def test_bad_min_precision(self, clean, tau):
        tree, _, _ = clean
        with pytest.raises(ValueError, match="min_precision"):
            GPEExplainer(tree, min_precision=tau)

    def test_regression_tree_is_rejected(self, clean):
        _, X_tr, _ = clean
        regressor = DecisionTreeRegressor(max_depth=2).fit(X_tr, X_tr[:, 0])
        with pytest.raises(ValueError, match="DecisionTreeClassifier"):
            GPEExplainer(regressor)

    def test_unfitted_tree_is_rejected(self):
        with pytest.raises(NotFittedError):
            GPEExplainer(DecisionTreeClassifier())

    def test_multi_output_is_rejected(self, clean):
        _, X_tr, _ = clean
        y = np.column_stack([X_tr[:, 0] > 0, X_tr[:, 1] > 0]).astype(int)
        with pytest.raises(ValueError, match="single-output"):
            GPEExplainer(DecisionTreeClassifier(max_depth=2).fit(X_tr, y))

    def test_ensemble_rejects_what_it_cannot_explain(self, clean):
        _, X_tr, _ = clean
        forest = RandomForestRegressor(n_estimators=3, max_depth=2).fit(X_tr, X_tr[:, 0])
        with pytest.raises(ValueError):
            GPEEnsemble(forest)
        with pytest.raises(NotFittedError):
            GPEEnsemble(RandomForestClassifier())

    def test_certified_takes_dataframe(self, clean):
        tree, X_tr, X_te = clean
        a = CertifiedGPE(tree, X_tr).explain(X_te[0])
        b = CertifiedGPE(tree, pd.DataFrame(X_tr)).explain(X_te[0])
        assert a.rule.conditions == b.rule.conditions


class TestTextAndHonesty:
    def test_rule_text_does_not_call_purity_confidence(self, clean):
        tree, X_tr, X_te = clean
        for x in X_te[:50]:
            exp = GPEExplainer(tree, X_train=X_tr).explain(x)
            assert "confidence" not in str(exp.rule)
            assert "leaf purity" in str(exp.rule)

    def test_optimal_keeps_purity_in_confidence(self, clean):
        tree, X_tr, X_te = clean
        for x in X_te[:20]:
            exp = GPEOptimal(tree, X_train=X_tr).explain(x)
            assert exp.rule.confidence == exp.original_rule.confidence

    def test_natural_language_does_not_claim_accuracy(self, clean):
        tree, X_tr, X_te = clean
        assert "accuracy" not in GPEExplainer(tree, X_train=X_tr).explain(X_te[0]).to_natural_language()

    @pytest.mark.parametrize("cls", [GPEStable, GPEMultiResolution, GPECounterfactual, GPEPareto])
    def test_no_unsupported_claims_in_docstrings(self, cls):
        assert "First " not in cls.__doc__
        assert "formal stability guarantees" not in cls.__doc__


class TestGreedyCore:
    """Facts about the algorithm itself, checked against independent recomputation."""

    def test_rule_is_a_subset_of_the_path(self, clean):
        tree, X_tr, X_te = clean
        for x in X_te[:100]:
            exp = GPEExplainer(tree, X_train=X_tr).explain(x)
            assert all(c in exp.original_rule.conditions for c in exp.rule.conditions)

    def test_no_further_literal_can_be_dropped(self, clean):
        tree, X_tr, X_te = clean
        pred = tree.predict(X_tr)
        for x in X_te[:100]:
            exp = GPEExplainer(tree, X_train=X_tr, min_precision=TAU).explain(x)
            conds = exp.rule.conditions
            if len(conds) < 2:
                continue
            for i in range(len(conds)):
                mask = Rule(conds[:i] + conds[i + 1:]).evaluate_batch(X_tr)
                assert (pred[mask] == exp.prediction).mean() < TAU

    def test_optimal_is_never_longer_and_is_minimal(self, clean):
        tree, X_tr, X_te = clean
        pred = tree.predict(X_tr)
        for x in X_te[:50]:
            greedy = GPEExplainer(tree, X_train=X_tr, min_precision=TAU).explain(x)
            optimal = GPEOptimal(tree, X_train=X_tr, min_precision=TAU).explain(x)
            assert optimal.complexity <= greedy.complexity
            path = optimal.original_rule.conditions
            for size in range(1, optimal.complexity):
                for subset in itertools.combinations(path, size):
                    mask = Rule(list(subset)).evaluate_batch(X_tr)
                    assert not mask.any() or (pred[mask] == optimal.prediction).mean() < TAU

    def test_background_row_order_and_duplicates_change_nothing(self, clean):
        tree, X_tr, X_te = clean
        order = np.random.default_rng(0).permutation(len(X_tr))
        explainer = GPEExplainer(tree, min_precision=TAU)
        for x in X_te[:50]:
            base = explainer.explain(x, X_tr)
            assert explainer.explain(x, X_tr[order]).rule.conditions == base.rule.conditions
            assert explainer.explain(x, np.vstack([X_tr, X_tr])).rule.conditions == base.rule.conditions

    def test_explanation_does_not_depend_on_x_beyond_its_leaf(self, clean):
        tree, X_tr, X_te = clean
        leaves = tree.apply(X_te)
        explainer = GPEExplainer(tree, X_train=X_tr, min_precision=TAU)
        for leaf in np.unique(leaves)[:15]:
            rows = X_te[leaves == leaf][:5]
            rules = [explainer.explain(x).rule.conditions for x in rows]
            assert all(r == rules[0] for r in rules)

    def test_tau_one_keeps_precision_one(self, clean):
        tree, X_tr, X_te = clean
        for x in X_te[:50]:
            exp = GPEExplainer(tree, X_train=X_tr, min_precision=1.0).explain(x)
            assert exp.metadata["n_background"] == 0 or exp.precision == 1.0

    def test_sklearn_wrapper(self, clean):
        from gpe.core import GPEExplainerSKLearn

        tree, X_tr, X_te = clean
        wrapper = GPEExplainerSKLearn(min_precision=TAU).fit(tree, X_tr)
        got = wrapper.transform(X_te[:5])
        assert [e.rule.conditions for e in got] == [
            GPEExplainer(tree, X_train=X_tr, min_precision=TAU).explain(x).rule.conditions for x in X_te[:5]
        ]
        with pytest.raises(ValueError):
            GPEExplainerSKLearn().transform(X_te[:1])

    def test_feature_importance_sums_to_one(self, clean):
        tree, X_tr, X_te = clean
        importance = GPEExplainer(tree, X_train=X_tr).get_feature_importance(X_te[:30])
        assert sum(importance.values()) == pytest.approx(1.0)

    def test_extracted_rules_partition_the_data(self, missing):
        tree, _, X_te = missing
        rules = extract_rules_from_tree(tree)
        assert len(rules) == tree.get_n_leaves()
        hits = np.sum([r.evaluate_batch(X_te) for r in rules], axis=0)
        assert (hits == 1).all()


class TestEnsemble:
    @pytest.fixture
    def forest(self, clean):
        _, X_tr, X_te = clean
        y = (X_tr[:, 0] + X_tr[:, 1] > 0).astype(int)
        return RandomForestClassifier(n_estimators=15, max_depth=5, random_state=0).fit(X_tr, y), X_tr, X_te

    def test_rule_is_true_on_its_own_row(self, forest):
        model, X_tr, X_te = forest
        explainer = GPEEnsemble(model, X_train=X_tr)
        for x in X_te[:50]:
            exp = explainer.explain(x)
            assert exp.rule.evaluate(x)

    def test_missing_value_stays_inside_the_rule(self, clean):
        _, X_tr, X_te = clean
        X = X_tr.copy()
        X[::5, 0] = np.nan
        y = (np.nan_to_num(X_tr[:, 0]) > 0).astype(int)
        model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0).fit(X, y)
        explainer = GPEEnsemble(model, X_train=X)
        for x in X[:25]:
            assert explainer.explain(x).rule.evaluate(x)

    def test_meets_threshold_tells_the_truth(self, forest):
        model, X_tr, X_te = forest
        explainer = GPEEnsemble(model, X_train=X_tr, min_precision=TAU)
        for x in X_te[:100]:
            exp = explainer.explain(x)
            assert exp.metadata["meets_threshold"] == bool(exp.precision >= TAU)
            mask = exp.rule.evaluate_batch(X_tr)
            if mask.any():
                assert exp.precision == pytest.approx((model.predict(X_tr[mask]) == exp.prediction).mean())

    def test_gradient_boosting(self, clean):
        _, X_tr, X_te = clean
        y = (X_tr[:, 0] > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=0).fit(X_tr, y)
        assert GPEEnsemble(model, X_train=X_tr).explain(X_te[0]).rule.evaluate(X_te[0])

    def test_dataframe_background(self, forest):
        model, X_tr, X_te = forest
        explainer = GPEEnsemble(model)
        a = explainer.explain(X_te[0], X_tr)
        b = explainer.explain(X_te[0], pd.DataFrame(X_tr))
        assert a.rule.conditions == b.rule.conditions


class TestCounterfactual:
    @pytest.fixture
    def small(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 10, size=(600, 3))
        y = ((X[:, 0] > 4) & (X[:, 1] < 6) | (X[:, 2] > 8)).astype(int)
        tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
        return tree, X

    def test_changes_the_class(self, small):
        tree, X = small
        explainer = GPECounterfactual(tree, X_train=X)
        for x in X[:150]:
            cf = explainer.explain_with_counterfactual(x)
            assert cf.found
            assert cf.counterfactual_prediction != cf.explanation.prediction
            assert tree.predict(cf.counterfactual.reshape(1, -1))[0] == cf.counterfactual_prediction
            assert cf.distance == pytest.approx(np.abs(cf.counterfactual - x).sum())
            for name, (old, new) in cf.changes.items():
                j = explainer.feature_names.index(name)
                assert old == x[j] and new == cf.counterfactual[j]

    def test_is_minimal_against_brute_force(self, small):
        """Scan every cell of the grid induced by the tree's thresholds."""
        tree, X = small
        explainer = GPECounterfactual(tree, X_train=X)
        low, high = X.min(axis=0), X.max(axis=0)
        cuts = [
            sorted(set(tree.tree_.threshold[tree.tree_.feature == j].tolist()))
            for j in range(X.shape[1])
        ]
        for x in X[:12]:
            cf = explainer.explain_with_counterfactual(x)
            eps = 1e-5
            values = [
                sorted({float(x[j])} | {t for t in cuts[j]} | {t + eps for t in cuts[j]})
                for j in range(X.shape[1])
            ]
            grid = np.array(list(itertools.product(*values)))
            other = tree.predict(grid) != cf.explanation.prediction
            cost = (np.abs(grid[other] - x) / (high - low)).sum(axis=1).min()
            found = (np.abs(cf.counterfactual - x) / (high - low)).sum()
            assert found <= cost + 2e-5

    def test_respects_actionable_features(self, small):
        tree, X = small
        explainer = GPECounterfactual(tree, X_train=X, actionable_features=["feature_2"])
        for x in X[:60]:
            cf = explainer.explain_with_counterfactual(x)
            assert set(cf.changes) <= {"feature_2"}
            assert not cf.found or cf.counterfactual_prediction != cf.explanation.prediction

    def test_respects_ranges(self, small):
        tree, X = small
        limits = {"feature_0": (0.0, 10.0), "feature_1": (0.0, 10.0), "feature_2": (0.0, 8.5)}
        explainer = GPECounterfactual(tree, X_train=X, feature_ranges=dict(limits))
        for x in X[:60]:
            cf = explainer.explain_with_counterfactual(x)
            for name, (_, new) in cf.changes.items():
                assert limits[name][0] <= new <= limits[name][1]

    def test_reports_when_there_is_none(self, small):
        tree, X = small
        explainer = GPECounterfactual(tree, X_train=X, actionable_features=[])
        cf = explainer.explain_with_counterfactual(X[0])
        assert not cf.found and cf.changes == {} and np.array_equal(cf.counterfactual, X[0])
        assert "no counterfactual" in str(cf)

    def test_single_class_tree_has_none(self):
        X = np.random.default_rng(0).normal(size=(50, 2))
        tree = DecisionTreeClassifier().fit(X, np.zeros(50, dtype=int))
        assert not GPECounterfactual(tree, X_train=X).explain_with_counterfactual(X[0]).found

    def test_missing_value_in_instance(self, missing):
        tree, X_tr, X_te = missing
        explainer = GPECounterfactual(tree, X_train=X_tr)
        for x in rows_with_missing(X_te, 40):
            cf = explainer.explain_with_counterfactual(x)
            assert cf.found
            assert cf.counterfactual_prediction != cf.explanation.prediction
            assert np.isfinite(cf.distance)

    def test_string_shows_changes(self, small):
        tree, X = small
        assert "WHAT would change" in str(GPECounterfactual(tree, X_train=X).explain_with_counterfactual(X[0]))


class TestStable:
    def test_is_reproducible_and_leaves_global_rng_alone(self, clean):
        tree, X_tr, X_te = clean
        background = X_tr[:300]
        make = lambda: GPEStable(tree, X_train=background, n_perturbations=3, random_state=7)
        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        a = make().explain(X_te[0])
        assert np.array_equal(np.random.get_state()[1], before)
        np.random.seed(999)
        b = make().explain(X_te[0])
        assert a.rule.conditions == b.rule.conditions
        assert a.metadata["stability"] == b.metadata["stability"]
        assert 0 <= a.metadata["stability"] <= 1

    def test_precision_reaches_tau(self, clean):
        tree, X_tr, X_te = clean
        explainer = GPEStable(tree, X_train=X_tr[:300], n_perturbations=3, min_precision=TAU)
        for x in X_te[:10]:
            exp = explainer.explain(x)
            assert exp.metadata["n_background"] == 0 or exp.precision >= TAU


class TestMultiResolution:
    def test_each_level_meets_its_threshold(self, clean):
        tree, X_tr, X_te = clean
        explainer = GPEMultiResolution(tree, X_train=X_tr)
        for x in X_te[:50]:
            multi = explainer.explain_multi(x)
            assert len(multi.levels) == 3
            for threshold, level in zip(explainer.precision_thresholds, multi.levels):
                assert level.rule.evaluate(x)
                assert np.isnan(level.precision) or level.precision >= threshold
        assert "Level 1" in str(multi)
        assert multi.get_level(2) is multi.levels[1]
        with pytest.raises(ValueError):
            multi.get_level(4)

    def test_does_not_change_the_explainer(self, clean):
        tree, X_tr, X_te = clean
        explainer = GPEMultiResolution(tree, X_train=X_tr)
        before = explainer.min_precision
        explainer.explain_multi(X_te[0])
        assert explainer.min_precision == before

    def test_threshold_count_must_match(self, clean):
        tree, X_tr, _ = clean
        with pytest.raises(ValueError, match="levels"):
            GPEMultiResolution(tree, X_train=X_tr, n_levels=2, precision_thresholds=[0.9])
        with pytest.raises(ValueError, match="min_precision"):
            GPEMultiResolution(tree, X_train=X_tr, n_levels=1, precision_thresholds=[1.2])
        assert len(GPEMultiResolution(tree, X_train=X_tr, n_levels=4).precision_thresholds) == 4


class TestPareto:
    def test_front_is_not_dominated(self, clean):
        tree, X_tr, X_te = clean
        explainer = GPEPareto(tree, X_train=X_tr, n_candidates=8)
        for x in X_te[:15]:
            front = explainer.find_pareto_front(x)
            assert front
            for a, b in itertools.permutations(front, 2):
                assert not explainer._is_dominated(a.explanation, b.explanation)
            assert all(p.is_pareto_optimal and p.dominated_by == 0 for p in front)
            assert [p.explanation.complexity for p in front] == sorted(p.explanation.complexity for p in front)
            assert "ParetoExplanation" in str(front[0])

    def test_does_not_change_the_explainer(self, clean):
        tree, X_tr, X_te = clean
        explainer = GPEPareto(tree, X_train=X_tr, n_candidates=4)
        before = explainer.min_precision
        exp = explainer.explain(X_te[0])
        assert explainer.min_precision == before
        assert exp.rule.evaluate(X_te[0]) and exp.method == "GPE-Pareto"

    def test_needs_background(self, clean):
        tree, _, X_te = clean
        with pytest.raises(ValueError, match="background"):
            GPEPareto(tree).find_pareto_front(X_te[0])


class TestMonotonic:
    def test_direction_follows_the_predicted_class(self, clean):
        tree, X_tr, X_te = clean
        constraints = {"feature_0": +1}
        explainer = GPEMonotonic(tree, X_train=X_tr, monotonic_constraints=constraints, min_precision=TAU)
        seen = set()
        for x in X_te[:150]:
            exp = explainer.explain(x)
            for cond in exp.rule.conditions:
                if cond.feature_name == "feature_0":
                    wanted = ">" if exp.prediction == tree.classes_[1] else "<="
                    assert cond.operator == wanted
                    seen.add(exp.prediction)
        assert len(seen) == 2

    def test_flag_shows_when_the_constraint_costs_precision(self, clean):
        tree, X_tr, X_te = clean
        explainer = GPEMonotonic(
            tree, X_train=X_tr, min_precision=TAU,
            monotonic_constraints={f"feature_{i}": +1 for i in range(6)},
        )
        flags = [explainer.explain(x) for x in X_te[:100]]
        for exp in flags:
            assert exp.metadata["meets_threshold"] == bool(exp.precision >= TAU)
        assert any(not exp.metadata["meets_threshold"] for exp in flags)


class TestWeighted:
    def test_reaches_tau_and_keeps_heavy_features_longer(self, clean):
        tree, X_tr, X_te = clean
        plain = GPEWeighted(tree, X_train=X_tr, min_precision=TAU)
        heavy = GPEWeighted(
            tree, X_train=X_tr, min_precision=TAU,
            feature_weights={"feature_1": 100.0, "feature_0": 0.01},
        )
        count = lambda ex, name: sum(
            name in ex.explain(x).rule.feature_names_used for x in X_te[:150]
        )
        assert count(heavy, "feature_1") >= count(plain, "feature_1")
        for x in X_te[:50]:
            exp = heavy.explain(x)
            assert exp.metadata["n_background"] == 0 or exp.precision >= TAU


class TestVisualization:
    def test_everything_runs(self, clean, tmp_path):
        import matplotlib
        import matplotlib.pyplot

        matplotlib.use("Agg")
        import gpe.visualization as viz

        tree, X_tr, X_te = clean
        explainer = GPEExplainer(tree, X_train=X_tr)
        exp = explainer.explain(X_te[0])
        empty = GPEExplainer(tree).explain(X_te[0])
        batch = explainer.explain_batch(X_te[:20]) + [empty]

        for item in (exp, empty):
            assert "rule" in viz.explanation_to_text(item).lower()
            assert "<div" in viz.explanation_to_html(item)
            viz.print_explanation(item)
            assert viz.plot_explanation(item) is not None
        assert viz.plot_comparison({"a": exp, "b": batch[1]}) is not None
        assert viz.plot_metrics_summary(batch) is not None
        matplotlib.pyplot.close("all")


class TestGapsFoundByMutation:
    """Each of these was a planted defect that no test noticed."""

    @staticmethod
    def certified(**kwargs):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1500, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X[:500], y[:500])
        return CertifiedGPE(tree, X[500:], min_precision=0.9, delta=0.05, **kwargs), X

    def test_chain_budget_is_divided_by_the_leaves(self):
        method, _ = self.certified(alpha_scheme="chain")
        level = method._level_function(4, 0.05, None, 5)
        assert level(2) == pytest.approx(0.05 / 5)

    @pytest.mark.parametrize("interval, promised", [("clopper_pearson", True), ("hoeffding", True), ("wilson", False)])
    def test_wilson_is_never_reported_as_a_guarantee(self, interval, promised):
        method, X = self.certified(interval=interval)
        assert method.explain(X[1000]).certificate.guarantee is promised

    def test_negative_infinity_survives_json(self):
        condition = Condition(0, "a", ">", -np.inf)
        restored = Condition.from_dict(json.loads(json.dumps(condition.to_dict())))
        assert restored.threshold == -np.inf
