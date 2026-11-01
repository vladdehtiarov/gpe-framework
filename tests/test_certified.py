"""
Tests for CertifiedGPE and compile_explanations.

Author: Vladyslav Dehtiarov
"""

import itertools
import math

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from gpe import (
    CertifiedGPE,
    GPEExplainer,
    GPEInformationTheoretic,
    GPEOptimal,
    compile_explanations,
    lower_bound,
    min_coverage,
)
from gpe.certified import _superset_sum, clopper_pearson_upper
from gpe.explanation import Explanation, Rule
from gpe.tree_utils import get_decision_path, get_leaf_paths, simplify_conditions

TAU = 0.95
DELTA = 0.05


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=30000, n_features=12, n_informative=6, flip_y=0.02, random_state=0
    )
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
    X_cal, X_te, _, _ = train_test_split(X_rest, y_rest, test_size=0.1, random_state=0)
    tree = DecisionTreeClassifier(max_depth=7, random_state=0).fit(X_tr, y_tr)
    return tree, X_tr, X_cal, X_te


def counts(rule, X_cal, pred, c):
    mask = rule.evaluate_batch(X_cal)
    return int((pred[mask] == c).sum()), int(mask.sum())


class TestBounds:
    def test_clopper_pearson_closed_form(self):
        for n, alpha in [(1, 0.05), (59, 0.05), (194, 0.05 / 1022), (5000, 1e-9)]:
            assert lower_bound(n, n, alpha) == pytest.approx(alpha ** (1 / n), rel=1e-10)

    def test_empty_cases_give_zero(self):
        for interval in ("clopper_pearson", "hoeffding", "wilson"):
            assert lower_bound(0, 0, 0.05, interval) == 0.0
            assert lower_bound(0, 100, 0.05, interval) == 0.0

    def test_vector_input(self):
        out = lower_bound(np.array([0, 10, 10]), np.array([0, 10, 20]), 0.05)
        assert out.shape == (3,)
        assert out[0] == 0 and out[1] > out[2] > 0

    def test_bound_stays_below_estimate(self):
        for interval in ("clopper_pearson", "hoeffding", "wilson"):
            for K, n in [(1, 1), (9, 10), (950, 1000), (1000, 1000)]:
                assert lower_bound(K, n, 0.01, interval) <= K / n

    def test_tiny_alpha_is_conservative(self):
        assert lower_bound(10 ** 6, 10 ** 6, 0.0) == 0.0

    def test_unknown_interval(self):
        with pytest.raises(ValueError):
            lower_bound(1, 1, 0.05, "agresti")

    @pytest.mark.parametrize("interval", ["clopper_pearson", "hoeffding"])
    def test_exact_coverage_for_all_outcomes(self, interval):
        """P(p < L) <= alpha, summing the binomial law over every outcome."""
        from scipy.stats import binom

        alpha = 0.05
        for n, p in itertools.product([1, 5, 20, 100, 400], [0.1, 0.5, 0.8, 0.95, 0.99]):
            K = np.arange(n + 1)
            miss = p < lower_bound(K, np.full(n + 1, n), alpha, interval)
            assert binom.pmf(K[miss], n, p).sum() <= alpha + 1e-12

    def test_upper_bound_coverage(self):
        from scipy.stats import binom

        for n, p in itertools.product([5, 50, 500], [0.1, 0.5, 0.95]):
            miss = [K for K in range(n + 1) if p > clopper_pearson_upper(K, n, 0.05)]
            assert binom.pmf(miss, n, p).sum() <= 0.05 + 1e-12

    def test_min_coverage_values(self):
        # Checked numerically in the research notes (F, section 5)
        assert min_coverage(6) == 139
        assert min_coverage(10) == 194
        assert min_coverage(15) == 262

    @pytest.mark.parametrize("k", [2, 6, 10, 15])
    def test_min_coverage_is_tight(self, k):
        alpha = DELTA / (2 ** k - 2)
        n = min_coverage(k)
        assert lower_bound(n, n, alpha) >= TAU
        assert lower_bound(n - 1, n - 1, alpha) < TAU


class TestSupersetSum:
    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_against_direct_count(self, k):
        rng = np.random.default_rng(k)
        counts_ = rng.integers(0, 5, 1 << k).astype(np.int64)
        expected = np.array([
            sum(counts_[t] for t in range(1 << k) if t & s == s) for s in range(1 << k)
        ])
        assert (_superset_sum(counts_.copy(), k) == expected).all()


def brute_force(explainer, x, level):
    """Enumerate subsets directly: shortest certified, largest coverage first."""
    path = get_decision_path(explainer.model, x)
    conditions = simplify_conditions(path.conditions)
    c = explainer.model.predict(x.reshape(1, -1))[0]
    k = len(conditions)
    for size in range(1, k):
        passed = []
        for subset in itertools.combinations(range(k), size):
            rule = Rule([conditions[i] for i in subset])
            K, n = counts(rule, explainer.X_cal, explainer._cal_pred, c)
            if n and lower_bound(K, n, level(k, size)) >= explainer.min_precision:
                passed.append((n, K / n))
        if passed:
            return size, max(passed)
    return None


class TestSearch:
    @pytest.mark.parametrize("scheme,level", [
        ("uniform", lambda k, s: DELTA / (2 ** k - 2)),
        ("layered", lambda k, s: DELTA / ((k - 1) * math.comb(k, s))),
    ])
    def test_exact_matches_brute_force(self, data, scheme, level):
        tree, _, X_cal, X_te = data
        explainer = CertifiedGPE(tree, X_cal, delta=DELTA, alpha_scheme=scheme, search="exact")
        seen = set()
        for x in X_te:
            leaf = tree.apply(x.reshape(1, -1))[0]
            if leaf in seen:
                continue
            seen.add(leaf)
            exp = explainer.explain(x)
            expected = brute_force(explainer, x, level)
            if expected is None:
                assert exp.certificate.trivial
            else:
                assert not exp.certificate.trivial
                assert exp.complexity == expected[0]
                assert exp.certificate.n == expected[1][0]
                assert exp.certificate.K / exp.certificate.n == pytest.approx(expected[1][1])
        assert len(seen) > 10

    @pytest.mark.parametrize("search", ["exact", "greedy"])
    @pytest.mark.parametrize("scheme", ["uniform", "layered"])
    def test_certificate_is_consistent(self, data, search, scheme):
        tree, _, X_cal, X_te = data
        explainer = CertifiedGPE(tree, X_cal, delta=DELTA, alpha_scheme=scheme, search=search)
        pred = tree.predict(X_cal)
        for x in X_te[:100]:
            exp = explainer.explain(x)
            cert = exp.certificate
            K, n = counts(exp.rule, X_cal, pred, exp.prediction)
            assert (cert.K, cert.n) == (K, n)
            assert exp.coverage == n / len(X_cal)
            assert exp.rule.evaluate(x)
            if cert.trivial:
                assert cert.lower_bound == 1.0 and cert.alpha is None
                assert exp.complexity == exp.original_rule.complexity
            else:
                assert cert.search == search
                assert cert.lower_bound == pytest.approx(lower_bound(K, n, cert.alpha))
                assert cert.lower_bound >= TAU
                assert exp.complexity < exp.original_rule.complexity

    def test_exact_never_longer_than_greedy(self, data):
        tree, _, X_cal, X_te = data
        exact = CertifiedGPE(tree, X_cal, search="exact")
        greedy = CertifiedGPE(tree, X_cal, search="greedy")
        for x in X_te[:100]:
            assert exact.explain(x).complexity <= greedy.explain(x).complexity

    def test_greedy_is_one_minimal(self, data):
        """No literal of a greedy rule can be dropped under the same certificate."""
        tree, _, X_cal, X_te = data
        explainer = CertifiedGPE(tree, X_cal, search="greedy")
        pred = tree.predict(X_cal)
        for x in X_te[:50]:
            exp = explainer.explain(x)
            k = exp.original_rule.complexity
            if exp.complexity < 2 or k < 2:
                continue
            alpha = DELTA / (2 ** k - 2)
            for i in range(exp.complexity):
                rule = Rule(exp.rule.conditions[:i] + exp.rule.conditions[i + 1:])
                K, n = counts(rule, X_cal, pred, exp.prediction)
                assert lower_bound(K, n, alpha) < TAU

    def test_auto_switches_to_greedy(self, data):
        tree, _, X_cal, X_te = data
        explainer = CertifiedGPE(tree, X_cal, max_exact_literals=1, search="auto")
        k = max(explainer.explain(x).original_rule.complexity for x in X_te[:50])
        assert k >= 2
        x = next(x for x in X_te if explainer.explain(x).original_rule.complexity >= 2)
        assert explainer.explain(x).certificate.search == "greedy"

    def test_exact_refuses_long_path(self, data):
        tree, _, X_cal, X_te = data
        explainer = CertifiedGPE(tree, X_cal, max_exact_literals=1, search="exact")
        x = next(
            x for x in X_te
            if len(simplify_conditions(get_decision_path(tree, x).conditions)) >= 2
        )
        with pytest.raises(ValueError):
            explainer.explain(x)

    def test_chain_order_ignores_calibration(self, data):
        tree, _, X_cal, X_te = data
        a = CertifiedGPE(tree, X_cal, alpha_scheme="chain")
        b = CertifiedGPE(tree, X_cal[:500], alpha_scheme="chain")
        for x in X_te[:20]:
            conditions = simplify_conditions(get_decision_path(tree, x).conditions)
            assert a._chain_order(x, conditions) == b._chain_order(x, conditions)

    def test_chain_rule_is_nested_and_certified(self, data):
        tree, _, X_cal, X_te = data
        explainer = CertifiedGPE(tree, X_cal, alpha_scheme="chain")
        shorter = 0
        for x in X_te[:100]:
            exp = explainer.explain(x)
            cert = exp.certificate
            assert cert.search in ("chain", "none")
            if not cert.trivial:
                shorter += 1
                assert cert.alpha == DELTA
                assert cert.lower_bound >= TAU
                conditions = simplify_conditions(get_decision_path(tree, x).conditions)
                order = explainer._chain_order(x, conditions)
                removed = exp.original_rule.complexity - exp.complexity
                kept = sorted(order[removed:])
                assert [conditions[i] for i in kept] == exp.rule.conditions
        assert shorter > 0

    def test_chain_uses_fewer_rows_than_uniform(self):
        assert lower_bound(59, 59, DELTA) >= TAU
        assert lower_bound(59, 59, DELTA / (2 ** 10 - 2)) < TAU


def reference_greedy(explainer, x):
    """Plain re-implementation of the greedy search, to compare against."""
    conditions = simplify_conditions(get_decision_path(explainer.model, x).conditions)
    c = explainer.model.predict(x.reshape(1, -1))[0]
    k = len(conditions)
    alpha = DELTA / (2 ** k - 2) if k >= 2 else None
    current = list(range(k))
    while len(current) > 1:
        best = None
        for i in current:
            rule = Rule([conditions[j] for j in current if j != i])
            K, n = counts(rule, explainer.X_cal, explainer._cal_pred, c)
            if n and lower_bound(K, n, alpha) >= explainer.min_precision:
                score = (n, K / n, -i)
                if best is None or score > best[0]:
                    best = (score, i)
        if best is None:
            break
        current.remove(best[1])
    return [conditions[i] for i in current]


def reference_chain_order(tree, x, conditions):
    """Impurity decrease computed from the tree arrays, separately from the package."""
    t = tree.tree_
    nodes = tree.decision_path(x.reshape(1, -1)).indices[:-1]
    gain = {}
    for node in nodes:
        left, right = t.children_left[node], t.children_right[node]
        gain[(int(t.feature[node]), float(t.threshold[node]))] = (
            t.weighted_n_node_samples[node] * t.impurity[node]
            - t.weighted_n_node_samples[left] * t.impurity[left]
            - t.weighted_n_node_samples[right] * t.impurity[right]
        )
    values = [gain[(c.feature_index, float(c.threshold))] for c in conditions]
    return sorted(range(len(conditions)), key=lambda i: (values[i], i))


class TestReferenceSearches:
    def test_greedy_matches_reference(self, data):
        tree, _, X_cal, X_te = data
        explainer = CertifiedGPE(tree, X_cal, search="greedy")
        changed = 0
        for x in X_te[:150]:
            exp = explainer.explain(x)
            expected = reference_greedy(explainer, x)
            assert exp.rule.conditions == expected
            changed += len(expected) < exp.original_rule.complexity
        assert changed > 0

    def test_greedy_keeps_largest_coverage(self):
        """Two literals can go; the one whose removal keeps more rows must go first."""
        explainer = CertifiedGPE.__new__(CertifiedGPE)
        explainer.interval = "clopper_pearson"
        # Rows: 2000 satisfy all, 1000 fail only literal 0, 3000 fail only literal 1
        matrix = np.array([[True, True, True]] * 2000 + [[False, True, True]] * 1000
                          + [[True, False, True]] * 3000 + [[False, False, False]] * 500)
        agrees = np.array([True] * 6000 + [False] * 500)
        chosen = explainer._search_greedy(
            matrix, agrees, 3, lambda size: 0.01, lambda lower: lower >= 0.9
        )
        # Dropping literal 1 first keeps 5000 rows, then dropping literal 0 keeps 6000.
        assert chosen[0] == [2] and chosen[2] == 6000

    def test_chain_order_matches_reference(self, data):
        tree, _, X_cal, X_te = data
        explainer = CertifiedGPE(tree, X_cal, alpha_scheme="chain")
        distinct = 0
        for x in X_te[:100]:
            conditions = simplify_conditions(get_decision_path(tree, x).conditions)
            order = explainer._chain_order(x, conditions)
            assert order == reference_chain_order(tree, x, conditions)
            distinct += order != list(range(len(conditions)))
        assert distinct > 0

    def test_chain_stops_at_first_failure(self):
        """A later subset that would pass must not rescue the chain."""
        explainer = CertifiedGPE.__new__(CertifiedGPE)
        explainer.interval = "clopper_pearson"
        explainer._chain_order = lambda x, conditions: [0, 1, 2]
        # {1, 2} holds 100 disagreeing rows; {2} adds 100000 agreeing rows and would pass.
        matrix = np.array([[True, True, True]] * 100 + [[True, False, True]] * 100000)
        agrees = np.array([False] * 100 + [True] * 100000)
        found = explainer._search_chain(
            None, [None] * 3, matrix, agrees, lambda size: 0.05, lambda lower: lower >= 0.95
        )
        assert found is None

    def test_chain_returns_last_passing_subset(self):
        explainer = CertifiedGPE.__new__(CertifiedGPE)
        explainer.interval = "clopper_pearson"
        explainer._chain_order = lambda x, conditions: [0, 1, 2]
        # {1, 2} passes with 1000 clean rows; {2} adds rows that spoil it.
        matrix = np.array([[True, True, True]] * 1000 + [[True, False, True]] * 1000)
        agrees = np.array([True] * 1000 + [False] * 1000)
        found = explainer._search_chain(
            None, [None] * 3, matrix, agrees, lambda size: 0.05, lambda lower: lower >= 0.95
        )
        assert found[0] == [1, 2] and found[1:3] == (1000, 1000)


class TestDegenerate:
    def test_single_leaf_tree(self):
        X = np.random.default_rng(0).normal(size=(100, 2))
        tree = DecisionTreeClassifier().fit(X, np.zeros(100, dtype=int))
        exp = CertifiedGPE(tree, X).explain(np.array([0.1, 0.2]))
        assert exp.complexity == 0
        assert exp.certificate.trivial and exp.certificate.family_size == 0
        assert exp.precision == 1.0

    def test_one_literal_path_needs_no_test(self):
        X = np.array([[0.0], [1.0]] * 50)
        y = np.array([0, 1] * 50)
        tree = DecisionTreeClassifier().fit(X, y)
        exp = CertifiedGPE(tree, np.array([[0.2], [0.8]])).explain(np.array([0.1]))
        assert exp.complexity == 1
        assert exp.certificate.trivial and exp.certificate.alpha is None

    def test_too_little_data_keeps_full_path(self, data):
        tree, _, X_cal, X_te = data
        explainer = CertifiedGPE(tree, X_cal[:20])
        for exp in explainer.explain_batch(X_te[:50]):
            assert exp.certificate.trivial
            assert exp.certificate.lower_bound == 1.0

    def test_empty_leaf_reports_exact_full_path(self, data):
        tree, _, X_cal, _ = data
        paths = get_leaf_paths(tree)
        leaf_ids = set(tree.apply(X_cal[:5]).tolist())
        rows = X_cal[:5]
        x = next(
            row for row, leaf in zip(X_cal[5:], tree.apply(X_cal[5:])) if int(leaf) not in leaf_ids
        )
        exp = CertifiedGPE(tree, rows).explain(x)
        assert exp.certificate.n == 0
        assert exp.precision == 1.0 and exp.coverage == 0.0
        assert exp.complexity == len(simplify_conditions([c for _, c in paths[int(tree.apply(x.reshape(1, -1))[0])]]))

    def test_very_long_path_family_does_not_overflow(self, data):
        tree, _, X_cal, _ = data
        explainer = CertifiedGPE(tree, X_cal)
        level = explainer._level_function(1100, DELTA, None, 1)
        assert level(5) == 0.0
        assert float(lower_bound(10, 10, level(5))) == 0.0


class TestValidation:
    def test_requires_classifier(self, data):
        _, X_tr, X_cal, _ = data
        reg = DecisionTreeRegressor(max_depth=2).fit(X_tr, X_tr[:, 0])
        with pytest.raises(ValueError):
            CertifiedGPE(reg, X_cal)

    @pytest.mark.parametrize("kwargs", [
        {"delta": 0}, {"delta": 1}, {"min_precision": 1.5}, {"interval": "x"},
        {"alpha_scheme": "x"}, {"search": "x"}, {"max_exact_literals": 40},
        {"alpha_scheme": "chain", "min_lift": 0.1}, {"min_lift": 0.1, "lift_budget": 1.0},
    ])
    def test_bad_arguments(self, data, kwargs):
        tree, _, X_cal, _ = data
        with pytest.raises(ValueError):
            CertifiedGPE(tree, X_cal, **kwargs)

    def test_bad_calibration_shape(self, data):
        tree, _, X_cal, _ = data
        with pytest.raises(ValueError):
            CertifiedGPE(tree, X_cal[:, :3])
        with pytest.raises(ValueError):
            CertifiedGPE(tree, X_cal[0])

    def test_background_rejected(self, data):
        tree, X_tr, X_cal, X_te = data
        with pytest.raises(ValueError):
            CertifiedGPE(tree, X_cal).explain(X_te[0], X_tr)

    def test_nan_instance_rejected(self, data):
        tree, _, X_cal, X_te = data
        x = X_te[0].copy()
        x[0] = np.nan
        with pytest.raises(ValueError):
            CertifiedGPE(tree, X_cal).explain(x)

    def test_calibration_row_warns(self, data):
        tree, _, X_cal, X_te = data
        explainer = CertifiedGPE(tree, X_cal)
        with pytest.warns(UserWarning):
            assert explainer.explain(X_cal[0]).metadata['x_in_calibration']
        assert not explainer.explain(X_te[0]).metadata['x_in_calibration']

    def test_training_overlap_warns(self, data):
        tree, X_tr, X_cal, _ = data
        with pytest.warns(UserWarning):
            explainer = CertifiedGPE(tree, X_tr[:100], X_fit=X_tr)
        assert explainer.n_fit_overlap == 100
        assert CertifiedGPE(tree, X_cal, X_fit=X_tr).n_fit_overlap == 0


class TestLift:
    @pytest.fixture(scope="class")
    def skewed(self):
        X, y = make_classification(
            n_samples=40000, n_features=10, n_informative=5, weights=[0.97], flip_y=0, random_state=1
        )
        X_tr, X_cal, y_tr, _ = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)
        return DecisionTreeClassifier(max_depth=6, random_state=1).fit(X_tr, y_tr), X_cal

    def test_precision_alone_accepts_base_rate(self, skewed):
        """Short majority-class rules pass on precision and add almost no lift."""
        tree, X_cal = skewed
        explainer = CertifiedGPE(tree, X_cal)
        majority = [
            exp for exp in explainer.explain_batch(X_cal[:200] + 1e-3)
            if exp.prediction == 0 and not exp.certificate.trivial
        ]
        assert majority
        assert all(exp.metadata['empirical_lift'] <= 1 - exp.certificate.base_rate for exp in majority)
        assert min(exp.certificate.base_rate for exp in majority) > 0.9

    def test_lift_mode_budget_and_fields(self, skewed):
        tree, X_cal = skewed
        explainer = CertifiedGPE(tree, X_cal, min_lift=0.01, lift_budget=0.2, search="exact")
        plain = CertifiedGPE(tree, X_cal, search="exact")
        pred = tree.predict(X_cal)
        for x in X_cal[:200] + 1e-3:
            exp = explainer.explain(x)
            cert = exp.certificate
            q = (pred == exp.prediction).sum()
            assert cert.base_rate_upper == pytest.approx(
                clopper_pearson_upper(int(q), len(X_cal), DELTA * 0.2)
            )
            assert cert.lift_lower_bound == pytest.approx(cert.lower_bound - cert.base_rate_upper)
            assert cert.meets_lift == (cert.lift_lower_bound >= 0.01)
            if not cert.trivial:
                k = exp.original_rule.complexity
                assert cert.alpha == pytest.approx(DELTA * 0.8 / (2 ** k - 2))
                assert cert.meets_lift
            assert exp.complexity >= plain.explain(x).complexity

    def test_full_path_may_fail_lift(self):
        """When the class share is near 1, no rule can beat it by gamma."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4000, 2))
        y = (X[:, 0] > 2.5).astype(int)
        tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
        X_cal = rng.normal(size=(4000, 2))
        exp = CertifiedGPE(tree, X_cal, min_lift=0.05).explain(np.array([0.0, 0.0]))
        assert exp.prediction == 0
        assert exp.certificate.trivial
        assert exp.certificate.meets_lift is False


class TestSerialization:
    def test_round_trip(self, data):
        tree, _, X_cal, X_te = data
        exp = CertifiedGPE(tree, X_cal).explain(X_te[0])
        restored = Explanation.from_dict(exp.to_dict())
        assert restored.certificate == exp.certificate

    def test_core_explanation_has_no_certificate(self, data):
        tree, _, X_cal, X_te = data
        exp = GPEExplainer(tree, X_train=X_cal).explain(X_te[0])
        assert exp.certificate is None
        assert Explanation.from_dict(exp.to_dict()).certificate is None


class TestGuarantee:
    """Small Monte Carlo with a known population, on the leaves of a fixed tree."""

    @staticmethod
    def population(seed=0):
        """A tree over discrete cells: the population law is known exactly."""
        rng = np.random.default_rng(seed)
        cells = np.array(list(itertools.product(range(6), range(6), range(6))), dtype=float)
        probs = rng.dirichlet(np.full(len(cells), 0.5))
        y = (cells[:, 0] + cells[:, 1] - cells[:, 2] + rng.normal(0, 1.5, len(cells)) > 2).astype(int)
        tree = DecisionTreeClassifier(random_state=seed).fit(cells, y, sample_weight=probs + 1e-6)
        return cells, probs, tree

    @pytest.mark.parametrize("kwargs", [
        {"search": "greedy"},
        {"search": "exact"},
        {"alpha_scheme": "layered", "search": "exact"},
        {"alpha_scheme": "chain"},
    ])
    def test_violations_stay_below_delta(self, kwargs):
        cells, probs, tree = self.population()
        pred = tree.predict(cells)
        rng = np.random.default_rng(1)
        leaf_ids = tree.apply(cells)
        tau = 0.8

        # Explain the cells of the largest leaves that have at least two literals
        targets = []
        for leaf in np.unique(leaf_ids):
            cell = cells[leaf_ids == leaf][0]
            if len(simplify_conditions(get_decision_path(tree, cell).conditions)) >= 2:
                targets.append((probs[leaf_ids == leaf].sum(), cell))
        targets = [cell for _, cell in sorted(targets, key=lambda t: -t[0])[:8]]

        assert targets
        reps = 150
        violations = np.zeros(len(targets))
        nontrivial = 0
        for _ in range(reps):
            X_cal = cells[rng.choice(len(cells), 1500, p=probs)]
            explainer = CertifiedGPE(tree, X_cal, min_precision=tau, delta=DELTA, **kwargs)
            for t, x in enumerate(targets):
                exp = explainer._explain_leaf(x)
                if exp.certificate.trivial:
                    continue
                nontrivial += 1
                region = exp.rule.evaluate_batch(cells)
                p = probs[region & (pred == exp.prediction)].sum() / probs[region].sum()
                violations[t] += p < tau

        # delta plus three standard errors of the frequency over the repetitions
        assert nontrivial > reps
        limit = DELTA + 3 * math.sqrt(DELTA * (1 - DELTA) / reps)
        assert (violations / reps).max() <= limit


class TestCompile:
    @pytest.mark.parametrize("make", [
        lambda t, X: GPEExplainer(t, X_train=X),
        lambda t, X: GPEInformationTheoretic(t, X_train=X),
        lambda t, X: CertifiedGPE(t, X),
        lambda t, X: CertifiedGPE(t, X, search="greedy"),
        lambda t, X: CertifiedGPE(t, X, alpha_scheme="chain"),
    ])
    def test_table_matches_online(self, data, make):
        tree, _, X_cal, X_te = data
        explainer = make(tree, X_cal)
        table = compile_explanations(explainer)
        assert set(table.entries) == set(int(l) for l in np.flatnonzero(tree.tree_.children_left == -1))
        for x in X_te[:300]:
            online, compiled = explainer.explain(x), table.explain(x)
            assert compiled.rule.conditions == online.rule.conditions
            assert compiled.prediction == online.prediction
            assert compiled.precision == pytest.approx(online.precision)
            assert compiled.coverage == pytest.approx(online.coverage)
            assert compiled.certificate == online.certificate
            assert compiled.method.endswith("-compiled")

    def test_masks_decode_to_rules(self, data):
        tree, _, X_cal, _ = data
        table = compile_explanations(CertifiedGPE(tree, X_cal))
        arrays = table.to_arrays()
        paths = get_leaf_paths(tree)
        for leaf, mask in zip(arrays['leaf_ids'], arrays['masks']):
            literals = [lit for d, (_, lit) in enumerate(paths[int(leaf)]) if int(mask) >> d & 1]
            decoded = Rule(literals).evaluate_batch(X_cal)
            assert (decoded == table.entries[int(leaf)].rule.evaluate_batch(X_cal)).all()

    def test_storage(self, data):
        tree, _, X_cal, _ = data
        table = compile_explanations(CertifiedGPE(tree, X_cal))
        leaves = tree.get_n_leaves()
        width = math.ceil(tree.get_depth() / 8)
        assert table.storage_bytes() == leaves * width
        assert table.storage_bytes(with_counts=True) == leaves * (width + 8)
        assert table.n_unique_rules <= leaves
        arrays = table.to_arrays()
        assert (arrays['n'] >= arrays['K']).all()
        assert arrays['n'].sum() > 0

    def test_simultaneous_uses_tree_wide_levels(self, data):
        tree, _, X_cal, _ = data
        table = compile_explanations(CertifiedGPE(tree, X_cal), simultaneous=True)
        lengths = [
            len(simplify_conditions([lit for _, lit in path])) for path in get_leaf_paths(tree).values()
        ]
        family = sum(2 ** k - 2 for k in lengths if k >= 2)
        checked = 0
        for entry in table.entries.values():
            if not entry.certificate.trivial:
                assert entry.certificate.alpha == pytest.approx(DELTA / family)
                checked += 1
        assert checked > 0
        assert table.simultaneous

    def test_simultaneous_budget_adds_up(self, data):
        tree, _, X_cal, _ = data
        explainer = CertifiedGPE(tree, X_cal, alpha_scheme="layered")
        lengths = [
            len(simplify_conditions([lit for _, lit in path])) for path in get_leaf_paths(tree).values()
        ]
        n_leaves = sum(k >= 2 for k in lengths)
        total = 0.0
        for k in lengths:
            if k < 2:
                continue
            level = explainer._level_function(k, DELTA, None, n_leaves)
            total += sum(math.comb(k, s) * level(s) for s in range(1, k))
        assert total == pytest.approx(DELTA)

    def test_simultaneous_is_never_shorter(self, data):
        tree, _, X_cal, _ = data
        explainer = CertifiedGPE(tree, X_cal)
        single = compile_explanations(explainer)
        joint = compile_explanations(explainer, simultaneous=True)
        for leaf, entry in joint.entries.items():
            assert entry.rule.complexity >= single.entries[leaf].rule.complexity

    def test_rejects_variants_without_leaf_proof(self, data):
        tree, _, X_cal, _ = data
        with pytest.raises(TypeError):
            compile_explanations(GPEOptimal(tree, X_train=X_cal))

    def test_argument_checks(self, data):
        tree, _, X_cal, _ = data
        with pytest.raises(ValueError):
            compile_explanations(GPEExplainer(tree, X_train=X_cal), simultaneous=True)
        with pytest.raises(ValueError):
            compile_explanations(CertifiedGPE(tree, X_cal), X_background=X_cal)
