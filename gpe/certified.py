"""
Certified-GPE: path-subset rules with a finite-sample precision guarantee.

A rule S (a subset of the literals on the decision path of x) is admissible when
a lower confidence bound on its population precision, computed on a calibration
set the tree has not seen, reaches the target tau:

    L_alpha_S(K_S, n_S) >= tau

where n_S counts calibration rows in the region of S, K_S counts those the model
assigns to the class of x, and the levels alpha_S are fixed in advance with
sum alpha_S <= delta over all proper non-empty subsets of the path.

By the union bound, with probability at least 1 - delta over the calibration set
all these bounds hold at once, so the guarantee covers any procedure that picks
a rule from the family: greedy, exhaustive, or anything else. The argument is the
one in Berk et al. (2013, Thm 4.1) and Angelopoulos et al. (2025, Thm 1), applied
to the family of path subsets.

The guarantee is about fidelity to the model under the calibration distribution,
for a fixed model and an instance independent of the calibration set. It is not
about the correctness of the prediction, and it does not survive a shift in the
distribution or a refit of the model.

Author: Vladyslav Dehtiarov
"""

import math
import time
import warnings
from functools import lru_cache
from typing import Callable, List, Optional

import numpy as np
from scipy.stats import beta, norm

from .core import GPEExplainer
from .explanation import Certificate, Condition, Explanation, Rule
from .tree_utils import literal_importance

INTERVALS = ('clopper_pearson', 'hoeffding', 'wilson')
ALPHA_SCHEMES = ('uniform', 'layered', 'chain')
SEARCHES = ('auto', 'exact', 'greedy')
MAX_EXACT_LITERALS = 26

# Below this level the quantile routines lose accuracy. A bound of 0 is always valid.
_MIN_ALPHA = 1e-300


def lower_bound(K, n, alpha: float, interval: str = 'clopper_pearson'):
    """
    One-sided lower confidence bound on a binomial proportion.

    Args:
        K: Number of successes (scalar or array)
        n: Number of trials (scalar or array)
        alpha: Level: P(p < bound) <= alpha for 'clopper_pearson' and 'hoeffding';
               only approximately for 'wilson'
        interval: 'clopper_pearson', 'hoeffding' or 'wilson'

    Returns:
        Lower bound, 0 when n = 0 (scalar or array)
    """
    K_arr, n_arr = np.broadcast_arrays(np.asarray(K, dtype=float), np.asarray(n, dtype=float))
    out = np.zeros(K_arr.shape, dtype=float)
    if alpha <= _MIN_ALPHA:
        return out[()]

    if interval == 'clopper_pearson':
        valid = (n_arr > 0) & (K_arr > 0)
        out[valid] = beta.ppf(alpha, K_arr[valid], n_arr[valid] - K_arr[valid] + 1)
    elif interval == 'hoeffding':
        valid = n_arr > 0
        out[valid] = K_arr[valid] / n_arr[valid] - np.sqrt(math.log(1 / alpha) / (2 * n_arr[valid]))
    elif interval == 'wilson':
        valid = (n_arr > 0) & (K_arr > 0)
        z = float(norm.isf(alpha))
        m, p = n_arr[valid], K_arr[valid] / n_arr[valid]
        center = p + z * z / (2 * m)
        radius = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
        out[valid] = (center - radius) / (1 + z * z / m)
    else:
        raise ValueError(f"Unknown interval: {interval}. Must be one of {INTERVALS}")
    return np.clip(out, 0.0, 1.0)[()]


def clopper_pearson_upper(K: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson upper bound: P(p > bound) <= alpha."""
    if n == 0 or K >= n or alpha <= _MIN_ALPHA:
        return 1.0
    return float(beta.isf(alpha, K + 1, n - K))


def min_coverage(k: int, min_precision: float = 0.95, delta: float = 0.05) -> int:
    """
    Smallest n at which a rule with K = n passes the uniform certificate.

    With K = n the Clopper-Pearson bound equals alpha^(1/n), which gives
    n >= ln((2^k - 2) / delta) / ln(1 / tau).

    Args:
        k: Number of literals on the path (k >= 2)
        min_precision: Target precision tau
        delta: Total error budget

    Returns:
        Minimal number of calibration rows in the rule region
    """
    return math.ceil(math.log((2 ** k - 2) / delta) / math.log(1 / min_precision))


@lru_cache(maxsize=MAX_EXACT_LITERALS + 1)
def _popcounts(k: int) -> np.ndarray:
    counts = np.zeros(1, dtype=np.uint8)
    for _ in range(k):
        counts = np.concatenate([counts, counts + 1])
    counts.setflags(write=False)
    return counts


def _superset_sum(counts: np.ndarray, k: int) -> np.ndarray:
    """For every mask S, sum counts[T] over all T that contain S."""
    for j in range(k):
        view = counts.reshape(-1, 2, 1 << j)
        view[:, 0, :] += view[:, 1, :]
    return counts


class CertifiedGPE(GPEExplainer):
    """
    GPE with a finite-sample guarantee on the population precision of the rule.

    Parameters:
        model: A fitted sklearn DecisionTreeClassifier
        X_cal: Calibration data. It must not have been used to fit the tree or to
               choose its hyperparameters, thresholds or preprocessing.
        feature_names: Optional list of feature names
        min_precision: Target precision tau
        delta: Total error budget
        interval: 'clopper_pearson' (default), 'hoeffding', or 'wilson'.
                  Wilson has no finite-sample coverage, so its certificates carry
                  guarantee=False.
        alpha_scheme:
            'uniform' - alpha_S = delta / (2^k - 2)
            'layered' - the budget is split evenly over rule lengths, then over
                        the subsets of each length (Webb, 2008)
            'chain'   - fixed-sequence testing (Bauer, 1991) at level delta along
                        one nested chain, ordered by the impurity decrease stored
                        in the tree. It needs far fewer rows, but the guarantee
                        holds only for this search at this tau.
        search: 'exact' enumerates all subsets and returns a certified rule of
                minimal length; 'greedy' removes literals one at a time, keeping
                the largest coverage; 'auto' uses 'exact' up to max_exact_literals.
                Ignored by alpha_scheme='chain'.
        max_exact_literals: Path length limit for 'exact' (at most 26)
        min_lift: Optional additive lift gamma: a rule must also satisfy
                  L_S - U_q >= gamma, where U_q is an upper bound on the share of
                  rows the model assigns to the class. Not available with 'chain'.
        lift_budget: Share of delta spent on U_q when min_lift is set
        simplify: Whether to merge redundant literals on the path
        X_fit: Optional training data, used only to warn about rows shared with X_cal

    Example:
        >>> explainer = CertifiedGPE(tree, X_cal, min_precision=0.95, delta=0.05)
        >>> exp = explainer.explain(x)
        >>> exp.certificate.lower_bound, exp.certificate.n
    """

    def __init__(
        self,
        model,
        X_cal: np.ndarray,
        feature_names: Optional[List[str]] = None,
        min_precision: float = 0.95,
        delta: float = 0.05,
        interval: str = 'clopper_pearson',
        alpha_scheme: str = 'uniform',
        search: str = 'auto',
        max_exact_literals: int = 20,
        min_lift: Optional[float] = None,
        lift_budget: float = 0.1,
        simplify: bool = True,
        X_fit: Optional[np.ndarray] = None
    ):
        super().__init__(model, feature_names, None, min_precision, simplify)
        if not hasattr(model, 'classes_'):
            raise ValueError("CertifiedGPE requires a DecisionTreeClassifier")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if not 0 <= min_precision <= 1:
            raise ValueError("min_precision must be in [0, 1]")
        if interval not in INTERVALS:
            raise ValueError(f"Unknown interval: {interval}. Must be one of {INTERVALS}")
        if alpha_scheme not in ALPHA_SCHEMES:
            raise ValueError(f"Unknown alpha_scheme: {alpha_scheme}. Must be one of {ALPHA_SCHEMES}")
        if search not in SEARCHES:
            raise ValueError(f"Unknown search: {search}. Must be one of {SEARCHES}")
        if not 1 <= max_exact_literals <= MAX_EXACT_LITERALS:
            raise ValueError(f"max_exact_literals must be in [1, {MAX_EXACT_LITERALS}]")
        if min_lift is not None and alpha_scheme == 'chain':
            raise ValueError(
                "min_lift needs bounds that hold for every subset at once; "
                "use alpha_scheme='uniform' or 'layered'"
            )
        if min_lift is not None and not 0 < lift_budget < 1:
            raise ValueError("lift_budget must be in (0, 1)")

        X_cal = np.asarray(X_cal)
        if X_cal.ndim != 2 or X_cal.shape[0] == 0:
            raise ValueError("X_cal must be a non-empty 2D array")
        if X_cal.shape[1] != model.n_features_in_:
            raise ValueError(
                f"X_cal has {X_cal.shape[1]} features, the model expects {model.n_features_in_}"
            )

        self.X_cal = X_cal
        self.delta = delta
        self.interval = interval
        self.alpha_scheme = alpha_scheme
        self.search = search
        self.max_exact_literals = max_exact_literals
        self.min_lift = min_lift
        self.lift_budget = lift_budget

        self._cal_pred = model.predict(X_cal)
        self._cal_leaf = model.apply(X_cal)
        self._cal32 = np.asarray(X_cal, dtype=np.float32)

        self.n_fit_overlap = 0
        if X_fit is not None:
            fit_rows = {row.tobytes() for row in np.asarray(X_fit, dtype=np.float32)}
            self.n_fit_overlap = sum(row.tobytes() in fit_rows for row in self._cal32)
            if self.n_fit_overlap:
                warnings.warn(
                    f"{self.n_fit_overlap} of {len(X_cal)} calibration rows also occur in the "
                    "training data; the guarantee assumes the two sets are independent"
                )

    def explain(self, x: np.ndarray, X_background: Optional[np.ndarray] = None) -> Explanation:
        """
        Generate a certified explanation for a single instance.

        Args:
            x: Feature vector to explain (1D array). Must not be a calibration row.
            X_background: Must be None: the certificate is tied to X_cal.

        Returns:
            Explanation with the rule and its Certificate
        """
        if X_background is not None:
            raise ValueError("CertifiedGPE evaluates rules on X_cal; X_background is not accepted")
        x = np.asarray(x, dtype=float).flatten()
        if np.isnan(x).any():
            raise ValueError("CertifiedGPE cannot explain an instance with missing values")

        start_time = time.time()
        explanation = self._explain_leaf(x)

        full_mask = explanation.original_rule.evaluate_batch(self.X_cal)
        x32 = x.astype(np.float32)
        in_cal = bool((self._cal32[full_mask] == x32).all(axis=1).any())
        explanation.metadata['x_in_calibration'] = in_cal
        if in_cal:
            warnings.warn(
                "The instance is a calibration row; the guarantee assumes it is independent of X_cal"
            )
        explanation.computation_time = time.time() - start_time
        return explanation

    def _explain_leaf(
        self,
        x: np.ndarray,
        family_total: Optional[int] = None,
        n_leaves: int = 1,
        n_classes: int = 1
    ) -> Explanation:
        """
        Build the explanation; x matters only through its leaf.

        The last three arguments split the budget over several leaves and classes
        for a table that must hold simultaneously for the whole tree. family_total
        is the sum of 2^k - 2 over leaves ('uniform'); n_leaves is the number of
        leaves with k >= 2 (other schemes).
        """
        full_rule = self._greedy_phase(x)
        leaf = self.model.apply(x.reshape(1, -1))[0]
        c = self.model.predict(x.reshape(1, -1))[0]
        full_rule.prediction = c

        agrees = self._cal_pred == c
        base_rate = float(agrees.mean())
        N = len(self.X_cal)

        full_mask = full_rule.evaluate_batch(self.X_cal)
        if (full_mask & (self._cal_leaf != leaf)).any():
            raise RuntimeError("The full path selects rows outside its leaf")

        upper = None
        path_budget = self.delta
        if self.min_lift is not None:
            beta_level = self.delta * self.lift_budget
            path_budget = self.delta - beta_level
            upper = clopper_pearson_upper(int(agrees.sum()), N, beta_level / n_classes)

        k = len(full_rule.conditions)
        chosen, K, n, bound, alpha, search = None, None, None, None, None, 'none'
        if k >= 2:
            matrix = np.column_stack([
                Rule(conditions=[cond]).evaluate_batch(self.X_cal) for cond in full_rule.conditions
            ])
            level = self._level_function(k, path_budget, family_total, n_leaves)

            def admissible(lower):
                ok = lower >= self.min_precision
                if self.min_lift is not None:
                    ok = ok & (lower - upper >= self.min_lift)
                return ok

            if self.alpha_scheme == 'chain':
                search = 'chain'
                found = self._search_chain(x, full_rule.conditions, matrix, agrees, level, admissible)
            elif self._use_exact(k):
                search = 'exact'
                found = self._search_exact(matrix, agrees, k, level, admissible)
            else:
                search = 'greedy'
                found = self._search_greedy(matrix, agrees, k, level, admissible)
            if found is not None:
                chosen, K, n, bound = found
                alpha = level(len(chosen))

        if chosen is None:
            # The region of the full path lies inside one leaf: precision is exactly 1.
            rule = Rule(conditions=list(full_rule.conditions), prediction=c)
            K, n, bound, alpha = int(full_mask.sum()), int(full_mask.sum()), 1.0, None
        else:
            rule = Rule(conditions=[full_rule.conditions[i] for i in chosen], prediction=c)
        trivial = chosen is None

        family = 0 if k < 2 else (k - 1 if self.alpha_scheme == 'chain' else 2 ** k - 2)
        certificate = Certificate(
            K=int(K),
            n=int(n),
            lower_bound=float(bound),
            alpha=None if alpha is None else float(alpha),
            delta=self.delta,
            min_precision=self.min_precision,
            interval=self.interval,
            alpha_scheme=self.alpha_scheme,
            search=search,
            family_size=family,
            trivial=trivial,
            guarantee=self.interval != 'wilson',
            base_rate=base_rate,
            min_lift=self.min_lift,
            base_rate_upper=upper,
            lift_lower_bound=None if upper is None else float(bound) - upper,
            meets_lift=None if upper is None else bool(float(bound) - upper >= self.min_lift),
        )
        precision = K / n if n else 1.0
        return Explanation(
            instance=x,
            rule=rule,
            prediction=c,
            precision=float(precision),
            coverage=n / N,
            original_rule=full_rule,
            method="Certified-GPE",
            metadata={
                'min_precision': self.min_precision,
                'simplify': self.simplify,
                'n_cal': N,
                'empirical_lift': float(precision) - base_rate,
                'x_in_calibration': False,
            },
            certificate=certificate,
        )

    def _prune_phase(
        self,
        x: np.ndarray,
        rule: Rule,
        X_background: Optional[np.ndarray] = None
    ) -> Rule:
        """Certified pruning; the rule argument is rebuilt from x."""
        return self._explain_leaf(x).rule

    def _use_exact(self, k: int) -> bool:
        if self.search == 'greedy':
            return False
        if k <= self.max_exact_literals:
            return True
        if self.search == 'exact':
            raise ValueError(
                f"The path has {k} literals, above max_exact_literals={self.max_exact_literals}"
            )
        return False

    def _level_function(
        self,
        k: int,
        budget: float,
        family_total: Optional[int],
        n_leaves: int
    ) -> Callable[[int], float]:
        """Return the map: rule length -> alpha_S. It never looks at X_cal."""
        if self.alpha_scheme == 'chain':
            return lambda size: budget / n_leaves
        if self.alpha_scheme == 'layered':
            share = budget / n_leaves / (k - 1)
            return lambda size: _divide(share, math.comb(k, size))
        family = family_total if family_total is not None else 2 ** k - 2
        alpha = _divide(budget, family)
        return lambda size: alpha

    def _search_exact(self, matrix, agrees, k, level, admissible):
        """Shortest certified subset; ties go to the larger coverage."""
        bits = matrix.astype(np.int64).dot(1 << np.arange(k, dtype=np.int64))
        size = 1 << k
        n_all = _superset_sum(np.bincount(bits, minlength=size).astype(np.int64), k)
        K_all = _superset_sum(np.bincount(bits[agrees], minlength=size).astype(np.int64), k)
        popcount = _popcounts(k)

        for length in range(1, k):
            candidates = np.flatnonzero(
                (popcount == length) & (n_all > 0) & (K_all >= self.min_precision * n_all)
            )
            if candidates.size == 0:
                continue
            K_c, n_c = K_all[candidates], n_all[candidates]
            lower = np.atleast_1d(lower_bound(K_c, n_c, level(length), self.interval))
            passed = np.flatnonzero(admissible(lower))
            if passed.size == 0:
                continue
            # Largest coverage, then highest precision, then lowest mask
            order = np.lexsort((candidates[passed], -K_c[passed] / n_c[passed], -n_c[passed]))
            best = passed[order[0]]
            mask = int(candidates[best])
            return (
                [j for j in range(k) if mask >> j & 1],
                int(K_c[best]), int(n_c[best]), float(lower[best]),
            )
        return None

    def _search_greedy(self, matrix, agrees, k, level, admissible):
        """Remove the literal that keeps the largest coverage among certified subsets."""
        fails = ~matrix
        current = list(range(k))
        result = None
        while len(current) > 1:
            sub = fails[:, current]
            fail_count = sub.sum(axis=1)
            inside = fail_count == 0
            one_short = fail_count == 1
            n_i = int(inside.sum()) + sub[one_short].sum(axis=0)
            K_i = int((inside & agrees).sum()) + sub[one_short & agrees].sum(axis=0)

            lower = np.atleast_1d(lower_bound(K_i, n_i, level(len(current) - 1), self.interval))
            passed = np.flatnonzero(admissible(lower) & (n_i > 0))
            if passed.size == 0:
                break
            ratio = K_i[passed] / n_i[passed]
            best = passed[np.lexsort((passed, -ratio, -n_i[passed]))[0]]
            current.pop(int(best))
            result = (list(current), int(K_i[best]), int(n_i[best]), float(lower[best]))
        return result

    def _search_chain(self, x, conditions, matrix, agrees, level, admissible):
        """Fixed-sequence testing: stop at the first subset of the chain that fails."""
        order = self._chain_order(x, conditions)
        current = list(range(len(conditions)))
        result = None
        for literal in order[:-1]:
            current.remove(literal)
            inside = matrix[:, current].all(axis=1)
            n = int(inside.sum())
            K = int((inside & agrees).sum())
            lower = float(lower_bound(K, n, level(len(current)), self.interval))
            if n == 0 or not admissible(lower):
                break
            result = (list(current), K, n, lower)
        return result

    def _chain_order(self, x: np.ndarray, conditions: List[Condition]) -> List[int]:
        """Literals by ascending impurity decrease, read from the fitted tree only."""
        from .tree_utils import literal_importance

        tree_ = self.model.tree_
        nodes = self.model.decision_path(x.reshape(1, -1)).indices[:-1]
        importance = []
        for cond in conditions:
            matches = [
                node for node in nodes
                if tree_.feature[node] == cond.feature_index and tree_.threshold[node] == cond.threshold
            ]
            importance.append(literal_importance(self.model, int(matches[0])))
        return sorted(range(len(conditions)), key=lambda i: (importance[i], i))

    def __repr__(self) -> str:
        return (f"CertifiedGPE(model={type(self.model).__name__}, min_precision={self.min_precision}, "
                f"delta={self.delta}, interval={self.interval!r}, alpha_scheme={self.alpha_scheme!r})")


def _divide(budget: float, family: int) -> float:
    """budget / family for a family too large to convert to float."""
    if family > 1e300:
        return 0.0
    return budget / family
