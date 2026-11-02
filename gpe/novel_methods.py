"""
GPE Novel Methods - Advanced Explanation Algorithms

This module implements novel variants of the GPE algorithm:
1. GPE-IT: Information-Theoretic pruning using mutual information
2. GPE-CF: Counterfactual-Integrated explanations
3. GPE-Stable: Stability-constrained explanations
4. GPE-Multi: Multi-resolution explanations

These methods represent the core scientific contribution of the GPE framework.

Author: Vladyslav Dehtiarov
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from copy import deepcopy
from dataclasses import dataclass
import time
from scipy import stats

from .explanation import Condition, Rule, Explanation
from .tree_utils import get_decision_path, get_leaf_paths, simplify_conditions, is_single_tree
from .core import GPEExplainer, as_instance, as_matrix, check_min_precision


# =============================================================================
# GPE-IT: Information-Theoretic Pruning
# =============================================================================

class GPEInformationTheoretic(GPEExplainer):
    """
    GPE with Information-Theoretic Pruning.
    
    Instead of greedy removal based on precision alone, this method adds
    literals in order of mutual information with the model's prediction,
    conditional on the literals selected so far, until the precision threshold
    is met on the background data. It is a heuristic for the order of the search
    and carries no statistical guarantee.
    
    The mutual information is calculated as:
        I(C; Y) = H(Y) - H(Y|C)
    
    where:
        - H(Y) is the entropy of predictions
        - H(Y|C) is the conditional entropy given the condition
    
    Parameters:
        model: Fitted decision tree
        feature_names: List of feature names
        X_train: Training data
        min_precision: Minimum precision threshold (default: 0.95)
        min_mutual_info: Kept for compatibility, has no effect since 1.3.0.
            Skipping low-information literals made rules miss min_precision.
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        min_precision: float = 0.95,
        min_mutual_info: float = 0.01
    ):
        super().__init__(model, feature_names, X_train, min_precision)
        self.min_mutual_info = min_mutual_info
    
    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate Shannon entropy of a distribution."""
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        probs = probs[probs > 0]  # Remove zeros
        
        return -np.sum(probs * np.log2(probs))
    
    def _calculate_mutual_information(
        self,
        condition: Condition,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Calculate mutual information I(condition; prediction).
        
        I(C; Y) = H(Y) - H(Y|C)
        """
        if len(y) == 0:
            return 0.0
        
        # H(Y) - entropy of predictions
        h_y = self._calculate_entropy(y)
        
        # Evaluate condition on all instances
        mask = Rule(conditions=[condition]).evaluate_batch(X)
        
        # H(Y|C) - conditional entropy
        p_true = mask.mean()
        p_false = 1 - p_true
        
        h_y_given_true = self._calculate_entropy(y[mask]) if mask.sum() > 0 else 0
        h_y_given_false = self._calculate_entropy(y[~mask]) if (~mask).sum() > 0 else 0
        
        h_y_given_c = p_true * h_y_given_true + p_false * h_y_given_false
        
        # Mutual information
        mi = h_y - h_y_given_c
        
        return max(0, mi)  # MI should be non-negative
    
    def _prune_phase(
        self,
        x: np.ndarray,
        rule: Rule,
        X_background: Optional[np.ndarray] = None
    ) -> Rule:
        """
        Information-theoretic literal selection.
        
        Adds literals in order of mutual information with the model's
        prediction, conditional on the literals already selected, until the
        precision threshold is met. Every literal stays available, so the
        search ends at the full path at worst, and the full path has
        precision 1 against the model.
        """
        if X_background is None or len(rule.conditions) <= 1:
            return rule
        
        prediction = self.model.predict(x.reshape(1, -1))[0]
        y_pred = self.model.predict(X_background)
        agrees = y_pred == prediction
        
        remaining = list(rule.conditions)
        selected: List[Condition] = []
        mask = np.ones(len(X_background), dtype=bool)
        
        while remaining:
            scores = [
                self._calculate_mutual_information(cond, X_background[mask], y_pred[mask])
                for cond in remaining
            ]
            best = remaining.pop(int(np.argmax(scores)))
            selected.append(best)
            mask &= Rule(conditions=[best]).evaluate_batch(X_background)
            if mask.any() and agrees[mask].mean() >= self.min_precision:
                break
        
        return Rule(conditions=selected, prediction=prediction)
    
    def explain(self, x: np.ndarray, X_background: Optional[np.ndarray] = None) -> Explanation:
        """Generate explanation with mutual information scores."""
        exp = super().explain(x, X_background)
        
        # Add MI scores to metadata
        if X_background is None:
            X_background = self.X_train
        if X_background is not None:
            X_background = as_matrix(X_background, self.model, 'X_background')
            y_pred = self.model.predict(X_background)
            mi_scores = {}
            for cond in exp.rule.conditions:
                mi = self._calculate_mutual_information(cond, X_background, y_pred)
                mi_scores[cond.feature_name] = mi
            exp.metadata['mutual_information'] = mi_scores
        
        exp.method = "GPE-IT"
        return exp


# =============================================================================
# GPE-CF: Counterfactual-Integrated Explanations
# =============================================================================

@dataclass
class CounterfactualExplanation:
    """
    Combined explanation with counterfactual.
    
    Provides both:
    1. Why the prediction was made (rule)
    2. What minimal change would alter the prediction (counterfactual)
    """
    explanation: Explanation
    counterfactual: np.ndarray
    counterfactual_prediction: Any
    changes: Dict[str, Tuple[float, float]]  # feature -> (original, new)
    distance: float  # L1 distance in the original feature units
    found: bool = True  # False: no reachable leaf of another class; x is returned unchanged
    
    def __str__(self) -> str:
        if not self.found:
            return (f"GPE-CF: no counterfactual within the allowed features and ranges\n"
                    f"WHY this prediction: {self.explanation.rule}")
        lines = [
            "=" * 60,
            "GPE-CF: Counterfactual-Integrated Explanation",
            "=" * 60,
            "",
            "WHY this prediction:",
            f"  {self.explanation.rule}",
            "",
            "WHAT would change the prediction:",
        ]
        
        for feature, (old, new) in self.changes.items():
            lines.append(f"  • Change {feature}: {old:.3f} → {new:.3f}")
        
        lines.extend([
            "",
            f"New prediction would be: {self.counterfactual_prediction}",
            f"Distance: {self.distance:.4f}",
            "=" * 60
        ])
        
        return "\n".join(lines)


class GPECounterfactual(GPEExplainer):
    """
    GPE with Integrated Counterfactual Explanations.
    
    For each explanation, also finds the minimal counterfactual - the
    smallest change to the instance that would alter the prediction.
    
    This provides actionable insights:
    - "You were rejected because income < 50000"
    - "If your income were 52000, you would be approved"
    
    A tree partitions the input space into boxes, one per leaf, so the closest
    point of another class is found exactly: for every leaf of another class,
    move x into the box by the smallest change, and keep the cheapest move. The
    cost is the L1 distance with each feature scaled by its range, when the range
    is known. The result is checked against the model. The search does not model
    causality or feature dependence, so the point may be unrealistic.
    
    Parameters:
        model: Fitted decision tree
        feature_names: List of feature names
        X_train: Training data
        min_precision: Minimum precision threshold
        actionable_features: Features that can be changed (optional)
        feature_ranges: Valid ranges for features (optional)
    
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        min_precision: float = 0.95,
        actionable_features: Optional[List[str]] = None,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        super().__init__(model, feature_names, X_train, min_precision)
        self.actionable_features = self.feature_names if actionable_features is None else actionable_features
        self.feature_ranges = feature_ranges or {}
        
        # Compute feature ranges from training data
        if self.X_train is not None:
            for i, name in enumerate(self.feature_names):
                if name not in self.feature_ranges:
                    column = self.X_train[:, i]
                    if not np.isnan(column).all():
                        self.feature_ranges[name] = (
                            float(np.nanmin(column)),
                            float(np.nanmax(column))
                        )
    
    def _move_into_box(
        self,
        x: np.ndarray,
        path: List[Tuple[int, Condition]]
    ) -> Optional[Tuple[np.ndarray, Dict[str, Tuple[float, float]], float]]:
        """
        Cheapest move of x into the box of a leaf.
        
        Returns:
            (new point, changes, cost), or None when the box is out of reach:
            it needs a change to a non-actionable feature or leaves a valid range
        """
        lows: Dict[int, float] = {}
        highs: Dict[int, float] = {}
        missing_ok: Dict[int, bool] = {}
        for _, literal in path:
            j = literal.feature_index
            missing_ok[j] = missing_ok.get(j, True) and literal.matches_missing
            if literal.operator == '>':
                lows[j] = max(lows.get(j, -np.inf), literal.threshold)
            else:
                highs[j] = min(highs.get(j, np.inf), literal.threshold)
        
        x_cf = x.copy()
        changes: Dict[str, Tuple[float, float]] = {}
        cost = 0.0
        for j in missing_ok:
            name = self.feature_names[j]
            value32 = float(np.float32(x[j]))
            low, high = lows.get(j, -np.inf), highs.get(j, np.inf)
            if np.isnan(x[j]) and missing_ok[j]:
                continue
            if low < value32 <= high:
                continue
            if name not in self.actionable_features:
                return None
            
            # The point of the box closest to x, representable in float32
            range_low, range_high = self.feature_ranges.get(name, (-np.inf, np.inf))
            target = range_low if np.isnan(x[j]) and np.isfinite(range_low) else x[j]
            if np.isnan(target) or target <= low:
                new = np.float32(low)
                if float(new) <= low:
                    new = np.nextafter(new, np.float32(np.inf))
            else:
                new = np.float32(high)
                if float(new) > high:
                    new = np.nextafter(new, np.float32(-np.inf))
            new = float(new)
            if not (low < new <= high) or not np.isfinite(new) or not range_low <= new <= range_high:
                return None
            
            scale = range_high - range_low
            step = abs(new - x[j]) if not np.isnan(x[j]) else 1.0
            cost += step / scale if np.isfinite(scale) and scale > 0 else step
            x_cf[j] = new
            changes[name] = (float(x[j]), new)
        return x_cf, changes, cost
    
    def _find_counterfactual(
        self,
        x: np.ndarray,
        rule: Rule,
        prediction: Any
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        """
        Find the closest point that the tree assigns to another class.
        
        Returns:
            (counterfactual, changes); x and an empty dict when there is none
        """
        tree_ = self.model.tree_
        best = None
        for leaf, path in get_leaf_paths(self.model, self.feature_names).items():
            if self.model.classes_[int(np.argmax(tree_.value[leaf][0]))] == prediction:
                continue
            move = self._move_into_box(x, path)
            if move is None:
                continue
            if best is None or move[2] < best[2]:
                best = move
        if best is None:
            return x.copy(), {}
        return best[0], best[1]
    
    def explain_with_counterfactual(
        self,
        x: np.ndarray,
        X_background: Optional[np.ndarray] = None
    ) -> CounterfactualExplanation:
        """
        Generate explanation with integrated counterfactual.
        
        Returns both the rule-based explanation and the minimal
        counterfactual that would change the prediction.
        """
        # Get base explanation
        x = as_instance(x, self.model)
        explanation = self.explain(x, X_background)
        explanation.method = "GPE-CF"
        
        # Find counterfactual
        x_cf, changes = self._find_counterfactual(
            x, explanation.rule, explanation.prediction
        )
        
        cf_prediction = self.model.predict(x_cf.reshape(1, -1))[0]
        found = bool(changes)
        if found and cf_prediction == explanation.prediction:
            raise RuntimeError("The counterfactual did not change the prediction")
        
        # Calculate distance over the changed features; a missing value has no distance
        difference = x - x_cf
        distance = float(np.abs(difference[~np.isnan(difference)]).sum())
        
        return CounterfactualExplanation(
            explanation=explanation,
            counterfactual=x_cf,
            counterfactual_prediction=cf_prediction,
            changes=changes,
            distance=distance,
            found=found
        )


# =============================================================================
# GPE-Stable: Stability-Constrained Explanations
# =============================================================================

class GPEStable(GPEExplainer):
    """
    GPE with Stability Constraints.
    
    Finds explanations that are stable under small perturbations to the
    input. This ensures that similar instances receive similar explanations.
    
    Stability is measured as the Jaccard similarity between explanations
    for nearby instances.
    
    The optimization problem:
        minimize complexity
        subject to: precision >= τ_p
                   stability >= τ_s
    
    Parameters:
        model: Fitted decision tree
        feature_names: List of feature names
        X_train: Training data
        min_precision: Minimum precision threshold (default: 0.95)
        min_stability: Minimum stability threshold (default: 0.8)
        n_perturbations: Number of perturbations for stability (default: 10)
        perturbation_std: Standard deviation of perturbations (default: 0.01)
        random_state: Seed; each stability estimate restarts from it, so results
            repeat and candidate rules are compared on the same perturbations
    
    Stability here is an empirical estimate on sampled perturbations. It is not a
    formal guarantee.
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        min_precision: float = 0.95,
        min_stability: float = 0.8,
        n_perturbations: int = 10,
        perturbation_std: float = 0.01,
        random_state: Optional[int] = 0
    ):
        super().__init__(model, feature_names, X_train, min_precision)
        self.min_stability = min_stability
        self.n_perturbations = n_perturbations
        self.perturbation_std = perturbation_std
        self.random_state = random_state
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        if len(set1) == 0 or len(set2) == 0:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union
    
    def _calculate_stability(
        self,
        x: np.ndarray,
        rule: Rule,
        X_background: Optional[np.ndarray]
    ) -> float:
        """
        Calculate stability of a rule under perturbations.
        
        Generates perturbed instances and checks if the same
        conditions appear in their explanations.
        """
        if X_background is None:
            return 1.0
        
        # Feature ranges for perturbation
        feature_stds = np.nanstd(X_background, axis=0)
        feature_stds[(feature_stds == 0) | np.isnan(feature_stds)] = 1
        rng = np.random.RandomState(self.random_state)
        
        original_features = set(c.feature_name for c in rule.conditions)
        similarities = []
        
        # Use simple GPE for stability calculation to avoid recursion
        simple_gpe = GPEExplainer(
            self.model, 
            feature_names=self.feature_names, 
            X_train=X_background,
            min_precision=self.min_precision
        )
        
        for _ in range(self.n_perturbations):
            # Generate perturbation
            noise = rng.standard_normal(len(x)) * self.perturbation_std * feature_stds
            x_perturbed = x + noise
            
            # Get explanation for perturbed instance using simple GPE
            exp_perturbed = simple_gpe.explain(x_perturbed, X_background)
            perturbed_features = set(c.feature_name for c in exp_perturbed.rule.conditions)
            
            # Calculate similarity
            similarity = self._jaccard_similarity(original_features, perturbed_features)
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _prune_phase(
        self,
        x: np.ndarray,
        rule: Rule,
        X_background: Optional[np.ndarray] = None
    ) -> Rule:
        """
        Stability-aware pruning.
        
        Removes conditions while maintaining both precision and stability.
        """
        if X_background is None or len(rule.conditions) <= 1:
            return rule
        
        prediction = self.model.predict(x.reshape(1, -1))[0]
        pruned_rule = deepcopy(rule)
        
        while len(pruned_rule.conditions) > 1:
            best_idx = None
            best_score = -1
            
            for i in range(len(pruned_rule.conditions)):
                # Create rule without condition i
                test_conditions = [c for j, c in enumerate(pruned_rule.conditions) if j != i]
                test_rule = Rule(conditions=test_conditions, prediction=prediction)
                
                # Check precision
                precision = self._calculate_precision(test_rule, prediction, X_background)
                if not precision >= self.min_precision:  # also skips NaN
                    continue
                
                # Check stability
                stability = self._calculate_stability(x, test_rule, X_background)
                if stability < self.min_stability:
                    continue
                
                # Score: prefer higher precision and stability
                score = precision * stability
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx is None:
                break
            
            pruned_rule.conditions.pop(best_idx)
        
        pruned_rule.prediction = prediction
        return pruned_rule
    
    def explain(self, x: np.ndarray, X_background: Optional[np.ndarray] = None) -> Explanation:
        """Generate stability-constrained explanation."""
        exp = super().explain(x, X_background)
        
        # Calculate and add stability to metadata
        if X_background is None:
            X_background = self.X_train
        if X_background is not None:
            X_background = as_matrix(X_background, self.model, 'X_background')
        stability = self._calculate_stability(x, exp.rule, X_background)
        exp.metadata['stability'] = stability
        exp.method = "GPE-Stable"
        
        return exp


# =============================================================================
# GPE-Multi: Multi-Resolution Explanations
# =============================================================================

@dataclass 
class MultiResolutionExplanation:
    """
    Explanation at multiple levels of detail.
    
    Provides explanations at different granularities:
    - Level 1: Simplest (1-2 conditions)
    - Level 2: Medium (3-4 conditions)
    - Level 3: Full detail (all conditions)
    """
    levels: List[Explanation]
    instance: np.ndarray
    prediction: Any
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "GPE-Multi: Multi-Resolution Explanation",
            "=" * 60,
            f"Prediction: {self.prediction}",
            ""
        ]
        
        for i, exp in enumerate(self.levels, 1):
            lines.append(f"Level {i} ({exp.complexity} conditions):")
            lines.append(f"  Rule: {exp.rule}")
            lines.append(f"  Precision: {exp.precision:.1%}, Coverage: {exp.coverage:.1%}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def get_level(self, level: int) -> Explanation:
        """Get explanation at specific level (1-indexed)."""
        if 1 <= level <= len(self.levels):
            return self.levels[level - 1]
        raise ValueError(f"Level must be between 1 and {len(self.levels)}")


class GPEMultiResolution(GPEExplainer):
    """
    GPE with Multi-Resolution Explanations.
    
    Generates explanations at multiple levels of detail, allowing users
    to drill down from simple to complex explanations.
    
    Useful for:
    - Progressive disclosure of information
    - Different audiences (expert vs novice)
    - Understanding the explanation-complexity trade-off
    
    Parameters:
        model: Fitted decision tree
        feature_names: List of feature names
        X_train: Training data
        n_levels: Number of resolution levels (default: 3)
        precision_thresholds: Precision threshold per level; its length must equal
            n_levels (default: [0.85, 0.95, 0.99], or evenly spaced for other n_levels)
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        n_levels: int = 3,
        precision_thresholds: Optional[List[float]] = None
    ):
        super().__init__(model, feature_names, X_train, min_precision=0.9)
        if n_levels < 1:
            raise ValueError("n_levels must be at least 1")
        if precision_thresholds is None:
            precision_thresholds = (
                [0.85, 0.95, 0.99] if n_levels == 3 else np.linspace(0.85, 0.99, n_levels).tolist()
            )
        elif len(precision_thresholds) != n_levels:
            raise ValueError(
                f"precision_thresholds has {len(precision_thresholds)} values for {n_levels} levels"
            )
        for threshold in precision_thresholds:
            check_min_precision(threshold)
        self.n_levels = n_levels
        self.precision_thresholds = list(precision_thresholds)
    
    def explain_multi(
        self,
        x: np.ndarray,
        X_background: Optional[np.ndarray] = None
    ) -> MultiResolutionExplanation:
        """
        Generate multi-resolution explanation.
        
        Returns explanations at multiple precision thresholds,
        from simplest to most detailed.
        """
        x = as_instance(x, self.model)
        
        if X_background is None:
            X_background = self.X_train
        else:
            X_background = as_matrix(X_background, self.model, 'X_background')
        
        prediction = self.model.predict(x.reshape(1, -1))[0]
        
        # Get full decision path
        original_rule = self._greedy_phase(x)
        
        levels = []
        
        for threshold in self.precision_thresholds:
            # Prune with this threshold
            pruned_rule = self._with_threshold(threshold, x, original_rule, X_background)
            
            # Calculate metrics
            precision = self._calculate_precision(pruned_rule, prediction, X_background) if X_background is not None else float('nan')
            coverage = self._calculate_coverage(pruned_rule, X_background) if X_background is not None else float('nan')
            
            exp = Explanation(
                instance=x,
                rule=pruned_rule,
                prediction=prediction,
                precision=precision,
                coverage=coverage,
                original_rule=original_rule,
                method=f"GPE-Multi-L{len(levels)+1}"
            )
            
            levels.append(exp)
        
        return MultiResolutionExplanation(
            levels=levels,
            instance=x,
            prediction=prediction
        )
    
    def _with_threshold(self, threshold, x, original_rule, X_background) -> Rule:
        """Prune at another threshold, keeping min_precision of the explainer."""
        saved = self.min_precision
        self.min_precision = threshold
        try:
            return self._prune_phase(x, deepcopy(original_rule), X_background)
        finally:
            self.min_precision = saved


# =============================================================================
# GPE-Pareto: Pareto-Optimal Explanations
# =============================================================================

@dataclass
class ParetoExplanation:
    """
    Pareto-optimal explanation.
    
    An explanation on the Pareto front of the precision-coverage-complexity
    trade-off space.
    """
    explanation: Explanation
    is_pareto_optimal: bool
    dominated_by: int  # Number of solutions that dominate this one
    
    def __str__(self) -> str:
        return (f"ParetoExplanation(precision={self.explanation.precision:.2%}, "
                f"coverage={self.explanation.coverage:.2%}, "
                f"complexity={self.explanation.complexity}, "
                f"pareto_optimal={self.is_pareto_optimal})")


class GPEPareto(GPEExplainer):
    """
    GPE with Pareto-Optimal Explanation Selection.
    
    Finds explanations that are Pareto-optimal with respect to multiple
    objectives: precision, coverage, and complexity.
    
    An explanation is Pareto-optimal if no other explanation is better
    in all objectives simultaneously.
    
    Parameters:
        model: Fitted decision tree
        feature_names: List of feature names
        X_train: Training data
        n_candidates: Number of candidate explanations to generate
    
    Scientific Contribution:
        Multi-objective optimization framework for explanation selection
        in decision trees.
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        n_candidates: int = 20
    ):
        super().__init__(model, feature_names, X_train, min_precision=0.8)
        self.n_candidates = n_candidates
    
    def _is_dominated(
        self,
        exp1: Explanation,
        exp2: Explanation
    ) -> bool:
        """
        Check if exp1 is dominated by exp2.
        
        exp1 is dominated if exp2 is at least as good in all objectives
        and strictly better in at least one.
        
        Objectives (maximize precision, maximize coverage, minimize complexity):
        """
        # exp2 should be >= in precision and coverage, <= in complexity
        at_least_as_good = (
            exp2.precision >= exp1.precision and
            exp2.coverage >= exp1.coverage and
            exp2.complexity <= exp1.complexity
        )
        
        # exp2 should be strictly better in at least one
        strictly_better = (
            exp2.precision > exp1.precision or
            exp2.coverage > exp1.coverage or
            exp2.complexity < exp1.complexity
        )
        
        return at_least_as_good and strictly_better
    
    def _generate_candidates(
        self,
        x: np.ndarray,
        original_rule: Rule,
        X_background: np.ndarray
    ) -> List[Explanation]:
        """Generate candidate explanations with different trade-offs."""
        prediction = self.model.predict(x.reshape(1, -1))[0]
        candidates = []
        
        # Different precision thresholds
        saved = self.min_precision
        for threshold in np.linspace(0.7, 0.99, self.n_candidates):
            self.min_precision = threshold
            try:
                pruned_rule = self._prune_phase(x, deepcopy(original_rule), X_background)
            finally:
                self.min_precision = saved
            
            precision = self._calculate_precision(pruned_rule, prediction, X_background)
            coverage = self._calculate_coverage(pruned_rule, X_background)
            
            exp = Explanation(
                instance=x,
                rule=pruned_rule,
                prediction=prediction,
                precision=precision,
                coverage=coverage,
                original_rule=original_rule,
                method="GPE-Pareto"
            )
            candidates.append(exp)
        
        return candidates
    
    def find_pareto_front(
        self,
        x: np.ndarray,
        X_background: Optional[np.ndarray] = None
    ) -> List[ParetoExplanation]:
        """
        Find all Pareto-optimal explanations.
        
        Returns explanations on the Pareto front, sorted by complexity.
        """
        x = as_instance(x, self.model)
        
        if X_background is None:
            X_background = self.X_train
        else:
            X_background = as_matrix(X_background, self.model, 'X_background')
        if X_background is None:
            raise ValueError("GPEPareto needs background data to compare candidates")
        
        # Get original rule
        original_rule = self._greedy_phase(x)
        
        # Generate candidates
        candidates = self._generate_candidates(x, original_rule, X_background)
        
        # Find Pareto front
        pareto_explanations = []
        
        for i, exp in enumerate(candidates):
            dominated_by = sum(1 for j, other in enumerate(candidates) 
                             if i != j and self._is_dominated(exp, other))
            
            pareto_exp = ParetoExplanation(
                explanation=exp,
                is_pareto_optimal=(dominated_by == 0),
                dominated_by=dominated_by
            )
            pareto_explanations.append(pareto_exp)
        
        # Return only Pareto-optimal, sorted by complexity
        pareto_front = [p for p in pareto_explanations if p.is_pareto_optimal]
        pareto_front.sort(key=lambda p: p.explanation.complexity)
        
        return pareto_front
    
    def explain(self, x: np.ndarray, X_background: Optional[np.ndarray] = None) -> Explanation:
        """
        Return the Pareto-optimal explanation with best balance.
        
        Uses a weighted sum of normalized objectives to select the best.
        """
        pareto_front = self.find_pareto_front(x, X_background)
        
        if not pareto_front:
            return super().explain(x, X_background)
        
        # Select best by weighted score
        best_score = -1
        best_exp = pareto_front[0].explanation
        
        for p in pareto_front:
            exp = p.explanation
            if exp.precision != exp.precision:
                continue  # A rule with no background rows cannot be scored
            # Normalize and weight: precision (0.4), coverage (0.3), 1/complexity (0.3)
            score = (0.4 * exp.precision + 
                    0.3 * exp.coverage + 
                    0.3 * (1 / max(exp.complexity, 1)))
            
            if score > best_score:
                best_score = score
                best_exp = exp
        
        best_exp.method = "GPE-Pareto"
        return best_exp

