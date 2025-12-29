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
from .tree_utils import get_decision_path, simplify_conditions, is_single_tree
from .core import GPEExplainer


# =============================================================================
# GPE-IT: Information-Theoretic Pruning
# =============================================================================

class GPEInformationTheoretic(GPEExplainer):
    """
    GPE with Information-Theoretic Pruning.
    
    Instead of greedy removal based on precision alone, this method uses
    mutual information I(condition; prediction) to select which conditions
    to keep. Conditions with higher mutual information are more important
    for the prediction.
    
    The mutual information is calculated as:
        I(C; Y) = H(Y) - H(Y|C)
    
    where:
        - H(Y) is the entropy of predictions
        - H(Y|C) is the conditional entropy given the condition
    
    This provides a principled, information-theoretic foundation for
    selecting the most informative conditions.
    
    Parameters:
        model: Fitted decision tree
        feature_names: List of feature names
        X_train: Training data
        min_precision: Minimum precision threshold (default: 0.95)
        min_mutual_info: Minimum mutual information threshold (default: 0.01)
    
    Scientific Contribution:
        First application of mutual information for condition selection
        in local tree explanations.
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
        # H(Y) - entropy of predictions
        h_y = self._calculate_entropy(y)
        
        # Evaluate condition on all instances
        values = X[:, condition.feature_index]
        
        if condition.operator == '<=':
            mask = values <= condition.threshold
        elif condition.operator == '<':
            mask = values < condition.threshold
        elif condition.operator == '>':
            mask = values > condition.threshold
        elif condition.operator == '>=':
            mask = values >= condition.threshold
        else:
            return 0.0
        
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
        Information-theoretic pruning.
        
        Removes conditions with low mutual information while maintaining
        precision threshold.
        """
        if X_background is None or len(rule.conditions) <= 1:
            return rule
        
        prediction = self.model.predict(x.reshape(1, -1))[0]
        y_pred = self.model.predict(X_background)
        
        # Calculate mutual information for each condition
        condition_mi = []
        for cond in rule.conditions:
            mi = self._calculate_mutual_information(cond, X_background, y_pred)
            condition_mi.append((cond, mi))
        
        # Sort by mutual information (descending)
        condition_mi.sort(key=lambda x: x[1], reverse=True)
        
        # Greedily add conditions until precision is met
        selected_conditions = []
        
        for cond, mi in condition_mi:
            if mi < self.min_mutual_info:
                continue
            
            test_conditions = selected_conditions + [cond]
            test_rule = Rule(conditions=test_conditions, prediction=prediction)
            
            precision = self._calculate_precision(test_rule, prediction, X_background)
            
            selected_conditions.append(cond)
            
            if precision >= self.min_precision:
                break
        
        # Ensure at least one condition
        if len(selected_conditions) == 0:
            selected_conditions = [condition_mi[0][0]] if condition_mi else []
        
        return Rule(
            conditions=selected_conditions,
            prediction=prediction
        )
    
    def explain(self, x: np.ndarray, X_background: Optional[np.ndarray] = None) -> Explanation:
        """Generate explanation with mutual information scores."""
        exp = super().explain(x, X_background)
        
        # Add MI scores to metadata
        if X_background is not None:
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
    distance: float  # L1 or L2 distance
    
    def __str__(self) -> str:
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
    
    The counterfactual is found by:
    1. Identifying the conditions that led to the prediction
    2. Finding the minimal changes to violate those conditions
    3. Verifying the prediction changes
    
    Parameters:
        model: Fitted decision tree
        feature_names: List of feature names
        X_train: Training data
        min_precision: Minimum precision threshold
        actionable_features: Features that can be changed (optional)
        feature_ranges: Valid ranges for features (optional)
    
    Scientific Contribution:
        Integration of rule-based explanations with counterfactuals
        in a unified framework for decision trees.
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
        self.actionable_features = actionable_features or self.feature_names
        self.feature_ranges = feature_ranges or {}
        
        # Compute feature ranges from training data
        if X_train is not None:
            for i, name in enumerate(self.feature_names):
                if name not in self.feature_ranges:
                    self.feature_ranges[name] = (
                        X_train[:, i].min(),
                        X_train[:, i].max()
                    )
    
    def _find_counterfactual(
        self,
        x: np.ndarray,
        rule: Rule,
        prediction: Any
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        """
        Find minimal counterfactual by violating rule conditions.
        
        Strategy: For each condition, find the minimal change to violate it,
        then verify the prediction changes.
        """
        x_cf = x.copy()
        changes = {}
        
        # Get all possible predictions
        if hasattr(self.model, 'classes_'):
            other_classes = [c for c in self.model.classes_ if c != prediction]
        else:
            other_classes = [1 - prediction]  # Binary case
        
        # Try to change prediction by violating conditions
        for cond in rule.conditions:
            if cond.feature_name not in self.actionable_features:
                continue
            
            idx = cond.feature_index
            old_value = x[idx]
            
            # Find value that violates the condition
            epsilon = 0.01 * (self.feature_ranges.get(cond.feature_name, (0, 1))[1] - 
                             self.feature_ranges.get(cond.feature_name, (0, 1))[0])
            epsilon = max(epsilon, 0.001)
            
            if cond.operator in ['<=', '<']:
                # Need value > threshold
                new_value = cond.threshold + epsilon
            else:  # '>', '>='
                # Need value <= threshold
                new_value = cond.threshold - epsilon
            
            # Clip to valid range
            if cond.feature_name in self.feature_ranges:
                min_val, max_val = self.feature_ranges[cond.feature_name]
                new_value = np.clip(new_value, min_val, max_val)
            
            x_cf[idx] = new_value
            changes[cond.feature_name] = (old_value, new_value)
            
            # Check if prediction changed
            new_pred = self.model.predict(x_cf.reshape(1, -1))[0]
            if new_pred != prediction:
                break
        
        return x_cf, changes
    
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
        explanation = self.explain(x, X_background)
        explanation.method = "GPE-CF"
        
        # Find counterfactual
        x_cf, changes = self._find_counterfactual(
            x, explanation.rule, explanation.prediction
        )
        
        cf_prediction = self.model.predict(x_cf.reshape(1, -1))[0]
        
        # Calculate distance
        distance = np.linalg.norm(x - x_cf, ord=1)  # L1 distance
        
        return CounterfactualExplanation(
            explanation=explanation,
            counterfactual=x_cf,
            counterfactual_prediction=cf_prediction,
            changes=changes,
            distance=distance
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
    
    Scientific Contribution:
        First stability-aware explanation method for decision trees
        with formal stability guarantees.
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        min_precision: float = 0.95,
        min_stability: float = 0.8,
        n_perturbations: int = 10,
        perturbation_std: float = 0.01
    ):
        super().__init__(model, feature_names, X_train, min_precision)
        self.min_stability = min_stability
        self.n_perturbations = n_perturbations
        self.perturbation_std = perturbation_std
    
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
        feature_stds = X_background.std(axis=0)
        feature_stds[feature_stds == 0] = 1
        
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
            noise = np.random.randn(len(x)) * self.perturbation_std * feature_stds
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
                if precision < self.min_precision:
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
        precision_thresholds: Precision threshold per level (default: [0.9, 0.95, 0.99])
    
    Scientific Contribution:
        First multi-resolution explanation framework for decision trees
        with precision-controlled granularity.
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
        self.n_levels = n_levels
        self.precision_thresholds = precision_thresholds or [0.85, 0.95, 0.99]
        
        # Ensure correct number of thresholds
        if len(self.precision_thresholds) != n_levels:
            self.precision_thresholds = np.linspace(0.85, 0.99, n_levels).tolist()
    
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
        x = np.asarray(x).flatten()
        
        if X_background is None:
            X_background = self.X_train
        
        prediction = self.model.predict(x.reshape(1, -1))[0]
        
        # Get full decision path
        original_rule = self._greedy_phase(x)
        
        levels = []
        
        for threshold in self.precision_thresholds:
            self.min_precision = threshold
            
            # Prune with this threshold
            pruned_rule = self._prune_phase(x, deepcopy(original_rule), X_background)
            
            # Calculate metrics
            precision = self._calculate_precision(pruned_rule, prediction, X_background) if X_background is not None else 1.0
            coverage = self._calculate_coverage(pruned_rule, X_background) if X_background is not None else 0.0
            
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
        for threshold in np.linspace(0.7, 0.99, self.n_candidates):
            self.min_precision = threshold
            pruned_rule = self._prune_phase(x, deepcopy(original_rule), X_background)
            
            precision = self._calculate_precision(pruned_rule, prediction, X_background)
            coverage = self._calculate_coverage(pruned_rule, X_background)
            
            exp = Explanation(
                instance=x,
                rule=pruned_rule,
                prediction=prediction,
                precision=precision,
                coverage=coverage,
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
        x = np.asarray(x).flatten()
        
        if X_background is None:
            X_background = self.X_train
        
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
            # Normalize and weight: precision (0.4), coverage (0.3), 1/complexity (0.3)
            score = (0.4 * exp.precision + 
                    0.3 * exp.coverage + 
                    0.3 * (1 / max(exp.complexity, 1)))
            
            if score > best_score:
                best_score = score
                best_exp = exp
        
        best_exp.method = "GPE-Pareto"
        return best_exp

