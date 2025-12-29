"""
GPE Algorithm Variants.

This module implements different variants of the GPE algorithm:
- GPEOptimal: Exhaustive search for the minimal rule (optimal but slower)
- GPEWeighted: Uses feature weights for domain-specific pruning
- GPEMonotonic: Ensures monotonic relationships in explanations
- GPEEnsemble: Extension for ensemble tree models (Random Forest, XGBoost, etc.)

Author: Vladyslav Dehtiarov
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any, Tuple, Callable
from copy import deepcopy
from itertools import combinations

from .explanation import Condition, Rule, Explanation
from .tree_utils import (
    get_decision_path,
    simplify_conditions,
    is_ensemble_tree,
    is_single_tree
)
from .core import GPEExplainer


class GPEOptimal(GPEExplainer):
    """
    Optimal GPE variant using exhaustive search.
    
    Unlike the greedy variant, this searches through all possible
    subsets of conditions to find the truly minimal rule that
    maintains the precision threshold.
    
    Warning: Exponential complexity O(2^n) where n is the number of
    conditions. Only recommended for rules with <= 15 conditions.
    
    Parameters:
        model: A fitted sklearn decision tree
        feature_names: Optional list of feature names
        X_train: Training data for precision/coverage calculation
        min_precision: Minimum precision threshold for pruning
        max_conditions: Maximum number of conditions to search (for efficiency)
    
    Example:
        >>> explainer = GPEOptimal(model, min_precision=0.95)
        >>> explanation = explainer.explain(x)
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        min_precision: float = 0.95,
        max_conditions: int = 15
    ):
        super().__init__(
            model=model,
            feature_names=feature_names,
            X_train=X_train,
            min_precision=min_precision,
            simplify=True
        )
        self.max_conditions = max_conditions
    
    def _prune_phase(
        self,
        x: np.ndarray,
        rule: Rule,
        X_background: Optional[np.ndarray] = None
    ) -> Rule:
        """
        Optimal pruning using exhaustive search.
        
        Searches through all possible subsets of conditions,
        starting from smallest, to find the minimal rule.
        """
        if X_background is None or len(rule.conditions) <= 1:
            return rule
        
        if len(rule.conditions) > self.max_conditions:
            # Fall back to greedy for very long rules
            return super()._prune_phase(x, rule, X_background)
        
        prediction = self.model.predict(x.reshape(1, -1))[0]
        
        # Try subsets from smallest to largest
        for size in range(1, len(rule.conditions) + 1):
            for subset_indices in combinations(range(len(rule.conditions)), size):
                subset_conditions = [rule.conditions[i] for i in subset_indices]
                test_rule = Rule(conditions=subset_conditions, prediction=prediction)
                
                precision = self._calculate_precision(test_rule, prediction, X_background)
                
                if precision >= self.min_precision:
                    # Found minimal rule
                    test_rule.confidence = precision
                    return test_rule
        
        # If no subset meets threshold, return full rule
        return rule


class GPEWeighted(GPEExplainer):
    """
    Weighted GPE variant with feature importance.
    
    Uses domain-specific feature weights to guide the pruning process.
    Features with higher weights are more likely to be kept in the explanation.
    
    Parameters:
        model: A fitted sklearn decision tree
        feature_names: Optional list of feature names
        X_train: Training data for precision/coverage calculation
        min_precision: Minimum precision threshold
        feature_weights: Dictionary mapping feature names to weights (0-1)
                        Higher weight = more important = less likely to be pruned
    
    Example:
        >>> weights = {'age': 0.8, 'income': 0.9, 'credit_score': 1.0}
        >>> explainer = GPEWeighted(model, feature_weights=weights)
        >>> explanation = explainer.explain(x)
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        min_precision: float = 0.95,
        feature_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(
            model=model,
            feature_names=feature_names,
            X_train=X_train,
            min_precision=min_precision,
            simplify=True
        )
        
        # Initialize feature weights
        if feature_weights is None:
            self.feature_weights = {name: 1.0 for name in self.feature_names}
        else:
            # Fill in missing weights with 1.0
            self.feature_weights = {name: 1.0 for name in self.feature_names}
            self.feature_weights.update(feature_weights)
    
    def _prune_phase(
        self,
        x: np.ndarray,
        rule: Rule,
        X_background: Optional[np.ndarray] = None
    ) -> Rule:
        """
        Weighted pruning that considers feature importance.
        
        Conditions on low-weight features are removed first.
        """
        if X_background is None or len(rule.conditions) <= 1:
            return rule
        
        prediction = self.model.predict(x.reshape(1, -1))[0]
        pruned_rule = deepcopy(rule)
        
        while len(pruned_rule.conditions) > 1:
            # Find condition with lowest weight that can be removed
            best_idx = None
            best_score = float('inf')  # Lower is better (want to remove)
            
            for i, cond in enumerate(pruned_rule.conditions):
                # Create rule without condition i
                test_conditions = [c for j, c in enumerate(pruned_rule.conditions) if j != i]
                test_rule = Rule(conditions=test_conditions, prediction=prediction)
                
                precision = self._calculate_precision(test_rule, prediction, X_background)
                
                if precision >= self.min_precision:
                    # Score combines weight (lower = remove first) and precision
                    weight = self.feature_weights.get(cond.feature_name, 1.0)
                    score = weight * (1.0 - precision + 0.01)  # Small offset to prefer lower weight
                    
                    if score < best_score:
                        best_score = score
                        best_idx = i
            
            if best_idx is None:
                break
            
            pruned_rule.conditions.pop(best_idx)
        
        pruned_rule.prediction = prediction
        return pruned_rule
    
    def set_feature_weights(self, weights: Dict[str, float]):
        """Update feature weights."""
        self.feature_weights.update(weights)


class GPEMonotonic(GPEExplainer):
    """
    Monotonic GPE variant that preserves monotonic relationships.
    
    For features with known monotonic relationships (e.g., higher income
    should not decrease approval probability), this variant ensures
    explanations respect these constraints.
    
    Parameters:
        model: A fitted sklearn decision tree
        feature_names: Optional list of feature names
        X_train: Training data
        min_precision: Minimum precision threshold
        monotonic_constraints: Dictionary mapping feature names to constraints
                              +1 = increasing (higher value → higher prediction)
                              -1 = decreasing (higher value → lower prediction)
                               0 = no constraint
    
    Example:
        >>> constraints = {'income': +1, 'debt': -1}
        >>> explainer = GPEMonotonic(model, monotonic_constraints=constraints)
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        min_precision: float = 0.95,
        monotonic_constraints: Optional[Dict[str, int]] = None
    ):
        super().__init__(
            model=model,
            feature_names=feature_names,
            X_train=X_train,
            min_precision=min_precision,
            simplify=True
        )
        
        self.monotonic_constraints = monotonic_constraints or {}
    
    def _prune_phase(
        self,
        x: np.ndarray,
        rule: Rule,
        X_background: Optional[np.ndarray] = None
    ) -> Rule:
        """
        Pruning that respects monotonic constraints.
        
        Only keeps conditions that align with monotonic constraints.
        """
        if X_background is None or len(rule.conditions) <= 1:
            return rule
        
        prediction = self.model.predict(x.reshape(1, -1))[0]
        
        # First, filter conditions that violate monotonic constraints
        valid_conditions = []
        for cond in rule.conditions:
            constraint = self.monotonic_constraints.get(cond.feature_name, 0)
            
            if constraint == 0:
                # No constraint, keep
                valid_conditions.append(cond)
            elif constraint == +1:
                # Increasing: only keep ">" or ">=" conditions
                if cond.operator in ['>', '>=']:
                    valid_conditions.append(cond)
            elif constraint == -1:
                # Decreasing: only keep "<" or "<=" conditions
                if cond.operator in ['<', '<=']:
                    valid_conditions.append(cond)
        
        # Apply standard pruning on remaining conditions
        filtered_rule = Rule(
            conditions=valid_conditions,
            prediction=prediction,
            confidence=rule.confidence
        )
        
        return super()._prune_phase(x, filtered_rule, X_background)


class GPEEnsemble:
    """
    GPE Explainer for Ensemble Tree Models.
    
    Generates explanations for Random Forest, Gradient Boosting,
    XGBoost, and LightGBM models by aggregating explanations from
    individual trees.
    
    Parameters:
        model: A fitted ensemble tree model
        feature_names: Optional list of feature names
        X_train: Training data
        min_precision: Minimum precision threshold
        aggregation: How to aggregate explanations ('majority', 'intersection', 'union')
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>> 
        >>> explainer = GPEEnsemble(model, aggregation='majority')
        >>> explanation = explainer.explain(x)
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        min_precision: float = 0.95,
        aggregation: str = 'majority'
    ):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.min_precision = min_precision
        self.aggregation = aggregation
        
        if not is_ensemble_tree(model):
            raise ValueError(
                "GPEEnsemble requires an ensemble tree model "
                "(RandomForest, GradientBoosting, XGBoost, LightGBM)"
            )
        
        # Set default feature names
        if self.feature_names is None:
            n_features = model.n_features_in_
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Extract individual trees
        self._trees = self._extract_trees()
    
    def _extract_trees(self) -> List:
        """Extract individual trees from the ensemble."""
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor
        )
        
        if isinstance(self.model, (RandomForestClassifier, RandomForestRegressor)):
            return self.model.estimators_
        elif isinstance(self.model, (GradientBoostingClassifier, GradientBoostingRegressor)):
            # Gradient boosting has a 2D array of trees
            trees = []
            for stage in self.model.estimators_:
                for tree in stage:
                    trees.append(tree)
            return trees
        else:
            # Try XGBoost/LightGBM
            try:
                return self.model.get_booster().get_dump()
            except:
                raise ValueError(f"Unsupported ensemble model: {type(self.model)}")
    
    def explain(
        self,
        x: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        n_trees: Optional[int] = None
    ) -> Explanation:
        """
        Generate an explanation by aggregating from multiple trees.
        
        Args:
            x: Feature vector to explain
            X_background: Background data
            n_trees: Number of trees to use (None = all trees)
        
        Returns:
            Aggregated explanation
        """
        start_time = time.time()
        
        x = np.asarray(x).flatten()
        
        if X_background is None:
            X_background = self.X_train
        
        # Get explanations from individual trees
        trees_to_use = self._trees[:n_trees] if n_trees else self._trees
        
        all_conditions = []
        condition_counts: Dict[str, int] = {}
        
        for tree in trees_to_use:
            if not is_single_tree(tree):
                continue  # Skip non-tree objects (e.g., XGBoost dump strings)
            
            explainer = GPEExplainer(
                model=tree,
                feature_names=self.feature_names,
                X_train=X_background,
                min_precision=self.min_precision
            )
            
            try:
                exp = explainer.explain(x, X_background)
                for cond in exp.rule.conditions:
                    cond_key = str(cond)
                    condition_counts[cond_key] = condition_counts.get(cond_key, 0) + 1
                    all_conditions.append(cond)
            except Exception:
                continue
        
        # Aggregate conditions based on strategy
        prediction = self.model.predict(x.reshape(1, -1))[0]
        
        if self.aggregation == 'majority':
            # Keep conditions that appear in majority of trees
            threshold = len(trees_to_use) / 2
            aggregated_conditions = []
            seen = set()
            for cond in all_conditions:
                cond_key = str(cond)
                if cond_key not in seen and condition_counts[cond_key] >= threshold:
                    aggregated_conditions.append(cond)
                    seen.add(cond_key)
        
        elif self.aggregation == 'intersection':
            # Keep only conditions that appear in ALL trees
            threshold = len(trees_to_use)
            aggregated_conditions = []
            seen = set()
            for cond in all_conditions:
                cond_key = str(cond)
                if cond_key not in seen and condition_counts[cond_key] >= threshold:
                    aggregated_conditions.append(cond)
                    seen.add(cond_key)
        
        elif self.aggregation == 'union':
            # Keep all unique conditions
            aggregated_conditions = []
            seen = set()
            for cond in all_conditions:
                cond_key = str(cond)
                if cond_key not in seen:
                    aggregated_conditions.append(cond)
                    seen.add(cond_key)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Simplify conditions
        aggregated_conditions = simplify_conditions(aggregated_conditions)
        
        # Create final rule
        aggregated_rule = Rule(
            conditions=aggregated_conditions,
            prediction=prediction
        )
        
        # Calculate metrics
        if X_background is not None:
            precision = self._calculate_precision(aggregated_rule, prediction, X_background)
            coverage = self._calculate_coverage(aggregated_rule, X_background)
        else:
            precision = 1.0
            coverage = 0.0
        
        computation_time = time.time() - start_time
        
        return Explanation(
            instance=x,
            rule=aggregated_rule,
            prediction=prediction,
            precision=precision,
            coverage=coverage,
            method=f"GPE-Ensemble-{self.aggregation}",
            computation_time=computation_time,
            metadata={
                'n_trees': len(trees_to_use),
                'aggregation': self.aggregation,
                'condition_counts': condition_counts
            }
        )
    
    def _calculate_precision(
        self,
        rule: Rule,
        prediction: Any,
        X: np.ndarray
    ) -> float:
        """Calculate precision using the ensemble model."""
        mask = rule.evaluate_batch(X)
        
        if mask.sum() == 0:
            return 1.0
        
        X_matching = X[mask]
        y_pred = self.model.predict(X_matching)
        
        return (y_pred == prediction).mean()
    
    def _calculate_coverage(self, rule: Rule, X: np.ndarray) -> float:
        """Calculate coverage."""
        mask = rule.evaluate_batch(X)
        return mask.mean()
    
    def explain_batch(
        self,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> List[Explanation]:
        """Generate explanations for multiple instances."""
        explanations = []
        
        for i, x in enumerate(X):
            if verbose and (i + 1) % 10 == 0:
                print(f"Explaining instance {i + 1}/{X.shape[0]}")
            
            explanation = self.explain(x, X_background)
            explanations.append(explanation)
        
        return explanations

