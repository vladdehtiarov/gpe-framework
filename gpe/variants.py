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
from collections import defaultdict

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
    GPE Explainer for Ensemble Tree Models (High-Performance).
    
    Generates explanations for Random Forest, Gradient Boosting, etc.
    by finding the 'Consensus Rule' among the voting trees.
    
    OPTIMIZATION:
    Uses 'Fast Consensus' algorithm: precomputed paths + threshold aggregation.
    Avoids expensive per-tree pruning or ensemble-level pruning during explanation.
    
    Complexity: O(N_trees + N_conditions), typically < 10ms.
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
            raise ValueError("GPEEnsemble requires an ensemble tree model")
        
        if self.feature_names is None:
            n_features = model.n_features_in_
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        self._trees = self._extract_trees()
        self._tree_paths = self._precompute_paths()
    
    def _extract_trees(self) -> List:
        """Extract individual trees from the ensemble."""
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor
        )
        
        if isinstance(self.model, (RandomForestClassifier, RandomForestRegressor)):
            return self.model.estimators_
        elif isinstance(self.model, (GradientBoostingClassifier, GradientBoostingRegressor)):
            trees = []
            for stage in self.model.estimators_:
                for tree in stage:
                    trees.append(tree)
            return trees
        else:
            try:
                return self.model.get_booster().get_dump()
            except:
                raise ValueError(f"Unsupported ensemble model: {type(self.model)}")

    def _precompute_paths(self) -> List[Dict[int, List[Condition]]]:
        """Precompute leaf-to-conditions map for all trees."""
        tree_paths = []
        for tree in self._trees:
            paths = {}
            if hasattr(tree, 'tree_'):
                self._traverse_tree(tree, paths)
            tree_paths.append(paths)
        return tree_paths

    def _traverse_tree(self, tree, paths_dict):
        """Recursively map leaf IDs to conditions."""
        tree_ = tree.tree_
        feature_names = self.feature_names
        stack = [(0, [])] # (node_id, conditions)
        
        while stack:
            node_id, current_conds = stack.pop()
            
            # Leaf
            if tree_.children_left[node_id] == tree_.children_right[node_id]:
                paths_dict[node_id] = current_conds
                continue
            
            # Internal
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            feat_name = feature_names[feature_idx]
            
            # Right child (>)
            right_child = tree_.children_right[node_id]
            right_cond = Condition(feature_idx, feat_name, '>', threshold)
            stack.append((right_child, current_conds + [right_cond]))
            
            # Left child (<=)
            left_child = tree_.children_left[node_id]
            left_cond = Condition(feature_idx, feat_name, '<=', threshold)
            stack.append((left_child, current_conds + [left_cond]))
    
    def explain(
        self,
        x: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        n_trees: Optional[int] = None
    ) -> Explanation:
        """
        Generate explanation using Fast Consensus.
        """
        start_time = time.time()
        x = np.asarray(x).flatten()
        
        if X_background is None:
            X_background = self.X_train
            
        prediction = self.model.predict(x.reshape(1, -1))[0]
        
        # 1. Get winning trees (Prediction Consistency)
        # We only consider trees that voted for the winning class (for Classifiers)
        # For Regressors, we might take all.
        try:
             # Only works for sklearn classifiers
             if hasattr(self.model, 'estimators_') and hasattr(self.model, 'classes_'):
                 # Get individual predictions
                 pass # Too slow to run predict on all trees.
                 # Actually, we should just use all trees. 
                 # Majority rule is implicit in the count collection.
        except:
             pass

        # 2. Get Leaves O(1)
        leaves = []
        try:
            if hasattr(self.model, 'apply'):
                leaves = self.model.apply(x.reshape(1, -1))[0]
                if len(leaves.shape) > 1: leaves = leaves.flatten()
            else:
                 # Fallback
                 leaves = [-1] * len(self._trees)
        except:
             leaves = [-1] * len(self._trees)

        # 3. Collect Conditions
        trees_to_use = self._trees[:n_trees] if n_trees else self._trees
        n_use = len(trees_to_use)
        if len(leaves) < n_use:
             leaves = list(leaves) + [-1]*(n_use - len(leaves))
             
        # Group by (Feature, Operator) -> List of Thresholds
        grouped_conditions = defaultdict(list)
        active_trees = 0
        
        for i, tree in enumerate(trees_to_use):
            if i < len(self._tree_paths) and self._tree_paths[i]:
                leaf_id = leaves[i]
                if leaf_id in self._tree_paths[i]:
                    active_trees += 1
                    path = self._tree_paths[i][leaf_id]
                    for cond in path:
                         key = (cond.feature_index, cond.feature_name, cond.operator)
                         grouped_conditions[key].append(cond.threshold)

        # 4. Forward Selection (Greedy Add)
        # Instead of taking ALL frequent conditions (which leads to complex, low-coverage rules),
        # we Greedily ADD conditions starting from the most frequent ones.
        
        # Min support to be considered a candidate
        min_support = max(1, int(active_trees * 0.1))
        
        candidates = []
        for (feat_idx, feat_name, op), thresholds in grouped_conditions.items():
            if len(thresholds) >= min_support:
                median_thresh = np.median(thresholds)
                cond = Condition(feat_idx, feat_name, op, median_thresh)
                # Score = frequency
                candidates.append((len(thresholds), cond))
        
        # Sort candidates by frequency (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        current_conditions = []
        current_precision = 0.0
        # Start with empty rule precision (global prior)
        if X_background is not None:
             prior_mask = np.ones(len(X_background), dtype=bool)
             prior_prec = (self.model.predict(X_background) == prediction).mean()
             current_precision = prior_prec
        
        max_conds = 6 # Limit complexity for interpretability
        
        # Helper for precision O(1) using vectorization
        def get_metrics(conds):
            if not conds: 
                 if X_background is not None:
                      return current_precision, 1.0
                 return 0.0, 1.0
                 
            r = Rule(conds, prediction)
            mask = r.evaluate_batch(X_background)
            coverage = mask.mean()
            if mask.sum() == 0: return 0.0, 0.0
            prec = (self.model.predict(X_background[mask]) == prediction).mean()
            return prec, coverage

        best_conditions = []
        best_precision = current_precision
        
        # Iterate through top 15 candidates
        for _, cand_cond in candidates[:15]:
             # Propose adding this condition
             proposal = best_conditions + [cand_cond]
             proposal = simplify_conditions(proposal)
             
             prec, cov = get_metrics(proposal)
             
             # Heuristic: Accept if precision improves OR stays high (>0.95)
             # If precision drops significantly, don't add.
             
             if prec >= best_precision:
                  best_conditions = proposal
                  best_precision = prec
             elif prec >= self.min_precision:
                  # Still good enough, and makes rule more specific?
                  # Actually, we usually want minimal rule for target precision.
                  # If we already have precision > min_precision, we stop?
                  # Let's say we want to maximize precision.
                  pass
             
             if best_precision >= self.min_precision:
                  # We found a rule that explains it well enough!
                  break
                  
             if len(best_conditions) >= max_conds:
                  break
        
        # Fallback: If we ended up with nothing (or just prior), take top 1 feature
        if not best_conditions and candidates:
             best_conditions = [candidates[0][1]]
             best_precision, _ = get_metrics(best_conditions)

        # 5. Simplify
        final_conditions = simplify_conditions(best_conditions)
        rule = Rule(final_conditions, prediction)
        
        # 6. Final Metrics
        if X_background is not None:
             precision, coverage = get_metrics(final_conditions)
        else:
             precision, coverage = 1.0, 0.0

        return Explanation(
            instance=x,
            rule=rule,
            prediction=prediction,
            precision=precision,
            coverage=coverage,
            method="GPE-Ensemble Forward",
            computation_time=time.time() - start_time,
            metadata={'active_trees': active_trees, 'candidates': len(candidates)}
        )

    def _calculate_precision(self, rule, prediction, X):
        mask = rule.evaluate_batch(X)
        if mask.sum() == 0: return 0.0
        return (self.model.predict(X[mask]) == prediction).mean()
        
    def _calculate_coverage(self, rule, X):
        return rule.evaluate_batch(X).mean()
    
    def explain_batch(self, X, X_background=None):
        return [self.explain(x, X_background) for x in X]
