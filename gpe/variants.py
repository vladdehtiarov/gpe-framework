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
    GPE Explainer for Ensemble Tree Models (Fast Consensus).

    Generates explanations for Random Forest, Gradient Boosting, etc.
    as a consensus rule agreed upon by the trees that voted for the
    ensemble's prediction.

    The algorithm has three phases:
    1. Path Extraction: decision path of x in every tree and the tree's vote
    2. Threshold Aggregation: per feature, the bounds of the voting trees are
       intersected into [low_j, high_j]; a feature enters the candidate pool
       when it is used by at least M/2 + 1 trees ('majority') or by trees
       holding more than half of the total weight ('weighted', trees weighted
       by their individual accuracy on the training data)
    3. Consensus Pruning: greedily remove the condition whose removal lowers
       precision the least, while precision stays >= min_precision

    Complexity: O(N_trees + N_conditions) per explanation.

    Parameters:
        model: A fitted sklearn ensemble (RandomForest*, GradientBoosting*)
        feature_names: Optional list of feature names
        X_train: Training data (default background, tree weights)
        min_precision: Precision threshold tau
        aggregation: 'majority' or 'weighted'
        y_train: Optional labels for tree weights; when omitted, the
                 ensemble's own predictions on X_train are used

    Example:
        >>> explainer = GPEEnsemble(rf_model, X_train=X_train)
        >>> explanation = explainer.explain(x)
    """

    AGGREGATIONS = ('majority', 'weighted')

    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        min_precision: float = 0.95,
        aggregation: str = 'majority',
        y_train: Optional[np.ndarray] = None
    ):
        if aggregation not in self.AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {self.AGGREGATIONS}, got '{aggregation}'"
            )
        if not is_ensemble_tree(model):
            raise ValueError("GPEEnsemble requires an ensemble tree model")

        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.min_precision = min_precision
        self.aggregation = aggregation

        if self.feature_names is None:
            n_features = model.n_features_in_
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        self._trees = self._extract_trees()
        self._tree_paths = self._precompute_paths()
        self._tree_weights = self._compute_tree_weights(y_train)
        self._bg_cache = (None, None)

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
            except Exception:
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
        """Map leaf IDs to the conditions on the path from the root."""
        tree_ = tree.tree_
        feature_names = self.feature_names
        stack = [(0, [])]  # (node_id, conditions)

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
            right_cond = Condition(feature_idx, feat_name, '>', threshold, float32_input=True)
            stack.append((right_child, current_conds + [right_cond]))

            # Left child (<=)
            left_child = tree_.children_left[node_id]
            left_cond = Condition(feature_idx, feat_name, '<=', threshold, float32_input=True)
            stack.append((left_child, current_conds + [left_cond]))

    def _votes_by_class(self) -> bool:
        """Whether individual trees cast class votes (bagging classifiers)."""
        from sklearn.ensemble import RandomForestClassifier
        return isinstance(self.model, RandomForestClassifier)

    def _compute_tree_weights(self, y_train: Optional[np.ndarray]) -> np.ndarray:
        """Vote weights: 1 per tree, or tree accuracy for 'weighted'."""
        n = len(self._trees)
        if self.aggregation == 'majority' or not self._votes_by_class():
            return np.ones(n)
        if self.X_train is None:
            raise ValueError("aggregation='weighted' requires X_train")

        X = np.asarray(self.X_train)
        labels = self.model.predict(X) if y_train is None else np.asarray(y_train)
        classes = self.model.classes_
        weights = np.empty(n)
        for i, tree in enumerate(self._trees):
            tree_pred = classes[tree.predict(X).astype(int)]
            weights[i] = (tree_pred == labels).mean()
        return weights

    def _tree_vote(self, i: int, leaf_id: int, class_idx: int) -> bool:
        """Whether tree i voted for the ensemble's class at this leaf."""
        if not self._votes_by_class():
            return True
        value = self._trees[i].tree_.value[leaf_id]
        return int(np.argmax(value[0])) == class_idx

    def _background_predictions(self, X_background: np.ndarray) -> np.ndarray:
        """Model predictions on the background set, cached between calls."""
        key, preds = self._bg_cache
        if key is not X_background:
            preds = self.model.predict(X_background)
            self._bg_cache = (X_background, preds)
        return preds

    def explain(
        self,
        x: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        n_trees: Optional[int] = None
    ) -> Explanation:
        """
        Generate a consensus explanation for a single instance.

        Args:
            x: Instance to explain (1D array)
            X_background: Background data for precision/coverage
                          (defaults to X_train)
            n_trees: Use only the first n_trees trees of the ensemble

        Returns:
            Explanation object
        """
        start_time = time.time()
        x = np.asarray(x).flatten()

        if X_background is None:
            X_background = self.X_train

        prediction = self.model.predict(x.reshape(1, -1))[0]
        class_idx = 0
        if hasattr(self.model, 'classes_'):
            class_idx = int(np.flatnonzero(self.model.classes_ == prediction)[0])

        # Phase 1: Path Extraction
        n_use = min(n_trees or len(self._trees), len(self._trees))
        leaves = np.asarray(self.model.apply(x.reshape(1, -1))).flatten()
        weights = self._tree_weights[:n_use]
        total_weight = weights.sum()

        # Phase 2: Threshold Aggregation over the trees that voted for the class
        lows: Dict[int, float] = {}
        highs: Dict[int, float] = {}
        support: Dict[int, float] = defaultdict(float)
        voting_trees = 0

        for i in range(n_use):
            path = self._tree_paths[i].get(int(leaves[i]))
            if path is None or not self._tree_vote(i, int(leaves[i]), class_idx):
                continue
            voting_trees += 1
            for feat_idx in {c.feature_index for c in path}:
                support[feat_idx] += weights[i]
            for cond in path:
                j = cond.feature_index
                if cond.operator == '>':
                    lows[j] = max(lows.get(j, -np.inf), cond.threshold)
                else:
                    highs[j] = min(highs.get(j, np.inf), cond.threshold)

        # Consensus: support of M/2 + 1 trees, or more than half the weight
        pool = [j for j, s in support.items() if s > total_weight / 2]
        consensus = bool(pool)
        if not consensus and support:
            # No feature reaches consensus: fall back to the most supported one
            pool = [max(support, key=support.get)]

        candidates = []
        for j in sorted(pool, key=lambda f: -support[f]):
            if j in lows:
                candidates.append(Condition(j, self.feature_names[j], '>', lows[j], float32_input=True))
            if j in highs:
                candidates.append(Condition(j, self.feature_names[j], '<=', highs[j], float32_input=True))

        # Phase 3: Consensus Pruning
        if X_background is not None:
            X_background = np.asarray(X_background)
            bg_pred = self._background_predictions(X_background)

            def metrics(conds):
                mask = Rule(conds, prediction).evaluate_batch(X_background)
                if mask.sum() == 0:
                    return 1.0, 0.0  # same convention as GPEExplainer
                return (bg_pred[mask] == prediction).mean(), mask.mean()

            conditions = list(candidates)
            precision, coverage = metrics(conditions)
            while len(conditions) > 1:
                best = None
                for i in range(len(conditions)):
                    trial = conditions[:i] + conditions[i + 1:]
                    p, c = metrics(trial)
                    if p >= self.min_precision and (best is None or (p, c) > best[1:]):
                        best = (i, p, c)
                if best is None:
                    break
                conditions.pop(best[0])
                precision, coverage = best[1], best[2]
        else:
            conditions = list(candidates)
            precision, coverage = 1.0, 0.0

        rule = Rule(conditions, prediction)

        return Explanation(
            instance=x,
            rule=rule,
            prediction=prediction,
            precision=float(precision),
            coverage=float(coverage),
            original_rule=Rule(list(candidates), prediction),
            method=f"GPE-Ensemble ({self.aggregation})",
            computation_time=time.time() - start_time,
            metadata={
                'n_trees': n_use,
                'voting_trees': voting_trees,
                'candidates': len(candidates),
                'consensus': consensus,
                'meets_threshold': bool(precision >= self.min_precision),
            }
        )

    def explain_batch(self, X, X_background=None):
        return [self.explain(x, X_background) for x in X]
