"""
Core GPE (Greedy-Prune-Explain) Algorithm Implementation.

The GPE algorithm generates minimal local explanations for decision tree predictions.
It works in three phases:
1. GREEDY: Extract the full decision path from root to leaf
2. PRUNE: Remove conditions that don't affect prediction precision
3. EXPLAIN: Return the minimal rule with metrics

Author: Vladyslav Dehtiarov
"""

import numpy as np
import time
from typing import List, Optional, Tuple, Dict, Any, Callable
from copy import deepcopy
from sklearn.utils.validation import check_is_fitted

from .explanation import Condition, Rule, Explanation
from .tree_utils import (
    get_decision_path,
    simplify_conditions,
    is_single_tree,
    is_ensemble_tree
)


def check_min_precision(min_precision: float) -> None:
    """Raise if the precision threshold is not a number in (0, 1]."""
    if not isinstance(min_precision, (int, float, np.number)) or not 0 < min_precision <= 1:
        raise ValueError(f"min_precision must be in (0, 1], got {min_precision!r}")


def check_feature_names(feature_names, model) -> List[str]:
    """Return the feature names as a list, defaulting to the ones the model saw."""
    if feature_names is None:
        if hasattr(model, 'feature_names_in_'):
            return [str(name) for name in model.feature_names_in_]
        return [f"feature_{i}" for i in range(model.n_features_in_)]
    feature_names = list(feature_names)
    if len(feature_names) != model.n_features_in_:
        raise ValueError(
            f"feature_names has {len(feature_names)} names, "
            f"the model expects {model.n_features_in_} features"
        )
    return feature_names


def as_matrix(X, model, name: str = 'X') -> np.ndarray:
    """
    Check a data matrix against the model and return it as a numeric 2D array.
    
    A NumPy array comes back as the same object, so caching by identity keeps
    working. A DataFrame must have the columns the model was fitted on, in order.
    
    Args:
        X: Array-like or DataFrame
        model: The fitted model
        name: Name of the argument, for error messages
        
    Returns:
        2D array with one column per model feature
    """
    if hasattr(X, 'iloc') and hasattr(X, 'columns'):
        expected = getattr(model, 'feature_names_in_', None)
        if expected is not None and list(X.columns) != list(expected):
            raise ValueError(f"{name} columns differ from the features the model was fitted on")
    if not isinstance(X, np.ndarray) or X.dtype == object:
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must hold numbers only: {error}") from error
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {X.ndim}D")
    if X.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if X.shape[1] != model.n_features_in_:
        raise ValueError(
            f"{name} has {X.shape[1]} features, the model expects {model.n_features_in_}"
        )
    return X


def as_instance(x, model) -> np.ndarray:
    """Check one instance against the model and return it as a float 1D array."""
    try:
        x = np.asarray(x, dtype=np.float64).flatten()
    except (TypeError, ValueError) as error:
        raise ValueError(f"The instance must hold numbers only: {error}") from error
    if x.shape[0] != model.n_features_in_:
        raise ValueError(
            f"The instance has {x.shape[0]} features, the model expects {model.n_features_in_}"
        )
    return x


class GPEExplainer:
    """
    Greedy-Prune-Explain Explainer for Decision Trees.
    
    This is the main class for generating local explanations for decision tree
    predictions using the GPE algorithm.
    
    Parameters:
        model: A fitted sklearn DecisionTreeClassifier. Regression trees are rejected:
            precision compares predicted classes, which a regressor does not have.
        feature_names: Optional list of feature names
        X_train: Default background data for pruning and for precision/coverage.
            Without background data nothing is pruned and both metrics are NaN.
        min_precision: Minimum precision threshold for pruning, in (0, 1] (default: 0.95)
        simplify: Whether to simplify conditions by merging redundant ones (default: True)
    
    Example:
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from gpe import GPEExplainer
        >>> 
        >>> model = DecisionTreeClassifier()
        >>> model.fit(X_train, y_train)
        >>> 
        >>> explainer = GPEExplainer(model, feature_names=['age', 'income'])
        >>> explanation = explainer.explain(X_test[0])
        >>> print(explanation)
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        X_train: Optional[np.ndarray] = None,
        min_precision: float = 0.95,
        simplify: bool = True
    ):
        # Validate model
        if not is_single_tree(model):
            if is_ensemble_tree(model):
                raise ValueError(
                    "GPEExplainer is designed for single decision trees. "
                    "For ensemble models, use GPEEnsemble instead."
                )
            raise ValueError(
                "GPEExplainer requires a fitted DecisionTreeClassifier. "
                "For other models, consider using "
                "model-agnostic methods like LIME or SHAP."
            )
        if not hasattr(model, 'predict_proba'):
            raise ValueError(
                "GPEExplainer requires a DecisionTreeClassifier: precision compares "
                "predicted classes, which a regression tree does not have."
            )
        check_is_fitted(model, 'tree_')
        if model.n_outputs_ != 1:
            raise ValueError("GPEExplainer requires a single-output classifier")
        check_min_precision(min_precision)
        
        self.model = model
        self.feature_names = check_feature_names(feature_names, model)
        self.X_train = None if X_train is None else as_matrix(X_train, model, 'X_train')
        self.min_precision = min_precision
        self.simplify = simplify
    
    def explain(
        self,
        x: np.ndarray,
        X_background: Optional[np.ndarray] = None
    ) -> Explanation:
        """
        Generate an explanation for a single instance.
        
        Args:
            x: Feature vector to explain (1D array)
            X_background: Background data for precision/coverage calculation.
                         If None, uses X_train if provided during initialization.
        
        Returns:
            Explanation object containing the minimal rule and metrics
        """
        start_time = time.time()
        
        x = as_instance(x, self.model)
        
        # Use X_train if no background provided
        if X_background is None:
            X_background = self.X_train
        else:
            X_background = as_matrix(X_background, self.model, 'X_background')
        
        # Phase 1: GREEDY - Extract full decision path
        original_rule = self._greedy_phase(x)
        
        # Phase 2: PRUNE - Remove unnecessary conditions
        pruned_rule = self._prune_phase(x, original_rule, X_background)
        
        # Phase 3: EXPLAIN - Calculate metrics and create explanation
        explanation = self._explain_phase(
            x, pruned_rule, original_rule, X_background, start_time
        )
        
        return explanation
    
    def _greedy_phase(self, x: np.ndarray) -> Rule:
        """
        Phase 1: Extract the full decision path from the tree.
        
        This phase follows the tree from root to leaf, collecting all
        conditions along the path.
        
        Args:
            x: Feature vector
            
        Returns:
            Rule representing the full decision path
        """
        rule = get_decision_path(self.model, x, self.feature_names)
        
        # Optionally simplify conditions
        if self.simplify:
            rule.conditions = simplify_conditions(rule.conditions)
        
        return rule
    
    def _prune_phase(
        self,
        x: np.ndarray,
        rule: Rule,
        X_background: Optional[np.ndarray] = None
    ) -> Rule:
        """
        Phase 2: Prune unnecessary conditions from the rule.
        
        This phase iteratively removes conditions that have minimal impact
        on precision, keeping only the most important ones.
        
        The algorithm:
        1. Start with the full rule
        2. For each condition, calculate precision without it
        3. Remove the condition with highest precision-without
        4. Repeat until precision drops below threshold
        
        Args:
            x: Feature vector
            rule: The full rule from greedy phase
            X_background: Background data for precision calculation
            
        Returns:
            Pruned rule with minimal conditions
        """
        if X_background is None or len(rule.conditions) <= 1:
            return rule
        
        prediction = self.model.predict(x.reshape(1, -1))[0]
        pruned_rule = deepcopy(rule)
        
        while len(pruned_rule.conditions) > 1:
            # Find the best condition to remove
            best_idx = None
            best_precision = -1
            
            for i in range(len(pruned_rule.conditions)):
                # Create rule without condition i
                test_conditions = [c for j, c in enumerate(pruned_rule.conditions) if j != i]
                test_rule = Rule(conditions=test_conditions, prediction=prediction)
                
                # Calculate precision
                precision = self._calculate_precision(test_rule, prediction, X_background)
                
                if precision >= self.min_precision and precision > best_precision:
                    best_precision = precision
                    best_idx = i
            
            # If no condition can be removed, stop
            if best_idx is None:
                break
            
            # Remove the condition
            pruned_rule.conditions.pop(best_idx)
        
        pruned_rule.prediction = prediction
        
        return pruned_rule
    
    def _explain_phase(
        self,
        x: np.ndarray,
        pruned_rule: Rule,
        original_rule: Rule,
        X_background: Optional[np.ndarray],
        start_time: float
    ) -> Explanation:
        """
        Phase 3: Create the final explanation with metrics.
        
        Args:
            x: Feature vector
            pruned_rule: The pruned rule
            original_rule: The original full rule
            X_background: Background data
            start_time: Start time for computation time calculation
            
        Returns:
            Complete Explanation object
        """
        prediction = self.model.predict(x.reshape(1, -1))[0]
        
        # Calculate metrics; without rows there is nothing to report
        if X_background is not None:
            precision = self._calculate_precision(pruned_rule, prediction, X_background)
            coverage = self._calculate_coverage(pruned_rule, X_background)
        else:
            precision = float('nan')
            coverage = float('nan')
        
        computation_time = time.time() - start_time
        
        return Explanation(
            instance=x,
            rule=pruned_rule,
            prediction=prediction,
            precision=precision,
            coverage=coverage,
            original_rule=original_rule,
            method="GPE",
            computation_time=computation_time,
            metadata={
                'min_precision': self.min_precision,
                'simplify': self.simplify,
                # Rows behind the reported precision; None without background data
                'n_background': (
                    int(round(coverage * len(X_background)))
                    if X_background is not None else None
                ),
                # Variants can miss the threshold; NaN precision does not meet it
                'meets_threshold': bool(precision >= self.min_precision)
            }
        )
    
    def _calculate_precision(
        self,
        rule: Rule,
        prediction: Any,
        X: np.ndarray
    ) -> float:
        """
        Calculate the precision of a rule.
        
        Precision = P(prediction | rule satisfied)
        
        This is the probability that an instance satisfying the rule
        will have the same prediction as the instance being explained.
        
        Args:
            rule: The rule to evaluate
            prediction: The target prediction
            X: Background data
            
        Returns:
            Precision value between 0 and 1. NaN when no row satisfies the rule:
            NaN fails every comparison with the threshold, so a rule without
            evidence is never accepted during pruning.
        """
        # Find instances that satisfy the rule
        mask = rule.evaluate_batch(X)
        
        if mask.sum() == 0:
            return float('nan')
        
        # Get predictions for matching instances
        X_matching = X[mask]
        y_pred = self.model.predict(X_matching)
        
        # Calculate precision
        return float((y_pred == prediction).mean())
    
    def _calculate_coverage(self, rule: Rule, X: np.ndarray) -> float:
        """
        Calculate the coverage of a rule.
        
        Coverage = proportion of instances that satisfy the rule
        
        Args:
            rule: The rule to evaluate
            X: Background data
            
        Returns:
            Coverage value between 0 and 1
        """
        mask = rule.evaluate_batch(X)
        return float(mask.mean())
    
    def explain_batch(
        self,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> List[Explanation]:
        """
        Generate explanations for multiple instances.
        
        Args:
            X: Feature matrix (2D array)
            X_background: Background data for precision/coverage calculation
            verbose: Whether to print progress
            
        Returns:
            List of Explanation objects
        """
        X = as_matrix(X, self.model, 'X')
        if X_background is not None:
            X_background = as_matrix(X_background, self.model, 'X_background')
        
        explanations = []
        n_samples = X.shape[0]
        
        for i, x in enumerate(X):
            if verbose and (i + 1) % 10 == 0:
                print(f"Explaining instance {i + 1}/{n_samples}")
            
            explanation = self.explain(x, X_background)
            explanations.append(explanation)
        
        return explanations
    
    def get_feature_importance(
        self,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate global feature importance based on explanations.
        
        Features that appear more frequently in explanations are considered
        more important.
        
        Args:
            X: Feature matrix to explain
            X_background: Background data
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        explanations = self.explain_batch(X, X_background)
        
        feature_counts = {name: 0 for name in self.feature_names}
        
        for exp in explanations:
            for cond in exp.rule.conditions:
                feature_counts[cond.feature_name] += 1
        
        # Normalize
        total = sum(feature_counts.values())
        if total > 0:
            feature_importance = {k: v / total for k, v in feature_counts.items()}
        else:
            feature_importance = {k: 0.0 for k in feature_counts}
        
        return feature_importance
    
    def __repr__(self) -> str:
        return (f"GPEExplainer(model={type(self.model).__name__}, "
                f"min_precision={self.min_precision})")


class GPEExplainerSKLearn:
    """
    Scikit-learn compatible wrapper for GPEExplainer.
    
    This class provides a familiar sklearn-like API for the GPE explainer.
    
    Parameters:
        min_precision: Minimum precision threshold for pruning
        simplify: Whether to simplify conditions
    """
    
    def __init__(
        self,
        min_precision: float = 0.95,
        simplify: bool = True
    ):
        self.min_precision = min_precision
        self.simplify = simplify
        self._explainer = None
        self._is_fitted = False
    
    def fit(
        self,
        model,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'GPEExplainerSKLearn':
        """
        Fit the explainer to a model and data.
        
        Args:
            model: A fitted sklearn decision tree
            X: Training data
            feature_names: Optional feature names
            
        Returns:
            self
        """
        self._explainer = GPEExplainer(
            model=model,
            feature_names=feature_names,
            X_train=X,
            min_precision=self.min_precision,
            simplify=self.simplify
        )
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> List[Explanation]:
        """
        Generate explanations for the given instances.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of explanations
        """
        if not self._is_fitted:
            raise ValueError("Explainer must be fitted before calling transform")
        
        return self._explainer.explain_batch(X)
    
    def fit_transform(
        self,
        model,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[Explanation]:
        """
        Fit and transform in one step.
        
        Args:
            model: A fitted sklearn decision tree
            X: Data to explain
            feature_names: Optional feature names
            
        Returns:
            List of explanations
        """
        self.fit(model, X, feature_names)
        return self.transform(X)

