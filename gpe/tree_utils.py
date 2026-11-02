"""
Utility functions for working with decision trees.

This module provides functions to:
- Extract decision paths from sklearn trees
- Convert tree nodes to conditions
- Navigate tree structures
"""

import numpy as np
from dataclasses import replace
from typing import List, Tuple, Optional, Dict, Any
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from .explanation import Condition, Rule


def _matches_missing(tree_, node_id: int, went_left: bool) -> bool:
    """Whether the tree sends a missing value to this branch of the split."""
    if not hasattr(tree_, 'missing_go_to_left'):  # scikit-learn before 1.3
        return False
    return bool(tree_.missing_go_to_left[node_id]) == went_left


def _path_nodes(tree, x: np.ndarray) -> List[int]:
    """
    Nodes from the root to the leaf of x, as predict routes it.
    
    tree.decision_path is not used: in scikit-learn 1.7 it sends a missing value
    to the right child whatever the split learned, unlike apply and predict.
    """
    tree_ = tree.tree_
    missing_left = getattr(tree_, 'missing_go_to_left', None)
    values = np.asarray(x, dtype=np.float32)
    node = 0
    nodes = [node]
    while tree_.children_left[node] != tree_.children_right[node]:
        value = values[tree_.feature[node]]
        if value != value:
            go_left = missing_left is not None and bool(missing_left[node])
        else:
            go_left = value <= tree_.threshold[node]
        node = int(tree_.children_left[node] if go_left else tree_.children_right[node])
        nodes.append(node)
    return nodes


def get_decision_path(
    tree,
    x: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Rule:
    """
    Extract the decision path for an instance from a decision tree.
    
    Args:
        tree: A fitted sklearn DecisionTreeClassifier or DecisionTreeRegressor
        x: Feature vector (1D array)
        feature_names: Optional list of feature names
        
    Returns:
        Rule object representing the decision path
    """
    tree_ = tree.tree_
    
    # Get the path through the tree
    node_indices = _path_nodes(tree, x)
    
    # Default feature names
    n_features = tree_.n_features
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    conditions = []
    
    for i, node_id in enumerate(node_indices[:-1]):  # Exclude leaf node
        # Get the feature and threshold at this node
        feature_idx = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        
        if feature_idx < 0:  # Leaf node
            continue
        
        # Determine if we went left (<=) or right (>)
        next_node = node_indices[i + 1]
        left_child = tree_.children_left[node_id]
        
        if next_node == left_child:
            operator = '<='
        else:
            operator = '>'
        
        condition = Condition(
            feature_index=int(feature_idx),
            feature_name=feature_names[feature_idx],
            operator=operator,
            threshold=float(threshold),
            is_categorical=False,
            float32_input=True,
            matches_missing=_matches_missing(tree_, node_id, operator == '<='),
            missing_in_instance=bool(np.isnan(float(x[feature_idx])))
        )
        conditions.append(condition)
    
    # Get prediction at the leaf
    leaf_node = node_indices[-1]
    
    if hasattr(tree, 'classes_'):  # Classification
        class_counts = tree_.value[leaf_node][0]
        prediction = tree.classes_[np.argmax(class_counts)]
        confidence = class_counts[np.argmax(class_counts)] / class_counts.sum()
    else:  # Regression
        prediction = tree_.value[leaf_node][0][0]
        confidence = None
    
    return Rule(
        conditions=conditions,
        prediction=prediction,
        confidence=confidence
    )


def get_leaf_paths(
    tree,
    feature_names: Optional[List[str]] = None
) -> Dict[int, List[Tuple[int, Condition]]]:
    """
    Map every leaf to its root-to-leaf literals, in depth order.
    
    Args:
        tree: A fitted sklearn decision tree
        feature_names: Optional list of feature names
        
    Returns:
        Dictionary: leaf node ID -> list of (internal node ID, literal)
    """
    tree_ = tree.tree_
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree_.n_features)]
    
    paths: Dict[int, List[Tuple[int, Condition]]] = {}
    stack: List[Tuple[int, List[Tuple[int, Condition]]]] = [(0, [])]
    while stack:
        node, path = stack.pop()
        left, right = tree_.children_left[node], tree_.children_right[node]
        if left == right:
            paths[int(node)] = path
            continue
        feature = int(tree_.feature[node])
        threshold = float(tree_.threshold[node])
        name = feature_names[feature]
        for child, operator in ((left, '<='), (right, '>')):
            literal = Condition(
                feature, name, operator, threshold, float32_input=True,
                matches_missing=_matches_missing(tree_, node, operator == '<=')
            )
            stack.append((child, path + [(int(node), literal)]))
    return paths


def representative_instance(tree, path: List[Tuple[int, Condition]]) -> np.ndarray:
    """
    Build an instance that reaches the leaf at the end of a path.
    
    Values are float32-representable, so the split tests see them unchanged.
    A feature gets NaN when only a missing value satisfies its literals, which
    happens in trees trained on missing data. Features not on the path are set to 0.
    
    Args:
        tree: A fitted sklearn decision tree
        path: Literals of one leaf, as returned by get_leaf_paths
        
    Returns:
        Feature vector (1D array)
    """
    lows: Dict[int, float] = {}
    highs: Dict[int, float] = {}
    for _, literal in path:
        j = literal.feature_index
        if literal.operator == '>':
            lows[j] = max(lows.get(j, -np.inf), literal.threshold)
        else:
            highs[j] = min(highs.get(j, np.inf), literal.threshold)
    
    x = np.zeros(tree.tree_.n_features, dtype=np.float64)
    for j in set(lows) | set(highs):
        low, high = lows.get(j, -np.inf), highs.get(j, np.inf)
        if np.isfinite(high):
            value = np.float32(high)
            if float(value) > high:
                value = np.nextafter(value, np.float32(-np.inf))
        elif np.isfinite(low):
            value = np.float32(low)
            if float(value) <= low:
                value = np.nextafter(value, np.float32(np.inf))
        else:
            value = np.float32(0.0)
        fits = low < float(value) <= high and np.isfinite(value)
        x[j] = float(value) if fits else np.nan
    return x


def literal_importance(tree, node_id: int) -> float:
    """
    Weighted impurity decrease of a split, taken from the fitted tree alone.
    
    Args:
        tree: A fitted sklearn decision tree
        node_id: Internal node ID
        
    Returns:
        Impurity decrease weighted by the training samples in the node
    """
    tree_ = tree.tree_
    left, right = tree_.children_left[node_id], tree_.children_right[node_id]
    weights = tree_.weighted_n_node_samples
    return float(
        weights[node_id] * tree_.impurity[node_id]
        - weights[left] * tree_.impurity[left]
        - weights[right] * tree_.impurity[right]
    )


def get_leaf_samples(
    tree,
    X: np.ndarray,
    leaf_id: int
) -> np.ndarray:
    """
    Get indices of samples that fall into a specific leaf.
    
    Args:
        tree: A fitted sklearn decision tree
        X: Feature matrix
        leaf_id: The leaf node ID
        
    Returns:
        Array of sample indices
    """
    leaf_ids = tree.apply(X)
    return np.where(leaf_ids == leaf_id)[0]


def get_leaf_id(tree, x: np.ndarray) -> int:
    """
    Get the leaf node ID for an instance.
    
    Args:
        tree: A fitted sklearn decision tree
        x: Feature vector (1D array)
        
    Returns:
        Leaf node ID
    """
    return tree.apply(x.reshape(1, -1))[0]


def get_tree_depth(tree) -> int:
    """
    Get the maximum depth of a decision tree.
    
    Args:
        tree: A fitted sklearn decision tree
        
    Returns:
        Maximum depth
    """
    return tree.tree_.max_depth


def get_n_leaves(tree) -> int:
    """
    Get the number of leaves in a decision tree.
    
    Args:
        tree: A fitted sklearn decision tree
        
    Returns:
        Number of leaves
    """
    return tree.tree_.n_leaves


def get_feature_importance_path(
    tree,
    x: np.ndarray
) -> Dict[int, float]:
    """
    Calculate feature importance along the decision path.
    
    This measures how much each feature contributed to reaching
    the final prediction.
    
    Args:
        tree: A fitted sklearn decision tree
        x: Feature vector (1D array)
        
    Returns:
        Dictionary mapping feature index to importance score
    """
    tree_ = tree.tree_
    
    # Get the path through the tree
    node_indices = _path_nodes(tree, x)
    
    importance = {}
    
    for i, node_id in enumerate(node_indices[:-1]):
        feature_idx = tree_.feature[node_id]
        
        if feature_idx < 0:  # Leaf node
            continue
        
        # Calculate importance based on impurity decrease
        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]
        
        n_samples = tree_.n_node_samples[node_id]
        n_left = tree_.n_node_samples[left_child]
        n_right = tree_.n_node_samples[right_child]
        
        impurity_decrease = (
            tree_.impurity[node_id] -
            (n_left / n_samples) * tree_.impurity[left_child] -
            (n_right / n_samples) * tree_.impurity[right_child]
        )
        
        if feature_idx in importance:
            importance[feature_idx] += impurity_decrease
        else:
            importance[feature_idx] = impurity_decrease
    
    return importance


def simplify_conditions(conditions: List[Condition]) -> List[Condition]:
    """
    Simplify a list of conditions by merging redundant ones.
    
    For example:
    - "x > 5" AND "x > 3" → "x > 5"
    - "x <= 10" AND "x <= 8" → "x <= 8"
    
    A merged condition matches a missing value only if all its sources do, so the
    conjunction selects the same rows as before.
    
    Args:
        conditions: List of conditions to simplify
        
    Returns:
        Simplified list of conditions
    """
    # Rounding to float32 changes what a threshold selects, so only like merges with like
    groups: Dict[Tuple[int, bool], List[Condition]] = {}
    for cond in conditions:
        groups.setdefault((cond.feature_index, cond.float32_input), []).append(cond)
    
    def tightest(sources: List[Condition], key) -> Condition:
        best = min(sources, key=key)
        missing = all(c.matches_missing for c in sources)
        return best if missing == best.matches_missing else replace(best, matches_missing=missing)
    
    simplified = []
    for conds in groups.values():
        lower_bounds = [c for c in conds if c.operator in ('>', '>=')]
        upper_bounds = [c for c in conds if c.operator in ('<', '<=')]
        # At the same threshold the strict bound selects fewer rows
        if lower_bounds:
            simplified.append(tightest(lower_bounds, lambda c: (-c.threshold, c.operator != '>')))
        if upper_bounds:
            simplified.append(tightest(upper_bounds, lambda c: (c.threshold, c.operator != '<')))
        simplified.extend(c for c in conds if c.operator not in ('>', '>=', '<', '<='))
    
    return simplified


def extract_rules_from_tree(
    tree,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None
) -> List[Rule]:
    """
    Extract all rules from a decision tree.
    
    Each rule corresponds to a path from root to leaf.
    
    Args:
        tree: A fitted sklearn decision tree
        feature_names: Optional list of feature names
        class_names: Optional list of class names
        
    Returns:
        List of Rule objects
    """
    tree_ = tree.tree_
    n_features = tree_.n_features
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    rules = []
    
    def recurse(node_id: int, conditions: List[Condition]):
        # Check if leaf
        if tree_.children_left[node_id] == tree_.children_right[node_id]:
            # Leaf node - create rule
            if hasattr(tree, 'classes_'):
                class_counts = tree_.value[node_id][0]
                prediction = tree.classes_[np.argmax(class_counts)]
                confidence = class_counts[np.argmax(class_counts)] / class_counts.sum()
            else:
                prediction = tree_.value[node_id][0][0]
                confidence = None
            
            rule = Rule(
                conditions=list(conditions),  # Copy conditions
                prediction=prediction,
                confidence=confidence
            )
            rules.append(rule)
            return
        
        feature_idx = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        
        # Left child (<=)
        left_condition = Condition(
            feature_index=int(feature_idx),
            feature_name=feature_names[feature_idx],
            operator='<=',
            threshold=float(threshold),
            float32_input=True,
            matches_missing=_matches_missing(tree_, node_id, True)
        )
        conditions.append(left_condition)
        recurse(tree_.children_left[node_id], conditions)
        conditions.pop()
        
        # Right child (>)
        right_condition = Condition(
            feature_index=int(feature_idx),
            feature_name=feature_names[feature_idx],
            operator='>',
            threshold=float(threshold),
            float32_input=True,
            matches_missing=_matches_missing(tree_, node_id, False)
        )
        conditions.append(right_condition)
        recurse(tree_.children_right[node_id], conditions)
        conditions.pop()
    
    recurse(0, [])
    return rules


def is_tree_model(model) -> bool:
    """
    Check if a model is a tree-based model.
    
    Args:
        model: A fitted model
        
    Returns:
        True if the model is tree-based
    """
    tree_types = (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor
    )
    
    # Check for XGBoost
    try:
        import xgboost as xgb
        tree_types = tree_types + (xgb.XGBClassifier, xgb.XGBRegressor)
    except ImportError:
        pass
    
    # Check for LightGBM
    try:
        import lightgbm as lgb
        tree_types = tree_types + (lgb.LGBMClassifier, lgb.LGBMRegressor)
    except ImportError:
        pass
    
    return isinstance(model, tree_types)


def is_single_tree(model) -> bool:
    """
    Check if a model is a single decision tree.
    
    Args:
        model: A fitted model
        
    Returns:
        True if the model is a single tree
    """
    return isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor))


def is_ensemble_tree(model) -> bool:
    """
    Check if a model is an ensemble of trees.
    
    Args:
        model: A fitted model
        
    Returns:
        True if the model is an ensemble of trees
    """
    ensemble_types = (
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor
    )
    
    try:
        import xgboost as xgb
        ensemble_types = ensemble_types + (xgb.XGBClassifier, xgb.XGBRegressor)
    except ImportError:
        pass
    
    try:
        import lightgbm as lgb
        ensemble_types = ensemble_types + (lgb.LGBMClassifier, lgb.LGBMRegressor)
    except ImportError:
        pass
    
    return isinstance(model, ensemble_types)

