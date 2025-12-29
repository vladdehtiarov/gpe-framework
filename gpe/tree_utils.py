"""
Utility functions for working with decision trees.

This module provides functions to:
- Extract decision paths from sklearn trees
- Convert tree nodes to conditions
- Navigate tree structures
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from .explanation import Condition, Rule


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
    node_indicator = tree.decision_path(x.reshape(1, -1))
    node_indices = node_indicator.indices
    
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
            feature_index=feature_idx,
            feature_name=feature_names[feature_idx],
            operator=operator,
            threshold=threshold,
            is_categorical=False
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
    node_indicator = tree.decision_path(x.reshape(1, -1))
    node_indices = node_indicator.indices
    
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
    
    Args:
        conditions: List of conditions to simplify
        
    Returns:
        Simplified list of conditions
    """
    # Group conditions by feature
    feature_conditions: Dict[int, List[Condition]] = {}
    for cond in conditions:
        if cond.feature_index not in feature_conditions:
            feature_conditions[cond.feature_index] = []
        feature_conditions[cond.feature_index].append(cond)
    
    simplified = []
    
    for feature_idx, conds in feature_conditions.items():
        # Find the tightest bounds
        lower_bounds = [c for c in conds if c.operator in ['>', '>=']]
        upper_bounds = [c for c in conds if c.operator in ['<', '<=']]
        equality = [c for c in conds if c.operator in ['==', '!=']]
        
        # Keep tightest lower bound
        if lower_bounds:
            if any(c.operator == '>' for c in lower_bounds):
                # All '>' conditions, keep highest threshold
                gt_conds = [c for c in lower_bounds if c.operator == '>']
                if gt_conds:
                    best = max(gt_conds, key=lambda c: c.threshold)
                    simplified.append(best)
                gte_conds = [c for c in lower_bounds if c.operator == '>=']
                if gte_conds:
                    best = max(gte_conds, key=lambda c: c.threshold)
                    # Only add if not dominated by a '>' condition
                    if not gt_conds or best.threshold > max(c.threshold for c in gt_conds):
                        simplified.append(best)
            else:
                best = max(lower_bounds, key=lambda c: c.threshold)
                simplified.append(best)
        
        # Keep tightest upper bound
        if upper_bounds:
            if any(c.operator == '<' for c in upper_bounds):
                lt_conds = [c for c in upper_bounds if c.operator == '<']
                if lt_conds:
                    best = min(lt_conds, key=lambda c: c.threshold)
                    simplified.append(best)
                lte_conds = [c for c in upper_bounds if c.operator == '<=']
                if lte_conds:
                    best = min(lte_conds, key=lambda c: c.threshold)
                    if not lt_conds or best.threshold < min(c.threshold for c in lt_conds):
                        simplified.append(best)
            else:
                best = min(upper_bounds, key=lambda c: c.threshold)
                simplified.append(best)
        
        # Keep equality conditions as-is
        simplified.extend(equality)
    
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
            feature_index=feature_idx,
            feature_name=feature_names[feature_idx],
            operator='<=',
            threshold=threshold
        )
        conditions.append(left_condition)
        recurse(tree_.children_left[node_id], conditions)
        conditions.pop()
        
        # Right child (>)
        right_condition = Condition(
            feature_index=feature_idx,
            feature_name=feature_names[feature_idx],
            operator='>',
            threshold=threshold
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

