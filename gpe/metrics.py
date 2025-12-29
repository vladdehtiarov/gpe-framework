"""
Evaluation Metrics for GPE Explanations.

This module provides functions to evaluate the quality of explanations:
- Precision: How accurate is the explanation?
- Coverage: How many instances does the explanation cover?
- Complexity: How simple is the explanation?
- Fidelity: How well does the explanation match the model?
- Stability: How consistent are explanations for similar instances?

Author: Vladyslav Dehtiarov
"""

import numpy as np
from typing import List, Optional, Any, Callable, Tuple
from scipy import stats

from .explanation import Explanation, Rule


def precision_score(
    explanation: Explanation,
    model,
    X: np.ndarray
) -> float:
    """
    Calculate the precision of an explanation.
    
    Precision = P(same prediction | rule satisfied)
    
    This measures the probability that an instance satisfying the
    explanation's conditions will have the same prediction as the
    explained instance.
    
    Args:
        explanation: The explanation to evaluate
        model: The model being explained
        X: Background data
        
    Returns:
        Precision score between 0 and 1
    """
    rule = explanation.rule
    prediction = explanation.prediction
    
    # Find instances that satisfy the rule
    mask = rule.evaluate_batch(X)
    
    if mask.sum() == 0:
        return 1.0  # Perfect precision if no instances match
    
    # Get predictions for matching instances
    X_matching = X[mask]
    y_pred = model.predict(X_matching)
    
    # Calculate precision
    return (y_pred == prediction).mean()


def coverage_score(
    explanation: Explanation,
    X: np.ndarray
) -> float:
    """
    Calculate the coverage of an explanation.
    
    Coverage = proportion of instances that satisfy the rule
    
    This measures how general the explanation is. Higher coverage
    means the explanation applies to more instances.
    
    Args:
        explanation: The explanation to evaluate
        X: Background data
        
    Returns:
        Coverage score between 0 and 1
    """
    mask = explanation.rule.evaluate_batch(X)
    return mask.mean()


def complexity_score(
    explanation: Explanation,
    normalize: bool = False,
    max_complexity: int = 10
) -> float:
    """
    Calculate the complexity of an explanation.
    
    Complexity = number of conditions in the rule
    
    Lower complexity means simpler, more interpretable explanations.
    
    Args:
        explanation: The explanation to evaluate
        normalize: If True, returns 1 - (complexity / max_complexity)
        max_complexity: Maximum expected complexity for normalization
        
    Returns:
        Complexity score (raw count or normalized 0-1)
    """
    complexity = explanation.complexity
    
    if normalize:
        return max(0, 1 - complexity / max_complexity)
    
    return float(complexity)


def fidelity_score(
    explanation: Explanation,
    model,
    X: np.ndarray,
    n_samples: int = 1000,
    radius: float = 0.1,
    random_state: Optional[int] = None
) -> float:
    """
    Calculate the fidelity (faithfulness) of an explanation.
    
    Fidelity measures how well the explanation's rule matches the
    model's behavior in the local neighborhood of the instance.
    
    We generate samples in a local neighborhood and check if the
    rule's predictions match the model's predictions.
    
    Args:
        explanation: The explanation to evaluate
        model: The model being explained
        X: Background data (used to estimate feature ranges)
        n_samples: Number of local samples to generate
        radius: Radius of local neighborhood (as fraction of feature range)
        random_state: Random seed for reproducibility
        
    Returns:
        Fidelity score between 0 and 1
    """
    rng = np.random.RandomState(random_state)
    
    # Get instance and feature ranges
    instance = explanation.instance
    feature_mins = X.min(axis=0)
    feature_maxs = X.max(axis=0)
    feature_ranges = feature_maxs - feature_mins
    
    # Generate local samples
    perturbations = rng.uniform(-radius, radius, (n_samples, len(instance)))
    local_samples = instance + perturbations * feature_ranges
    
    # Clip to valid range
    local_samples = np.clip(local_samples, feature_mins, feature_maxs)
    
    # Get model predictions
    model_predictions = model.predict(local_samples)
    
    # Get rule predictions
    rule_satisfied = explanation.rule.evaluate_batch(local_samples)
    
    # Rule predicts the explanation's prediction when satisfied,
    # and we consider it "correct" if both agree
    correct = 0
    for i in range(n_samples):
        if rule_satisfied[i]:
            if model_predictions[i] == explanation.prediction:
                correct += 1
        else:
            if model_predictions[i] != explanation.prediction:
                correct += 1
    
    return correct / n_samples


def stability_score(
    explainer,
    x: np.ndarray,
    X: np.ndarray,
    n_perturbations: int = 10,
    noise_level: float = 0.01,
    random_state: Optional[int] = None
) -> float:
    """
    Calculate the stability of explanations.
    
    Stability measures how consistent explanations are for similar
    instances. We perturb the input slightly and check if the
    explanation changes.
    
    Args:
        explainer: GPE explainer instance
        x: Instance to explain
        X: Background data (used to estimate feature ranges)
        n_perturbations: Number of perturbed instances to generate
        noise_level: Level of noise to add (as fraction of feature range)
        random_state: Random seed for reproducibility
        
    Returns:
        Stability score between 0 and 1 (1 = perfectly stable)
    """
    rng = np.random.RandomState(random_state)
    
    # Get feature ranges
    feature_ranges = X.max(axis=0) - X.min(axis=0)
    feature_ranges[feature_ranges == 0] = 1  # Avoid division by zero
    
    # Get original explanation
    original_exp = explainer.explain(x, X)
    original_features = set(original_exp.rule.feature_names_used)
    
    # Generate and explain perturbed instances
    stability_scores = []
    
    for _ in range(n_perturbations):
        # Add small noise
        noise = rng.uniform(-noise_level, noise_level, len(x))
        x_perturbed = x + noise * feature_ranges
        
        # Get explanation for perturbed instance
        perturbed_exp = explainer.explain(x_perturbed, X)
        perturbed_features = set(perturbed_exp.rule.feature_names_used)
        
        # Calculate Jaccard similarity
        if len(original_features) == 0 and len(perturbed_features) == 0:
            similarity = 1.0
        elif len(original_features) == 0 or len(perturbed_features) == 0:
            similarity = 0.0
        else:
            intersection = len(original_features & perturbed_features)
            union = len(original_features | perturbed_features)
            similarity = intersection / union
        
        stability_scores.append(similarity)
    
    return np.mean(stability_scores)


def evaluate_explanation(
    explanation: Explanation,
    model,
    X: np.ndarray,
    explainer=None,
    compute_fidelity: bool = True,
    compute_stability: bool = True,
    random_state: Optional[int] = None
) -> dict:
    """
    Comprehensive evaluation of an explanation.
    
    Computes all available metrics for the given explanation.
    
    Args:
        explanation: The explanation to evaluate
        model: The model being explained
        X: Background data
        explainer: GPE explainer (required for stability)
        compute_fidelity: Whether to compute fidelity (slower)
        compute_stability: Whether to compute stability (slower, requires explainer)
        random_state: Random seed
        
    Returns:
        Dictionary with all metric scores
    """
    metrics = {
        'precision': precision_score(explanation, model, X),
        'coverage': coverage_score(explanation, X),
        'complexity': complexity_score(explanation),
        'complexity_normalized': complexity_score(explanation, normalize=True),
    }
    
    if compute_fidelity:
        metrics['fidelity'] = fidelity_score(
            explanation, model, X, random_state=random_state
        )
    
    if compute_stability and explainer is not None:
        metrics['stability'] = stability_score(
            explainer, explanation.instance, X, random_state=random_state
        )
    
    return metrics


def compare_explanations(
    explanations1: List[Explanation],
    explanations2: List[Explanation],
    model,
    X: np.ndarray
) -> dict:
    """
    Compare two sets of explanations (e.g., from different methods).
    
    Performs statistical tests to determine if there are significant
    differences in explanation quality.
    
    Args:
        explanations1: First set of explanations
        explanations2: Second set of explanations
        model: The model being explained
        X: Background data
        
    Returns:
        Dictionary with comparison statistics and p-values
    """
    # Compute metrics for both sets
    metrics1 = {
        'precision': [precision_score(e, model, X) for e in explanations1],
        'coverage': [coverage_score(e, X) for e in explanations1],
        'complexity': [complexity_score(e) for e in explanations1],
    }
    
    metrics2 = {
        'precision': [precision_score(e, model, X) for e in explanations2],
        'coverage': [coverage_score(e, X) for e in explanations2],
        'complexity': [complexity_score(e) for e in explanations2],
    }
    
    # Statistical comparison
    comparison = {}
    
    for metric_name in metrics1.keys():
        vals1 = metrics1[metric_name]
        vals2 = metrics2[metric_name]
        
        # Paired t-test (assumes same instances)
        t_stat, p_value = stats.ttest_rel(vals1, vals2)
        
        # Effect size (Cohen's d)
        diff = np.array(vals1) - np.array(vals2)
        d = diff.mean() / diff.std() if diff.std() > 0 else 0
        
        comparison[metric_name] = {
            'mean_1': np.mean(vals1),
            'std_1': np.std(vals1),
            'mean_2': np.mean(vals2),
            'std_2': np.std(vals2),
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': d,
            'significant': p_value < 0.05
        }
    
    return comparison


def aggregate_metrics(
    explanations: List[Explanation],
    model,
    X: np.ndarray
) -> dict:
    """
    Compute aggregate metrics across multiple explanations.
    
    Args:
        explanations: List of explanations
        model: The model being explained
        X: Background data
        
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    metrics = {
        'precision': [precision_score(e, model, X) for e in explanations],
        'coverage': [coverage_score(e, X) for e in explanations],
        'complexity': [complexity_score(e) for e in explanations],
        'computation_time': [e.computation_time for e in explanations],
    }
    
    aggregated = {}
    
    for metric_name, values in metrics.items():
        aggregated[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    return aggregated

