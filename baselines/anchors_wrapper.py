"""
Anchors Wrapper for GPE Benchmark.

Provides a unified interface for Anchors explanations that matches
the GPE Explanation format.

Author: Vladyslav Dehtiarov
"""

import re
import numpy as np
import time
from typing import List, Optional, Dict, Any

import sys
sys.path.insert(0, '..')
from gpe.explanation import Condition, Rule, Explanation


_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _snap(printed: float, edges: np.ndarray, value: float, operator: str) -> float:
    """Closest quartile edge to a rounded bound, among those keeping x in the region."""
    if edges.size == 0:
        return printed
    if operator in ('>', '>='):
        fitting = edges[edges < value]
    else:
        fitting = edges[edges >= value]
    if fitting.size == 0:
        return printed
    return float(fitting[np.argmin(np.abs(fitting - printed))])


class AnchorsWrapper:
    """
    Wrapper for Anchors explanations.
    
    Anchors generate rule-based explanations that "anchor" the prediction,
    providing sufficient conditions for the prediction to hold.
    
    Parameters:
        model: The model to explain
        X_train: Training data
        feature_names: Feature names
        categorical_names: Dict mapping categorical feature indices to value names
        threshold: Precision threshold for anchors (default: 0.95)
        
    Example:
        >>> wrapper = AnchorsWrapper(model, X_train, feature_names)
        >>> explanation = wrapper.explain(x)
    """
    
    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        categorical_names: Optional[Dict[int, List[str]]] = None,
        threshold: float = 0.95
    ):
        self.model = model
        self.X_train = X_train
        self.threshold = threshold
        self.categorical_names = categorical_names or {}
        
        # Set feature names
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Initialize Anchors
        self._init_anchors()
    
    def _init_anchors(self):
        """Initialize the Anchors explainer."""
        try:
            from anchor import anchor_tabular
        except ImportError:
            raise ImportError(
                "anchor-exp is not installed. Install with: pip install anchor-exp"
            )
        
        self.explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=[str(c) for c in getattr(self.model, 'classes_', ['0', '1'])],
            feature_names=self.feature_names,
            train_data=self.X_train,
            categorical_names=self.categorical_names
        )
    
    def explain(
        self,
        x: np.ndarray,
        **kwargs
    ) -> Explanation:
        """
        Generate an explanation using Anchors.
        
        Args:
            x: Instance to explain
            **kwargs: Additional arguments for Anchors
            
        Returns:
            Explanation object in GPE format
        """
        start_time = time.time()
        
        x = np.asarray(x).flatten()
        
        # Get Anchors explanation
        anchor_exp = self.explainer.explain_instance(
            x,
            self.model.predict,
            threshold=self.threshold,
            **kwargs
        )
        
        prediction = self.model.predict(x.reshape(1, -1))[0]
        
        # Convert Anchors to GPE format
        conditions = self._parse_anchor_rules(anchor_exp, x)
        
        # Create rule
        rule = Rule(
            conditions=conditions,
            prediction=prediction,
            confidence=anchor_exp.precision() if hasattr(anchor_exp, 'precision') else None
        )
        
        computation_time = time.time() - start_time
        
        # Metrics as Anchors reports them, on its own perturbation distribution.
        # Comparisons must recompute them on data: see anchor issue #82.
        precision = anchor_exp.precision() if hasattr(anchor_exp, 'precision') else float('nan')
        coverage = anchor_exp.coverage() if hasattr(anchor_exp, 'coverage') else float('nan')
        
        return Explanation(
            instance=x,
            rule=rule,
            prediction=prediction,
            precision=precision,
            coverage=coverage,
            method="Anchors",
            computation_time=computation_time,
            metadata={
                'anchor_names': anchor_exp.names() if hasattr(anchor_exp, 'names') else [],
                'threshold': self.threshold
            }
        )
    
    def _parse_anchor_rules(
        self,
        anchor_exp,
        x: np.ndarray
    ) -> List[Condition]:
        """
        Turn an anchor into conditions.
        
        Feature indices come from anchor_exp.features(). Only the strings hold the
        bounds, rounded to two decimals, so each bound is snapped to the quartile edge
        of the discretizer it was printed from. The feature name is cut out of the
        string first, so it may contain any characters.
        """
        conditions = []
        for j, text in zip(anchor_exp.features(), anchor_exp.names()):
            j = int(j)
            name = self.feature_names[j]
            before, _, after = text.partition(name)
            if j in self.explainer.categorical_names and j not in self.explainer.ordinal_features:
                conditions.append(Condition(
                    j, name, '==', float(x[j]), is_categorical=False
                ))
                continue
            
            edges = np.asarray(self.explainer.disc.mins[j][1:], dtype=float)
            low = re.search(rf"({_NUMBER})\s*<\s*$", before)
            if low:
                conditions.append(Condition(j, name, '>', _snap(float(low.group(1)), edges, x[j], '>')))
            high = re.search(rf"^\s*(<=|>|<|>=)\s*({_NUMBER})", after)
            if high:
                operator = high.group(1)
                conditions.append(Condition(
                    j, name, operator, _snap(float(high.group(2)), edges, x[j], operator)
                ))
        return conditions
    
    def explain_batch(
        self,
        X: np.ndarray,
        verbose: bool = False
    ) -> List[Explanation]:
        """Generate explanations for multiple instances."""
        explanations = []
        
        for i, x in enumerate(X):
            if verbose and (i + 1) % 10 == 0:
                print(f"Anchors: Explaining instance {i + 1}/{X.shape[0]}")
            
            try:
                explanation = self.explain(x)
                explanations.append(explanation)
            except Exception as e:
                # Anchors can fail for some instances
                print(f"Warning: Anchors failed for instance {i}: {e}")
                # Create empty explanation
                explanations.append(Explanation(
                    instance=x,
                    rule=Rule(conditions=[], prediction=self.model.predict(x.reshape(1, -1))[0]),
                    prediction=self.model.predict(x.reshape(1, -1))[0],
                    method="Anchors-failed"
                ))
        
        return explanations

