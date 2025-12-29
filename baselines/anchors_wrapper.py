"""
Anchors Wrapper for GPE Benchmark.

Provides a unified interface for Anchors explanations that matches
the GPE Explanation format.

Author: Vladyslav Dehtiarov
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any

import sys
sys.path.insert(0, '..')
from gpe.explanation import Condition, Rule, Explanation


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
            class_names=['0', '1'],  # Will be updated based on model
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
        
        # Get metrics from Anchors
        precision = anchor_exp.precision() if hasattr(anchor_exp, 'precision') else 1.0
        coverage = anchor_exp.coverage() if hasattr(anchor_exp, 'coverage') else 0.0
        
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
        Parse Anchors explanation into Conditions.
        
        Anchors format examples:
        - "feature_0 > 3.50"
        - "feature_1 <= 2.00"
        - "3.50 < feature_0 <= 5.00"
        """
        import re
        
        conditions = []
        
        if not hasattr(anchor_exp, 'names'):
            return conditions
        
        for rule_str in anchor_exp.names():
            # Try different patterns
            
            # Pattern: "value < feature <= value"
            match = re.match(r"([\d.-]+)\s*<\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*<=\s*([\d.-]+)", rule_str)
            if match:
                low = float(match.group(1))
                feature_name = match.group(2)
                high = float(match.group(3))
                
                try:
                    idx = self.feature_names.index(feature_name)
                except ValueError:
                    continue
                
                # Add both bounds
                conditions.append(Condition(
                    feature_index=idx,
                    feature_name=feature_name,
                    operator='>',
                    threshold=low
                ))
                conditions.append(Condition(
                    feature_index=idx,
                    feature_name=feature_name,
                    operator='<=',
                    threshold=high
                ))
                continue
            
            # Pattern: "feature <= value" or "feature > value"
            match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(<=|<|>=|>|=)\s*([\d.-]+)", rule_str)
            if match:
                feature_name = match.group(1)
                operator = match.group(2)
                threshold = float(match.group(3))
                
                try:
                    idx = self.feature_names.index(feature_name)
                except ValueError:
                    continue
                
                # Normalize operator
                if operator == '=':
                    operator = '=='
                
                conditions.append(Condition(
                    feature_index=idx,
                    feature_name=feature_name,
                    operator=operator,
                    threshold=threshold
                ))
                continue
            
            # Categorical pattern: "feature = category_name"
            match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)", rule_str)
            if match:
                feature_name = match.group(1)
                category = match.group(2)
                
                try:
                    idx = self.feature_names.index(feature_name)
                except ValueError:
                    continue
                
                conditions.append(Condition(
                    feature_index=idx,
                    feature_name=feature_name,
                    operator='==',
                    threshold=category,
                    is_categorical=True
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

