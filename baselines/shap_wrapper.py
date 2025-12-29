"""
SHAP Wrapper for GPE Benchmark.

Provides a unified interface for SHAP explanations that matches
the GPE Explanation format.

Author: Vladyslav Dehtiarov
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any

import sys
sys.path.insert(0, '..')
from gpe.explanation import Condition, Rule, Explanation


class SHAPWrapper:
    """
    Wrapper for SHAP (SHapley Additive exPlanations).
    
    Provides a unified interface that converts SHAP explanations to
    the GPE Explanation format for comparison.
    
    Parameters:
        model: The model to explain
        X_train: Training/background data
        feature_names: Feature names
        explainer_type: 'tree', 'linear', 'kernel', or 'auto'
        n_features: Number of top features to include
        
    Example:
        >>> wrapper = SHAPWrapper(model, X_train, feature_names)
        >>> explanation = wrapper.explain(x)
    """
    
    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        explainer_type: str = 'auto',
        n_features: int = 10
    ):
        self.model = model
        self.X_train = X_train
        self.explainer_type = explainer_type
        self.n_features = n_features
        
        # Set feature names
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Initialize SHAP
        self._init_shap()
    
    def _init_shap(self):
        """Initialize the SHAP explainer."""
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            )
        
        if self.explainer_type == 'auto':
            # Auto-detect best explainer type
            model_type = type(self.model).__name__
            
            if 'Tree' in model_type or 'Forest' in model_type or 'Gradient' in model_type:
                self.explainer_type = 'tree'
            elif 'Linear' in model_type or 'Logistic' in model_type:
                self.explainer_type = 'linear'
            else:
                self.explainer_type = 'kernel'
        
        if self.explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(
                self.model,
                self.X_train
            )
        elif self.explainer_type == 'kernel':
            # Sample background data for efficiency
            if len(self.X_train) > 100:
                background = shap.sample(self.X_train, 100)
            else:
                background = self.X_train
            
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') 
                else self.model.predict,
                background
            )
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")
    
    def explain(
        self,
        x: np.ndarray,
        **kwargs
    ) -> Explanation:
        """
        Generate an explanation using SHAP.
        
        Args:
            x: Instance to explain
            **kwargs: Additional arguments
            
        Returns:
            Explanation object in GPE format
        """
        start_time = time.time()
        
        x = np.asarray(x).flatten()
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(x.reshape(1, -1))
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-class classification
            prediction = self.model.predict(x.reshape(1, -1))[0]
            if hasattr(self.model, 'classes_'):
                pred_idx = np.where(self.model.classes_ == prediction)[0][0]
            else:
                pred_idx = int(prediction)
            shap_values = shap_values[pred_idx].flatten()
        else:
            shap_values = shap_values.flatten()
            prediction = self.model.predict(x.reshape(1, -1))[0]
        
        # Convert SHAP values to conditions
        # Select top n_features by absolute SHAP value
        n_features = min(self.n_features, len(x), len(shap_values))
        top_indices = np.argsort(np.abs(shap_values))[-n_features:][::-1]
        
        conditions = []
        for idx in top_indices:
            if idx >= len(x):  # Skip if index out of bounds
                continue
            if np.abs(shap_values[idx]) < 1e-6:
                continue
            
            # Create condition based on SHAP value direction
            value = x[idx]
            
            # Determine threshold as a percentage offset from the value
            offset = 0.1 * (self.X_train[:, idx].max() - self.X_train[:, idx].min())
            
            if shap_values[idx] > 0:
                # Positive contribution - feature value being high is important
                operator = '>'
                threshold = value - offset
            else:
                # Negative contribution - feature value being low is important
                operator = '<='
                threshold = value + offset
            
            condition = Condition(
                feature_index=idx,
                feature_name=self.feature_names[idx],
                operator=operator,
                threshold=threshold
            )
            conditions.append(condition)
        
        # Create rule
        rule = Rule(
            conditions=conditions,
            prediction=prediction
        )
        
        computation_time = time.time() - start_time
        
        # Calculate metrics
        precision, coverage = self._estimate_metrics(rule, prediction)
        
        return Explanation(
            instance=x,
            rule=rule,
            prediction=prediction,
            precision=precision,
            coverage=coverage,
            method=f"SHAP-{self.explainer_type}",
            computation_time=computation_time,
            metadata={
                'shap_values': shap_values.tolist(),
                'explainer_type': self.explainer_type,
                'n_features': self.n_features
            }
        )
    
    def _estimate_metrics(
        self,
        rule: Rule,
        prediction: Any
    ) -> tuple:
        """Estimate precision and coverage from training data."""
        mask = rule.evaluate_batch(self.X_train)
        
        if mask.sum() == 0:
            return 1.0, 0.0
        
        coverage = mask.mean()
        
        y_pred = self.model.predict(self.X_train[mask])
        precision = (y_pred == prediction).mean()
        
        return precision, coverage
    
    def explain_batch(
        self,
        X: np.ndarray,
        verbose: bool = False
    ) -> List[Explanation]:
        """Generate explanations for multiple instances."""
        explanations = []
        
        for i, x in enumerate(X):
            if verbose and (i + 1) % 10 == 0:
                print(f"SHAP: Explaining instance {i + 1}/{X.shape[0]}")
            
            explanation = self.explain(x)
            explanations.append(explanation)
        
        return explanations
    
    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """
        Calculate global feature importance using SHAP.
        
        Args:
            X: Data to compute importance over
            
        Returns:
            Dictionary mapping feature names to importance
        """
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            # Average across classes
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance = {}
        for i, name in enumerate(self.feature_names):
            importance[name] = mean_abs_shap[i]
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance

