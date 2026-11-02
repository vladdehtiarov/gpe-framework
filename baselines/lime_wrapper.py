"""
LIME Wrapper for GPE Benchmark.

Provides a unified interface for LIME explanations that matches
the GPE Explanation format.

Author: Vladyslav Dehtiarov
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any

# Import GPE structures
import sys
sys.path.insert(0, '..')
from gpe.explanation import Condition, Rule, Explanation


class LIMEWrapper:
    """
    Wrapper for LIME (Local Interpretable Model-agnostic Explanations).
    
    Provides a unified interface that converts LIME explanations to
    the GPE Explanation format for comparison.
    
    Parameters:
        model: The model to explain
        X_train: Training data for LIME
        feature_names: Feature names
        mode: 'classification' or 'regression'
        n_features: Number of features to include in explanation
        
    Example:
        >>> wrapper = LIMEWrapper(model, X_train, feature_names)
        >>> explanation = wrapper.explain(x)
    """
    
    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        mode: str = 'classification',
        n_features: int = 10,
        kernel_width: Optional[float] = None
    ):
        self.model = model
        self.X_train = X_train
        self.mode = mode
        self.n_features = n_features
        self.kernel_width = kernel_width
        
        # Set feature names
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Initialize LIME
        self._init_lime()
    
    def _init_lime(self):
        """Initialize the LIME explainer."""
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            raise ImportError(
                "LIME is not installed. Install with: pip install lime"
            )
        
        if self.mode == 'classification':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                mode='classification',
                kernel_width=self.kernel_width
            )
        else:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                mode='regression',
                kernel_width=self.kernel_width
            )
    
    def explain(
        self,
        x: np.ndarray,
        **kwargs
    ) -> Explanation:
        """
        Generate an explanation using LIME.
        
        Args:
            x: Instance to explain
            **kwargs: Additional arguments passed to LIME
            
        Returns:
            Explanation object in GPE format
        """
        start_time = time.time()
        
        x = np.asarray(x).flatten()
        
        # Get LIME explanation
        if self.mode == 'classification':
            lime_exp = self.explainer.explain_instance(
                x,
                self.model.predict_proba,
                num_features=self.n_features,
                **kwargs
            )
            prediction = self.model.predict(x.reshape(1, -1))[0]
        else:
            lime_exp = self.explainer.explain_instance(
                x,
                self.model.predict,
                num_features=self.n_features,
                **kwargs
            )
            prediction = self.model.predict(x.reshape(1, -1))[0]
        
        # Convert LIME explanation to GPE format
        conditions = self._conditions_from_exp(lime_exp, x)
        
        # Create rule
        rule = Rule(
            conditions=conditions,
            prediction=prediction
        )
        
        computation_time = time.time() - start_time
        
        # Calculate approximate precision/coverage
        precision, coverage = self._estimate_metrics(rule, prediction)
        
        return Explanation(
            instance=x,
            rule=rule,
            prediction=prediction,
            precision=precision,
            coverage=coverage,
            method="LIME",
            computation_time=computation_time,
            metadata={
                'lime_score': lime_exp.score if hasattr(lime_exp, 'score') else None,
                'n_features': self.n_features
            }
        )
    
    def _conditions_from_exp(self, lime_exp, x: np.ndarray) -> List[Condition]:
        """
        Turn LIME's selected features into conditions.
        
        Indices and bin edges come from the explainer. The strings of as_list()
        round the edges to two decimals and depend on feature names. A selected
        feature gives both edges of the quartile bin of x, which is the region
        LIME reports, so the rule is true on x. LIME selects n_features features
        whatever the data, so the length of this rule is a setting, not a result.
        """
        label = sorted(lime_exp.local_exp)[0] if self.mode == 'classification' else 0
        selected = [int(j) for j, _ in lime_exp.local_exp[label]]
        discretizer = self.explainer.discretizer
        bins = discretizer.discretize(x.reshape(1, -1))[0]
        conditions = []
        for j in selected:
            name = self.feature_names[j]
            if j not in discretizer.to_discretize:
                conditions.append(Condition(j, name, '==', float(x[j])))
                continue
            index = int(bins[j])
            edges = discretizer.mins[j], discretizer.maxs[j]
            if index > 0:
                conditions.append(Condition(j, name, '>', float(edges[0][index])))
            if index < len(edges[0]) - 1:
                conditions.append(Condition(j, name, '<=', float(edges[1][index])))
        return conditions
    
    def _estimate_metrics(
        self,
        rule: Rule,
        prediction: Any
    ) -> tuple:
        """Estimate precision and coverage from training data."""
        mask = rule.evaluate_batch(self.X_train)
        
        if mask.sum() == 0:
            return float('nan'), 0.0
        
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
                print(f"LIME: Explaining instance {i + 1}/{X.shape[0]}")
            
            explanation = self.explain(x)
            explanations.append(explanation)
        
        return explanations

