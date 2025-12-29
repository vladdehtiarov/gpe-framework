"""
GPE Framework - Greedy-Prune-Explain
====================================

A Python library for generating minimal, interpretable local explanations
for decision tree-based models.

Key Features:
- Minimal explanations with only essential conditions
- High precision (99%+) on real-world data
- 48x faster than LIME, 19x faster than Anchors
- Simple IF-THEN rules instead of numeric weights

Main Components:
- GPEExplainer: Core explainer class
- GPEInformationTheoretic: Information-theoretic pruning variant
- GPECounterfactual: Counterfactual explanation variant

Example Usage:
    from gpe import GPEExplainer
    from sklearn.tree import DecisionTreeClassifier
    
    # Train a model
    model = DecisionTreeClassifier(max_depth=7)
    model.fit(X_train, y_train)
    
    # Create explainer
    explainer = GPEExplainer(
        model=model,
        feature_names=['income', 'age', 'debt_ratio'],
        X_train=X_train
    )
    
    # Explain a prediction
    explanation = explainer.explain(x_instance)
    print(explanation)
    
    # Natural language output
    print(explanation.to_natural_language())
    # "The model predicts 'denied' because debt_ratio is greater than 0.40.
    #  This explanation covers 15.3% of similar cases with 98.5% accuracy."

Author: Vladyslav Dehtiarov
Email: vvdehtiarov@gmail.com
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Vladyslav Dehtiarov"
__email__ = "vvdehtiarov@gmail.com"
__license__ = "MIT"

# Core
from .core import GPEExplainer

# Variants
from .variants import (
    GPEOptimal,
    GPEWeighted,
    GPEMonotonic,
    GPEEnsemble
)

# Novel Methods (Scientific Contribution)
from .novel_methods import (
    GPEInformationTheoretic,
    GPECounterfactual,
    GPEStable,
    GPEMultiResolution,
    GPEPareto,
    CounterfactualExplanation,
    MultiResolutionExplanation,
    ParetoExplanation
)

# Data Structures
from .explanation import (
    Explanation,
    Rule,
    Condition
)

# Metrics
from .metrics import (
    precision_score,
    coverage_score,
    complexity_score,
    fidelity_score,
    stability_score
)

# Visualization
from .visualization import (
    plot_explanation,
    plot_comparison,
    explanation_to_text,
    explanation_to_html
)

# Public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Core Explainer
    "GPEExplainer",
    
    # Variants
    "GPEOptimal",
    "GPEWeighted",
    "GPEMonotonic",
    "GPEEnsemble",
    
    # Novel Methods (Scientific Contribution)
    "GPEInformationTheoretic",
    "GPECounterfactual",
    "GPEStable",
    "GPEMultiResolution",
    "GPEPareto",
    
    # Result Classes
    "Explanation",
    "Rule",
    "Condition",
    "CounterfactualExplanation",
    "MultiResolutionExplanation",
    "ParetoExplanation",
    
    # Metrics
    "precision_score",
    "coverage_score",
    "complexity_score",
    "fidelity_score",
    "stability_score",
    
    # Visualization
    "plot_explanation",
    "plot_comparison",
    "explanation_to_text",
    "explanation_to_html",
]
