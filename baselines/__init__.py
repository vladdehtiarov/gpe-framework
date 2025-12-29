"""
Baseline Explainer Wrappers.

This module provides unified wrappers for comparing GPE with other
explanation methods:
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Anchors

Author: Vladyslav Dehtiarov
"""

from .lime_wrapper import LIMEWrapper
from .shap_wrapper import SHAPWrapper
from .anchors_wrapper import AnchorsWrapper

__all__ = ['LIMEWrapper', 'SHAPWrapper', 'AnchorsWrapper']

