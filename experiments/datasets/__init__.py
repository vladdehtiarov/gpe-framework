"""
Dataset Loaders for GPE Benchmarks.

This module provides functions to load various datasets for
benchmarking explanation methods.

Datasets included:
- UCI datasets (Iris, Wine, Breast Cancer, Adult, German Credit)
- Synthetic datasets for controlled experiments
- Financial datasets (Credit Default)

Author: Vladyslav Dehtiarov
"""

from .loaders import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_adult,
    load_german_credit,
    load_credit_default,
    load_synthetic_simple,
    load_synthetic_complex,
    get_dataset,
    list_datasets
)

__all__ = [
    'load_iris',
    'load_wine',
    'load_breast_cancer',
    'load_adult',
    'load_german_credit',
    'load_credit_default',
    'load_synthetic_simple',
    'load_synthetic_complex',
    'get_dataset',
    'list_datasets'
]

