"""
GPE Experiments Module.

Contains benchmarking tools and dataset loaders for evaluating
GPE against other explanation methods.
"""

from .benchmark import BenchmarkRunner, BenchmarkConfig, run_quick_benchmark

__all__ = ['BenchmarkRunner', 'BenchmarkConfig', 'run_quick_benchmark']

