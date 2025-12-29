"""
Unit Tests for GPE Metrics.

Author: Vladyslav Dehtiarov
"""

import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.insert(0, '..')
from gpe import GPEExplainer
from gpe.metrics import (
    precision_score,
    coverage_score,
    complexity_score,
    fidelity_score,
    stability_score,
    evaluate_explanation,
    compare_explanations,
    aggregate_metrics
)
from gpe.explanation import Condition, Rule, Explanation


class TestPrecisionScore:
    """Tests for precision_score function."""
    
    @pytest.fixture
    def setup(self):
        """Create test data and model."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)
        
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        
        return X, y, model
    
    def test_perfect_precision(self, setup):
        """Test precision with perfect rule."""
        X, y, model = setup
        
        # Rule that perfectly matches the decision boundary
        rule = Rule(conditions=[Condition(0, "x", ">", 0)], prediction=1)
        explanation = Explanation(
            instance=X[0],
            rule=rule,
            prediction=1
        )
        
        precision = precision_score(explanation, model, X)
        
        # Should be very high (close to 1) since rule matches decision boundary
        assert precision >= 0.9
    
    def test_empty_coverage_precision(self, setup):
        """Test precision when no instances match."""
        X, y, model = setup
        
        # Impossible rule (no instances match)
        rule = Rule(conditions=[
            Condition(0, "x", ">", 1000),  # Very high threshold
        ], prediction=1)
        
        explanation = Explanation(
            instance=X[0],
            rule=rule,
            prediction=1
        )
        
        precision = precision_score(explanation, model, X)
        
        # Should be 1.0 when no instances match
        assert precision == 1.0


class TestCoverageScore:
    """Tests for coverage_score function."""
    
    def test_full_coverage(self):
        """Test coverage with empty rule (matches all)."""
        X = np.random.randn(100, 4)
        
        rule = Rule(conditions=[])  # Empty rule matches all
        explanation = Explanation(
            instance=X[0],
            rule=rule,
            prediction=1
        )
        
        coverage = coverage_score(explanation, X)
        
        assert coverage == 1.0
    
    def test_partial_coverage(self):
        """Test partial coverage."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        
        # Rule that matches ~50% of instances
        rule = Rule(conditions=[Condition(0, "x", ">", 0)])
        explanation = Explanation(
            instance=X[0],
            rule=rule,
            prediction=1
        )
        
        coverage = coverage_score(explanation, X)
        
        # Should be around 0.5 for standard normal data
        assert 0.3 <= coverage <= 0.7


class TestComplexityScore:
    """Tests for complexity_score function."""
    
    def test_complexity_count(self):
        """Test that complexity equals number of conditions."""
        rule = Rule(conditions=[
            Condition(0, "x", ">", 0),
            Condition(1, "y", "<=", 5),
            Condition(2, "z", ">", -1)
        ])
        
        explanation = Explanation(
            instance=np.zeros(3),
            rule=rule,
            prediction=1
        )
        
        complexity = complexity_score(explanation)
        
        assert complexity == 3
    
    def test_normalized_complexity(self):
        """Test normalized complexity."""
        rule = Rule(conditions=[
            Condition(0, "x", ">", 0),
            Condition(1, "y", "<=", 5)
        ])
        
        explanation = Explanation(
            instance=np.zeros(2),
            rule=rule,
            prediction=1
        )
        
        # With max_complexity=10, 2 conditions = 1 - 2/10 = 0.8
        normalized = complexity_score(explanation, normalize=True, max_complexity=10)
        
        assert normalized == 0.8


class TestFidelityScore:
    """Tests for fidelity_score function."""
    
    @pytest.fixture
    def setup(self):
        """Create test data and model."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)
        
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        
        return X, y, model
    
    def test_fidelity_reasonable_range(self, setup):
        """Test that fidelity is in valid range."""
        X, y, model = setup
        
        explainer = GPEExplainer(model=model, X_train=X)
        explanation = explainer.explain(X[0])
        
        fidelity = fidelity_score(explanation, model, X, random_state=42)
        
        assert 0 <= fidelity <= 1


class TestStabilityScore:
    """Tests for stability_score function."""
    
    @pytest.fixture
    def setup(self):
        """Create test data and model."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)
        
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        
        return X, y, model
    
    def test_stability_reasonable_range(self, setup):
        """Test that stability is in valid range."""
        X, y, model = setup
        
        explainer = GPEExplainer(model=model, X_train=X)
        x = X[0]
        
        stability = stability_score(explainer, x, X, random_state=42)
        
        assert 0 <= stability <= 1


class TestAggregateMetrics:
    """Tests for aggregate_metrics function."""
    
    @pytest.fixture
    def explanations(self):
        """Create test explanations."""
        explanations = []
        
        for i in range(10):
            rule = Rule(
                conditions=[Condition(0, "x", ">", 0)] * (i % 3 + 1),
                prediction=1
            )
            exp = Explanation(
                instance=np.random.randn(4),
                rule=rule,
                prediction=1,
                precision=0.8 + i * 0.01,
                coverage=0.1 + i * 0.02,
                computation_time=0.001 * (i + 1)
            )
            explanations.append(exp)
        
        return explanations
    
    def test_aggregate_returns_all_metrics(self, explanations):
        """Test that aggregation returns all expected metrics."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)
        
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        
        aggregated = aggregate_metrics(explanations, model, X)
        
        expected_metrics = ['precision', 'coverage', 'complexity', 'computation_time']
        for metric in expected_metrics:
            assert metric in aggregated
            assert 'mean' in aggregated[metric]
            assert 'std' in aggregated[metric]
            assert 'min' in aggregated[metric]
            assert 'max' in aggregated[metric]


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_tests()

