"""
Unit Tests for GPE Variants.

Author: Vladyslav Dehtiarov
"""

import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, '..')
from gpe import GPEExplainer, GPEOptimal, GPEWeighted, GPEMonotonic, GPEEnsemble
from gpe.explanation import Explanation


class TestGPEOptimal:
    """Tests for GPEOptimal variant."""
    
    @pytest.fixture
    def setup(self):
        """Create test data and model."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
        
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X, y)
        
        return X, y, model
    
    def test_optimal_returns_explanation(self, setup):
        """Test that GPEOptimal returns valid explanations."""
        X, y, model = setup
        
        explainer = GPEOptimal(
            model=model,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X
        )
        
        explanation = explainer.explain(X[0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.method == "GPE"  # Base class method name
    
    def test_optimal_finds_minimal(self, setup):
        """Test that GPEOptimal finds minimal rules."""
        X, y, model = setup
        
        greedy_explainer = GPEExplainer(
            model=model,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X,
            min_precision=0.9
        )
        
        optimal_explainer = GPEOptimal(
            model=model,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X,
            min_precision=0.9
        )
        
        # Optimal should find rules at least as simple as greedy
        for x in X[:10]:
            greedy_exp = greedy_explainer.explain(x)
            optimal_exp = optimal_explainer.explain(x)
            
            assert optimal_exp.complexity <= greedy_exp.complexity


class TestGPEWeighted:
    """Tests for GPEWeighted variant."""
    
    @pytest.fixture
    def setup(self):
        """Create test data and model."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
        
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X, y)
        
        return X, y, model
    
    def test_weighted_respects_weights(self, setup):
        """Test that high-weight features are more likely to be kept."""
        X, y, model = setup
        
        # Give feature 'a' very high weight
        feature_weights = {'a': 1.0, 'b': 0.1, 'c': 0.1, 'd': 0.1}
        
        explainer = GPEWeighted(
            model=model,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X,
            feature_weights=feature_weights,
            min_precision=0.8
        )
        
        # Count how often 'a' appears in explanations
        a_count = 0
        total = 0
        
        for x in X[:20]:
            explanation = explainer.explain(x)
            features_used = explanation.rule.feature_names_used
            
            if 'a' in features_used:
                a_count += 1
            total += 1
        
        # Feature 'a' should appear frequently
        # (not a strict assertion due to stochastic nature)
        assert a_count >= total * 0.3  # At least 30% of the time
    
    def test_set_feature_weights(self, setup):
        """Test updating feature weights."""
        X, y, model = setup
        
        explainer = GPEWeighted(
            model=model,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X
        )
        
        # Update weights
        new_weights = {'a': 0.5, 'b': 1.0}
        explainer.set_feature_weights(new_weights)
        
        assert explainer.feature_weights['a'] == 0.5
        assert explainer.feature_weights['b'] == 1.0


class TestGPEMonotonic:
    """Tests for GPEMonotonic variant."""
    
    @pytest.fixture
    def setup(self):
        """Create test data and model."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        # Feature 0: positive relationship, Feature 1: negative relationship
        y = ((X[:, 0] > 0) & (X[:, 1] < 0)).astype(int)
        
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X, y)
        
        return X, y, model
    
    def test_monotonic_constraints(self, setup):
        """Test that monotonic constraints are respected."""
        X, y, model = setup
        
        # Feature 0 should be increasing (+1), Feature 1 decreasing (-1)
        constraints = {'a': +1, 'b': -1, 'c': 0, 'd': 0}
        
        explainer = GPEMonotonic(
            model=model,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X,
            monotonic_constraints=constraints
        )
        
        for x in X[:10]:
            explanation = explainer.explain(x)
            
            for cond in explanation.rule.conditions:
                constraint = constraints.get(cond.feature_name, 0)
                
                if constraint == +1:
                    # Should only have '>' or '>=' conditions
                    assert cond.operator in ['>', '>='], \
                        f"Feature {cond.feature_name} with +1 constraint has {cond.operator}"
                elif constraint == -1:
                    # Should only have '<' or '<=' conditions
                    assert cond.operator in ['<', '<='], \
                        f"Feature {cond.feature_name} with -1 constraint has {cond.operator}"


class TestGPEEnsemble:
    """Tests for GPEEnsemble variant."""
    
    @pytest.fixture
    def setup(self):
        """Create test data and ensemble model."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
        model.fit(X, y)
        
        return X, y, model
    
    def test_ensemble_returns_explanation(self, setup):
        """Test that GPEEnsemble returns valid explanations."""
        X, y, model = setup
        
        explainer = GPEEnsemble(
            model=model,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X
        )
        
        explanation = explainer.explain(X[0])
        
        assert isinstance(explanation, Explanation)
        assert "Ensemble" in explanation.method
    
    def test_ensemble_aggregation_methods(self, setup):
        """Test different aggregation methods."""
        X, y, model = setup
        
        for aggregation in ['majority', 'intersection', 'union']:
            explainer = GPEEnsemble(
                model=model,
                feature_names=['a', 'b', 'c', 'd'],
                X_train=X,
                aggregation=aggregation
            )
            
            explanation = explainer.explain(X[0])
            
            assert isinstance(explanation, Explanation)
            assert aggregation in explanation.method
    
    def test_ensemble_invalid_model(self):
        """Test that single trees raise error."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)
        
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        
        with pytest.raises(ValueError):
            GPEEnsemble(model=model, X_train=X)


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_tests()

