"""
Unit Tests for GPE Core Algorithm.

Author: Vladyslav Dehtiarov
"""

import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, load_iris

import sys
sys.path.insert(0, '..')
from gpe import GPEExplainer
from gpe.explanation import Condition, Rule, Explanation


class TestCondition:
    """Tests for Condition class."""
    
    def test_condition_creation(self):
        """Test basic condition creation."""
        cond = Condition(
            feature_index=0,
            feature_name="age",
            operator="<=",
            threshold=30.0
        )
        
        assert cond.feature_index == 0
        assert cond.feature_name == "age"
        assert cond.operator == "<="
        assert cond.threshold == 30.0
    
    def test_condition_evaluate_leq(self):
        """Test <= operator evaluation."""
        cond = Condition(0, "x", "<=", 5.0)
        
        assert cond.evaluate(np.array([3.0, 0, 0])) == True
        assert cond.evaluate(np.array([5.0, 0, 0])) == True
        assert cond.evaluate(np.array([6.0, 0, 0])) == False
    
    def test_condition_evaluate_gt(self):
        """Test > operator evaluation."""
        cond = Condition(0, "x", ">", 5.0)
        
        assert cond.evaluate(np.array([3.0, 0, 0])) == False
        assert cond.evaluate(np.array([5.0, 0, 0])) == False
        assert cond.evaluate(np.array([6.0, 0, 0])) == True
    
    def test_condition_str(self):
        """Test string representation."""
        cond = Condition(0, "age", "<=", 30.5)
        assert "age" in str(cond)
        assert "<=" in str(cond)
    
    def test_condition_invalid_operator(self):
        """Test that invalid operators raise error."""
        with pytest.raises(ValueError):
            Condition(0, "x", "~=", 5.0)


class TestRule:
    """Tests for Rule class."""
    
    def test_rule_creation(self):
        """Test basic rule creation."""
        conditions = [
            Condition(0, "x", ">", 0),
            Condition(1, "y", "<=", 10)
        ]
        rule = Rule(conditions=conditions, prediction=1)
        
        assert len(rule.conditions) == 2
        assert rule.prediction == 1
    
    def test_rule_evaluate_single(self):
        """Test rule evaluation on single instance."""
        conditions = [
            Condition(0, "x", ">", 0),
            Condition(1, "y", "<=", 10)
        ]
        rule = Rule(conditions=conditions)
        
        # Both conditions satisfied
        assert rule.evaluate(np.array([5, 5])) == True
        
        # First condition not satisfied
        assert rule.evaluate(np.array([-1, 5])) == False
        
        # Second condition not satisfied
        assert rule.evaluate(np.array([5, 15])) == False
    
    def test_rule_evaluate_batch(self):
        """Test rule evaluation on multiple instances."""
        conditions = [
            Condition(0, "x", ">", 0),
        ]
        rule = Rule(conditions=conditions)
        
        X = np.array([
            [5, 0],
            [-1, 0],
            [3, 0],
            [-5, 0]
        ])
        
        mask = rule.evaluate_batch(X)
        expected = np.array([True, False, True, False])
        
        np.testing.assert_array_equal(mask, expected)
    
    def test_rule_complexity(self):
        """Test complexity calculation."""
        rule = Rule(conditions=[
            Condition(0, "x", ">", 0),
            Condition(1, "y", ">", 0),
            Condition(2, "z", ">", 0)
        ])
        
        assert rule.complexity == 3
    
    def test_empty_rule(self):
        """Test empty rule (always true)."""
        rule = Rule(conditions=[])
        
        assert rule.evaluate(np.array([1, 2, 3])) == True
        assert rule.complexity == 0


class TestGPEExplainer:
    """Tests for GPEExplainer class."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
        return X, y
    
    @pytest.fixture
    def trained_tree(self, simple_data):
        """Create a trained decision tree."""
        X, y = simple_data
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X, y)
        return model
    
    def test_explainer_creation(self, trained_tree, simple_data):
        """Test explainer initialization."""
        X, y = simple_data
        explainer = GPEExplainer(
            model=trained_tree,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X
        )
        
        assert explainer.model == trained_tree
        assert len(explainer.feature_names) == 4
    
    def test_explain_returns_explanation(self, trained_tree, simple_data):
        """Test that explain returns an Explanation object."""
        X, y = simple_data
        explainer = GPEExplainer(
            model=trained_tree,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X
        )
        
        explanation = explainer.explain(X[0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.prediction in [0, 1]
        assert 0 <= explanation.precision <= 1
        assert 0 <= explanation.coverage <= 1
        assert explanation.complexity >= 0
    
    def test_explain_prediction_matches_model(self, trained_tree, simple_data):
        """Test that explanation prediction matches model prediction."""
        X, y = simple_data
        explainer = GPEExplainer(
            model=trained_tree,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X
        )
        
        for x in X[:10]:
            explanation = explainer.explain(x)
            model_pred = trained_tree.predict(x.reshape(1, -1))[0]
            
            assert explanation.prediction == model_pred
    
    def test_pruning_reduces_complexity(self, trained_tree, simple_data):
        """Test that pruning reduces rule complexity."""
        X, y = simple_data
        explainer = GPEExplainer(
            model=trained_tree,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X,
            min_precision=0.9
        )
        
        explanation = explainer.explain(X[0])
        
        # Pruned rule should be smaller or equal to original
        if explanation.original_rule is not None:
            assert explanation.complexity <= explanation.original_rule.complexity
    
    def test_precision_threshold(self, trained_tree, simple_data):
        """Test that explanations meet precision threshold."""
        X, y = simple_data
        min_precision = 0.9
        
        explainer = GPEExplainer(
            model=trained_tree,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X,
            min_precision=min_precision
        )
        
        for x in X[:10]:
            explanation = explainer.explain(x)
            # Precision should be at least min_precision (with some tolerance)
            assert explanation.precision >= min_precision - 0.1
    
    def test_explain_batch(self, trained_tree, simple_data):
        """Test batch explanation."""
        X, y = simple_data
        explainer = GPEExplainer(
            model=trained_tree,
            feature_names=['a', 'b', 'c', 'd'],
            X_train=X
        )
        
        explanations = explainer.explain_batch(X[:5])
        
        assert len(explanations) == 5
        assert all(isinstance(e, Explanation) for e in explanations)
    
    def test_invalid_model_raises_error(self, simple_data):
        """Test that non-tree models raise error."""
        X, y = simple_data
        
        # Random Forest should raise error (use GPEEnsemble instead)
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X, y)
        
        with pytest.raises(ValueError):
            GPEExplainer(model=rf, X_train=X)


class TestExplanation:
    """Tests for Explanation class."""
    
    def test_explanation_to_dict(self):
        """Test serialization to dictionary."""
        rule = Rule(
            conditions=[Condition(0, "x", ">", 0)],
            prediction=1
        )
        
        explanation = Explanation(
            instance=np.array([1, 2, 3]),
            rule=rule,
            prediction=1,
            precision=0.95,
            coverage=0.1
        )
        
        d = explanation.to_dict()
        
        assert 'instance' in d
        assert 'rule' in d
        assert 'precision' in d
        assert 'coverage' in d
    
    def test_explanation_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            'instance': [1, 2, 3],
            'rule': {
                'conditions': [{
                    'feature_index': 0,
                    'feature_name': 'x',
                    'operator': '>',
                    'threshold': 0,
                    'is_categorical': False
                }],
                'prediction': 1,
                'confidence': 0.9
            },
            'prediction': 1,
            'precision': 0.95,
            'coverage': 0.1
        }
        
        explanation = Explanation.from_dict(d)
        
        assert explanation.prediction == 1
        assert explanation.precision == 0.95
        assert len(explanation.rule.conditions) == 1
    
    def test_natural_language(self):
        """Test natural language generation."""
        rule = Rule(
            conditions=[
                Condition(0, "age", ">", 30),
                Condition(1, "income", "<=", 50000)
            ],
            prediction="approved"
        )
        
        explanation = Explanation(
            instance=np.array([35, 45000]),
            rule=rule,
            prediction="approved",
            precision=0.95,
            coverage=0.2
        )
        
        text = explanation.to_natural_language()
        
        assert "age" in text
        assert "income" in text
        assert "approved" in text


class TestIntegration:
    """Integration tests using real datasets."""
    
    def test_iris_dataset(self):
        """Test on Iris dataset."""
        from sklearn.datasets import load_iris
        
        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names
        
        # Train tree
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X, y)
        
        # Create explainer
        explainer = GPEExplainer(
            model=model,
            feature_names=list(feature_names),
            X_train=X
        )
        
        # Explain all instances
        for x in X[:20]:
            explanation = explainer.explain(x)
            
            # Basic checks
            assert explanation.prediction in [0, 1, 2]
            assert 0 <= explanation.precision <= 1
            assert explanation.complexity >= 0
            
            # Prediction should match model
            assert explanation.prediction == model.predict(x.reshape(1, -1))[0]


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_tests()

