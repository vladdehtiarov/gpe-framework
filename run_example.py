#!/usr/bin/env python3
"""
GPE Framework - Quick Example Script

Run this to verify the installation works correctly.

Usage:
    python run_example.py
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Import GPE
from gpe import GPEExplainer, GPEOptimal
from gpe.visualization import explanation_to_text


def main():
    print("=" * 70)
    print("GPE Framework - Quick Example")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Train model
    print("\n2. Training Decision Tree...")
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"   Accuracy: {accuracy:.2%}")
    
    # Create explainer
    print("\n3. Creating GPE Explainer...")
    explainer = GPEExplainer(
        model=model,
        feature_names=feature_names,
        X_train=X_train,
        min_precision=0.95
    )
    
    # Explain instances
    print("\n4. Generating Explanations...")
    print("-" * 70)
    
    for i in range(3):
        instance = X_test[i]
        prediction = model.predict(instance.reshape(1, -1))[0]
        
        print(f"\nInstance {i+1}:")
        print(f"  Values: {dict(zip(feature_names, [f'{v:.2f}' for v in instance]))}")
        print(f"  Actual class: {class_names[y_test[i]]}")
        print(f"  Predicted class: {class_names[prediction]}")
        
        explanation = explainer.explain(instance)
        
        print(f"\n  GPE Explanation:")
        print(f"  Rule: {explanation.rule}")
        print(f"  Precision: {explanation.precision:.2%}")
        print(f"  Coverage: {explanation.coverage:.2%}")
        print(f"  Complexity: {explanation.complexity} conditions")
        print(f"  Time: {explanation.computation_time*1000:.2f} ms")
        
        if explanation.original_rule:
            print(f"  Reduction: {explanation.reduction_ratio:.1%} "
                  f"({explanation.original_rule.complexity} → {explanation.complexity})")
    
    # Batch statistics
    print("\n" + "-" * 70)
    print("\n5. Batch Explanation Statistics (all test instances)...")
    
    explanations = explainer.explain_batch(X_test)
    
    complexities = [e.complexity for e in explanations]
    precisions = [e.precision for e in explanations]
    times = [e.computation_time * 1000 for e in explanations]
    
    print(f"   Average Complexity: {np.mean(complexities):.2f} ± {np.std(complexities):.2f}")
    print(f"   Average Precision: {np.mean(precisions):.2%}")
    print(f"   Average Time: {np.mean(times):.2f} ms")
    print(f"   Total Time: {sum(times):.2f} ms")
    
    # Compare with GPE-Optimal
    print("\n6. Comparing GPE vs GPE-Optimal...")
    
    gpe_optimal = GPEOptimal(
        model=model,
        feature_names=feature_names,
        X_train=X_train,
        min_precision=0.95
    )
    
    exp_greedy = explainer.explain(X_test[0])
    exp_optimal = gpe_optimal.explain(X_test[0])
    
    print(f"   GPE-Greedy:  Complexity={exp_greedy.complexity}, Time={exp_greedy.computation_time*1000:.2f}ms")
    print(f"   GPE-Optimal: Complexity={exp_optimal.complexity}, Time={exp_optimal.computation_time*1000:.2f}ms")
    
    print("\n" + "=" * 70)
    print("✅ GPE Framework is working correctly!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

