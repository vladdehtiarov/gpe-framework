
import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gpe.variants import GPEEnsemble

# Load simple dataset for sanity check
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train RF
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Explain
explainer = GPEEnsemble(rf, feature_names=feature_names, X_train=X_train[:100], aggregation='majority')

print("\n--- GPE Ensemble Explanation Sanity Check ---")
for i in range(3):
    x = X_test[i]
    exp = explainer.explain(x, X_background=X_train[:50]) # Small background for speed
    print(f"\nInstance {i}: Pred={exp.prediction}")
    print(f"Rule: {exp.rule}")
    print(f"Precision: {exp.precision:.3f}")
    print(f"Coverage: {exp.coverage:.3f}")
    print(f"Active Trees: {exp.metadata.get('active_trees')}")
