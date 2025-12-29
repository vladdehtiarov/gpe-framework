# GPE Framework - Greedy-Prune-Explain

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/gpe-framework.svg)](https://badge.fury.io/py/gpe-framework)

**GPE (Greedy-Prune-Explain)** is a novel method for generating minimal, interpretable local explanations for decision tree predictions. Unlike existing methods like LIME, SHAP, or Anchors, GPE leverages the inherent structure of decision trees to produce explanations that are:

- ✅ **Minimal** — Contains only essential conditions (1-2 instead of 5)
- ✅ **Precise** — 99.4% precision on real-world financial data
- ✅ **Fast** — 48x faster than LIME, 19x faster than Anchors
- ✅ **Actionable** — Simple IF-THEN rules like "income < 50000 AND debt_ratio > 0.4"

## 📊 Benchmark Results

Tested on 3 financial datasets (632K records total):

| Method | Time (ms) | Complexity | Precision | Speedup |
|--------|-----------|------------|-----------|---------|
| **GPE-Core** | **4.4** | **1.4** | **99.4%** | **48x** |
| GPE-IT | 3.0 | 1.4 | 97.9% | 71x |
| LIME | 213 | 5.0 | — | 1x |
| Anchors | 82 | 0.7 | 99.2% | 3x |

*All results are statistically significant (p < 0.001)*

## 🚀 Installation

```bash
pip install gpe-framework
```

Or install from source:

```bash
git clone https://github.com/vladdehtiarov/gpe-framework.git
cd gpe-framework
pip install -e .
```

## 📖 Quick Start

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from gpe import GPEExplainer

# Load data and train model
iris = load_iris()
X, y = iris.data, iris.target
model = DecisionTreeClassifier(max_depth=5)
model.fit(X, y)

# Create explainer
explainer = GPEExplainer(
    model=model,
    feature_names=iris.feature_names,
    X_train=X,
    min_precision=0.95
)

# Explain a prediction
explanation = explainer.explain(X[0])
print(explanation)
```

**Output:**
```
============================================================
GPE Explanation (method: GPE)
============================================================
Prediction: 0
Rule: petal length (cm) <= 2.45
------------------------------------------------------------
Precision: 100.00%
Coverage: 33.33%
Complexity: 1 conditions
Reduction: 75.0% (4 → 1 conditions)
============================================================
```

### Natural Language Output

```python
print(explanation.to_natural_language())
```
```
The model predicts 'setosa' because petal length is at most 2.45.
This explanation covers 33.3% of similar cases with 100.0% accuracy.
```

## 🔧 GPE Variants

```python
from gpe import (
    GPEExplainer,            # Standard (fast, greedy)
    GPEInformationTheoretic, # Uses mutual information for pruning
    GPECounterfactual,       # Adds counterfactual explanations
    GPEOptimal,              # Exhaustive search for minimal rule
    GPEEnsemble              # For Random Forest, XGBoost
)

# GPE-IT: Uses mutual information I(condition; prediction)
gpe_it = GPEInformationTheoretic(model, feature_names=features, X_train=X)
explanation = gpe_it.explain(x)

# GPE-CF: Includes counterfactual explanation
gpe_cf = GPECounterfactual(model, feature_names=features, X_train=X)
cf_explanation = gpe_cf.explain_with_counterfactual(x)
print(f"To change the decision: {cf_explanation.changes}")
```

## 📖 How It Works

GPE operates in three phases:

### 1. GREEDY Phase
Extract the full decision path from root to leaf:
```
Root → income <= 50000 → debt_ratio > 0.4 → ... → Leaf (denied)
```

### 2. PRUNE Phase
Iteratively remove conditions that don't affect precision:
```python
while conditions > 1:
    for condition in rule:
        precision_without = calculate_precision(rule - condition)
        if precision_without >= threshold:
            remove(condition)
```

### 3. EXPLAIN Phase
Return the minimal rule with metrics:
- **Precision** — Accuracy for instances satisfying the rule
- **Coverage** — Proportion of dataset satisfying the rule
- **Complexity** — Number of conditions

## 📊 Metrics

```python
from gpe import (
    precision_score,
    coverage_score,
    complexity_score,
    fidelity_score,
    stability_score
)

# Evaluate explanation quality
precision = precision_score(explanation, model, X)
coverage = coverage_score(explanation, X)
complexity = complexity_score(explanation)
```

## 🔬 Scientific Novelty

1. **GPE-Core** — First local explanation method specifically designed for decision trees
2. **GPE-IT** — Novel use of mutual information I(condition; prediction) for condition selection
3. **Theoretical guarantees** — Proven precision bounds and O(n·d) complexity
4. **Practical efficiency** — 48x faster than LIME on real data

## 📁 Project Structure

```
gpe-framework/
├── gpe/
│   ├── __init__.py          # Public API
│   ├── core.py              # GPEExplainer
│   ├── novel_methods.py     # GPE-IT, GPE-CF (scientific contribution)
│   ├── variants.py          # GPEOptimal, GPEWeighted
│   ├── explanation.py       # Data structures
│   ├── metrics.py           # Evaluation metrics
│   ├── tree_utils.py        # Tree utilities
│   └── visualization.py     # Plotting functions
├── tests/                   # Unit tests
├── experiments/             # Benchmark scripts
└── docs/                    # Documentation
```

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black gpe/
```

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Vladyslav Dehtiarov**
- Email: vvdehtiarov@gmail.com
- ORCID: [0000-0002-1578-8588](https://orcid.org/0000-0002-1578-8588)
- Affiliation: Sumy State University, Ukraine

## 📚 Citation

If you use GPE in your research, please cite:

```bibtex
@article{dehtiarov2025gpe,
  title={Greedy-Prune-Explain: Minimal Local Explanations for Decision Tree Predictions},
  author={Dehtiarov, Vladyslav and Borovyk, Valentyna},
  journal={International Journal of Artificial Intelligence Research},
  year={2025}
}
```

## 🔗 Links

- [GitHub Repository](https://github.com/vladdehtiarov/gpe-framework)
- [PyPI Package](https://pypi.org/project/gpe-framework/)
- [Documentation](https://github.com/vladdehtiarov/gpe-framework#readme)
- [Issue Tracker](https://github.com/vladdehtiarov/gpe-framework/issues)
