# GPE Framework - Greedy-Prune-Explain

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/gpe-framework.svg)](https://badge.fury.io/py/gpe-framework)
[![Tests](https://github.com/vladdehtiarov/gpe-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/vladdehtiarov/gpe-framework/actions/workflows/tests.yml)

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
    CertifiedGPE,            # Finite-sample precision guarantee
    GPEInformationTheoretic, # Uses mutual information for pruning
    GPECounterfactual,       # Adds counterfactual explanations
    GPEOptimal,              # Exhaustive search for minimal rule
    GPEEnsemble              # For Random Forest, Gradient Boosting
)

# GPE-IT: Uses mutual information I(condition; prediction)
gpe_it = GPEInformationTheoretic(model, feature_names=features, X_train=X)
explanation = gpe_it.explain(x)

# GPE-CF: Includes counterfactual explanation
gpe_cf = GPECounterfactual(model, feature_names=features, X_train=X)
cf_explanation = gpe_cf.explain_with_counterfactual(x)
print(f"To change the decision: {cf_explanation.changes}")
```

### GPE-Ensemble

`GPEEnsemble` explains Random Forest predictions with a consensus rule in three phases:

1. **Path Extraction** — the decision path of `x` in every tree, together with the tree's vote
2. **Threshold Aggregation** — for each feature, the bounds of the trees that voted for the ensemble's class are intersected into `[low_j, high_j]`; a feature becomes a candidate when at least `M/2 + 1` trees use it (`aggregation='majority'`) or when the trees using it hold more than half of the total weight (`aggregation='weighted'`, trees weighted by their accuracy on the training data)
3. **Consensus Pruning** — conditions are greedily removed while precision on the background set stays at or above `min_precision`

```python
from gpe import GPEEnsemble

explainer = GPEEnsemble(rf_model, feature_names=features, X_train=X_bg,
                        min_precision=0.95, aggregation='majority')
explanation = explainer.explain(x)
explanation.metadata  # voting_trees, candidates, consensus, meets_threshold
```

RandomForest with 50 trees (max_depth=8), 50 test instances × 3 runs, 100 background samples
(`experiments/run_ensemble_benchmark.py`, v1.2.0, Apple M1 Pro):

| Dataset | Time (ms) | Complexity | Precision | Coverage |
|---------|-----------|------------|-----------|----------|
| Credit Risk | 28.8 ± 2.8 | 1.8 | 0.982 | 0.223 |
| Credit Card Fraud | 30.8 ± 6.1 | 1.0 | 1.000 | 0.902 |
| Lending Club | 33.6 ± 9.0 | 1.0 | 1.000 | 0.885 |

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

## 🔒 Certified-GPE

`GPEExplainer` checks precision on the background data, so the rule it picks can fall below the threshold on new data. `CertifiedGPE` gives a finite-sample guarantee instead: with probability at least `1 - delta` over the calibration set, the rule's population precision is at least `min_precision`.

```python
from gpe import CertifiedGPE

# X_cal must not have been used to fit or tune the tree
explainer = CertifiedGPE(model, X_cal, min_precision=0.95, delta=0.05)
explanation = explainer.explain(x)

cert = explanation.certificate
print(explanation.rule, cert.lower_bound, cert.K, cert.n, cert.trivial)
```

The rule comes from the subsets of the decision path. The levels of a Clopper-Pearson bound are split over all `2^k - 2` subsets (Berk et al., 2013; Angelopoulos et al., 2025), so the guarantee covers any search. Up to 20 literals `search='auto'` enumerates all subsets and returns a shortest certified rule; above that it is greedy.

What it does not cover:

- It concerns fidelity to the model under the calibration distribution, not the correctness of the prediction. It does not survive distribution shift or a refit.
- It holds for one explanation at a time. For the whole tree at once use `compile_explanations(explainer, simultaneous=True)`.
- It is conservative and needs data. A rule with no errors needs `min_coverage(k)` calibration rows: 194 at `k=10`, `tau=0.95`, `delta=0.05`. When nothing can be certified, the full path is returned with `cert.trivial=True`; its precision is exactly 1 because its region lies inside one leaf.
- `interval='wilson'` has no finite-sample coverage, and its certificates carry `guarantee=False`.
- With imbalanced classes, high precision can just mean the base rate. Use `min_lift` to require the rule to beat it.

## 📦 Compiled Explanations

For `GPEExplainer`, `GPEInformationTheoretic` and `CertifiedGPE`, the explanation of an instance depends only on its leaf. All rules can be computed once and looked up without background data:

```python
from gpe import compile_explanations

table = compile_explanations(explainer)
explanation = table.explain(x)
arrays = table.to_arrays()     # leaf_ids, bit masks over the path, K, n
print(table.storage_bytes())   # size on top of the tree itself
```

## 🔬 Method Properties

1. **GPE-Core** — rules from subsets of the decision path, evaluated on real data rather than perturbations
2. **GPE-IT** — literal order from conditional mutual information; a heuristic with no guarantee
3. **Certified-GPE** — finite-sample lower bound on population precision

## 📁 Project Structure

```
gpe-framework/
├── gpe/
│   ├── __init__.py          # Public API
│   ├── core.py              # GPEExplainer
│   ├── certified.py         # CertifiedGPE, confidence bounds
│   ├── compiled.py          # compile_explanations: leaf-to-rule table
│   ├── novel_methods.py     # GPE-IT, GPE-CF (scientific contribution)
│   ├── variants.py          # GPEOptimal, GPEWeighted, GPEMonotonic, GPEEnsemble
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
@article{dehtiarov2026gpe,
  title={The Greedy-Prune-Explain Method for Fast Generation of Local Explanations from Decision Trees in Binary Classification Tasks},
  author={Dehtiarov, Vladyslav and Borovyk, Valentyna},
  journal={Nauka i tekhnika sohodni (Science and Technology Today)},
  year={2026},
  note={Accepted for publication}
}

@article{dehtiarov2026gpeensemble,
  title={The GPE-Ensemble Algorithm: Extending Local Explainability to Ensemble Decision-Tree Models},
  author={Dehtiarov, Vladyslav and Borovyk, Valentyna},
  journal={Nauka i tekhnika sohodni (Science and Technology Today)},
  year={2026}
}
```

## 🔗 Links

- [GitHub Repository](https://github.com/vladdehtiarov/gpe-framework)
- [PyPI Package](https://pypi.org/project/gpe-framework/)
- [Documentation](https://github.com/vladdehtiarov/gpe-framework#readme)
- [Issue Tracker](https://github.com/vladdehtiarov/gpe-framework/issues)
