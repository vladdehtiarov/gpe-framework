# Changelog

## 1.3.1 — 2026-11-02

Rules differ from 1.3.0 only where the full path has no background rows: 0 of 6168 explanations per method otherwise (`experiments/results_release/`).

### Fixed
- **Missing values.** On a tree trained with NaN, a rule could be false on its own row. `Condition.matches_missing` now follows the tree's routing, so a path rule selects exactly the rows of its leaf, NaN included. This holds for `GPEExplainer` and its variants, `CertifiedGPE`, which now accepts instances with missing values, `compile_explanations`, including leaves reached only by NaN, and `GPEEnsemble`.
- **Precision without evidence.** A rule with no background rows reported precision 1.0. It now reports NaN and `metadata['n_background'] == 0`, and NaN fails every threshold, so pruning never accepts such a rule. Previously pruning could pass through empty regions, which is the source of all rule changes above. Without any background data, precision and coverage are NaN. `precision_score` follows, and `aggregate_metrics` and `compare_explanations` skip undefined values and report how many remained.
- `Explanation.to_dict()` passes strict JSON (`json.dumps(..., allow_nan=False)`) for all methods and round-trips through `from_dict`. NaN becomes null, an infinity becomes the string `Infinity`.
- `str(rule)` called the purity of the leaf "confidence", which read as the precision of the rule. It now says "leaf purity". `GPEOptimal` no longer stores precision in that field. `to_natural_language()` no longer calls agreement with the model "accuracy".
- `GPECounterfactual` could return a point with the same class. It now finds the closest point of another class exactly, by moving x into the box of each leaf, checks the result against the model, and sets `found=False` when no point is reachable. `actionable_features=[]` was treated as all features.
- `GPEMultiResolution` and `GPEPareto` overwrote `min_precision` of the explainer. `GPEMultiResolution` silently replaced thresholds of the wrong length; it now raises `ValueError`.
- `GPEStable` used the global NumPy random state. It takes `random_state`, and results repeat.
- `GPEMonotonic` ignored the predicted class: "higher value, higher class" also argues for the lower class with a low value. It now swaps the direction for the lower class of a binary model.
- GPE-IT no longer warns on empty subsets.
- `simplify_conditions` on hand-made conditions: it dropped `in` conditions, ignored the missing-value flag when merging `>` with `>=` or `<` with `<=`, and merged conditions with different `float32_input`, all of which changed the selected rows. Paths read from a tree have none of these, and their rules are unchanged.

### Changed
- `GPEExplainer` and all variants reject regression trees, multi-output trees and unfitted models with a clear error. Regression trees never pruned, because precision compares classes.
- `GPEEnsemble` accepts only `RandomForestClassifier` and `GradientBoostingClassifier`. XGBoost models returned empty rules, and regressors could not prune.
- `min_precision` must be in (0, 1], and `feature_names` must match the model. Default names come from `feature_names_in_` if present.
- Background data, calibration data and batches may be DataFrames; an empty or misshapen matrix raises a clear `ValueError`.
- `metadata['meets_threshold']` on `GPEExplainer` and its variants, because some of them can miss `min_precision`.
- Removed claims of priority and of "formal stability guarantees" from docstrings: stability is an empirical estimate.

### Baselines
Published comparisons with these wrappers should be read with the notes below.
- `SHAPWrapper` flattened the 3D output of current `shap`, which mixed the values of features and classes. It now takes one value per feature for the predicted class and stores `metadata['top_features']`. Its rule is a conversion with an arbitrary margin and does not measure SHAP.
- `LIMEWrapper` kept one edge of the bin of x, which widened the rule and lowered its precision, and it lost features whose names contain spaces or hyphens. It now uses the indices and exact edges of the explainer. The number of features in a LIME rule is the `n_features` setting, not a result of the method.
- `AnchorsWrapper` lost conditions on such names and used bounds rounded to two decimals. It now uses `features()` and snaps bounds to the quartile edges. Its `precision` and `coverage` are those Anchors reports on its perturbation distribution; comparisons must recompute them on data.

### Tests
- New `tests/test_robustness.py`, `tests/test_baselines.py`, `tests/test_tree_utils.py`. All variants, the visualization module and the baseline wrappers now run in the test suite.

## 1.3.0 — 2026-11-02

### Added
- `CertifiedGPE`: rules from the decision path with a finite-sample lower bound on population precision, computed on a separate calibration set `X_cal`. Parameters: `delta`, `interval` (`clopper_pearson`, `hoeffding`, `wilson`), `alpha_scheme` (`uniform`, `layered`, `chain`), `search` (`exact` up to 20 literals by default, `greedy`), `min_lift`.
- `Certificate` on `Explanation`: K, n, lower bound, alpha, budget and lift fields.
- `compile_explanations()` and `CompiledExplanations`: a leaf-to-rule table with bit masks over the path, for lookups without background data. `simultaneous=True` makes certificates hold for the whole tree at once.
- `lower_bound()`, `min_coverage()`.
- `Explanation.metadata['n_background']` in `GPEExplainer`: 0 means the reported precision of 1.0 rests on no rows.
- `experiments/stage0_facts.py`.

### Fixed
- `GPEInformationTheoretic` could return rules below `min_precision` (in 1.2.0, up to 98.7% of explanations on a synthetic set): literals under `min_mutual_info` were skipped, so the search could stop short of the full path. It now selects by conditional mutual information and always reaches `min_precision` on the background data. `min_mutual_info` has no effect any more. Published GPE-IT numbers came from the old behaviour.
- Conditions taken from a tree now round the feature value to float32 before comparing, as sklearn does, so a path rule selects exactly the rows of its leaf.

## 1.2.0 — 2026-10-30

### Changed
- `GPEEnsemble` implements the three-phase consensus algorithm: path extraction with per-tree votes, threshold aggregation over the trees that voted for the ensemble's class (features used by at least M/2 + 1 trees), and consensus pruning down to `min_precision`. Replaces the forward-selection heuristic from 1.1.0.
- `GPEEnsemble` background predictions are computed once per background set (≈3× faster on the ensemble benchmark).
- `numpy<2` pin removed.

### Added
- `aggregation='weighted'`: tree votes weighted by individual accuracy on the training data (`y_train` optional; ensemble predictions are used when omitted).
- `Explanation.metadata` for `GPEEnsemble`: `voting_trees`, `candidates`, `consensus`, `meets_threshold`.
- Student Performance dataset loader and benchmark; ensemble benchmark script.
- CI on Python 3.10–3.13.

### Fixed
- `GPEMonotonic` ignored constraints on single-condition rules.
- `gpe.__version__` reported 1.0.1 in the 1.1.0 release.
- Lending Club file name in experiment scripts contained a Cyrillic «с».
- Unknown `aggregation` values now raise `ValueError` instead of being ignored.

## 1.1.0 — 2026-02-08
- Real-time `GPEEnsemble` and defense benchmarks.

## 1.0.1 — 2025-12-29
- Project URLs.

## 1.0.0 — 2025-12-29
- Initial release.
