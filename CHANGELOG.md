# Changelog

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
