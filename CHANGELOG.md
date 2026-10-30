# Changelog

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
