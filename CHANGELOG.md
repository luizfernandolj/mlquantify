# Changelog

All notable changes to mlquantify will be documented in this file.

---

## [Unreleased] — v0.3.0

### New Modules

#### `compose`
Added a new `compose` module that wraps the [`qunfold`](https://github.com/mirkobunse/qunfold) package, exposing its quantification methods as first-class estimators in mlquantify. Includes `ComposeQuantifier` with an enhanced `fit`/`predict` interface and full test coverage validating equivalence with QuaPy's baseline methods.

#### `solvers`
Introduced a dedicated `solvers` module with modular, reusable solver components:

| Solver | Description |
|---|---|
| `BinarySolver` | Solver for binary quantification problems |
| `LeastSquaresSolver` | Least-squares constrained optimization |
| `SimplexSolver` | Simplex-projected optimization |

These underpin the optimization steps across aggregative methods and can be composed independently of any specific quantifier.

#### `representations`
Added a standalone `representations` module providing pluggable feature representations previously scattered across method-specific implementations:

- `HistogramRepresentation`
- `DensityRepresentation`
- `KernelRepresentation`
- `DistanceRepresentation`
- `PredictionRepresentation`

#### `losses`
Extracted loss functions into their own `losses` module, making them composable and reusable across different quantifier families:

- Distance-based losses (`_distances.py`)
- Likelihood-based losses (`_likelihood.py`)

---

### Breaking Changes

#### `learner` renamed to `estimator`
The parameter and attribute name `learner` has been renamed to `estimator` across the entire library to align with scikit-learn conventions.

**Before:**
```python
quantifier = CC(learner=LogisticRegression())
```

**After:**
```python
quantifier = CC(estimator=LogisticRegression())
```

> Any code passing `learner=` to quantifier constructors must be updated to `estimator=`.

---

### Improvements

- **`matching`** — Methods now leverage the new `representations` and `solvers` modules internally, reducing duplication.
- **`model_selection`** — Comprehensive docstring rewrites and improved protocol/search API documentation.
- **`neighbors`** — Updated API documentation and docstrings.
- **`neural`** — Rewrote docstrings and fixed QuaNet import warnings; improved class structure documentation.
- **`utils`** — Improved validation logic (`_validation.py`) and added scikit-learn-style tags support (`_tags.py`).
- **`metrics`** — Added distance metrics to the `metrics` module (`_distances.py`).

---

### Tests

- Added `test_compose.py` with full coverage for the new `compose` module.
- Added `test_representations.py` for the new `representations` module.
- Added `test_solvers.py` for the new `solvers` module.
- Added `test_matching.py` for matching-based quantifiers.
- Added `test_losses.py` for loss functions.
- Renamed `test_adjust_counting.py` → `test_counting.py` for consistency.
- Added comparison tests between QuaPy and qunfold-based methods to validate equivalent results.

---

### Documentation

- New user guide covering the updated library structure.
- Added `solvers` API reference page.
- Updated aggregative quantification usage guide.
- Full API docstrings added across `model_selection`, `neighbors`, and `neural` modules.

---

[Unreleased]: https://github.com/luizfernandolj/QuantifyML/compare/v0.2.1...HEAD
