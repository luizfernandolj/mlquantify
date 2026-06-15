# Changelog

All notable changes to mlquantify will be documented in this file.

---

## [v0.4.0]

This release focuses on **performance** (a compiled Cython kernel + several
vectorisations), **multiclass extensibility**, an easy **evaluation helper**, and
a full **documentation standardisation** across every module.

### New Features

- **`apply_protocol`** (`model_selection`) — a scikit-learn `cross_validate`-style
  helper that runs the whole quantification-evaluation loop in one call: fit the
  quantifier, sample test batches with a protocol (`'app'`/`'npp'`/`'upp'`/`'ppp'`),
  predict each, and return the true/predicted prevalences plus one score array per
  metric.
- **Cython acceleration for distribution matching** — a compiled histogram α-sweep
  kernel (`mlquantify.matching._histogram_sweep`) speeds up `DyS`/`HDy`/`HDx`. It is
  **optional**: when the extension is not built, the library transparently falls
  back to a pure-Python sweep (a one-time `PerformanceWarning` is emitted) and to
  the generic solver. Toggle it at runtime with
  `mlquantify.matching._histogram.USE_SWEEP_KERNEL`.
- **Pluggable multiclass strategies** (`multiclass`) — One-vs-Rest / One-vs-One now
  live in a registry. Add custom decompositions (ECOC, hierarchical, …) with
  `@register_strategy(...)` on a `MulticlassStrategy` subclass; inspect them with
  `available_strategies()`. No change to the dispatch is required.
- **`strategy` parameter on `APP`** (`model_selection`) — the way prevalences are
  sampled over the simplex is now selectable: `'grid'` (the classic evenly-spaced
  lattice), `'kraemer'` (uniform Kraemer sampling), `'uniform'` (flat Dirichlet), or
  `'dirichlet'` (biased by a new `dirichlet_alpha` concentration: `<1` favours
  extreme, one-class-dominant prevalences, `>1` balanced ones). `UPP` is now a thin
  `APP` subclass with the strategy pinned to `'kraemer'`; its old `algorithm=`
  argument is kept as a **deprecated** alias for `strategy`.
- **`simplex_dirichlet_sampling`** (`utils`) — new Dirichlet sampler over the simplex
  with a tunable `alpha` concentration. `simplex_uniform_sampling` is now a thin
  `alpha=1` wrapper around it.
- **`classes` argument on `aggregate`** — every aggregative quantifier's `aggregate`
  accepts an optional `classes=` listing the class set the output must report, so a
  class absent from the predictions still appears (with prevalence 0). Defaults to
  `None` (inferred as before).

### Performance

- **`evaluate_thresholds`** vectorised from `O(n_thresholds · n)` to `O(n log n)`
  via sorted survival counts — removes a quadratic cliff and speeds **all**
  threshold methods (`TAC`, `TX`, `TMAX`, `T50`, `MS`, `MS2`); ~160–490× faster on
  large inputs.
- **Median Sweep** (`MS`/`MS2`) vectorised — the per-threshold `CC` re-count loop is
  replaced by a single cumulative computation.
- **`GridSearchQ`** now precomputes the (seeded) protocol batches and their true
  prevalences once instead of regenerating them for every parameter combination
  (identical results).
- The Cython sweep kernel roughly halves `DyS`/`HDy` `predict` time versus the
  generic Python solver, with bit-identical estimates.

### Behavior Changes

- **`MLPE`** now matches its documented role as the *trivial baseline*: it returns
  the **training prevalence** for any test set (previously it responded to the test
  set, contradicting the docs). Use `EMQ` for the non-trivial maximum-likelihood
  estimator.
- **`CC`/`PCC` `aggregate` signature** — the `y_train` argument (which only supplied
  the class set) is replaced by `classes`. On the predict path they now report the
  classes seen at `fit`, and their `requires_train_labels` tag is `False`. Callers
  passing `y_train=` positionally are unaffected (it maps to `classes`); update
  `y_train=` keyword calls to `classes=`.
- **`GridSearchQ` now takes a quantifier *instance*** (e.g. `GridSearchQ(CC(...))`)
  rather than a zero-argument factory, matching its documented `quantifier`
  parameter. Its fitted attributes are standardised to the trailing-underscore
  convention — `best_score_`, `best_params_`, `best_model_` (were `best_score` /
  `best_params`) — and the redundant `best_params()` / `best_model()` accessor
  methods (which collided with the attributes) are removed.
- **`PWK`** is now an aggregative `CC` subclass built on a quantification-modified
  k-NN classifier, so it shares the `fit`/`predict`/`aggregate` interface (and the
  new `classes` argument). It no longer accepts an external estimator parameter —
  the modified k-NN is intrinsic to the method.

### Fixes

- `normalize_prevalence` no longer emits a numpy "uninitialised memory" warning
  (it used `where=` without `out=`); zero/negative-sum inputs now return a valid
  distribution.
- Corrected the `DyS` docstring (it defaults to the **Topsøe** distance, not
  Hellinger) and documented previously-undocumented constructor parameters
  (`distance`, `solver`, `bin_strategy`, `laplace_smoothing`).
- `QuaDapt` reference corrected to the LeQua 2025 paper.
- Added the missing `ACC` entry to the API reference.
- **`CC.aggregate` on multiclass data** no longer shrinks `classes_` to the classes
  present in a single batch's predictions. This previously produced ragged
  prevalence vectors and crashed `apply_protocol` (and `UPP`/sampled-prevalence
  protocols) on multiclass problems; absent classes now correctly report 0.
- **`simplex_uniform_sampling`** now respects `random_state` — it previously drew
  from the global NumPy RNG and ignored the seed, making "uniform" UPP runs
  non-reproducible.
- Removed dead code: the unused `neighbors._base` (`BaseKDE`) and
  `solvers._least_squares` modules.
- **All docstring doctests now run and pass** (122), with a seeded, format-stable
  doctest setup; fixed numerous broken examples (wrong `aggregate` calls, stale
  attribute names, incomplete snippets). Added a GitHub Actions workflow that runs
  the test suite and the doctests across Python 3.9–3.13 on Linux/macOS/Windows.
- **Restored Python 3.9 import** — `utils._tags` and `utils._constraints` used
  `X | Y` class annotations, which 3.9 evaluates at runtime (the `|` union syntax
  is 3.10+), breaking the whole package import. Added
  `from __future__ import annotations` to the affected modules.
- **`MS2`/`TMAX` doctests** are now skipped (`# doctest: +SKIP`): their threshold
  selection is platform-sensitive and produced non-reproducible prevalences on
  macOS.

### Documentation

- **Standardised docstrings and user guide across every module** following a
  two-surface model (API docstrings = interface, User Guide = theory): each method
  leads with its shift assumption, and gained `Attributes`/`Notes`/`See Also`
  sections and per-option parameter descriptions.
- New User Guide pages: **Multiclass Quantification** (OvR/OvO, setting a binary
  method, registering new strategies), **Prevalence Normalization** (all
  `normalize_prevalence` / config options), and **Cython acceleration** (what the
  histogram sweep is and where it saves time and memory).
- Augmented the **Solvers**, **Representations** and **Losses** guides, and added
  live `.. plot::` figures (histogram `bins`/`bin_edges`/class-conditional, KDE
  `bandwidth`, Prediction soft-vs-hard, and the EMQ optimisation).
- Removed duplicated content and fixed heading-level nesting in several guide pages;
  suppressed benign `ref.citation` build warnings.
- Rewrote the **Protocols** guide's `APP`/`UPP` parameter tables to document the new
  `strategy`/`dirichlet_alpha` options (grid vs Kraemer vs uniform vs Dirichlet), and
  added `simplex_dirichlet_sampling` to the API reference.

### Build & Packaging

- Added the Cython build infrastructure: `pyproject.toml` build backend, `setup.py`
  `cythonize` of the optional extension, and `MANIFEST.in` shipping `*.pyx`/`*.pxd`
  in the sdist so source installs can compile (or use the fallback).
- CI now builds **portable binary wheels** with `cibuildwheel`
  (Linux / macOS x86_64+arm64 / Windows × CPython 3.9–3.13), verifies each wheel
  ships the compiled kernel, builds an sdist, and publishes to PyPI on a `v*` tag.
- **Source builds no longer fail on the package version** — the docs workflow now
  writes a PEP 440-compliant `0.4.0.dev0` to `VERSION.txt` instead of a bare `dev`
  (rejected by newer setuptools), and `setup.py` falls back to that version when
  `VERSION.txt` is absent, so local `pip install .`/`-e .` works again.

### Tests

- Added `tests/test_optimizations.py` with parity tests for `evaluate_thresholds`,
  Median Sweep, `GridSearchQ` caching, and the Cython kernel (compiled == pure
  Python == generic solver).
- Added a cross-library comparison harness (`comparisons/`) validating prevalence
  agreement and speed against QuaPy, qunfold and quantificationlib.

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

[v0.4.0]: https://github.com/luizfernandolj/QuantifyML/compare/v0.3.0...v0.4.0
[Unreleased]: https://github.com/luizfernandolj/QuantifyML/compare/v0.2.1...HEAD
