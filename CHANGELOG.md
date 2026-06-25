# Changelog

All notable changes to mlquantify will be documented in this file.

---

## [Unreleased]

_Nothing yet. Add user-visible changes here as they land._

---

## [v0.5.0]

This release adds two new subpackages — **`mlquantify.visualization`**
(scikit-learn-style plotting) and **`mlquantify.datasets`** (synthetic and
real-world quantification data) — plus an example gallery and a redesigned
documentation homepage. It also aligns the evaluation-metrics API with
scikit-learn (a breaking change, below) and modernises packaging.

### Breaking Changes

- **Evaluation metrics now follow the scikit-learn argument order
  ``metric(y_true, y_pred)``** — the *true* prevalence comes first and the
  prediction second (previously the order was reversed, ``metric(pred, true)``).
  Symmetric metrics (`AE`, `MAE`, `SE`, `MSE`, `NMD`, `RNOD`, `CvM_L1`) are
  unaffected numerically, but **asymmetric metrics (`KLD`, `NKLD`, `RAE`,
  `NRAE`, `NAE`, `VSE`) now return different values** for callers that relied on
  the old order. Most of the documentation and the README already used the
  ``(true, pred)`` order and so become correct; update any code that passed the
  prediction first. The `apply_protocol` scoring-callable contract is likewise
  ``metric(true, pred)``.

### New Features

- **`mlquantify.visualization`** — a new plotting subpackage of scikit-learn-style
  ``*Display`` classes for quantification results. Each exposes
  ``from_predictions`` / ``from_estimator`` / ``from_protocol`` constructors, a
  ``plot()`` method, stored ``ax_`` / ``figure_`` attributes, and forwards
  matplotlib styling kwargs.

  - *Multiple-sample diagnostics* (summarise a protocol run): ``DiagonalDisplay``
    (true vs. predicted prevalence), ``BiasDisplay`` (signed-error boxplots,
    global or binned), ``ErrorByShiftDisplay`` (error vs. prior-probability
    shift, with Forman smoothing for RAE).
  - *Single-sample displays*: ``PrevalenceDisplay`` (per-class prevalence bars)
    and ``ConfidenceRegionDisplay`` (per-class intervals, or a ternary
    confidence ellipse for 3-class problems, hooking into
    ``mlquantify.confidence``).

  The subpackage is not imported by ``import mlquantify`` so matplotlib stays
  off the top-level import path; import it explicitly with
  ``from mlquantify.visualization import DiagonalDisplay``.

- **`mlquantify.datasets`** — a new subpackage (mirroring `sklearn.datasets`) for
  quantification data, both synthetic and real-world. Unlike `visualization` it
  *is* part of the top-level namespace (`import mlquantify` exposes
  `mlquantify.datasets`). It provides:

  - **`make_quantification`** — a synthetic-data generator, the quantification
    analogue of `make_classification`: it builds one labelled population, then
    draws `n_batches` *bags* whose class balance varies, returning each bag's
    **true prevalence** alongside the data. Three shift kinds via `shift_type` —
    `'prior'` (P(y) changes, clusters stay put), `'covariate'` (feature clouds
    translated past a fixed boundary, tuned by `covariate_scale`), and `'concept'`
    (the decision boundary rotated per bag, tuned by `concept_strength`); pass a
    list to **stack** several per bag. Also: selectable per-bag prevalence sampling
    (`prevalence=` `'uniform'` / Dirichlet `concentration` / `'grid'` / a fixed
    `target_prevalence`), population difficulty knobs (`class_sep`, `flip_y`,
    `n_features`, `weights`), an optional training split (`return_train`), the
    decision boundary (`return_boundary` → `coef`/`intercept`), and output-layout
    controls (`stack`, `pack`, `as_frame`).
  - **25 real-world dataset fetchers** — `fetch_*` loaders that download and cache
    well-known quantification benchmarks, following scikit-learn conventions
    (keyword-only; return a `Bunch` or a `(X, y)` tuple with `return_X_y=True`;
    cached once under `data_home`). Each also accepts an optional quantification
    `protocol=` (e.g. `"app"`), in which case the returned `Bunch` additionally
    carries `.samples` and `.prevalences` — test bags with known class prevalence,
    ready to score against. Coverage spans tabular (`fetch_mushroom`,
    `fetch_covertype`, `fetch_dry_bean`, `fetch_miniboone`, `fetch_wine_quality`,
    … 15 in total), text (`fetch_newsgroups20`, `fetch_imdb`, `fetch_rcv1_v2`,
    `fetch_sentiment140`, `fetch_multidomain_sentiment`), image (`fetch_mnist_usps`,
    `fetch_cifar10`), graph (`fetch_planetoid_cora_citeseer_pubmed`), a concept-drift
    stream (`fetch_sea_concepts`), and the LeQua 2024 competition
    (`fetch_lequa2024`).
  - **Dataset helpers** — `Bunch` (attribute-style dict), `get_data_home` /
    `fetch_remote` (cache location and cached download with retries),
    `make_protocol` (turn a loaded dataset into protocol bags), and a
    download-progress hook (`set_progress_hook` / `get_progress_hook`).

### Fixed

- Removed stray ``print()`` calls in the metrics input helper that dumped
  prevalence vectors to stdout whenever a metric was given a Python ``list``.

### Documentation

- New User Guide pages: **Synthetic Datasets** (the `make_quantification`
  generator and the prior/covariate/concept shift types, with a 3-class
  visualisation) and **Real-World Datasets** (the 25 fetchers, their shared
  configuration, and protocol mode). Added a `datasets` API reference page.
- New **example gallery** with runnable `.. plot::` examples: synthetic-data
  walkthroughs (intro, prevalence sampling, shift, the three shift types,
  difficulty, end-to-end quantifiers) plus quantification examples across modules
  (CC under shift, distribution matching, EMQ convergence, error by shift, grid
  search, method comparison, multiclass, protocols, confidence regions, intro).
- Redesigned documentation homepage with per-module showcase cards (generated
  figures) and a dev banner flagging the non-stable version.

### Build & Packaging

- **`matplotlib` is no longer a required dependency.** Plotting moves behind a
  ``viz`` extra (``pip install mlquantify[viz]``), and the neural quantifiers
  behind a ``neural`` extra; ``pip install mlquantify[all]`` pulls in both. A
  bare ``import mlquantify`` never imports matplotlib.
- Removed the unused ``xlrd`` dependency.
- Project metadata moved from ``setup.py`` into ``pyproject.toml`` (PEP 621
  ``[project]`` table). ``setup.py`` now only resolves the version and builds
  the optional Cython kernel.

### Internal

- De-duplicated the metrics ``process_inputs`` helper into a single
  ``mlquantify.metrics._utils`` module (was copy-pasted across three files).
- Removed the dead ``mlquantify/model_selection/_split.py`` stub (empty ``# TODO``,
  imported nowhere).

### Tests

- Added `tests/test_datasets.py` covering the generator and the fetchers; fixed
  the fetch tests.
- Added `tests/test_confidence.py` (percentile intervals, simplex/CLR ellipses,
  and the factory). Added ``xfail`` specification tests in
  `tests/test_calibration.py` for the still-unimplemented calibration stubs.

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
  The kernel check inspects the wheel's contents for the compiled
  `_histogram_sweep` extension rather than importing `mlquantify` — importing
  would reinstall the full runtime stack, and recent scipy/scikit-learn no longer
  ship `manylinux2014` wheels, so that install tried to build scipy from source
  and failed (no OpenBLAS) on Python 3.11+.
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
- Removed the redundant `tests/test_mixture.py` (superseded by `test_matching.py`)
  and folded its remaining unique coverage (`SMM` binary fit/predict) into it.

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

[v0.5.0]: https://github.com/luizfernandolj/mlquantify/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/luizfernandolj/mlquantify/compare/v0.3.0...v0.4.0
[Unreleased]: https://github.com/luizfernandolj/mlquantify/compare/v0.5.0...HEAD
