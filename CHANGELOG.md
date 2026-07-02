# Changelog

All notable changes to mlquantify will be documented in this file.

---

## [Unreleased]

This cycle adds three new method families to `mlquantify`. **Neural
quantification**: two symmetric neural quantifiers (**HistNetQ** and
**GMNet**), the differentiable bag representations they are built on, a
wrapper to plug any PyTorch model into `QuaNet`, and a way to train directly
on prevalence-labelled bags — faithful ports of the authors' code (Esuli et
al. 2018; Pérez-Mon et al. 2024, 2025), numerically checked against it.
**Quantification trees** (`tree`): decision trees and forests whose split
criterion targets quantification directly (Milli et al. 2013). **ReadMe
methods** (`readme`): the classifier-free quantifiers from automated content
analysis (Hopkins & King 2010; Jerzak, King & Strezhnev 2022). Everything
torch-based lives behind the `neural` extra
(`pip install mlquantify[neural]`); a bare `import mlquantify` never imports
torch.

### New Features

- **ReadMe methods** (`readme`, new family) — the classifier-free quantifiers
  from automated content analysis: **`ReadMe`** (Hopkins & King 2010; King &
  Lu 2008) solves the accounting identity `P(S) = P(S|D) P(D)` by
  simplex-constrained least squares over the joint distribution of random
  binary-feature subsets (with median auto-binarization and variance-weighted
  feature sampling), and **`ReadMe2`** (Jerzak, King & Strezhnev 2022) — the
  continuous-feature successor — learns softsign feature projections by SGD
  (maximizing category and feature distinctiveness), kNN-matches the labeled
  set to the unlabeled documents to resist semantic change, and averages
  bootstrapped constrained-LS estimates. `ReadMe2` requires torch and is
  exposed only when it is installed (same optional pattern as `neural`);
  `ReadMe` is pure numpy/scipy.
- **Quantification Trees** (`tree`, new family) — decision trees whose split
  criterion is optimized for quantification rather than classification
  (Milli et al. 2013): a split is chosen to balance false positives against
  false negatives per class (criterion `'eb'`, or `'cqb'` to also trade off
  classification error), so that counting the leaf predictions estimates the
  prevalences directly. Three classes: `QuantificationTreeClassifier` (the
  raw sklearn-style tree, composable with any aggregative quantifier, e.g.
  `ACC(estimator=QuantificationTreeClassifier())` — the paper's AC(Q)),
  `QuantificationTree` (Classify-and-Count over a single tree) and
  `QuantificationForest` (the paper's Random Forest quantifier: averages
  per-tree **Adjusted Count** estimates, with per-tree tpr/fpr rates from
  cross-validation; the paper's best configuration).
- **`HistNetQ`** (`neural`) — a symmetric neural quantifier that trains
  end-to-end on bags labelled by prevalence, summarising each bag with a
  **differentiable histogram** over learnable feature extractor outputs
  (Pérez-Mon et al. 2024). You supply the feature-extraction `nn.Module`; the
  network adds the histogram bag-representation and a softmax quantification
  head, and optimises a quantification loss (`'rae'`/`'ae'`/`'mse'`) directly.
- **`GMNet`** (`neural`) — a symmetric neural quantifier that represents each bag
  with a mixture of **full-covariance Gaussians** across several independent
  latent spaces (Pérez-Mon et al. 2025). The full covariance (Cholesky-
  parameterised, so positive-definite without a `geotorch` dependency) captures
  feature correlations a per-feature histogram cannot; a CKA penalty
  (`cka_lambda`) keeps the latent spaces diverse.
- **Training directly on prevalence-labelled bags** — `PrevalenceBagMixin` and
  the ready-made `HistNetQBags` / `GMNetBags` accept `fit(Xs, ps)` (a collection
  of bags and their class prevalences, no per-instance labels), the native
  format of the LeQua competitions / learning-from-label-proportions. New bags
  are synthesised by mixing the real ones (Bag Mixer).
- **Differentiable bag representations** (`representations`, torch-only) —
  `TorchRepresentation` (base class for permutation-invariant, differentiable
  bag descriptors), `DifferentiableHistogramRepresentation` (learnable hard
  histogram, the HistNetQ BRM) and `GaussianRepresentation` (full-covariance
  Gaussian mixture, the GMNet BRM). Exported only when torch is available.
- **`TorchClassifierWrapper`** (`neural`) — wraps any `torch.nn.Module` in the
  scikit-learn `fit` / `predict_proba` / `transform` interface so a custom
  PyTorch model can be used as the `estimator` (and embedding source) of
  `QuaNet`.
- **`QuaNet` gains an `embedder=` parameter** — supply a separate transformer
  (e.g. `PCA`, `TfidfVectorizer`, a `TorchClassifierWrapper`) for the dense
  embeddings when the base `estimator` has no `transform` method. The two bare
  `assert`s were replaced with clear `ValueError`s.
- **Training controls on the symmetric quantifiers** — configurable
  `optimizer` (`'adam'`/`'adamw'`), `weight_decay`, an optional
  `ReduceLROnPlateau` schedule (`end_lr` + `lr_factor`, stopping when the LR
  bottoms out), multi-bag mini-batching (`batch_size`) with
  `gradient_accumulation`, a `tqdm` progress bar (`verbose`), and
  **checkpoint/resume** (`checkpoint_path` + `checkpoint_every`): re-fitting with
  the same checkpoint continues where training stopped.

### Fixed

- **Metrics no longer divide by zero on degenerate prevalences** — every
  metric in `mlquantify.metrics` is now finite on inputs with absent
  classes (e.g. the extreme bags of an APP sweep) and on degenerate
  single-class / zero-padded vectors, instead of emitting
  `inf`/`NaN`/`RuntimeWarning`:
  - `RAE`, `NRAE`, `KLD` and `NKLD` apply additive (Forman) smoothing,
    `(p + eps) / (1 + n_classes * eps)` — new `eps` parameter, default
    `1e-3`; pass `1/(2*sample_size)` for the LeQua convention, or `0` to
    disable;
  - `RAE` also fixed to normalise by the *per-class* absolute error rather
    than the mean one (the two coincide only in the binary case);
  - `NAE` and `NRAE` return `0.0` when the normaliser is zero (single-class
    input, where the maximum attainable error is itself zero);
  - `NMD` and `RNOD` return `0.0` for single-class input, and `RNOD` falls
    back to the full class set instead of dividing by an empty support when
    the true prevalence vector is all zeros.
- **`QuaNet.fit` no longer crashes on multiclass aggregation** — the auxiliary
  quantifier estimates are now read whether `aggregate` returns a dict or an
  array (it depends on the global `prevalence_return_type`), fixing an
  `AttributeError: 'numpy.ndarray' object has no attribute 'values'`.
- **`QuaNet` boolean parameter constraints** — `fit_estimator` and
  `bidirectional` were declared as positive-number intervals, which rejected
  `False`; they are now `"boolean"` constraints.

### Documentation

- New **Quantification Trees** user guide (why FP/FN balance matters for
  quantification, the EB/CQB criteria, the gain/stopping rule, the
  Adjusted-Count forest, and the `ACC(estimator=QuantificationTreeClassifier())`
  composition) under *Aggregative Quantification*, and a new **ReadMe
  Methods** guide (the accounting identity, its `P(S|D)`-stability
  assumption, feature-subset smoothing, ReadMe2's learned projections and
  matching) under *Non-Aggregative Quantification*; both families registered
  in the API reference and the MLQuantify Methods table.
- Three new example-gallery pages: **QuaNet** (LSTM correction network over a
  classifier, diagonal-plot read-out), **Quantification trees and forests**
  (vs. a standard CART under shift), and **Quantifying without a classifier
  (ReadMe)** (ReadMe vs. ReadMe2 across the prevalence range).
- The example gallery now generates its data with
  `mlquantify.datasets.make_quantification` (training sample + bags at
  explicitly controlled prevalence values) instead of
  `sklearn.datasets.make_classification` plus manual resampling — 12 pages
  converted, every plot block re-executed and verified. Pages whose subject
  is orthogonal (sampling protocols on raw data, classifier calibration)
  keep their original data generation.
- Honest performance narrative for quantification trees: the example page
  and user-guide intro now explain that a single tree's `FP = FN` guarantee
  makes it unbiased *at the training prevalence* only — CC's response slope
  is the classifier's `tpr − fpr`, so the shallow tree errs more at extreme
  prevalences than a deep CART, and the Adjusted-Count forest (or an `ACC`
  composition) is the configuration that actually tracks the diagonal,
  consistent with the paper's own results.
- **Neural Quantifiers** (and **Calibration**) are now top-level User Guide
  entries instead of nested under *Core Components*.
- Rewrote the **Neural Quantifiers** guide: the symmetric FEM → BRM → QM
  architecture, per-parameter tables for HistNetQ and GMNet (including how to
  match the authors' training setup), training directly on labelled bags, and
  the `TorchClassifierWrapper`, with an architecture diagram.
- Extended the **Representations** guide with the differentiable histogram and
  full-covariance Gaussian bag representations (live `.. plot::` figures).
- New example-gallery page **Neural quantifiers (HistNetQ & GMNet)** with a
  plotted diagonal comparison, plus `experiments/lequa_neural.py` — a runnable
  LeQua reproduction script (auto-downloads the data, GPU-ready, checkpointing).

### Build & Packaging

- The CI doctest step (`pytest --doctest-modules mlquantify/`) no longer
  crashes on torch-less environments: a package-level `conftest.py` skips
  collecting the torch-only modules (`readme/_readme2.py`,
  `representations/_torch_*.py`) when torch is not installed — the same
  mechanism scikit-learn uses. Public imports were already guarded by the
  package `__init__` files; only direct module collection was affected.
- The docs build now installs a CPU build of torch so the neural `.. plot::`
  examples execute, and `conf.py` only mocks torch for autodoc when it is not
  installed.

### Tests

- New `tests/test_tree.py` (20 tests): classifier behaviour (criteria,
  `max_depth`, determinism, the FP/FN-balancing property), quantifier
  prevalence validity on binary and multiclass data, forest subsampling /
  parallelism / per-tree adjustment rates, the `ACC` composition, and
  parameter validation.
- New `tests/test_readme.py` (14 tests): prevalence validity, shifted-bag
  recovery, auto-binarization and the `binarize=False` guard, determinism,
  `get_params`/`set_params` round-trips; `ReadMe2` tests skip cleanly when
  torch is missing and the module imports without it.
- `tests/test_metrics.py` gained divide-by-zero regression tests: RAE
  per-class normalisation on multiclass input (the bug was invisible in
  binary cases), finiteness of `RAE`/`KLD`/`NKLD` with zero-prevalence
  classes under the default smoothing, exact `eps=0` opt-out behaviour, and
  the degenerate single-class / all-zero guards of `NAE`, `NMD` and `RNOD`.
- Extended `tests/test_neural.py` with HistNetQ/GMNet bag-training coverage
  (list and 3-D array inputs, prevalence renormalisation, the equal-bag-size
  guard) and fixed the stale QuaNet test.

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

- **`mlquantify.calibration`** — a posterior-calibration subpackage.
  `ClassifierCalibrator` implements the scaling family of post-hoc calibrators —
  Temperature Scaling (`'ts'`), Bias-Corrected Temperature Scaling (`'bcts'`),
  Vector Scaling (`'vs'`) and No-Bias Vector Scaling (`'nbvs'`) — fitting each by
  minimising the held-out negative log-likelihood (Guo et al. 2017; Alexandari,
  Kundaje & Shrikumar 2020). It accepts probabilities or logits and returns
  calibrated probabilities, and is what `EMQ(calib_function=...)` now uses
  internally. `QuantifierCalibrator` is reserved for a future release.

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
- Added a **Calibration** User Guide page and a calibration example (reliability
  diagram + confidence histogram before/after Bias-Corrected Temperature
  Scaling).

### Build & Packaging

- **`matplotlib` is no longer a required dependency.** Plotting moves behind a
  ``viz`` extra (``pip install mlquantify[viz]``), and the neural quantifiers
  behind a ``neural`` extra; ``pip install mlquantify[all]`` pulls in both. A
  bare ``import mlquantify`` never imports matplotlib.
- Removed the unused ``xlrd`` dependency.
- Project metadata moved from ``setup.py`` into ``pyproject.toml`` (PEP 621
  ``[project]`` table). ``setup.py`` now only resolves the version and builds
  the optional Cython kernel.
- **Dropped the `abstention` dependency.** EMQ's posterior calibration now uses
  the native `mlquantify.calibration` (identical results to ~1e-7), removing a
  latent break under scipy 2.0 (`abstention` imported the removed
  ``scipy.misc``).

### Internal

- De-duplicated the metrics ``process_inputs`` helper into a single
  ``mlquantify.metrics._utils`` module (was copy-pasted across three files).
- Removed the dead ``mlquantify/model_selection/_split.py`` stub (empty ``# TODO``,
  imported nowhere).

### Tests

- Added `tests/test_datasets.py` covering the generator and the fetchers; fixed
  the fetch tests.
- Added `tests/test_confidence.py` (percentile intervals, simplex/CLR ellipses,
  and the factory) and `tests/test_calibration.py` (the four scaling methods,
  temperature recoverability, and EMQ integration).

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
