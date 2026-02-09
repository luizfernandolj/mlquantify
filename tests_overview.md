# Tests Overview

This document contains a list of all tests in the `tests` directory with a brief description of what each test does.

## `conftest.py`
Configuration file for pytest.
- `base_config`: Fixture that provides a base configuration dictionary for tests.
- `dataset_base`: Fixture that loads the 'base' dataset.
- `dataset_binary`: Fixture that converts the 'base' dataset to a binary classification problem (class '2' vs rest).
- `dataset_binary_sp`: Fixture similar to `dataset_binary` but for 'sp' (possibly Semi-Supervised or Special Purpose, context implies a specific dataset variant).
- `dataset_3_classes`: Fixture that filters the 'base' dataset to include only classes '1', '2', and '3'.

## `test_adjust_counting.py`
Tests for the `adjust_counting` module.
- `test_CC_binary`, `test_CC_aggregate_binary`: Test Classify and Count (CC) for binary classification (fit/predict and aggregation).
- `test_PCC_binary`, `test_PCC_aggregate_binary`: Test Probabilistic Classify and Count (PCC) for binary classification.
- `test_matrix_adjustment_binary`, `test_matrix_adjustment_multiclass`: Parametrized tests for AC, PAC, and FM methods on binary and multiclass datasets.
- `test_threshold_adjustment_binary`: Parametrized tests for TAC, TX, and TMAX methods on binary datasets.
- `test_CDE_binary`: Tests for CDE-Iterate method on binary datasets.

## `test_base.py`
Tests for the base classes and mixins.
- `test_base_quantifier_params`: Verifies `get_params` method of `BaseQuantifier`.
- `test_aggregation_mixin_params`: Verifies parameter handling in `AggregationMixin`, specifically when wrapping a learner.

## `test_extensive_adjust_counting.py`
Comprehensive extensive tests for the `adjust_counting` module covering CC, PCC, AC, PAC, FM, TAC, TX, TMAX, T50, MS, MS2, and CDE.
- `TestCC`: Detailed tests for Classify and Count (binary, multiclass, aggregation with hard/soft labels, pandas input, string labels).
- `TestPCC`: Detailed tests for Probabilistic Classify and Count (fit/predict, aggregation, pandas input).
- `TestMatrixAdjustment`: Tests for AC, PAC, FM (fit/predict, prevalence range, learner compatibility, pandas input, output keys, string labels).
- `TestThresholdAdjustment`: Tests for TAC, TX, TMAX, T50, MS, MS2 (fit/predict, multiclass via OVR, prevalence range).
- `TestCDE`: Tests for CDE-Iterate (fit/predict, multiclass, tolerance/max_iter variations).
- `TestMulticlassStrategies`: Tests OVR strategy for threshold-based and CDE quantifiers.
- `TestEdgeCases`: Tests for edge cases like tiny datasets (n<10), extreme imbalance, constant predictions, and single samples.
- `TestPrevalenceNormalization`: Verifies prevalence normalization (summing to 1).
- `TestUtils`: Unit tests for helper functions `compute_table`, `compute_tpr`, `compute_fpr`, and `evaluate_thresholds`.
- `TestLabelTypeVariations`: Tests ensuring compatibility with int, string, and float labels.
- `TestAggregateDirectly`: Tests `aggregate` method directly for CC and PCC.

## `test_extensive_base_config.py`
Comprehensive tests for base modules (`base.py`, `base_aggregative.py`, `calibration.py`, `confidence.py`, `multiclass.py`, `_config.py`).
- `TestBaseQuantifier`: Tests core functionality of `BaseQuantifier` (params, tags, saving/loading).
- `TestMetaquantifierMixin`: Tests `MetaquantifierMixin` inheritance and behavior.
- `TestProtocolMixin`: Tests `ProtocolMixin` tags and behavior.
- `TestAggregationMixin`, `TestSoftLearnerQMixin`, `TestCrispLearnerQMixin`: Tests for mixins handling aggregation and learner types.
- `TestGetConfig`, `TestSetConfig`, `TestConfigContext`: Tests for configuration management (get/set config, context manager).
- `TestCalibration`: Tests for calibration wrapper classes.
- `TestConfidenceInterval`, `TestConfidenceEllipseSimplex`, `TestConfidenceEllipseCLR`, `TestConstructConfidenceRegion`: Tests for confidence region construction and properties.

## `test_extensive_input_variants.py`
Tests for different input types and validations.
- `test_validate_data_...`: Tests validation of numpy/pandas inputs, categorical features, and rejection of NaNs.
- `test_get_prev_from_labels_...`: Tests extracting prevalence from labels (string, float, categorical).
- `test_validate_prevalences_...`: Tests prevalence validation and normalization.
- `test_define_binary_string_labels`: Tests `define_binary` decorator with string labels.

## `test_extensive_likelihood.py`
Comprehensive tests for the `mlquantify.likelihood` module (EMQ).
- `TestEMQBasicBinary`, `TestEMQBasicMulticlass`: Basic fit/predict tests for EMQ on binary and multiclass datasets.
- `TestEMQWithDifferentLearners`: Tests EMQ with different underlying learners (LogReg, RF, DT).
- `TestEMQCalibration`: Tests EMQ with various calibration methods (ts, bcts, vs, nbvs).
- `TestEMQInputTypes`: Tests EMQ with pandas input and string labels.
- `TestEMStatic`: Tests the static `EM` method directly.
- `TestEMQEdgeCases`: Tests edge cases (tiny dataset, imbalance, single class, constant features).
- `TestEMQErrors`: Tests error handling (predict before fit, invalid params).

## `test_extensive_meta.py`
Comprehensive tests for the `mlquantify.meta` module (EnsembleQ, AggregativeBootstrap, QuaDapt).
- `TestEnsembleQBinary`, `TestEnsembleQMulticlass`: Tests for EnsembleQ (fit/predict, protocols, selection metrics, size variations).
- `TestAggregativeBootstrapBinary`, `TestAggregativeBootstrapMulticlass`: Tests for AggregativeBootstrap (bootstraps, region types, confidence levels).
- `TestQuaDaptBinary`: Tests for QuaDapt (fit/predict, measures, merging factors).
- `TestMoSS`: Tests for the static `MoSS` method.
- `TestMetaIntegration`: Integration tests using multiple meta quantifiers together.

## `test_extensive_metrics_full.py`
Comprehensive tests for the `mlquantify.metrics` module.
- Tests specific metric classes: `AE` (Absolute Error), `MAE` (Mean Absolute Error), `KLD` (Kullback-Leibler Divergence), `SE` (Squared Error), `MSE` (Mean Squared Error), `NAE` (Normalized Absolute Error), `NKLD` (Normalized KLD), `RAE` (Relative Absolute Error), `NRAE` (Normalized RAE), `NMD` (Normalized Match Distance), `RNOD` (Root Normalised Order-aware Divergence), `VSE` (Variance-normalised Squared Error), `CvM_L1` (L1 Cramér–von Mises).
- Covers edge cases: perfect predictions, known values, non-negativity, symmetry, input types (list, dict, numpy), and multiclass scenarios.
- Tests handling of length mismatches (padding) and extreme imbalance.

## `test_extensive_metrics_param.py`
Parametrized tests for metrics to ensure consistency and correctness across various inputs.
- Verifies non-negativity for array and scalar metrics on core cases.
- Tests dictionary inputs for metrics.
- Validates properties of `NMD`, `RNOD` (including custom distance matrices), `VSE` (variance-normalized), and `CvM_L1`.

## `test_extensive_mixture.py`
Comprehensive tests for the `mlquantify.mixture` module.
- Covers Aggregative Mixture Models (`DyS`, `HDy`, `SMM`, `SORD`) and Non-aggregative Models (`HDx`, `MMD_RKHS`).
- Tests binary and multiclass (OVR, OVO) scenarios.
- Tests input variations: Pandas DataFrames, string labels.
- Tests specific features: `DyS` distance measures, custom bin sizes, different learners, `MMD_RKHS` kernels.
- Tests internal utilities: `getHist`, `ternary_search`, distance functions (`hellinger`, `topsoe`, etc.).

## `test_extensive_model_selection.py`
Comprehensive tests for the `mlquantify.model_selection` module.
- `GridSearchQ`: Tests hyperparameter tuning for quantifiers (binary/multiclass, fit/predict, best params/score/model).
- Protocols: Tests `APP` (Artificial Prevalence), `NPP` (Natural Prevalence), `UPP` (Uniform Prevalence), and `PPP` (Personalized Prevalence) for split generation, batch sizes, and reproducibility.
- Covers integration with different input types (pandas) and scoring functions.

## `test_extensive_multiclass.py`
Tests the `define_binary` decorator and multiclass wrapper functionality.
- Verifies that binary quantifiers can be adapted for multiclass problems using `ovr` (One-vs-Rest) and `ovo` (One-vs-One) strategies.
- Checks consistency of predictions (sum to 1, correct keys) and invalid strategy handling.

## `test_extensive_neighbors.py`
Comprehensive tests for the `mlquantify.neighbors` module.
- `PWK`: Tests Parzen Window Kernel quantifier (binary/multiclass, parameters, input types, edge cases).
- `PWKCLF`: Tests the underlying classifier (`PWKCLF`).
- KDE Quantifiers: Tests `KDEyML` (Maximum Likelihood), `KDEyHD` (Hellinger Distance), and `KDEyCS` (Cauchy-Schwarz).
- Covers bandwidths, kernels, session-scoped fixtures, and internal utilities (`gaussian_kernel`, `_simplex_constraints`).

## `test_extensive_sampling_prevalence.py`
Tests sampling and prevalence utility functions.
- `get_indexes_with_prevalence`: Tests index selection for specific prevalences (binary/multiclass).
- Simplex Sampling: Tests `simplex_uniform_kraemer`, `simplex_grid_sampling`, and `simplex_uniform_sampling` for generating valid prevalence vectors.
- `bootstrap_sample_indices`: Tests bootstrapping logic.
- Prevalence Helpers: Tests `get_prev_from_labels` and `normalize_prevalence`.

## `test_extensive_utils_full.py`
Comprehensive tests for the `mlquantify.utils` subpackage.
- Constraints: Tests `Interval`, `Options`, `HasMethods`, `Hidden`, `CallableConstraint`, `StringConstraint`.
- Tags: Tests partial tags (`TargetInputTags`, `PredictionRequirements`) and full `Tags` objects.
- Validation: Tests `validate_predictions` (soft/crisp), `validate_y` (dimensions/types), `validate_prevalences`, and `check_is_fitted`.
- Utilities: Covers `check_random_state`, `resolve_n_jobs`, and module-specific helpers.

## `test_extensive_validation.py`
Specific tests for validation logic and edge cases.
- Detailed tests for `validate_y` with various input tags (1D/2D, categorical).
- Tests `validate_predictions` for type rejection (int vs float) and shape handling.
- Tests `validate_prevalences` return types (dict vs array) and normalization methods.
- Tests `validate_parameter_constraints` for valid/invalid parameter combinations.

## `test_integration.py`
High-level integration tests for the library.
- Runs full quantification pipelines: data generation -> training (AC, PAC, DyS, EnsembleQ) -> prediction -> evaluation.
- Tests `GridSearchQ` in a realistic scenario with custom quantifier classes.

## `test_likelihood.py`
Basic tests for the `mlquantify.likelihood` module.
- `EMQ`: Basic binary fit/predict and static `EM` method tests.
- (simpler version of `test_extensive_likelihood.py`)

## `test_meta.py`
Basic tests for the `mlquantify.meta` module.
- Tests `EnsembleQ`, `AggregativeBootstrap`, and `QuaDapt` on binary data.
- (simpler version of `test_extensive_meta.py`)

## `test_metrics.py`
Basic tests for the `mlquantify.metrics` module.
- Verifies implementation of `AE`, `MAE`, `SE`, `MSE`, `KLD`, `NKLD`, `NMD`, `RNOD`, `CvM_L1` against known simple examples.
- (simpler version of `test_extensive_metrics_full.py`)

## `test_mixture.py`
Basic tests for the `mlquantify.mixture` module.
- Tests aggregative (`DyS`, `HDy`, `SMM`, `SORD`) and non-aggregative (`HDx`, `MMD_RKHS`) models on binary data.
- (simpler version of `test_extensive_mixture.py`)

## `test_model_selection.py`
Basic tests for `GridSearchQ`.
- Tests grid search with binary quantifier `TAC` and custom parameter grid.
- Demonstrates handling of learner initialization within `GridSearchQ`.

## `test_multiclass.py`
Basic tests for multiclass wrapper strategies.
- Tests `BinaryQuantifier` with `ovr` (One-vs-Rest) and `ovo` (One-vs-One) strategies.
- Verifies usage of `define_binary` decorator.

## `test_neighbors.py`
Basic tests for `PWK` (Parzen Window Kernel).
- Tests binary fit/predict and `classify` method.
- (simpler version of `test_extensive_neighbors.py`)

## `test_neural.py.disabled`
Tests for `QuaNet` (Neural Network Quantifier).
- **Note**: This file is disabled/renamed, likely due to optional PyTorch dependency.
- Contains tests for `QuaNet` training/prediction and `EarlyStop` mechanism.

## `test_utils.py`
Basic tests for `mlquantify.utils`.
- Tests sampling functions (`get_indexes_with_prevalence`, simplex sampling).
- Tests prevalence helpers (`get_prev_from_labels`, `normalize_prevalence`) and validation logic.
- (simpler version of `test_extensive_utils_full.py`)







