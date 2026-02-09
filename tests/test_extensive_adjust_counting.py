"""
Comprehensive tests for mlquantify.adjust_counting module.

Covers CC, PCC, AC, PAC, FM, TAC, TX, TMAX, T50, MS, MS2, CDE
with binary and multiclass datasets, varied input types, label types,
learner types, strategy variations, edge cases, and error handling.
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlquantify import config_context
from mlquantify.adjust_counting import (
    CC,
    PCC,
    AC,
    PAC,
    FM,
    TAC,
    TX,
    TMAX,
    T50,
    MS,
    MS2,
    CDE,
)
from mlquantify.adjust_counting._utils import (
    compute_table,
    compute_tpr,
    compute_fpr,
    evaluate_thresholds,
)

# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------

COUNT_CLASSES = [CC, PCC]
MATRIX_CLASSES = [AC, PAC, FM]
THRESHOLD_CLASSES = [TAC, TX, TMAX, T50, MS, MS2]
ALL_CLASSES = COUNT_CLASSES + MATRIX_CLASSES + THRESHOLD_CLASSES + [CDE]

LEARNER_FACTORIES = [
    pytest.param(lambda: LogisticRegression(random_state=42, solver="liblinear", max_iter=200), id="LogReg"),
    pytest.param(lambda: DecisionTreeClassifier(random_state=42), id="DTree"),
    pytest.param(lambda: RandomForestClassifier(n_estimators=10, random_state=42), id="RF"),
]


# ---------------------------------------------------------------------------
# Small inline dataset helpers
# ---------------------------------------------------------------------------

def _small_binary_dataset(n=100, random_state=42):
    X, y = make_classification(
        n_samples=n, n_features=5, n_classes=2,
        weights=[0.6, 0.4], random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.3, random_state=random_state)


def _small_multiclass_dataset(n=150, random_state=42):
    X, y = make_classification(
        n_samples=n, n_features=10, n_classes=3,
        n_informative=6, weights=[0.3, 0.4, 0.3],
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.3, random_state=random_state)


def _tiny_binary_dataset(n=8, random_state=0):
    """Very small dataset (n < 10) for edge-case testing."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n, 2)
    y = np.array([0, 1] * (n // 2))[:n]
    return X, X, y, y  # train == test for simplicity


def _imbalanced_binary_dataset(n=200, minority_frac=0.05, random_state=42):
    """Extremely imbalanced binary dataset."""
    X, y = make_classification(
        n_samples=n, n_features=5, n_classes=2,
        weights=[1 - minority_frac, minority_frac],
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.3, random_state=random_state)


def _constant_predictions_dataset():
    """Dataset where all predictions will be the same class."""
    rng = np.random.RandomState(99)
    X = rng.randn(50, 3)
    y = np.zeros(50, dtype=int)  # all class 0
    y[0] = 1  # at least 2 classes to avoid errors during fit
    return X, X, y, y


# =========================================================================
# 1. CC – Classify and Count
# =========================================================================

class TestCC:
    """Tests for Classify and Count (CC)."""

    def test_binary_fit_predict(self, binary_dataset, binary_classifier):
        X_train, X_test, y_train, y_test = binary_dataset
        q = CC(learner=binary_classifier)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_multiclass_fit_predict(self, multiclass_dataset, multiclass_classifier):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = CC(learner=multiclass_classifier)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_aggregate_hard_labels(self):
        q = CC()
        preds = np.array([0, 0, 1, 1, 1])
        prev = q.aggregate(preds)
        assert prev[0] == pytest.approx(0.4)
        assert prev[1] == pytest.approx(0.6)

    def test_aggregate_with_probabilities_and_threshold(self):
        q = CC(threshold=0.5)
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.6, 0.4],
        ])
        prev = q.aggregate(probs)
        # threshold 0.5 → classes [0, 0, 1, 1, 0]
        assert prev[0] == pytest.approx(0.6)
        assert prev[1] == pytest.approx(0.4)

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_threshold_variations(self, threshold):
        q = CC(threshold=threshold)
        probs = np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5], [0.1, 0.9]])
        prev = q.aggregate(probs)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_pandas_input(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        X_train_df = pd.DataFrame(X_train)
        y_train_s = pd.Series(y_train)
        X_test_df = pd.DataFrame(X_test)
        q = CC(learner=LogisticRegression(random_state=42, solver="liblinear"))
        q.fit(X_train_df, y_train_s)
        prev = q.predict(X_test_df)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_string_labels(self):
        rng = np.random.RandomState(0)
        X = rng.randn(60, 3)
        y = np.where(rng.rand(60) > 0.5, "pos", "neg")
        q = CC(learner=LogisticRegression(random_state=42, solver="liblinear"))
        q.fit(X, y)
        prev = q.predict(X)
        assert isinstance(prev, dict)
        assert set(prev.keys()) == {"neg", "pos"}
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_output_keys_match_classes(self):
        q = CC()
        preds = np.array([0, 1, 2, 0, 1, 2])
        y_train = np.array([0, 1, 2, 0, 1, 2])
        prev = q.aggregate(preds, y_train)
        assert set(prev.keys()) == {0, 1, 2}

    @pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
    def test_different_learners(self, learner_factory, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = learner_factory()
        q = CC(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)


# =========================================================================
# 2. PCC – Probabilistic Classify and Count
# =========================================================================

class TestPCC:
    """Tests for Probabilistic Classify and Count (PCC)."""

    def test_binary_fit_predict(self, binary_dataset, binary_classifier):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PCC(learner=binary_classifier)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_multiclass_fit_predict(self, multiclass_dataset, multiclass_classifier):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = PCC(learner=multiclass_classifier)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_aggregate_probabilities(self):
        q = PCC()
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.4, 0.6],
        ])
        prev = q.aggregate(probs)
        assert prev[0] == pytest.approx(0.48)
        assert prev[1] == pytest.approx(0.52)

    def test_aggregate_3class(self):
        q = PCC()
        probs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ])
        prev = q.aggregate(probs)
        assert len(prev) == 3
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_pandas_input(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PCC(learner=LogisticRegression(random_state=42, solver="liblinear"))
        q.fit(pd.DataFrame(X_train), pd.Series(y_train))
        prev = q.predict(pd.DataFrame(X_test))
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
    def test_different_learners(self, learner_factory, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = learner_factory()
        q = PCC(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)


# =========================================================================
# 3. Matrix adjustment methods – AC, PAC, FM
# =========================================================================

class TestMatrixAdjustment:
    """Tests for AC, PAC, and FM quantifiers."""

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_binary_fit_predict(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_multiclass_fit_predict(self, Quantifier, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear", max_iter=200)
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_prevalence_values_in_range(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        for v in prev.values():
            assert 0.0 <= v <= 1.0

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    @pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
    def test_different_learners(self, Quantifier, learner_factory, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = learner_factory()
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_pandas_input(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(pd.DataFrame(X_train), pd.Series(y_train))
        prev = q.predict(pd.DataFrame(X_test))
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_output_keys_match_classes(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert set(prev.keys()) == set(np.unique(y_train))

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_string_labels(self, Quantifier):
        rng = np.random.RandomState(0)
        X = rng.randn(80, 4)
        y = np.where(rng.rand(80) > 0.5, "pos", "neg")
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X, y)
        prev = q.predict(X)
        assert isinstance(prev, dict)
        assert set(prev.keys()) == {"neg", "pos"}
        assert sum(prev.values()) == pytest.approx(1.0)


# =========================================================================
# 4. Threshold adjustment methods – TAC, TX, TMAX, T50, MS, MS2
# =========================================================================

class TestThresholdAdjustment:
    """Tests for TAC, TX, TMAX, T50, MS, and MS2 quantifiers."""

    @pytest.mark.parametrize("Quantifier", THRESHOLD_CLASSES)
    def test_binary_fit_predict(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", THRESHOLD_CLASSES)
    def test_multiclass_fit_predict(self, Quantifier, multiclass_dataset):
        """Threshold methods use @define_binary and should support multiclass via OVR."""
        X_train, X_test, y_train, y_test = multiclass_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear", max_iter=200)
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        with config_context(prevalence_normalization="sum"):
            prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", THRESHOLD_CLASSES)
    def test_prevalence_in_range(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        for v in prev.values():
            assert 0.0 <= v <= 1.0

    @pytest.mark.parametrize("Quantifier", THRESHOLD_CLASSES)
    @pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
    def test_different_learners(self, Quantifier, learner_factory):
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        learner = learner_factory()
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", THRESHOLD_CLASSES)
    def test_pandas_input(self, Quantifier):
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(pd.DataFrame(X_train), pd.Series(y_train))
        prev = q.predict(pd.DataFrame(X_test))
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", THRESHOLD_CLASSES)
    def test_output_keys_match_binary_classes(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert set(prev.keys()) == {0, 1}


# =========================================================================
# 5. CDE – CDE-Iterate
# =========================================================================

class TestCDE:
    """Tests for CDE-Iterate quantifier."""

    def test_binary_fit_predict(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = CDE(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_multiclass_fit_predict(self, multiclass_dataset):
        """CDE uses @define_binary so multiclass should work via OVR."""
        X_train, X_test, y_train, y_test = multiclass_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear", max_iter=200)
        q = CDE(learner=learner)
        q.fit(X_train, y_train)
        with config_context(prevalence_normalization="sum"):
            prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("learner_factory", LEARNER_FACTORIES)
    def test_different_learners(self, learner_factory):
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        learner = learner_factory()
        q = CDE(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_output_keys(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = CDE(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert set(prev.keys()) == {0, 1}

    @pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-8])
    def test_tolerance_variations(self, tol):
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = CDE(learner=learner, tol=tol)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("max_iter", [1, 10, 200])
    def test_max_iter_variations(self, max_iter):
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = CDE(learner=learner, max_iter=max_iter)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_pandas_input(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = CDE(learner=learner)
        q.fit(pd.DataFrame(X_train), pd.Series(y_train))
        prev = q.predict(pd.DataFrame(X_test))
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)


# =========================================================================
# 6. Multiclass strategy variations
# =========================================================================

class TestMulticlassStrategies:
    """Test OVR strategy for threshold-based and CDE quantifiers on multiclass data."""

    @pytest.mark.parametrize("Quantifier", THRESHOLD_CLASSES + [CDE])
    def test_ovr_strategy(self, Quantifier, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear", max_iter=200)
        q = Quantifier(learner=learner, strategy="ovr")
        q.fit(X_train, y_train)
        with config_context(prevalence_normalization="sum"):
            prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert sum(prev.values()) == pytest.approx(1.0)
        assert set(prev.keys()) == set(np.unique(y_train))


# =========================================================================
# 7. Edge cases
# =========================================================================

class TestEdgeCases:
    """Edge-case tests: tiny datasets, extreme imbalance, constant predictions."""

    @pytest.mark.parametrize("Quantifier", [CC, PCC])
    def test_tiny_dataset_count(self, Quantifier):
        """Fit/predict on n < 10 samples."""
        X_train, X_test, y_train, y_test = _tiny_binary_dataset()
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", [AC, PAC, FM])
    def test_tiny_dataset_matrix(self, Quantifier):
        """Fit/predict on n < 10 samples — matrix methods should still work."""
        X_train, X_test, y_train, y_test = _tiny_binary_dataset()
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train, cv=2)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", [TAC, TX, TMAX])
    def test_small_dataset_threshold(self, Quantifier):
        """Threshold methods need enough data for cross-validation."""
        X_train, X_test, y_train, y_test = _small_binary_dataset(n=50, random_state=7)
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", [CC, PCC])
    def test_extreme_imbalance_count(self, Quantifier):
        X_train, X_test, y_train, y_test = _imbalanced_binary_dataset()
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)
        for v in prev.values():
            assert 0.0 <= v <= 1.0

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_extreme_imbalance_matrix(self, Quantifier):
        X_train, X_test, y_train, y_test = _imbalanced_binary_dataset()
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", [TAC, TMAX])
    def test_extreme_imbalance_threshold(self, Quantifier):
        X_train, X_test, y_train, y_test = _imbalanced_binary_dataset()
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_constant_predictions_cc(self):
        """All predictions are the same class."""
        q = CC()
        preds = np.zeros(20, dtype=int)
        y_train = np.array([0, 1])  # ensure two classes known
        prev = q.aggregate(preds, y_train)
        assert prev[0] == pytest.approx(1.0)
        assert prev[1] == pytest.approx(0.0)

    def test_constant_predictions_pcc(self):
        """All probabilities heavily favour one class."""
        q = PCC()
        probs = np.column_stack([np.ones(20) * 0.99, np.ones(20) * 0.01])
        prev = q.aggregate(probs)
        assert prev[0] > 0.9
        assert prev[1] < 0.1
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_single_sample_cc(self):
        q = CC()
        preds = np.array([1])
        prev = q.aggregate(preds)
        assert prev[1] == pytest.approx(1.0)

    def test_single_sample_pcc(self):
        q = PCC()
        probs = np.array([[0.3, 0.7]])
        prev = q.aggregate(probs)
        assert prev[0] == pytest.approx(0.3)
        assert prev[1] == pytest.approx(0.7)


# =========================================================================
# 8. Prevalence normalization with config_context
# =========================================================================

class TestPrevalenceNormalization:
    """Test that prevalences sum to 1 using config_context normalization."""

    @pytest.mark.parametrize("Quantifier", THRESHOLD_CLASSES + [CDE])
    def test_normalization_multiclass(self, Quantifier, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear", max_iter=200)
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        with config_context(prevalence_normalization="sum"):
            prev = q.predict(X_test)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_normalization_matrix_multiclass(self, Quantifier, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear", max_iter=200)
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert sum(prev.values()) == pytest.approx(1.0)


# =========================================================================
# 9. Utility functions
# =========================================================================

class TestUtils:
    """Tests for adjust_counting._utils helpers."""

    def test_compute_table(self):
        y = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        classes = np.array([0, 1])
        TP, FP, FN, TN = compute_table(y, y_pred, classes)
        assert TP == 2
        assert FP == 1
        assert FN == 1
        assert TN == 1

    def test_compute_tpr(self):
        assert compute_tpr(3, 1) == pytest.approx(0.75)
        assert compute_tpr(0, 0) == 0  # edge case

    def test_compute_fpr(self):
        assert compute_fpr(1, 4) == pytest.approx(0.2)
        assert compute_fpr(0, 0) == 0  # edge case

    def test_evaluate_thresholds_shape(self):
        rng = np.random.RandomState(42)
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        probs = rng.rand(8)
        thresholds, tprs, fprs = evaluate_thresholds(y, probs)
        assert len(thresholds) == 101
        assert len(tprs) == 101
        assert len(fprs) == 101

    def test_evaluate_thresholds_extremes(self):
        y = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        thresholds, tprs, fprs = evaluate_thresholds(y, probs)
        # At threshold 0 everything is predicted positive
        assert tprs[0] == pytest.approx(1.0)
        assert fprs[0] == pytest.approx(1.0)
        # At threshold 1 everything is predicted negative
        assert tprs[-1] == pytest.approx(0.0)
        assert fprs[-1] == pytest.approx(0.0)

    def test_compute_table_all_correct(self):
        y = np.array([0, 0, 1, 1])
        classes = np.array([0, 1])
        TP, FP, FN, TN = compute_table(y, y, classes)
        assert TP == 2
        assert TN == 2
        assert FP == 0
        assert FN == 0

    def test_compute_table_all_wrong(self):
        y = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        classes = np.array([0, 1])
        TP, FP, FN, TN = compute_table(y, y_pred, classes)
        assert TP == 0
        assert TN == 0
        assert FP == 2
        assert FN == 2


# =========================================================================
# 10. Label type variations across all classes
# =========================================================================

class TestLabelTypeVariations:
    """Test different label encodings: int, string, float."""

    @pytest.mark.parametrize("Quantifier", [CC, PCC])
    def test_int_labels(self, Quantifier):
        rng = np.random.RandomState(1)
        X = rng.randn(80, 4)
        y = rng.randint(0, 2, 80)
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X, y)
        prev = q.predict(X)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_string_labels_cc(self):
        rng = np.random.RandomState(2)
        X = rng.randn(80, 4)
        y = np.where(rng.rand(80) > 0.5, "pos", "neg")
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = CC(learner=learner)
        q.fit(X, y)
        prev = q.predict(X)
        assert isinstance(prev, dict)
        assert set(prev.keys()) == {"neg", "pos"}

    def test_string_labels_pcc(self):
        """PCC uses predict_proba; keys are column indices, not label names."""
        rng = np.random.RandomState(2)
        X = rng.randn(80, 4)
        y = np.where(rng.rand(80) > 0.5, "pos", "neg")
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = PCC(learner=learner)
        q.fit(X, y)
        prev = q.predict(X)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_int_labels_matrix(self, Quantifier):
        rng = np.random.RandomState(3)
        X = rng.randn(100, 5)
        y = rng.randint(0, 2, 100)
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X, y)
        prev = q.predict(X)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)


# =========================================================================
# 11. Aggregate method direct tests
# =========================================================================

class TestAggregateDirectly:
    """Test the aggregate method independently from fit/predict."""

    def test_cc_aggregate_with_y_train(self):
        q = CC()
        preds = np.array([0, 0, 1, 1, 0])
        y_train = np.array([0, 1, 0, 1, 0])
        prev = q.aggregate(preds, y_train)
        assert prev[0] == pytest.approx(0.6)
        assert prev[1] == pytest.approx(0.4)

    def test_pcc_aggregate_uniform(self):
        q = PCC()
        probs = np.array([[0.5, 0.5]] * 10)
        prev = q.aggregate(probs)
        assert prev[0] == pytest.approx(0.5)
        assert prev[1] == pytest.approx(0.5)

    def test_cc_aggregate_multiclass(self):
        q = CC()
        preds = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2])
        y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        prev = q.aggregate(preds, y_train)
        assert len(prev) == 3
        assert sum(prev.values()) == pytest.approx(1.0)

    def test_pcc_aggregate_2d_two_columns(self):
        """PCC aggregate with 2D probability array."""
        q = PCC()
        probs = np.array([[0.4, 0.6], [0.3, 0.7], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
        prev = q.aggregate(probs)
        assert len(prev) == 2
        assert sum(prev.values()) == pytest.approx(1.0)
        assert prev[1] == pytest.approx(0.5)


# =========================================================================
# 12. Cross-validation parameter tests for adjustment methods
# =========================================================================

class TestCVParameters:
    """Test cv, stratified, random_state, shuffle parameters in fit."""

    @pytest.mark.parametrize("cv", [2, 3, 5])
    @pytest.mark.parametrize("Quantifier", [AC, PAC])
    def test_cv_folds(self, Quantifier, cv, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train, cv=cv)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("Quantifier", [AC, PAC, FM])
    def test_learner_fitted_flag(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        learner.fit(X_train, y_train)
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train, learner_fitted=True)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)


# =========================================================================
# 13. Specific threshold value tests for TAC
# =========================================================================

class TestTACThreshold:
    """Test TAC with various threshold values."""

    @pytest.mark.parametrize("threshold", [0.25, 0.5])
    def test_threshold_values(self, threshold, binary_dataset):
        """TAC threshold must exist in linspace(0,1,101). 0.25 and 0.5 are safe choices."""
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = TAC(learner=learner, threshold=threshold)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0)


# =========================================================================
# 14. Consistency tests – fit_predict if available
# =========================================================================

class TestConsistency:
    """Consistency checks: predict returns same result on repeated calls."""

    @pytest.mark.parametrize("Quantifier", [CC, PCC])
    def test_predict_deterministic_count(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev1 = q.predict(X_test)
        prev2 = q.predict(X_test)
        assert prev1 == prev2

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_predict_deterministic_matrix(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        q.fit(X_train, y_train)
        prev1 = q.predict(X_test)
        prev2 = q.predict(X_test)
        assert prev1 == prev2


# =========================================================================
# 15. Error cases
# =========================================================================

class TestErrorCases:
    """Test expected errors and boundary conditions."""

    @pytest.mark.parametrize("Quantifier", [CC, PCC])
    def test_unfitted_predict_raises(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        with pytest.raises(Exception):
            q.predict(X_test)

    @pytest.mark.parametrize("Quantifier", MATRIX_CLASSES)
    def test_unfitted_predict_raises_matrix(self, Quantifier, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        learner = LogisticRegression(random_state=42, solver="liblinear")
        q = Quantifier(learner=learner)
        with pytest.raises(Exception):
            q.predict(X_test)

    def test_cc_no_learner_predict_raises(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = CC()
        with pytest.raises(Exception):
            q.predict(X_test)

    def test_pcc_no_learner_predict_raises(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PCC()
        with pytest.raises(Exception):
            q.predict(X_test)


# =========================================================================
# 16. Parametrized fit-predict over ALL quantifiers with binary data
# =========================================================================

@pytest.mark.parametrize("Quantifier", [CC, PCC, AC, PAC, FM, TAC, TX, TMAX, T50, MS, MS2, CDE])
def test_all_quantifiers_binary(Quantifier, binary_dataset):
    """Every quantifier should fit + predict correctly on binary data."""
    X_train, X_test, y_train, y_test = binary_dataset
    learner = LogisticRegression(random_state=42, solver="liblinear")
    q = Quantifier(learner=learner)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert sum(prev.values()) == pytest.approx(1.0)
    for v in prev.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.parametrize("Quantifier", [CC, PCC, AC, PAC, FM])
def test_all_quantifiers_multiclass(Quantifier, multiclass_dataset):
    """Quantifiers without @define_binary should directly handle multiclass."""
    X_train, X_test, y_train, y_test = multiclass_dataset
    learner = LogisticRegression(random_state=42, solver="liblinear", max_iter=200)
    q = Quantifier(learner=learner)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 3
    assert sum(prev.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("Quantifier", [TAC, TX, TMAX, T50, MS, MS2, CDE])
def test_binary_defined_quantifiers_multiclass(Quantifier, multiclass_dataset):
    """@define_binary quantifiers on multiclass data need normalization."""
    X_train, X_test, y_train, y_test = multiclass_dataset
    learner = LogisticRegression(random_state=42, solver="liblinear", max_iter=200)
    q = Quantifier(learner=learner)
    q.fit(X_train, y_train)
    with config_context(prevalence_normalization="sum"):
        prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 3
    assert sum(prev.values()) == pytest.approx(1.0)


# =========================================================================
# 17. DataFrame-only pipeline – end to end
# =========================================================================

@pytest.mark.parametrize("Quantifier", [CC, PCC, AC, PAC, FM])
def test_full_pandas_pipeline(Quantifier):
    """Full pipeline using only pandas objects."""
    rng = np.random.RandomState(7)
    X = pd.DataFrame(rng.randn(120, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.randint(0, 2, 120))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    learner = LogisticRegression(random_state=42, solver="liblinear")
    q = Quantifier(learner=learner)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert sum(prev.values()) == pytest.approx(1.0)
