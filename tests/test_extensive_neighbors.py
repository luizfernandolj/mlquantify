"""
Comprehensive tests for the mlquantify.neighbors module.

Covers:
- PWK quantifier (binary + multiclass, various parameters)
- PWKCLF classifier
- KDEyML, KDEyHD, KDEyCS quantifiers
- Utility functions: gaussian_kernel, negative_log_likelihood, _simplex_constraints
- Edge cases, error handling, input types, label types
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from mlquantify import config_context
from mlquantify.neighbors import PWK, KDEyML, KDEyHD, KDEyCS
from mlquantify.neighbors._classification import PWKCLF
from mlquantify.neighbors._utils import (
    gaussian_kernel,
    negative_log_likelihood,
    _simplex_constraints,
)


# ============================================================
# Helpers
# ============================================================

def _make_tiny_binary(n=30, random_state=0):
    """Small binary dataset for quick tests."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n, 4)
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    return X, y


def _make_tiny_multiclass(n=60, n_classes=3, random_state=0):
    """Small multiclass dataset for quick tests."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n, 4)
    y = np.array([c for c in range(n_classes) for _ in range(n // n_classes)])
    return X, y


def _make_string_label_binary(n=40, random_state=0):
    """Binary dataset with string labels."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n, 4)
    y = np.array(["cat"] * (n // 2) + ["dog"] * (n - n // 2))
    return X, y


def _make_string_label_multiclass(n=60, random_state=0):
    """Multiclass dataset with string labels."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n, 5)
    y = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
    return X, y


def _make_imbalanced_binary(n=100, minority_frac=0.05, random_state=42):
    """Extremely imbalanced binary dataset."""
    rng = np.random.RandomState(random_state)
    n_min = max(2, int(n * minority_frac))
    n_maj = n - n_min
    X = rng.randn(n, 4)
    y = np.array([0] * n_maj + [1] * n_min)
    return X, y


# ============================================================
# PWK – basic binary
# ============================================================

class TestPWKBinary:

    def test_fit_predict_basic(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PWK(n_neighbors=5)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_classify_returns_valid_labels(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PWK(n_neighbors=5)
        q.fit(X_train, y_train)
        labels = q.classify(X_test)
        assert len(labels) == len(X_test)
        assert set(np.unique(labels)).issubset(set(np.unique(y_train)))

    def test_prevalence_keys_match_classes(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PWK(n_neighbors=7)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert set(prev.keys()) == set(np.unique(y_train))

    def test_config_context_array_return(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PWK(n_neighbors=5)
        q.fit(X_train, y_train)
        with config_context(prevalence_return_type="array"):
            prev = q.predict(X_test)
            assert isinstance(prev, np.ndarray)
            assert pytest.approx(np.sum(prev), abs=1e-6) == 1.0


# ============================================================
# PWK – basic multiclass
# ============================================================

class TestPWKMulticlass:

    def test_fit_predict_multiclass(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = PWK(n_neighbors=5)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_classify_multiclass(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = PWK(n_neighbors=5)
        q.fit(X_train, y_train)
        labels = q.classify(X_test)
        assert len(labels) == len(X_test)
        assert set(np.unique(labels)).issubset(set(np.unique(y_train)))


# ============================================================
# PWK – parameter variations
# ============================================================

class TestPWKParams:

    @pytest.mark.parametrize("k", [1, 3, 5, 10, 20])
    def test_various_k(self, binary_dataset, k):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PWK(n_neighbors=k)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("alpha", [1, 2, 5, 10])
    def test_various_alpha(self, binary_dataset, alpha):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PWK(alpha=alpha, n_neighbors=5)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
    def test_various_algorithms(self, binary_dataset, algorithm):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PWK(n_neighbors=5, algorithm=algorithm)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev", "minkowski"])
    def test_various_metrics(self, binary_dataset, metric):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PWK(n_neighbors=5, metric=metric)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_minkowski_p(self, binary_dataset, p):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PWK(n_neighbors=5, metric="minkowski", p=p)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0


# ============================================================
# PWK – input type variations
# ============================================================

class TestPWKInputTypes:

    def test_numpy_arrays(self):
        X, y = _make_tiny_binary(50)
        X_train, X_test = X[:35], X[35:]
        y_train = y[:35]
        q = PWK(n_neighbors=5)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_pandas_dataframes(self):
        X, y = _make_tiny_binary(50)
        X_train_df = pd.DataFrame(X[:35], columns=[f"f{i}" for i in range(X.shape[1])])
        X_test_df = pd.DataFrame(X[35:], columns=[f"f{i}" for i in range(X.shape[1])])
        y_train_s = pd.Series(y[:35])
        q = PWK(n_neighbors=5)
        q.fit(X_train_df, y_train_s)
        prev = q.predict(X_test_df)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_string_labels_binary(self):
        X, y = _make_string_label_binary(50)
        X_train, X_test = X[:35], X[35:]
        y_train = y[:35]
        q = PWK(n_neighbors=5)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert set(prev.keys()) == {"cat", "dog"}
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_string_labels_multiclass(self):
        X, y = _make_string_label_multiclass(60)
        X_train, X_test = X[:45], X[45:]
        y_train = y[:45]
        q = PWK(n_neighbors=5)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert set(prev.keys()) == {"A", "B", "C"}
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0


# ============================================================
# PWK – edge cases
# ============================================================

class TestPWKEdgeCases:

    def test_k_greater_than_n_samples(self):
        """PWKCLF adjusts k when k > n_samples, so this should work."""
        X, y = _make_tiny_binary(10)
        q = PWK(n_neighbors=50)
        q.fit(X[:7], y[:7])
        prev = q.predict(X[7:])
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_extreme_imbalance(self):
        X, y = _make_imbalanced_binary(100, minority_frac=0.05)
        X_train, X_test = X[:80], X[80:]
        y_train = y[:80]
        q = PWK(n_neighbors=5)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0
        # All prevalences non-negative
        for v in prev.values():
            assert v >= 0.0

    def test_single_test_sample(self):
        X, y = _make_tiny_binary(30)
        q = PWK(n_neighbors=5)
        q.fit(X[:25], y[:25])
        prev = q.predict(X[25:26])
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_very_small_training_set(self):
        X, y = _make_tiny_binary(6)
        q = PWK(n_neighbors=3)
        q.fit(X[:4], y[:4])
        prev = q.predict(X[4:])
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0


# ============================================================
# PWK – unfitted / error cases
# ============================================================

class TestPWKErrors:

    def test_predict_unfitted_raises(self):
        q = PWK(n_neighbors=5)
        X_test = np.random.randn(10, 4)
        with pytest.raises(Exception):
            q.predict(X_test)

    def test_classify_unfitted_raises(self):
        q = PWK(n_neighbors=5)
        X_test = np.random.randn(10, 4)
        with pytest.raises(Exception):
            q.classify(X_test)


# ============================================================
# PWKCLF – direct classifier tests
# ============================================================

class TestPWKCLF:

    def test_fit_predict_binary(self):
        X, y = _make_tiny_binary(40)
        clf = PWKCLF(n_neighbors=5)
        clf.fit(X[:30], y[:30])
        preds = clf.predict(X[30:])
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})

    def test_fit_predict_multiclass(self):
        X, y = _make_tiny_multiclass(60)
        clf = PWKCLF(n_neighbors=5)
        clf.fit(X[:45], y[:45])
        preds = clf.predict(X[45:])
        assert len(preds) == 15
        assert set(preds).issubset({0, 1, 2})

    @pytest.mark.parametrize("alpha", [1, 2, 5])
    def test_alpha_affects_weights(self, alpha):
        X, y = _make_tiny_binary(40)
        clf = PWKCLF(alpha=alpha, n_neighbors=5)
        clf.fit(X[:30], y[:30])
        # class_weights should exist after fit
        assert clf.class_weights is not None
        assert len(clf.class_weights) == 2
        # Weights should be positive
        assert np.all(clf.class_weights > 0)

    def test_classes_attribute(self):
        X, y = _make_tiny_binary(40)
        clf = PWKCLF(n_neighbors=5)
        clf.fit(X[:30], y[:30])
        assert clf.classes_ is not None
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))

    def test_class_to_index_mapping(self):
        X, y = _make_tiny_multiclass(60, n_classes=3)
        clf = PWKCLF(n_neighbors=5)
        clf.fit(X[:45], y[:45])
        assert clf.class_to_index == {0: 0, 1: 1, 2: 2}

    def test_k_adjusted_when_larger_than_n(self):
        """When n_neighbors > n_samples, PWKCLF adjusts k internally."""
        X, y = _make_tiny_binary(10)
        clf = PWKCLF(n_neighbors=100)
        clf.fit(X[:5], y[:5])
        preds = clf.predict(X[5:])
        assert len(preds) == 5

    def test_string_labels(self):
        X, y = _make_string_label_binary(40)
        clf = PWKCLF(n_neighbors=5)
        clf.fit(X[:30], y[:30])
        preds = clf.predict(X[30:])
        assert set(preds).issubset({"cat", "dog"})

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
    def test_various_metrics(self, metric):
        X, y = _make_tiny_binary(40)
        clf = PWKCLF(n_neighbors=5, metric=metric)
        clf.fit(X[:30], y[:30])
        preds = clf.predict(X[30:])
        assert len(preds) == 10


# ============================================================
# KDEyML – Maximum Likelihood
# ============================================================

class TestKDEyML:

    @pytest.fixture()
    def fitted_binary_lr(self):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        return X[:60], X[60:], y[:60], y[60:], lr

    @pytest.fixture()
    def fitted_multiclass_lr(self):
        X, y = _make_tiny_multiclass(90, n_classes=3)
        lr = LogisticRegression(random_state=0, solver="liblinear", max_iter=500)
        return X[:70], X[70:], y[:70], y[70:], lr

    def test_fit_predict_binary(self, fitted_binary_lr):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_fit_predict_multiclass(self, fitted_multiclass_lr):
        X_train, X_test, y_train, y_test, lr = fitted_multiclass_lr
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    @pytest.mark.parametrize("bw", [0.01, 0.1, 0.5, 1.0])
    def test_various_bandwidths(self, fitted_binary_lr, bw):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyML(learner=lr, bandwidth=bw)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    @pytest.mark.parametrize("kernel", ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"])
    def test_various_kernels(self, fitted_binary_lr, kernel):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyML(learner=lr, bandwidth=0.2, kernel=kernel)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_prevalence_keys_match_classes(self, fitted_binary_lr):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert set(prev.keys()) == set(np.unique(y_train))

    def test_array_return_type(self, fitted_binary_lr):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        with config_context(prevalence_return_type="array"):
            prev = q.predict(X_test)
            assert isinstance(prev, np.ndarray)
            assert pytest.approx(np.sum(prev), abs=1e-5) == 1.0

    def test_learner_fitted_flag(self, fitted_binary_lr):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        lr.fit(X_train, y_train)
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train, learner_fitted=True)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_predict_unfitted_raises(self):
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyML(learner=lr, bandwidth=0.1)
        X_test = np.random.randn(10, 4)
        with pytest.raises(Exception):
            q.predict(X_test)


# ============================================================
# KDEyHD – Hellinger Distance
# ============================================================

class TestKDEyHD:

    @pytest.fixture()
    def fitted_binary_lr(self):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        return X[:60], X[60:], y[:60], y[60:], lr

    @pytest.fixture()
    def fitted_multiclass_lr(self):
        X, y = _make_tiny_multiclass(90, n_classes=3)
        lr = LogisticRegression(random_state=0, solver="liblinear", max_iter=500)
        return X[:70], X[70:], y[:70], y[70:], lr

    def test_fit_predict_binary(self, fitted_binary_lr):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyHD(learner=lr, bandwidth=0.1, montecarlo_trials=500, random_state=42)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_fit_predict_multiclass(self, fitted_multiclass_lr):
        X_train, X_test, y_train, y_test, lr = fitted_multiclass_lr
        q = KDEyHD(learner=lr, bandwidth=0.1, montecarlo_trials=500, random_state=42)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    @pytest.mark.parametrize("bw", [0.05, 0.1, 0.5])
    def test_various_bandwidths(self, fitted_binary_lr, bw):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyHD(learner=lr, bandwidth=bw, montecarlo_trials=500, random_state=42)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    @pytest.mark.parametrize("mc_trials", [100, 500, 2000])
    def test_various_montecarlo_trials(self, fitted_binary_lr, mc_trials):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyHD(learner=lr, bandwidth=0.1, montecarlo_trials=mc_trials, random_state=42)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_random_state_reproducibility(self, fitted_binary_lr):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q1 = KDEyHD(learner=lr, bandwidth=0.1, montecarlo_trials=500, random_state=42)
        q1.fit(X_train, y_train)
        prev1 = q1.predict(X_test)

        q2 = KDEyHD(learner=lr, bandwidth=0.1, montecarlo_trials=500, random_state=42)
        q2.fit(X_train, y_train)
        prev2 = q2.predict(X_test)

        for k in prev1:
            assert pytest.approx(prev1[k], abs=1e-4) == prev2[k]

    def test_prevalence_keys_match_classes(self, fitted_binary_lr):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyHD(learner=lr, bandwidth=0.1, montecarlo_trials=500, random_state=42)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert set(prev.keys()) == set(np.unique(y_train))

    def test_predict_unfitted_raises(self):
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyHD(learner=lr, bandwidth=0.1)
        X_test = np.random.randn(10, 4)
        with pytest.raises(Exception):
            q.predict(X_test)


# ============================================================
# KDEyCS – Cauchy–Schwarz
# ============================================================

class TestKDEyCS:

    @pytest.fixture()
    def fitted_binary_lr(self):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        return X[:60], X[60:], y[:60], y[60:], lr

    @pytest.fixture()
    def fitted_multiclass_lr(self):
        X, y = _make_tiny_multiclass(90, n_classes=3)
        lr = LogisticRegression(random_state=0, solver="liblinear", max_iter=500)
        return X[:70], X[70:], y[:70], y[70:], lr

    def test_fit_predict_binary(self, fitted_binary_lr):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyCS(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_fit_predict_multiclass(self, fitted_multiclass_lr):
        X_train, X_test, y_train, y_test, lr = fitted_multiclass_lr
        q = KDEyCS(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    @pytest.mark.parametrize("bw", [0.05, 0.1, 0.5, 1.0])
    def test_various_bandwidths(self, fitted_binary_lr, bw):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyCS(learner=lr, bandwidth=bw)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    @pytest.mark.parametrize("kernel", ["gaussian", "tophat", "epanechnikov"])
    def test_various_kernels(self, fitted_binary_lr, kernel):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyCS(learner=lr, bandwidth=0.2, kernel=kernel)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_prevalence_keys_match_classes(self, fitted_binary_lr):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyCS(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert set(prev.keys()) == set(np.unique(y_train))

    def test_array_return_type(self, fitted_binary_lr):
        X_train, X_test, y_train, y_test, lr = fitted_binary_lr
        q = KDEyCS(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        with config_context(prevalence_return_type="array"):
            prev = q.predict(X_test)
            assert isinstance(prev, np.ndarray)
            assert pytest.approx(np.sum(prev), abs=1e-5) == 1.0

    def test_predict_unfitted_raises(self):
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyCS(learner=lr, bandwidth=0.1)
        X_test = np.random.randn(10, 4)
        with pytest.raises(Exception):
            q.predict(X_test)


# ============================================================
# KDE quantifiers with session-scoped fixtures
# ============================================================

class TestKDEWithSessionFixtures:
    """Use the session-scoped binary_dataset and multiclass_dataset fixtures."""

    def test_kdeyml_binary_session(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_kdeyml_multiclass_session(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        lr = LogisticRegression(random_state=0, solver="liblinear", max_iter=500)
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 3
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_kdeycs_binary_session(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyCS(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_kdeyhd_binary_session(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyHD(learner=lr, bandwidth=0.1, montecarlo_trials=500, random_state=42)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# ============================================================
# KDE – learner_fitted=True path
# ============================================================

class TestKDELearnerFitted:

    def test_kdeycs_learner_fitted(self):
        X, y = _make_tiny_binary(80)
        X_train, X_test, y_train, y_test = X[:60], X[60:], y[:60], y[60:]
        lr = LogisticRegression(random_state=0, solver="liblinear")
        lr.fit(X_train, y_train)
        q = KDEyCS(learner=lr, bandwidth=0.1)
        q.fit(X_train, y_train, learner_fitted=True)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0

    def test_kdeyhd_learner_fitted(self):
        X, y = _make_tiny_binary(80)
        X_train, X_test, y_train, y_test = X[:60], X[60:], y[:60], y[60:]
        lr = LogisticRegression(random_state=0, solver="liblinear")
        lr.fit(X_train, y_train)
        q = KDEyHD(learner=lr, bandwidth=0.1, montecarlo_trials=500, random_state=42)
        q.fit(X_train, y_train, learner_fitted=True)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# ============================================================
# Utility functions
# ============================================================

class TestGaussianKernel:

    def test_identical_points_high_similarity(self):
        X = np.array([[1.0, 2.0]])
        K = gaussian_kernel(X, X, bandwidth=1.0)
        assert K.shape == (1, 1)
        assert K[0, 0] > 0

    def test_distant_points_low_similarity(self):
        X = np.array([[0.0, 0.0]])
        Y = np.array([[100.0, 100.0]])
        K = gaussian_kernel(X, Y, bandwidth=0.1)
        assert K[0, 0] < 1e-10

    def test_symmetric(self):
        rng = np.random.RandomState(0)
        X = rng.randn(5, 3)
        K = gaussian_kernel(X, X, bandwidth=1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_shape(self):
        X = np.random.randn(10, 3)
        Y = np.random.randn(7, 3)
        K = gaussian_kernel(X, Y, bandwidth=0.5)
        assert K.shape == (10, 7)

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0])
        K = gaussian_kernel(X, X, bandwidth=1.0)
        # atleast_2d converts 1D to (1, 3)
        assert K.ndim == 2

    @pytest.mark.parametrize("bw", [0.01, 0.1, 1.0, 10.0])
    def test_all_positive(self, bw):
        rng = np.random.RandomState(42)
        X = rng.randn(5, 2)
        Y = rng.randn(3, 2)
        K = gaussian_kernel(X, Y, bandwidth=bw)
        assert np.all(K >= 0)


class TestNegativeLogLikelihood:

    def test_uniform_likelihood(self):
        likes = np.ones(10)
        nll = negative_log_likelihood(likes)
        assert pytest.approx(nll, abs=1e-10) == 0.0

    def test_positive_output(self):
        likes = np.array([0.5, 0.3, 0.2])
        nll = negative_log_likelihood(likes)
        assert nll > 0

    def test_very_small_values_clipped(self):
        likes = np.array([1e-20, 1e-30, 0.0])
        nll = negative_log_likelihood(likes)
        assert np.isfinite(nll)

    def test_all_ones(self):
        nll = negative_log_likelihood(np.ones(100))
        assert pytest.approx(nll, abs=1e-10) == 0.0


class TestSimplexConstraints:

    def test_returns_constraints_and_bounds(self):
        cons, bounds = _simplex_constraints(3)
        assert len(cons) == 1
        assert cons[0]["type"] == "eq"
        assert len(bounds) == 3
        assert all(b == (0.0, 1.0) for b in bounds)

    def test_constraint_satisfied_at_uniform(self):
        cons, bounds = _simplex_constraints(4)
        alpha = np.ones(4) / 4
        assert pytest.approx(cons[0]["fun"](alpha), abs=1e-12) == 0.0

    def test_constraint_violated_off_simplex(self):
        cons, _ = _simplex_constraints(3)
        alpha = np.array([0.5, 0.5, 0.5])
        assert cons[0]["fun"](alpha) != 0.0


# ============================================================
# KDE – best_distance attribute
# ============================================================

class TestBestDistance:

    def test_kdeyml_stores_best_distance(self):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X[:60], y[:60])
        _ = q.predict(X[60:])
        assert q.best_distance is not None
        assert np.isfinite(q.best_distance)

    def test_kdeycs_stores_best_distance(self):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyCS(learner=lr, bandwidth=0.1)
        q.fit(X[:60], y[:60])
        _ = q.predict(X[60:])
        assert q.best_distance is not None
        assert np.isfinite(q.best_distance)

    def test_kdeyhd_stores_best_distance(self):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyHD(learner=lr, bandwidth=0.1, montecarlo_trials=500, random_state=42)
        q.fit(X[:60], y[:60])
        _ = q.predict(X[60:])
        assert q.best_distance is not None
        assert np.isfinite(q.best_distance)


# ============================================================
# PWK + KDE – pandas DataFrame input
# ============================================================

class TestPandasInput:

    def test_pwk_pandas_binary(self):
        X, y = _make_tiny_binary(50)
        X_train_df = pd.DataFrame(X[:35])
        X_test_df = pd.DataFrame(X[35:])
        y_train_s = pd.Series(y[:35])
        q = PWK(n_neighbors=5)
        q.fit(X_train_df, y_train_s)
        prev = q.predict(X_test_df)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_kdeyml_pandas(self):
        X, y = _make_tiny_binary(80)
        X_train_df = pd.DataFrame(X[:60])
        X_test_df = pd.DataFrame(X[60:])
        y_train_s = pd.Series(y[:60])
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X_train_df, y_train_s)
        prev = q.predict(X_test_df)
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# ============================================================
# Cross-validation cv parameter in KDE
# ============================================================

class TestKDECVParam:

    @pytest.mark.parametrize("cv", [2, 3, 5])
    def test_kdeyml_various_cv(self, cv):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X[:60], y[:60], cv=cv)
        prev = q.predict(X[60:])
        assert isinstance(prev, dict)
        assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# ============================================================
# Normalization strategies via config_context
# ============================================================

class TestNormalizationStrategies:

    @pytest.mark.parametrize("norm", ["sum", "mean", "softmax"])
    def test_pwk_normalization_strategies(self, norm):
        X, y = _make_tiny_binary(50)
        q = PWK(n_neighbors=5)
        q.fit(X[:35], y[:35])
        with config_context(prevalence_normalization=norm):
            prev = q.predict(X[35:])
            assert isinstance(prev, dict)
            assert pytest.approx(sum(prev.values()), abs=1e-4) == 1.0

    @pytest.mark.parametrize("norm", ["sum", "mean"])
    def test_kdeyml_normalization_strategies(self, norm):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X[:60], y[:60])
        with config_context(prevalence_normalization=norm):
            prev = q.predict(X[60:])
            assert isinstance(prev, dict)
            assert pytest.approx(sum(prev.values()), abs=1e-4) == 1.0


# ============================================================
# PWK get_params / set_params (sklearn interface)
# ============================================================

class TestPWKSklearnInterface:

    def test_get_params(self):
        q = PWK(n_neighbors=7, alpha=2, metric="manhattan")
        params = q.get_params()
        assert params["n_neighbors"] == 7
        assert params["alpha"] == 2
        assert params["metric"] == "manhattan"

    def test_set_params(self):
        q = PWK(n_neighbors=5)
        q.set_params(n_neighbors=10, alpha=3)
        assert q.n_neighbors == 10
        assert q.alpha == 3


# ============================================================
# Multiple predict calls (stability)
# ============================================================

class TestMultiplePredictions:

    def test_pwk_multiple_predicts(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = PWK(n_neighbors=5)
        q.fit(X_train, y_train)
        prev1 = q.predict(X_test)
        prev2 = q.predict(X_test)
        for k in prev1:
            assert pytest.approx(prev1[k], abs=1e-10) == prev2[k]

    def test_kdeyml_multiple_predicts(self):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X[:60], y[:60])
        prev1 = q.predict(X[60:])
        prev2 = q.predict(X[60:])
        for k in prev1:
            assert pytest.approx(prev1[k], abs=1e-6) == prev2[k]


# ============================================================
# All prevalences non-negative
# ============================================================

class TestNonNegativePrevalences:

    @pytest.mark.parametrize("QuantifierClass,extra_kwargs", [
        (PWK, {"n_neighbors": 5}),
    ])
    def test_prevalences_non_negative(self, binary_dataset, QuantifierClass, extra_kwargs):
        X_train, X_test, y_train, y_test = binary_dataset
        q = QuantifierClass(**extra_kwargs)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        for v in prev.values():
            assert v >= 0.0

    def test_kdeyml_prevalences_non_negative(self):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyML(learner=lr, bandwidth=0.1)
        q.fit(X[:60], y[:60])
        prev = q.predict(X[60:])
        for v in prev.values():
            assert v >= 0.0

    def test_kdeycs_prevalences_non_negative(self):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyCS(learner=lr, bandwidth=0.1)
        q.fit(X[:60], y[:60])
        prev = q.predict(X[60:])
        for v in prev.values():
            assert v >= 0.0

    def test_kdeyhd_prevalences_non_negative(self):
        X, y = _make_tiny_binary(80)
        lr = LogisticRegression(random_state=0, solver="liblinear")
        q = KDEyHD(learner=lr, bandwidth=0.1, montecarlo_trials=500, random_state=42)
        q.fit(X[:60], y[:60])
        prev = q.predict(X[60:])
        for v in prev.values():
            assert v >= 0.0
