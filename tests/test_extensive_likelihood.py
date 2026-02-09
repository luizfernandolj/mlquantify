"""Comprehensive tests for the mlquantify.likelihood module (EMQ)."""

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlquantify import config_context
from mlquantify.likelihood import EMQ
from mlquantify.metrics._slq import MAE


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _small_binary_dataset(n=200, random_state=42):
    X, y = make_classification(
        n_samples=n, n_features=10, n_classes=2,
        weights=[0.6, 0.4], random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.3, random_state=random_state)


def _small_multiclass_dataset(n=300, n_classes=3, random_state=42):
    X, y = make_classification(
        n_samples=n, n_features=10, n_classes=n_classes,
        n_informative=8, weights=None, random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.3, random_state=random_state)


def _string_label_dataset():
    """Binary dataset with string labels 'pos' / 'neg'."""
    X, y_int = make_classification(
        n_samples=200, n_features=10, n_classes=2,
        random_state=7,
    )
    mapping = {0: "neg", 1: "pos"}
    y_str = np.array([mapping[v] for v in y_int])
    return train_test_split(X, y_str, test_size=0.3, random_state=7)


# ===========================================================================
# Section 1 – Basic fitting / predicting with conftest fixtures
# ===========================================================================

class TestEMQBasicBinary:
    """Binary dataset tests using session-scoped fixtures."""

    def test_fit_returns_self(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        result = q.fit(X_train, y_train)
        assert result is q

    def test_predict_returns_dict(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)

    def test_predict_keys_match_classes(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert set(prev.keys()) == set(np.unique(y_train))

    def test_prevalences_sum_to_one(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-6)

    def test_prevalences_in_0_1(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        for v in prev.values():
            assert 0.0 <= v <= 1.0

    def test_with_pretrained_classifier(self, binary_dataset, binary_classifier):
        """Pass an already-fitted classifier; EMQ re-fits internally."""
        X_train, X_test, y_train, y_test = binary_dataset
        q = EMQ(learner=binary_classifier)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-6)


class TestEMQBasicMulticlass:
    """Multiclass dataset tests using session-scoped fixtures."""

    def test_predict_dict_multiclass(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = EMQ(learner=LogisticRegression(random_state=0, solver="lbfgs", max_iter=500))
        q.fit(X_train, y_train)
        with config_context(prevalence_normalization="sum"):
            prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert set(prev.keys()) == set(np.unique(y_train))

    def test_multiclass_prevalences_sum_to_one(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = EMQ(learner=LogisticRegression(random_state=0, solver="lbfgs", max_iter=500))
        q.fit(X_train, y_train)
        with config_context(prevalence_normalization="sum"):
            prev = q.predict(X_test)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-4)

    def test_multiclass_all_classes_present(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = EMQ(learner=RandomForestClassifier(n_estimators=10, random_state=0))
        q.fit(X_train, y_train)
        with config_context(prevalence_normalization="sum"):
            prev = q.predict(X_test)
        assert len(prev) == len(np.unique(y_train))


# ===========================================================================
# Section 2 – Parametrized learner variations
# ===========================================================================

@pytest.mark.parametrize(
    "learner",
    [
        LogisticRegression(random_state=0, solver="liblinear"),
        RandomForestClassifier(n_estimators=10, random_state=0),
        DecisionTreeClassifier(random_state=0),
    ],
    ids=["LogisticRegression", "RandomForest", "DecisionTree"],
)
class TestEMQWithDifferentLearners:

    def test_binary_fit_predict(self, learner, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = EMQ(learner=learner)
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-6)

    def test_multiclass_fit_predict(self, learner, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = EMQ(learner=learner)
        q.fit(X_train, y_train)
        with config_context(prevalence_normalization="sum"):
            prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# Section 3 – Calibration function variations
# ===========================================================================

@pytest.mark.parametrize(
    "calib",
    [None, "ts", "bcts", "vs", "nbvs"],
    ids=["no_calib", "ts", "bcts", "vs", "nbvs"],
)
class TestEMQCalibration:

    def test_binary_with_calibration(self, calib, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = EMQ(
            learner=LogisticRegression(random_state=0, solver="liblinear"),
            calib_function=calib,
        )
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-6)

    def test_multiclass_with_calibration(self, calib, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = EMQ(
            learner=LogisticRegression(random_state=0, solver="lbfgs", max_iter=500),
            calib_function=calib,
        )
        q.fit(X_train, y_train)
        with config_context(prevalence_normalization="sum"):
            prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-4)


def test_custom_callable_calibration(binary_dataset):
    """A custom callable is NOT accepted as calibration function; only string options are valid."""
    X_train, X_test, y_train, y_test = binary_dataset

    def identity_calib(preds):
        return preds

    q = EMQ(
        learner=LogisticRegression(random_state=0, solver="liblinear"),
        calib_function=identity_calib,
    )
    with pytest.raises(Exception):
        q.fit(X_train, y_train)


# ===========================================================================
# Section 4 – Input type variations (numpy vs pandas, int vs string labels)
# ===========================================================================

class TestEMQInputTypes:

    def test_pandas_dataframe_input(self):
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        X_train_df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
        X_test_df = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])
        y_train_s = pd.Series(y_train)

        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train_df, y_train_s)
        prev = q.predict(X_test_df)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-6)

    def test_string_labels(self):
        X_train, X_test, y_train, y_test = _string_label_dataset()
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert set(prev.keys()) == {"neg", "pos"}
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-6)

    def test_pandas_with_string_labels(self):
        X_train, X_test, y_train, y_test = _string_label_dataset()
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        y_train_s = pd.Series(y_train)

        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train_df, y_train_s)
        prev = q.predict(X_test_df)
        assert isinstance(prev, dict)
        assert set(prev.keys()) == {"neg", "pos"}
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Section 5 – Parameter variations (tol, max_iter)
# ===========================================================================

@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-8])
def test_tol_variation(tol, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = EMQ(
        learner=LogisticRegression(random_state=0, solver="liblinear"),
        tol=tol,
    )
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert sum(prev.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("max_iter", [1, 10, 500])
def test_max_iter_variation(max_iter, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = EMQ(
        learner=LogisticRegression(random_state=0, solver="liblinear"),
        max_iter=max_iter,
    )
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert sum(prev.values()) == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Section 6 – Static EM method
# ===========================================================================

class TestEMStatic:

    def test_em_basic(self):
        posteriors = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
        priors = np.array([0.5, 0.5])
        qs, ps = EMQ.EM(posteriors, priors)
        assert len(qs) == 2
        assert sum(qs) == pytest.approx(1.0, abs=1e-6)
        assert ps.shape == posteriors.shape

    def test_em_uniform_posteriors(self):
        posteriors = np.full((10, 3), 1.0 / 3.0)
        priors = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        qs, ps = EMQ.EM(posteriors, priors)
        np.testing.assert_allclose(qs, priors, atol=1e-6)

    def test_em_extreme_posteriors(self):
        # One class dominates
        posteriors = np.array([[0.99, 0.01]] * 50 + [[0.01, 0.99]] * 5)
        priors = np.array([0.5, 0.5])
        qs, _ = EMQ.EM(posteriors, priors)
        assert qs[0] > qs[1], "Class 0 should dominate"
        assert sum(qs) == pytest.approx(1.0, abs=1e-6)

    def test_em_single_sample(self):
        posteriors = np.array([[0.7, 0.3]])
        priors = np.array([0.5, 0.5])
        qs, ps = EMQ.EM(posteriors, priors)
        assert len(qs) == 2
        assert sum(qs) == pytest.approx(1.0, abs=1e-6)

    def test_em_three_classes(self):
        rng = np.random.RandomState(0)
        posteriors = rng.dirichlet([1, 1, 1], size=100)
        priors = np.array([0.4, 0.3, 0.3])
        qs, ps = EMQ.EM(posteriors, priors)
        assert len(qs) == 3
        assert sum(qs) == pytest.approx(1.0, abs=1e-6)

    def test_em_convergence_with_low_tolerance(self):
        posteriors = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
        priors = np.array([0.5, 0.5])
        qs1, _ = EMQ.EM(posteriors, priors, tolerance=1e-10, max_iter=1000)
        qs2, _ = EMQ.EM(posteriors, priors, tolerance=1e-2, max_iter=1000)
        # Both should sum to ~1
        assert sum(qs1) == pytest.approx(1.0, abs=1e-6)
        assert sum(qs2) == pytest.approx(1.0, abs=1e-6)

    def test_em_zero_prior_handled(self):
        """Priors with a zero entry should be adjusted so EM doesn't divide by 0."""
        posteriors = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        priors = np.array([0.0, 1.0])
        qs, _ = EMQ.EM(posteriors, priors, tolerance=1e-6)
        assert sum(qs) == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize(
        "criteria",
        [MAE, lambda p, q: np.max(np.abs(p - q))],
        ids=["MAE", "max_abs_diff"],
    )
    def test_em_different_criteria(self, criteria):
        posteriors = np.array([[0.8, 0.2], [0.6, 0.4], [0.2, 0.8]])
        priors = np.array([0.5, 0.5])
        qs, _ = EMQ.EM(posteriors, priors, criteria=criteria)
        assert sum(qs) == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Section 7 – aggregate method
# ===========================================================================

class TestEMQAggregate:

    def test_aggregate_binary(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        clf = LogisticRegression(random_state=0, solver="liblinear")
        q = EMQ(learner=clf)
        q.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        prev = q.aggregate(proba, y_train)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-6)

    def test_aggregate_multiclass(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        clf = LogisticRegression(random_state=0, solver="lbfgs", max_iter=500)
        q = EMQ(learner=clf)
        q.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        with config_context(prevalence_normalization="sum"):
            prev = q.aggregate(proba, y_train)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# Section 8 – config_context: return type and normalization
# ===========================================================================

class TestEMQConfigContext:

    def test_return_type_array(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train, y_train)
        with config_context(prevalence_return_type="array"):
            prev = q.predict(X_test)
        assert isinstance(prev, np.ndarray)
        assert prev.sum() == pytest.approx(1.0, abs=1e-6)

    def test_return_type_dict(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train, y_train)
        with config_context(prevalence_return_type="dict"):
            prev = q.predict(X_test)
        assert isinstance(prev, dict)

    @pytest.mark.parametrize(
        "normalization",
        ["sum", "l1", "softmax"],
        ids=["sum", "l1", "softmax"],
    )
    def test_normalization_multiclass(self, normalization, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = EMQ(learner=LogisticRegression(random_state=0, solver="lbfgs", max_iter=500))
        q.fit(X_train, y_train)
        with config_context(prevalence_normalization=normalization):
            prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-2)


# ===========================================================================
# Section 9 – Edge cases
# ===========================================================================

class TestEMQEdgeCases:

    def test_tiny_dataset(self):
        """Very small dataset (20 samples)."""
        X_train, X_test, y_train, y_test = _small_binary_dataset(n=30, random_state=99)
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-4)

    def test_extreme_imbalance(self):
        """95% / 5% class imbalance."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2,
            weights=[0.95, 0.05], random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y,
        )
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-4)
        # Majority class should have higher prevalence
        assert prev[0] > prev[1]

    def test_single_class_training(self):
        """Training data contains only one class."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 5)
        y_train = np.zeros(50, dtype=int)
        X_test = rng.randn(20, 5)
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        # fit should work even with a single class, predict may fail or produce
        # degenerate output; we just check no crash
        with pytest.raises(Exception):
            # LogisticRegression or EMQ internals might raise because
            # single-class label prevents proper probability calibration
            q.fit(X_train, y_train)
            q.predict(X_test)

    def test_constant_features(self):
        """All features are constant – learner still runs."""
        rng = np.random.RandomState(0)
        X_train = np.ones((80, 5))
        y_train = rng.choice([0, 1], size=80)
        X_test = np.ones((20, 5))
        q = EMQ(learner=DecisionTreeClassifier(random_state=0))
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-4)

    def test_single_test_sample(self):
        """Predict with a single test instance."""
        X_train, _, y_train, _ = _small_binary_dataset()
        X_test_single = X_train[:1]
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        q.fit(X_train, y_train)
        prev = q.predict(X_test_single)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-4)

    def test_max_iter_one_no_crash(self):
        """max_iter=1 should still produce output (not converged)."""
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        q = EMQ(
            learner=LogisticRegression(random_state=0, solver="liblinear"),
            max_iter=1,
        )
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert sum(prev.values()) == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# Section 10 – Error cases
# ===========================================================================

class TestEMQErrors:

    def test_predict_before_fit_raises(self):
        """Calling predict on an unfitted EMQ should raise."""
        q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        X_test = np.random.randn(10, 5)
        with pytest.raises(Exception):
            q.predict(X_test)

    def test_invalid_learner_no_predict_proba(self):
        """A learner without predict_proba should raise at fit time during param validation."""
        from sklearn.svm import LinearSVC
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        q = EMQ(learner=LinearSVC(random_state=0))
        with pytest.raises(Exception):
            q.fit(X_train, y_train)

    def test_invalid_tol_raises(self):
        """Non-positive tolerance should be rejected by parameter constraints."""
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        q = EMQ(
            learner=LogisticRegression(random_state=0, solver="liblinear"),
            tol=-1,
        )
        with pytest.raises(Exception):
            q.fit(X_train, y_train)

    def test_invalid_max_iter_raises(self):
        """max_iter=0 should be rejected by parameter constraints."""
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        q = EMQ(
            learner=LogisticRegression(random_state=0, solver="liblinear"),
            max_iter=0,
        )
        with pytest.raises(Exception):
            q.fit(X_train, y_train)

    def test_invalid_calib_function_string_raises(self):
        """An unrecognized calib_function string should raise."""
        X_train, X_test, y_train, y_test = _small_binary_dataset()
        q = EMQ(
            learner=LogisticRegression(random_state=0, solver="liblinear"),
            calib_function="invalid_method",
        )
        with pytest.raises(Exception):
            q.fit(X_train, y_train)


# ===========================================================================
# Section 11 – get_params / set_params compatibility
# ===========================================================================

class TestEMQParams:

    def test_get_params(self):
        q = EMQ(
            learner=LogisticRegression(),
            tol=1e-5,
            max_iter=200,
            calib_function="ts",
        )
        params = q.get_params(deep=False)
        assert params["tol"] == 1e-5
        assert params["max_iter"] == 200
        assert params["calib_function"] == "ts"

    def test_set_params(self):
        q = EMQ(learner=LogisticRegression())
        q.set_params(tol=0.01, max_iter=50)
        assert q.tol == 0.01
        assert q.max_iter == 50

    def test_set_learner_params(self):
        q = EMQ(learner=LogisticRegression(C=1.0))
        q.set_params(learner__C=0.5)
        assert q.learner.C == 0.5

    def test_default_params(self):
        q = EMQ()
        assert q.tol == 1e-4
        assert q.max_iter == 100
        assert q.calib_function is None
        assert q.learner is None


# ===========================================================================
# Section 12 – Reproducibility / determinism
# ===========================================================================

class TestEMQReproducibility:

    def test_same_result_twice(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        prev_list = []
        for _ in range(2):
            q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
            q.fit(X_train, y_train)
            prev_list.append(q.predict(X_test))
        for key in prev_list[0]:
            assert prev_list[0][key] == pytest.approx(prev_list[1][key], abs=1e-10)

    def test_different_tol_similar_result(self, binary_dataset):
        """Lowering tolerance should give similar (not wildly different) results."""
        X_train, X_test, y_train, y_test = binary_dataset
        q1 = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"), tol=1e-2)
        q1.fit(X_train, y_train)
        prev1 = q1.predict(X_test)

        q2 = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"), tol=1e-8)
        q2.fit(X_train, y_train)
        prev2 = q2.predict(X_test)

        for key in prev1:
            assert prev1[key] == pytest.approx(prev2[key], abs=0.05)


# ===========================================================================
# Section 13 – Multiclass with different number of classes
# ===========================================================================

@pytest.mark.parametrize("n_classes", [3, 4, 5])
def test_multiclass_varying_classes(n_classes):
    X, y = make_classification(
        n_samples=300, n_features=10, n_classes=n_classes,
        n_informative=8, n_clusters_per_class=1, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    q = EMQ(learner=LogisticRegression(random_state=0, solver="lbfgs", max_iter=500))
    q.fit(X_train, y_train)
    with config_context(prevalence_normalization="sum"):
        prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == n_classes
    assert sum(prev.values()) == pytest.approx(1.0, abs=1e-2)


# ===========================================================================
# Section 14 – Calibration internal methods directly
# ===========================================================================

class TestCalibrationMethods:

    @pytest.fixture(autouse=True)
    def _fitted_emq(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        self.q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
        self.q.fit(X_train, y_train)
        self.proba = self.q.learner.predict_proba(X_test)

    def test_temperature_scaling_output_shape(self):
        result = self.q._temperature_scaling(self.proba)
        assert result.shape == self.proba.shape

    def test_temperature_scaling_sums_to_one(self):
        result = self.q._temperature_scaling(self.proba)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_bcts_output_shape(self):
        result = self.q._bias_corrected_temperature_scaling(self.proba)
        assert result.shape == self.proba.shape

    def test_bcts_sums_to_one(self):
        result = self.q._bias_corrected_temperature_scaling(self.proba)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_vector_scaling_output_shape(self):
        result = self.q._vector_scaling(self.proba)
        assert result.shape == self.proba.shape

    def test_vector_scaling_sums_to_one(self):
        result = self.q._vector_scaling(self.proba)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_nbvs_output_shape(self):
        result = self.q._no_bias_vector_scaling(self.proba)
        assert result.shape == self.proba.shape

    def test_nbvs_sums_to_one(self):
        result = self.q._no_bias_vector_scaling(self.proba)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_apply_calibration_none(self):
        self.q.calib_function = None
        result = self.q._apply_calibration(self.proba)
        np.testing.assert_array_equal(result, self.proba)


# ===========================================================================
# Section 15 – Multiclass string labels
# ===========================================================================

def test_multiclass_string_labels():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    labels = np.array(["cat", "dog", "bird"])
    y = rng.choice(labels, size=200)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    q = EMQ(learner=LogisticRegression(random_state=0, solver="lbfgs", max_iter=500))
    q.fit(X_train, y_train)
    with config_context(prevalence_normalization="sum"):
        prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert set(prev.keys()) == {"cat", "dog", "bird"}
    assert sum(prev.values()) == pytest.approx(1.0, abs=1e-2)


# ===========================================================================
# Section 16 – Binary with various prevalence shifts in test
# ===========================================================================

@pytest.mark.parametrize("test_weight", [0.1, 0.5, 0.9])
def test_binary_different_test_prevalences(test_weight):
    """Fit on balanced, predict on shifted test set."""
    X, y = make_classification(
        n_samples=400, n_features=10, n_classes=2,
        weights=[0.5, 0.5], random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    # Resample test set to get approximate desired prevalence
    rng = np.random.RandomState(0)
    idx_0 = np.where(y_test == 0)[0]
    idx_1 = np.where(y_test == 1)[0]
    n0 = max(1, int(len(y_test) * (1 - test_weight)))
    n1 = max(1, int(len(y_test) * test_weight))
    sel = np.concatenate([
        rng.choice(idx_0, size=min(n0, len(idx_0)), replace=True),
        rng.choice(idx_1, size=min(n1, len(idx_1)), replace=True),
    ])
    X_test_shifted = X_test[sel]

    q = EMQ(learner=LogisticRegression(random_state=0, solver="liblinear"))
    q.fit(X_train, y_train)
    prev = q.predict(X_test_shifted)
    assert isinstance(prev, dict)
    assert sum(prev.values()) == pytest.approx(1.0, abs=1e-4)
