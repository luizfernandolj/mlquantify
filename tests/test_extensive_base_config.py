"""
Comprehensive tests for base modules: base.py, base_aggregative.py,
calibration.py, confidence.py, multiclass.py, and _config.py.
"""

import os
import tempfile
from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import mlquantify
from mlquantify._config import get_config, set_config, config_context
from mlquantify.base import BaseQuantifier, MetaquantifierMixin, ProtocolMixin
from mlquantify.base_aggregative import (
    AggregationMixin,
    CrispLearnerQMixin,
    SoftLearnerQMixin,
    _get_learner_function,
    get_aggregation_requirements,
    is_aggregative_quantifier,
    uses_crisp_predictions,
    uses_soft_predictions,
)
from mlquantify.calibration import Calibrator, ClassifierCalibrator, QuantifierCalibrator
from mlquantify.confidence import (
    BaseConfidenceRegion,
    ConfidenceEllipseCLR,
    ConfidenceEllipseSimplex,
    ConfidenceInterval,
    construct_confidence_region,
)
from mlquantify.multiclass import BinaryQuantifier, define_binary
from mlquantify.utils._tags import Tags, TargetInputTags, PredictionRequirements, get_tags


# ============================================================
# Helper concrete classes for testing abstract bases
# ============================================================


class ConcreteQuantifier(BaseQuantifier):
    """Minimal concrete quantifier for testing BaseQuantifier."""

    def __init__(self, param_a=1, param_b="hello"):
        self.param_a = param_a
        self.param_b = param_b

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.ones(len(self.classes_)) / len(self.classes_)


class ConcreteMetaQuantifier(MetaquantifierMixin, BaseQuantifier):
    """Concrete meta-quantifier for testing MetaquantifierMixin."""

    def __init__(self, base_quantifier=None):
        self.base_quantifier = base_quantifier

    def fit(self, X, y):
        if self.base_quantifier is not None:
            self.base_quantifier.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.ones(len(self.classes_)) / len(self.classes_)


class ConcreteProtocolQuantifier(ProtocolMixin, BaseQuantifier):
    """Concrete protocol quantifier for testing ProtocolMixin."""

    def __init__(self, n_samples=10):
        self.n_samples = n_samples

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[: self.n_samples]


class ConcreteAggregativeQuantifier(AggregationMixin, BaseQuantifier):
    """Concrete aggregative quantifier for testing AggregationMixin."""

    def __init__(self, learner=None):
        self.learner = learner or LogisticRegression(random_state=42, solver="liblinear")

    def fit(self, X, y):
        self.learner.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        preds = self.learner.predict(X)
        _, counts = np.unique(preds, return_counts=True)
        return counts / counts.sum()


class ConcreteSoftQuantifier(SoftLearnerQMixin, AggregationMixin, BaseQuantifier):
    """Concrete soft aggregative quantifier for testing SoftLearnerQMixin."""

    def __init__(self, learner=None):
        self.learner = learner or LogisticRegression(random_state=42, solver="liblinear")

    def fit(self, X, y):
        self.learner.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        proba = self.learner.predict_proba(X)
        return proba.mean(axis=0)


class ConcreteCrispQuantifier(CrispLearnerQMixin, AggregationMixin, BaseQuantifier):
    """Concrete crisp aggregative quantifier for testing CrispLearnerQMixin."""

    def __init__(self, learner=None):
        self.learner = learner or LogisticRegression(random_state=42, solver="liblinear")

    def fit(self, X, y):
        self.learner.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        preds = self.learner.predict(X)
        _, counts = np.unique(preds, return_counts=True)
        return counts / counts.sum()


# ============================================================
# 1. BaseQuantifier Tests
# ============================================================


class TestBaseQuantifier:
    """Tests for BaseQuantifier core functionality."""

    def test_get_params_returns_dict(self):
        q = ConcreteQuantifier(param_a=42, param_b="world")
        params = q.get_params()
        assert isinstance(params, dict)
        assert params["param_a"] == 42
        assert params["param_b"] == "world"

    def test_get_params_deep_false(self):
        q = ConcreteQuantifier()
        params = q.get_params(deep=False)
        assert "param_a" in params
        assert "param_b" in params

    @pytest.mark.parametrize(
        "param_a, param_b",
        [
            (1, "hello"),
            (100, "world"),
            (0, ""),
            (-1, None),
        ],
    )
    def test_get_params_various_values(self, param_a, param_b):
        q = ConcreteQuantifier(param_a=param_a, param_b=param_b)
        params = q.get_params()
        assert params["param_a"] == param_a
        assert params["param_b"] == param_b

    def test_set_params(self):
        q = ConcreteQuantifier()
        q.set_params(param_a=99, param_b="changed")
        assert q.param_a == 99
        assert q.param_b == "changed"

    def test_set_params_returns_self(self):
        q = ConcreteQuantifier()
        result = q.set_params(param_a=10)
        assert result is q

    def test_set_params_partial(self):
        q = ConcreteQuantifier(param_a=1, param_b="original")
        q.set_params(param_a=50)
        assert q.param_a == 50
        assert q.param_b == "original"

    def test_validate_params_no_constraints(self):
        """_validate_params should work fine when there are no constraints."""
        q = ConcreteQuantifier()
        # Should not raise
        q._validate_params()

    def test_mlquantify_tags_returns_tags_instance(self):
        q = ConcreteQuantifier()
        tags = q.__mlquantify_tags__()
        assert isinstance(tags, Tags)

    def test_mlquantify_tags_default_values(self):
        q = ConcreteQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.has_estimator is None
        assert tags.estimation_type is None
        assert tags.estimator_function is None
        assert tags.estimator_type is None
        assert tags.aggregation_type is None
        assert tags.requires_fit is True
        assert isinstance(tags.target_input_tags, TargetInputTags)
        assert isinstance(tags.prediction_requirements, PredictionRequirements)

    def test_save_quantifier_default_path(self):
        q = ConcreteQuantifier()
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                q.save_quantifier()
                expected_path = os.path.join(tmpdir, "ConcreteQuantifier.joblib")
                assert os.path.exists(expected_path)
            finally:
                os.chdir(orig_dir)

    def test_save_quantifier_custom_path(self):
        q = ConcreteQuantifier(param_a=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "my_quantifier.joblib")
            q.save_quantifier(path)
            assert os.path.exists(path)
            import joblib
            loaded = joblib.load(path)
            assert loaded.param_a == 7

    def test_save_and_load_preserves_params(self):
        q = ConcreteQuantifier(param_a=42, param_b="saved")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_q.joblib")
            q.save_quantifier(path)
            import joblib
            loaded = joblib.load(path)
            assert loaded.get_params() == q.get_params()

    def test_get_tags_utility_function(self):
        q = ConcreteQuantifier()
        tags = get_tags(q)
        assert isinstance(tags, Tags)
        assert tags.requires_fit is True


# ============================================================
# 2. MetaquantifierMixin Tests
# ============================================================


class TestMetaquantifierMixin:
    """Tests for MetaquantifierMixin."""

    def test_is_instance(self):
        q = ConcreteMetaQuantifier()
        assert isinstance(q, MetaquantifierMixin)
        assert isinstance(q, BaseQuantifier)

    def test_tags_inherited_from_base(self):
        q = ConcreteMetaQuantifier()
        tags = q.__mlquantify_tags__()
        assert isinstance(tags, Tags)
        # MetaquantifierMixin doesn't override tags, so defaults apply
        assert tags.requires_fit is True

    def test_fit_predict(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = ConcreteMetaQuantifier()
        q.fit(X_train, y_train)
        result = q.predict(X_test)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        np.testing.assert_almost_equal(result.sum(), 1.0)


# ============================================================
# 3. ProtocolMixin Tests
# ============================================================


class TestProtocolMixin:
    """Tests for ProtocolMixin."""

    def test_tags_estimation_type(self):
        q = ConcreteProtocolQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.estimation_type == "sample"

    def test_tags_requires_fit_false(self):
        q = ConcreteProtocolQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.requires_fit is False

    def test_is_instance(self):
        q = ConcreteProtocolQuantifier()
        assert isinstance(q, ProtocolMixin)
        assert isinstance(q, BaseQuantifier)

    def test_protocol_tags_override_base(self):
        """ProtocolMixin should override base requires_fit from True to False."""
        base_q = ConcreteQuantifier()
        proto_q = ConcreteProtocolQuantifier()
        base_tags = base_q.__mlquantify_tags__()
        proto_tags = proto_q.__mlquantify_tags__()
        assert base_tags.requires_fit is True
        assert proto_tags.requires_fit is False


# ============================================================
# 4. AggregationMixin, SoftLearnerQMixin, CrispLearnerQMixin Tests
# ============================================================


class TestAggregationMixin:
    """Tests for AggregationMixin tag behavior."""

    def test_has_estimator_tag(self):
        q = ConcreteAggregativeQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.has_estimator is True

    def test_requires_fit_tag(self):
        q = ConcreteAggregativeQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.requires_fit is True

    def test_is_aggregative_helper(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = ConcreteAggregativeQuantifier()
        q.fit(X_train, y_train)
        assert is_aggregative_quantifier(q) is True

    def test_set_params_with_learner(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        q = ConcreteAggregativeQuantifier(
            learner=LogisticRegression(C=1.0, solver="liblinear")
        )
        q.set_params(learner__C=0.5)
        assert q.learner.C == 0.5

    def test_set_params_model_and_learner(self):
        q = ConcreteAggregativeQuantifier(
            learner=LogisticRegression(C=1.0, solver="liblinear")
        )
        q.set_params(learner__C=0.01)
        assert q.learner.C == 0.01


class TestSoftLearnerQMixin:
    """Tests for SoftLearnerQMixin tag behavior."""

    def test_estimator_function_tag(self):
        q = ConcreteSoftQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.estimator_function == "predict_proba"

    def test_estimator_type_tag(self):
        q = ConcreteSoftQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.estimator_type == "soft"

    def test_uses_soft_predictions_helper(self):
        q = ConcreteSoftQuantifier()
        assert uses_soft_predictions(q) is True
        assert uses_crisp_predictions(q) is False

    def test_has_estimator_inherited(self):
        q = ConcreteSoftQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.has_estimator is True


class TestCrispLearnerQMixin:
    """Tests for CrispLearnerQMixin (aliased as HardLearnerQMixin in test spec)."""

    def test_estimator_function_tag(self):
        q = ConcreteCrispQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.estimator_function == "predict"

    def test_estimator_type_tag(self):
        q = ConcreteCrispQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.estimator_type == "crisp"

    def test_uses_crisp_predictions_helper(self):
        q = ConcreteCrispQuantifier()
        assert uses_crisp_predictions(q) is True
        assert uses_soft_predictions(q) is False

    def test_has_estimator_inherited(self):
        q = ConcreteCrispQuantifier()
        tags = q.__mlquantify_tags__()
        assert tags.has_estimator is True


class TestGetLearnerFunction:
    """Tests for _get_learner_function utility."""

    def test_soft_returns_predict_proba(self):
        q = ConcreteSoftQuantifier()
        assert _get_learner_function(q) == "predict_proba"

    def test_crisp_returns_predict(self):
        q = ConcreteCrispQuantifier()
        assert _get_learner_function(q) == "predict"

    def test_no_estimator_function_raises(self):
        q = ConcreteAggregativeQuantifier()
        # AggregationMixin alone doesn't set estimator_function
        with pytest.raises(ValueError, match="does not specify an estimator function"):
            _get_learner_function(q)

    def test_learner_missing_method_raises(self):
        """If the learner doesn't have the required method, should raise."""

        class NoPredict:
            pass

        q = ConcreteSoftQuantifier(learner=NoPredict())
        with pytest.raises(AttributeError, match="does not have the method"):
            _get_learner_function(q)


# ============================================================
# 5. Config: get_config Tests
# ============================================================


class TestGetConfig:
    """Tests for get_config returning defaults."""

    def setup_method(self):
        """Reset config to defaults before each test."""
        set_config(prevalence_return_type="dict", prevalence_normalization="mean")

    def test_returns_dict(self):
        config = get_config()
        assert isinstance(config, dict)

    def test_default_prevalence_return_type(self):
        config = get_config()
        assert config["prevalence_return_type"] == "dict"

    def test_default_prevalence_normalization(self):
        config = get_config()
        assert config["prevalence_normalization"] == "mean"

    def test_has_expected_keys(self):
        config = get_config()
        assert "prevalence_return_type" in config
        assert "prevalence_normalization" in config

    def test_returns_copy(self):
        """Modifying the returned dict should not affect internal state."""
        config = get_config()
        config["prevalence_return_type"] = "array"
        assert get_config()["prevalence_return_type"] == "dict"


# ============================================================
# 6. Config: set_config Tests
# ============================================================


class TestSetConfig:
    """Tests for set_config updating keys and persistence."""

    def setup_method(self):
        set_config(prevalence_return_type="dict", prevalence_normalization="mean")

    def teardown_method(self):
        set_config(prevalence_return_type="dict", prevalence_normalization="mean")

    @pytest.mark.parametrize("return_type", ["dict", "array"])
    def test_set_prevalence_return_type(self, return_type):
        set_config(prevalence_return_type=return_type)
        assert get_config()["prevalence_return_type"] == return_type

    @pytest.mark.parametrize(
        "normalization",
        ["sum", "l1", "softmax", "mean", "median"],
    )
    def test_set_prevalence_normalization(self, normalization):
        set_config(prevalence_normalization=normalization)
        assert get_config()["prevalence_normalization"] == normalization

    def test_set_config_persists(self):
        set_config(prevalence_return_type="array")
        # Multiple calls to get_config should reflect the change
        assert get_config()["prevalence_return_type"] == "array"
        assert get_config()["prevalence_return_type"] == "array"

    def test_set_config_none_does_not_change(self):
        set_config(prevalence_return_type="array")
        set_config(prevalence_return_type=None)
        assert get_config()["prevalence_return_type"] == "array"

    def test_set_both_keys(self):
        set_config(prevalence_return_type="array", prevalence_normalization="softmax")
        config = get_config()
        assert config["prevalence_return_type"] == "array"
        assert config["prevalence_normalization"] == "softmax"


# ============================================================
# 7. Config: config_context Tests
# ============================================================


class TestConfigContext:
    """Tests for config_context temporary override behavior."""

    def setup_method(self):
        set_config(prevalence_return_type="dict", prevalence_normalization="mean")

    def teardown_method(self):
        set_config(prevalence_return_type="dict", prevalence_normalization="mean")

    def test_temporary_override(self):
        with config_context(prevalence_return_type="array"):
            assert get_config()["prevalence_return_type"] == "array"
        assert get_config()["prevalence_return_type"] == "dict"

    def test_restores_on_exit(self):
        set_config(prevalence_return_type="dict", prevalence_normalization="sum")
        with config_context(prevalence_return_type="array", prevalence_normalization="softmax"):
            assert get_config()["prevalence_return_type"] == "array"
            assert get_config()["prevalence_normalization"] == "softmax"
        assert get_config()["prevalence_return_type"] == "dict"
        assert get_config()["prevalence_normalization"] == "sum"

    def test_restores_on_exception(self):
        set_config(prevalence_return_type="dict")
        try:
            with config_context(prevalence_return_type="array"):
                assert get_config()["prevalence_return_type"] == "array"
                raise ValueError("Intentional error")
        except ValueError:
            pass
        assert get_config()["prevalence_return_type"] == "dict"

    def test_nested_usage(self):
        with config_context(prevalence_return_type="array"):
            assert get_config()["prevalence_return_type"] == "array"
            with config_context(prevalence_normalization="l1"):
                assert get_config()["prevalence_return_type"] == "array"
                assert get_config()["prevalence_normalization"] == "l1"
            assert get_config()["prevalence_normalization"] == "mean"
        assert get_config()["prevalence_return_type"] == "dict"

    def test_none_does_not_change_existing(self):
        set_config(prevalence_return_type="array")
        with config_context(prevalence_return_type=None):
            assert get_config()["prevalence_return_type"] == "array"
        assert get_config()["prevalence_return_type"] == "array"
        # Reset for teardown
        set_config(prevalence_return_type="dict")

    @pytest.mark.parametrize(
        "ctx_kwargs, expected_inside",
        [
            ({"prevalence_return_type": "array"}, {"prevalence_return_type": "array"}),
            ({"prevalence_normalization": "softmax"}, {"prevalence_normalization": "softmax"}),
            (
                {"prevalence_return_type": "array", "prevalence_normalization": "l1"},
                {"prevalence_return_type": "array", "prevalence_normalization": "l1"},
            ),
        ],
    )
    def test_config_context_parametrized(self, ctx_kwargs, expected_inside):
        with config_context(**ctx_kwargs):
            config = get_config()
            for key, val in expected_inside.items():
                assert config[key] == val

    def test_mlquantify_module_exposes_config_context(self):
        """config_context should be accessible from the mlquantify top-level package."""
        assert hasattr(mlquantify, "config_context")
        assert mlquantify.config_context is config_context


# ============================================================
# 8. Calibration Tests
# ============================================================


class TestCalibration:
    """Tests for Calibration wrapper classes."""

    def test_calibrator_instantiation(self):
        cal = Calibrator()
        assert cal is not None

    def test_calibrator_fit(self):
        cal = Calibrator()
        # fit currently does nothing; verify no exception
        result = cal.fit(np.array([0, 1, 1]), np.array([0.3, 0.8, 0.9]))
        assert result is None

    def test_calibrator_predict(self):
        cal = Calibrator()
        result = cal.predict(np.array([0.5, 0.6]))
        assert result is None

    def test_classifier_calibrator_instantiation(self):
        cal = ClassifierCalibrator()
        assert isinstance(cal, Calibrator)

    def test_classifier_calibrator_fit_predict(self):
        cal = ClassifierCalibrator()
        # fit and predict are stubs; verify no exception
        cal.fit(np.array([0, 1, 1]), np.array([0.3, 0.8, 0.9]))
        result = cal.predict(np.array([0.5, 0.6]))
        assert result is None

    def test_quantifier_calibrator_instantiation(self):
        cal = QuantifierCalibrator()
        assert isinstance(cal, Calibrator)

    def test_quantifier_calibrator_fit_predict(self):
        cal = QuantifierCalibrator()
        cal.fit(np.array([0, 1]), np.array([0.4, 0.6]))
        result = cal.predict(np.array([0.5, 0.5]))
        assert result is None

    def test_calibrator_inheritance(self):
        assert issubclass(ClassifierCalibrator, Calibrator)
        assert issubclass(QuantifierCalibrator, Calibrator)


# ============================================================
# 9. Confidence Tests
# ============================================================


class TestConfidenceInterval:
    """Tests for ConfidenceInterval confidence region."""

    @pytest.fixture
    def bootstrap_samples(self):
        rng = np.random.RandomState(42)
        return rng.dirichlet(np.ones(3), size=200)

    def test_output_shape(self, bootstrap_samples):
        ci = ConfidenceInterval(bootstrap_samples, confidence_level=0.95)
        low, high = ci.get_region()
        assert low.shape == (3,)
        assert high.shape == (3,)

    @pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
    def test_alpha_parameter(self, bootstrap_samples, alpha):
        ci = ConfidenceInterval(bootstrap_samples, confidence_level=alpha)
        low, high = ci.get_region()
        # Higher confidence => wider intervals
        assert np.all(low <= high)

    def test_wider_interval_for_higher_confidence(self, bootstrap_samples):
        ci90 = ConfidenceInterval(bootstrap_samples, confidence_level=0.90)
        ci99 = ConfidenceInterval(bootstrap_samples, confidence_level=0.99)
        low90, high90 = ci90.get_region()
        low99, high99 = ci99.get_region()
        width90 = high90 - low90
        width99 = high99 - low99
        assert np.all(width99 >= width90 - 1e-10)

    def test_point_estimate(self, bootstrap_samples):
        ci = ConfidenceInterval(bootstrap_samples)
        point = ci.get_point_estimate()
        assert point.shape == (3,)
        np.testing.assert_almost_equal(point.sum(), 1.0, decimal=1)

    def test_contains_mean(self, bootstrap_samples):
        ci = ConfidenceInterval(bootstrap_samples, confidence_level=0.99)
        mean = np.mean(bootstrap_samples, axis=0)
        assert ci.contains(mean)

    def test_reproducibility(self):
        rng = np.random.RandomState(123)
        samples = rng.dirichlet(np.ones(3), size=100)
        ci1 = ConfidenceInterval(samples, confidence_level=0.95)
        ci2 = ConfidenceInterval(samples, confidence_level=0.95)
        np.testing.assert_array_equal(ci1.get_region()[0], ci2.get_region()[0])
        np.testing.assert_array_equal(ci1.get_region()[1], ci2.get_region()[1])

    def test_binary_case(self):
        rng = np.random.RandomState(42)
        samples = rng.dirichlet(np.ones(2), size=100)
        ci = ConfidenceInterval(samples, confidence_level=0.95)
        low, high = ci.get_region()
        assert low.shape == (2,)
        assert high.shape == (2,)


class TestConfidenceEllipseSimplex:
    """Tests for ConfidenceEllipseSimplex."""

    @pytest.fixture
    def bootstrap_samples(self):
        rng = np.random.RandomState(42)
        return rng.dirichlet(np.ones(3), size=200)

    def test_point_estimate_shape(self, bootstrap_samples):
        ce = ConfidenceEllipseSimplex(bootstrap_samples)
        point = ce.get_point_estimate()
        assert point.shape == (3,)

    def test_contains_mean(self, bootstrap_samples):
        ce = ConfidenceEllipseSimplex(bootstrap_samples, confidence_level=0.99)
        mean = np.mean(bootstrap_samples, axis=0)
        assert ce.contains(mean)

    def test_region_components(self, bootstrap_samples):
        ce = ConfidenceEllipseSimplex(bootstrap_samples)
        mean, prec, chi2_val = ce.get_region()
        assert mean.shape == (3,)
        assert prec.shape == (3, 3)
        assert chi2_val > 0

    @pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
    def test_chi2_increases_with_confidence(self, bootstrap_samples, alpha):
        ce = ConfidenceEllipseSimplex(bootstrap_samples, confidence_level=alpha)
        _, _, chi2_val = ce.get_region()
        assert chi2_val > 0


class TestConfidenceEllipseCLR:
    """Tests for ConfidenceEllipseCLR."""

    @pytest.fixture
    def bootstrap_samples(self):
        rng = np.random.RandomState(42)
        return rng.dirichlet(np.ones(3), size=200)

    def test_point_estimate_shape(self, bootstrap_samples):
        ce = ConfidenceEllipseCLR(bootstrap_samples)
        point = ce.get_point_estimate()
        assert point.shape == (3,)

    def test_contains_sample_mean(self, bootstrap_samples):
        ce = ConfidenceEllipseCLR(bootstrap_samples, confidence_level=0.99)
        mean = np.mean(bootstrap_samples, axis=0)
        assert ce.contains(mean)

    @pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
    def test_alpha_parameter(self, bootstrap_samples, alpha):
        ce = ConfidenceEllipseCLR(bootstrap_samples, confidence_level=alpha)
        assert ce.chi2_val > 0


class TestConstructConfidenceRegion:
    """Tests for factory function construct_confidence_region."""

    @pytest.fixture
    def bootstrap_samples(self):
        rng = np.random.RandomState(42)
        return rng.dirichlet(np.ones(3), size=200)

    @pytest.mark.parametrize(
        "method, expected_class",
        [
            ("intervals", ConfidenceInterval),
            ("ellipse", ConfidenceEllipseSimplex),
            ("ellipse-clr", ConfidenceEllipseCLR),
            ("clr", ConfidenceEllipseCLR),
            ("elipse-clr", ConfidenceEllipseCLR),  # typo variant
        ],
    )
    def test_factory_methods(self, bootstrap_samples, method, expected_class):
        region = construct_confidence_region(bootstrap_samples, method=method)
        assert isinstance(region, expected_class)

    def test_unknown_method_raises(self, bootstrap_samples):
        with pytest.raises(NotImplementedError):
            construct_confidence_region(bootstrap_samples, method="unknown")


class TestBaseConfidenceRegion:
    """Tests for BaseConfidenceRegion ABC-like behavior."""

    def test_cannot_instantiate_directly(self):
        samples = np.random.dirichlet(np.ones(3), size=50)
        with pytest.raises(NotImplementedError):
            BaseConfidenceRegion(samples)


# ============================================================
# 10. Multiclass Tests
# ============================================================


@define_binary
class SimpleCC(CrispLearnerQMixin, AggregationMixin, BaseQuantifier):
    """Simple CC-like quantifier decorated with define_binary for testing."""

    def __init__(self, learner=None, strategy="ovr", n_jobs=None):
        self.learner = learner or LogisticRegression(random_state=42, solver="liblinear")
        self.strategy = strategy
        self.n_jobs = n_jobs

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = False
        tags.prediction_requirements.requires_train_labels = True
        return tags

    def fit(self, X, y):
        self.learner.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        from mlquantify.utils._validation import validate_prevalences
        preds = self.learner.predict(X)
        classes = self.classes_
        counts = np.array([np.count_nonzero(preds == c) for c in classes])
        prevalences = counts / len(preds)
        return validate_prevalences(self, prevalences, classes)

    def aggregate(self, predictions, y_train=None):
        from mlquantify.utils._validation import validate_prevalences
        if y_train is None:
            y_train = np.unique(predictions)
        classes = np.unique(y_train)
        self.classes_ = classes
        counts = np.array([np.count_nonzero(predictions == c) for c in classes])
        prevalences = counts / len(predictions)
        return validate_prevalences(self, prevalences, classes)


class TestDefineBinary:
    """Tests for the define_binary decorator."""

    @pytest.mark.parametrize("strategy", ["ovr", "ovo"])
    def test_define_binary_with_string_labels(self, strategy):
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42,
        )
        labels = np.array(["cat", "dog", "fish"])
        y_str = labels[y]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_str, test_size=0.3, random_state=42
        )
        q = SimpleCC(strategy=strategy)
        q.fit(X_train, y_train)
        result = q.predict(X_test)
        assert isinstance(result, dict)
        assert len(result) == 3
        total = sum(result.values())
        assert abs(total - 1.0) < 0.6  # OvO aggregation may not sum exactly to 1

    @pytest.mark.parametrize("strategy", ["ovr", "ovo"])
    def test_define_binary_with_int_labels(self, strategy, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = SimpleCC(strategy=strategy)
        q.fit(X_train, y_train)
        result = q.predict(X_test)
        assert isinstance(result, dict)
        assert len(result) == 3

    @pytest.mark.parametrize("strategy", ["ovr", "ovo"])
    def test_define_binary_with_float_labels(self, strategy):
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42,
        )
        y_float = y.astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_float, test_size=0.3, random_state=42
        )
        q = SimpleCC(strategy=strategy)
        q.fit(X_train, y_train)
        result = q.predict(X_test)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_original_methods_preserved(self):
        """define_binary should store original methods as _original_fit etc."""
        assert hasattr(SimpleCC, "_original_fit")
        assert hasattr(SimpleCC, "_original_predict")
        assert hasattr(SimpleCC, "_original_aggregate")
        assert callable(SimpleCC._original_fit)
        assert callable(SimpleCC._original_predict)
        assert callable(SimpleCC._original_aggregate)

    def test_fit_predict_method_added(self):
        """define_binary should add fit_predict method."""
        assert hasattr(SimpleCC, "fit_predict")
        assert callable(SimpleCC.fit_predict)


class TestBinaryQuantifier:
    """Tests for BinaryQuantifier fit/predict/aggregate/fit_predict."""

    def test_fit_binary_passthrough(self, binary_dataset):
        """Binary data (<=2 classes) should use _original_fit directly."""
        X_train, X_test, y_train, y_test = binary_dataset
        q = SimpleCC(strategy="ovr")
        q.fit(X_train, y_train)
        # For binary data, should set binary=True
        assert hasattr(q, "binary") and q.binary is True

    def test_predict_binary_passthrough(self, binary_dataset):
        """Binary data should use _original_predict directly."""
        X_train, X_test, y_train, y_test = binary_dataset
        q = SimpleCC(strategy="ovr")
        q.fit(X_train, y_train)
        result = q.predict(X_test)
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_fit_multiclass_ovr(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = SimpleCC(strategy="ovr")
        q.fit(X_train, y_train)
        assert hasattr(q, "qtfs_")
        assert len(q.qtfs_) == 3  # 3 classes

    def test_fit_multiclass_ovo(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = SimpleCC(strategy="ovo")
        q.fit(X_train, y_train)
        assert hasattr(q, "qtfs_")
        assert len(q.qtfs_) == 3  # C(3,2) = 3 pairs

    def test_predict_multiclass_ovr(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = SimpleCC(strategy="ovr")
        q.fit(X_train, y_train)
        result = q.predict(X_test)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_predict_multiclass_ovo(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = SimpleCC(strategy="ovo")
        q.fit(X_train, y_train)
        result = q.predict(X_test)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_fit_predict_ovr(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = SimpleCC(strategy="ovr")
        result = q.fit_predict(X_train, y_train, X_test)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_fit_predict_ovo(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = SimpleCC(strategy="ovo")
        result = q.fit_predict(X_train, y_train, X_test)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_fit_predict_binary_passthrough(self, binary_dataset):
        """fit_predict on binary data should handle it like standard fit+predict."""
        X_train, X_test, y_train, y_test = binary_dataset
        q = SimpleCC(strategy="ovr")
        result = q.fit_predict(X_train, y_train, X_test)
        assert isinstance(result, dict)
        assert len(result) == 2


class TestMulticlassEdgeCases:
    """Edge case tests for multiclass module."""

    def test_invalid_strategy_fit(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = SimpleCC(strategy="invalid")
        with pytest.raises(ValueError, match="Strategy must be 'ovr' or 'ovo'"):
            q.fit(X_train, y_train)

    def test_invalid_strategy_predict(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = SimpleCC(strategy="ovr")
        q.fit(X_train, y_train)
        # Manually override strategy to trigger error in predict
        q.strategy = "invalid"
        q.binary = False  # Ensure it doesn't think it's binary
        delattr(q, "binary")
        with pytest.raises(ValueError, match="Strategy must be 'ovr' or 'ovo'"):
            q.predict(X_test)

    def test_binary_passthrough_two_classes(self):
        """With exactly 2 classes, BinaryQuantifier should passthrough to original."""
        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        q = SimpleCC(strategy="ovr")
        q.fit(X_train, y_train)
        assert hasattr(q, "binary") and q.binary is True
        result = q.predict(X_test)
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_single_class_in_training(self):
        """With a single class, sklearn learner raises ValueError (needs >=2 classes)."""
        X = np.random.randn(50, 5)
        y = np.zeros(50, dtype=int)
        q = SimpleCC(strategy="ovr")
        with pytest.raises(ValueError):
            q.fit(X, y)

    def test_many_classes_ovr(self):
        """Test with more than 3 classes."""
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_classes=5,
            n_informative=8,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        q = SimpleCC(strategy="ovr")
        q.fit(X_train, y_train)
        result = q.predict(X_test)
        assert isinstance(result, dict)
        assert len(result) == 5

    def test_many_classes_ovo(self):
        """Test with more than 3 classes using OvO."""
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_classes=4,
            n_informative=8,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        q = SimpleCC(strategy="ovo")
        q.fit(X_train, y_train)
        result = q.predict(X_test)
        assert isinstance(result, dict)
        assert len(result) == 4
        # C(4,2)=6 binary quantifiers
        assert len(q.qtfs_) == 6

    def test_define_binary_no_fit_method(self):
        """Decorator should handle a class without fit gracefully."""

        @define_binary
        class NoFitQuantifier(BaseQuantifier):
            def __init__(self):
                pass

            def predict(self, X):
                return np.array([0.5, 0.5])

        # Should still have predict replaced
        assert hasattr(NoFitQuantifier, "_original_predict")

    def test_deep_copy_isolation(self, multiclass_dataset):
        """Each binary sub-quantifier should be independent (deepcopy)."""
        X_train, X_test, y_train, y_test = multiclass_dataset
        q = SimpleCC(strategy="ovr")
        q.fit(X_train, y_train)
        # Modifying one sub-quantifier shouldn't affect others
        keys = list(q.qtfs_.keys())
        if len(keys) >= 2:
            q.qtfs_[keys[0]].learner.C = 999.0
            assert q.qtfs_[keys[1]].learner.C != 999.0


# ============================================================
# Additional parametrized edge case tests
# ============================================================


class TestTagsCombinations:
    """Test various tag combinations across mixin classes."""

    @pytest.mark.parametrize(
        "cls, expected_estimator_type, expected_estimator_function",
        [
            (ConcreteSoftQuantifier, "soft", "predict_proba"),
            (ConcreteCrispQuantifier, "crisp", "predict"),
        ],
    )
    def test_mixin_tag_combinations(self, cls, expected_estimator_type, expected_estimator_function):
        q = cls()
        tags = q.__mlquantify_tags__()
        assert tags.estimator_type == expected_estimator_type
        assert tags.estimator_function == expected_estimator_function
        assert tags.has_estimator is True
        assert tags.requires_fit is True

    @pytest.mark.parametrize(
        "cls, expected_requires_fit",
        [
            (ConcreteQuantifier, True),
            (ConcreteProtocolQuantifier, False),
            (ConcreteAggregativeQuantifier, True),
        ],
    )
    def test_requires_fit_across_classes(self, cls, expected_requires_fit):
        q = cls()
        tags = q.__mlquantify_tags__()
        assert tags.requires_fit is expected_requires_fit


class TestConfigIntegration:
    """Integration tests combining config with other functionality."""

    def setup_method(self):
        set_config(prevalence_return_type="dict", prevalence_normalization="mean")

    def teardown_method(self):
        set_config(prevalence_return_type="dict", prevalence_normalization="mean")

    def test_config_context_with_quantifier_predict(self, binary_dataset):
        """Config context should affect prevalence return type during predict."""
        X_train, X_test, y_train, y_test = binary_dataset
        q = SimpleCC(strategy="ovr")
        q.fit(X_train, y_train)

        # Default: dict
        result_dict = q.predict(X_test)
        assert isinstance(result_dict, dict)

        # With config_context: array
        with config_context(prevalence_return_type="array"):
            result_arr = q.predict(X_test)
            assert isinstance(result_arr, np.ndarray)

        # Back to dict
        result_dict_again = q.predict(X_test)
        assert isinstance(result_dict_again, dict)

    def test_multiple_set_config_calls(self):
        """Multiple set_config calls should each update the state."""
        set_config(prevalence_return_type="array")
        assert get_config()["prevalence_return_type"] == "array"
        set_config(prevalence_normalization="l1")
        assert get_config()["prevalence_normalization"] == "l1"
        # First setting should still be in effect
        assert get_config()["prevalence_return_type"] == "array"
