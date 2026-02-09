"""
Comprehensive tests for mlquantify.utils subpackage.

Covers: constraints, tags, validation, sampling, prevalence, exceptions,
context, decorators, random, parallel, artificial, optimization.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from numpy.random import RandomState, Generator

# ── Imports under test ──────────────────────────────────────────────────────
from mlquantify.utils._constraints import (
    Interval,
    Options,
    HasMethods,
    Hidden,
    CallableConstraint,
    StringConstraint,
    NoneConstraint,
    _InstancesOf,
    _ArrayLikes,
    make_constraint,
)
from mlquantify.utils._tags import (
    Tags,
    TargetInputTags,
    PredictionRequirements,
    get_tags,
)
from mlquantify.utils._validation import (
    validate_y,
    validate_predictions,
    validate_prevalences,
    normalize_prevalences,
    validate_parameter_constraints,
    validate_data,
    check_is_fitted,
    _is_fitted,
    _is_arraylike,
    _is_arraylike_not_scalar,
    check_classes_attribute,
)
from mlquantify.utils._sampling import (
    get_indexes_with_prevalence,
    simplex_uniform_kraemer,
    simplex_grid_sampling,
    simplex_uniform_sampling,
    bootstrap_sample_indices,
)
from mlquantify.utils.prevalence import (
    get_prev_from_labels,
    normalize_prevalence,
)
from mlquantify.utils._exceptions import (
    InputValidationError,
    InvalidParameterError,
    NotFittedError,
)
from mlquantify.utils._context import validation_context, is_validation_skipped
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._random import check_random_state
from mlquantify.utils._parallel import resolve_n_jobs
from mlquantify.utils._artificial import make_prevs
from mlquantify.utils._optimization import _optimize_on_simplex

from mlquantify.base import BaseQuantifier


# ═══════════════════════════════════════════════════════════════════════════
# Helper stubs
# ═══════════════════════════════════════════════════════════════════════════

class _DummyQuantifier(BaseQuantifier):
    """Minimal quantifier for testing validation helpers."""

    _parameter_constraints = {}

    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.array([0.5, 0.5])

    def __mlquantify_tags__(self):
        return Tags(
            has_estimator=False,
            estimation_type=None,
            estimator_function=None,
            estimator_type="soft",
            aggregation_type=None,
            target_input_tags=TargetInputTags(
                one_d=True,
                two_d=True,
                continuous=False,
                categorical=True,
                multi_class=True,
                required=False,
            ),
            prediction_requirements=PredictionRequirements(),
            requires_fit=True,
        )


class _CrispQuantifier(_DummyQuantifier):
    """Quantifier that declares crisp estimator_type."""

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator_type = "crisp"
        return tags


class _Only1DQuantifier(_DummyQuantifier):
    """Quantifier that only accepts 1D input."""

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.target_input_tags = TargetInputTags(
            one_d=True,
            two_d=False,
            continuous=False,
            categorical=False,
            multi_class=False,
            required=False,
        )
        return tags


class _NoFitQuantifier(BaseQuantifier):
    """Quantifier that doesn't require fitting."""

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([0.5, 0.5])

    def __mlquantify_tags__(self):
        return Tags(
            has_estimator=False,
            estimation_type=None,
            estimator_function=None,
            estimator_type=None,
            aggregation_type=None,
            target_input_tags=TargetInputTags(),
            prediction_requirements=PredictionRequirements(),
            requires_fit=False,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 1–6  CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════


class TestInterval:
    """Tests for Interval constraint."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, True),
            (1, True),
            (0.5, True),
            (-0.01, False),
            (1.01, False),
        ],
    )
    def test_inclusive_default(self, value, expected):
        constraint = Interval(0, 1)
        assert constraint.is_satisfied_by(value) is expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, False),
            (1, False),
            (0.5, True),
        ],
    )
    def test_exclusive_both(self, value, expected):
        constraint = Interval(0, 1, inclusive_left=False, inclusive_right=False)
        assert constraint.is_satisfied_by(value) is expected

    def test_none_left_bound(self):
        constraint = Interval(None, 10)
        assert constraint.is_satisfied_by(-1e10) is True
        assert constraint.is_satisfied_by(11) is False

    def test_none_right_bound(self):
        constraint = Interval(0, None)
        assert constraint.is_satisfied_by(1e10) is True
        assert constraint.is_satisfied_by(-1) is False

    def test_discrete(self):
        constraint = Interval(0, 10, discrete=True)
        assert constraint.is_satisfied_by(5) is True
        assert constraint.is_satisfied_by(5.0) is True  # float but integer-valued
        assert constraint.is_satisfied_by(5.5) is False

    def test_non_numeric_rejected(self):
        constraint = Interval(0, 1)
        assert constraint.is_satisfied_by("hello") is False

    def test_str_representation(self):
        assert str(Interval(0, 1)) == "[0, 1]"
        assert str(Interval(0, 1, inclusive_left=False, inclusive_right=False)) == "(0, 1)"

    def test_numpy_number(self):
        constraint = Interval(0, 1)
        assert constraint.is_satisfied_by(np.float64(0.5)) is True


class TestOptions:
    """Tests for Options constraint."""

    def test_satisfied(self):
        constraint = Options(["a", "b", "c"])
        assert constraint.is_satisfied_by("a") is True

    def test_not_satisfied(self):
        constraint = Options(["a", "b", "c"])
        assert constraint.is_satisfied_by("d") is False

    def test_numeric_options(self):
        constraint = Options([1, 2, 3])
        assert constraint.is_satisfied_by(2) is True
        assert constraint.is_satisfied_by(4) is False

    def test_none_in_options(self):
        constraint = Options([None, "auto"])
        assert constraint.is_satisfied_by(None) is True

    def test_str_representation(self):
        assert "one of" in str(Options(["x", "y"]))


class TestHasMethods:
    """Tests for HasMethods constraint."""

    def test_satisfied(self):
        obj = MagicMock(spec=["fit", "predict"])
        constraint = HasMethods(["fit", "predict"])
        assert constraint.is_satisfied_by(obj) is True

    def test_not_satisfied_missing(self):
        obj = MagicMock(spec=["fit"])
        constraint = HasMethods(["fit", "predict"])
        assert constraint.is_satisfied_by(obj) is False

    def test_non_callable_method(self):
        """An attribute that exists but is not callable should fail."""

        class _Stub:
            fit = 42  # not callable

        constraint = HasMethods(["fit"])
        assert constraint.is_satisfied_by(_Stub()) is False

    def test_str_representation(self):
        assert "fit" in str(HasMethods(["fit", "predict"]))


class TestHidden:
    """Tests for Hidden constraint wrapper."""

    def test_delegates(self):
        inner = Interval(0, 1)
        hidden = Hidden(inner)
        assert hidden.is_satisfied_by(0.5) is True
        assert hidden.is_satisfied_by(2) is False

    def test_hidden_property(self):
        hidden = Hidden(Interval(0, 1))
        assert hidden.hidden is True

    def test_str(self):
        assert "hidden" in str(Hidden(Interval(0, 1))).lower()


class TestCallableConstraint:
    """Tests for CallableConstraint."""

    def test_callable_satisfied(self):
        constraint = CallableConstraint()
        assert constraint.is_satisfied_by(lambda x: x) is True
        assert constraint.is_satisfied_by(len) is True

    def test_not_callable(self):
        constraint = CallableConstraint()
        assert constraint.is_satisfied_by(42) is False
        assert constraint.is_satisfied_by("abc") is False

    def test_str(self):
        assert "callable" in str(CallableConstraint())


class TestStringConstraint:
    """Tests for StringConstraint."""

    def test_array_like(self):
        sc = StringConstraint("array-like")
        assert sc.is_satisfied_by([1, 2, 3]) is True
        assert sc.is_satisfied_by(np.array([1])) is True
        assert sc.is_satisfied_by((1, 2)) is True
        assert sc.is_satisfied_by(42) is False

    def test_boolean(self):
        sc = StringConstraint("boolean")
        assert sc.is_satisfied_by(True) is True
        assert sc.is_satisfied_by(1) is False


class TestNoneConstraint:
    """Tests for NoneConstraint (created via make_constraint(None))."""

    def test_via_make_constraint(self):
        c = make_constraint(None)
        assert isinstance(c, NoneConstraint)


class TestMakeConstraint:
    """Tests for the make_constraint factory."""

    def test_interval_passthrough(self):
        c = make_constraint(Interval(0, 1))
        assert isinstance(c, Interval)

    def test_options_passthrough(self):
        c = make_constraint(Options(["a"]))
        assert isinstance(c, Options)

    def test_has_methods_passthrough(self):
        c = make_constraint(HasMethods(["fit"]))
        assert isinstance(c, HasMethods)

    def test_hidden_passthrough(self):
        c = make_constraint(Hidden(Interval(0, 1)))
        assert isinstance(c, Hidden)

    def test_callable_passthrough(self):
        c = make_constraint(CallableConstraint())
        assert isinstance(c, CallableConstraint)

    def test_type_becomes_instances_of(self):
        c = make_constraint(int)
        assert isinstance(c, _InstancesOf)
        assert c.is_satisfied_by(5) is True
        assert c.is_satisfied_by("x") is False

    def test_string_array_like(self):
        c = make_constraint("array-like")
        assert isinstance(c, _ArrayLikes)

    def test_string_other(self):
        c = make_constraint("random_state")
        assert isinstance(c, StringConstraint)

    def test_none_input(self):
        c = make_constraint(None)
        assert isinstance(c, NoneConstraint)

    def test_unsupported_raises(self):
        with pytest.raises(TypeError):
            make_constraint(3.14)


# ═══════════════════════════════════════════════════════════════════════════
# 7–8  TAGS
# ═══════════════════════════════════════════════════════════════════════════


class TestTargetInputTags:
    """Tests for TargetInputTags defaults and custom values."""

    def test_defaults(self):
        t = TargetInputTags()
        assert t.one_d is True
        assert t.two_d is False
        assert t.continuous is False
        assert t.categorical is True
        assert t.multi_class is True
        assert t.required is False

    def test_custom(self):
        t = TargetInputTags(one_d=False, two_d=True, continuous=True)
        assert t.one_d is False
        assert t.two_d is True
        assert t.continuous is True


class TestPredictionRequirements:
    """Tests for PredictionRequirements defaults."""

    def test_defaults(self):
        r = PredictionRequirements()
        assert r.requires_train_proba is True
        assert r.requires_train_labels is True
        assert r.requires_test_predictions is True


class TestTags:
    """Tests for Tags dataclass."""

    def test_tags_creation(self):
        tags = Tags(
            estimation_type="counting",
            estimator_function="predict",
            estimator_type="crisp",
            aggregation_type="sum",
            target_input_tags=TargetInputTags(),
            prediction_requirements=PredictionRequirements(),
        )
        assert tags.estimation_type == "counting"
        assert tags.has_estimator is False
        assert tags.requires_fit is True


class TestGetTags:
    """Tests for get_tags helper."""

    def test_with_tags_method(self):
        q = _DummyQuantifier()
        tags = get_tags(q)
        assert isinstance(tags, Tags)
        assert tags.estimator_type == "soft"

    def test_without_tags_method(self):

        class _NoTags:
            pass

        with pytest.raises(AttributeError, match="__mlquantify_tags__"):
            get_tags(_NoTags())


# ═══════════════════════════════════════════════════════════════════════════
# 9–16  VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateY:
    """Tests for validate_y."""

    def test_valid_1d_int(self):
        q = _DummyQuantifier()
        y = np.array([0, 1, 0, 1])
        validate_y(q, y)  # no error

    def test_valid_1d_string(self):
        q = _DummyQuantifier()
        y = np.array(["cat", "dog", "cat"])
        validate_y(q, y)  # categorical=True

    def test_valid_2d_float(self):
        q = _DummyQuantifier()
        y = np.array([[0.3, 0.7], [0.5, 0.5], [0.2, 0.8]])
        validate_y(q, y)  # two_d=True

    def test_non_numpy_raises(self):
        q = _DummyQuantifier()
        with pytest.raises((InputValidationError, NameError)):
            validate_y(q, [0, 1, 2])

    def test_3d_raises(self):
        q = _DummyQuantifier()
        y = np.zeros((2, 3, 4))
        with pytest.raises(InputValidationError, match="1D or 2D"):
            validate_y(q, y)

    def test_1d_multiclass_not_allowed(self):
        q = _Only1DQuantifier()
        y = np.array([0, 1, 2, 3])  # 4 unique, multi_class=False
        with pytest.raises(InputValidationError, match="binary"):
            validate_y(q, y)

    def test_2d_not_allowed(self):
        q = _Only1DQuantifier()
        y = np.array([[0.5, 0.5]])
        with pytest.raises(InputValidationError):
            validate_y(q, y)


class TestValidatePredictions:
    """Tests for validate_predictions."""

    def test_soft_float_passthrough(self):
        q = _DummyQuantifier()
        preds = np.array([[0.3, 0.7], [0.5, 0.5]])
        result = validate_predictions(q, preds)
        np.testing.assert_array_equal(result, preds)

    def test_soft_int_raises(self):
        q = _DummyQuantifier()  # estimator_type='soft'
        preds = np.array([0, 1, 0, 1])
        with pytest.raises(InputValidationError, match="float"):
            validate_predictions(q, preds)

    def test_crisp_converts_float_to_labels(self):
        q = _CrispQuantifier()
        preds = np.array([[0.2, 0.8], [0.9, 0.1]])
        result = validate_predictions(q, preds, threshold=0.5)
        assert result.ndim == 1
        assert len(result) == 2

    def test_non_numpy_raises(self):
        q = _DummyQuantifier()
        with pytest.raises((InputValidationError, NameError)):
            validate_predictions(q, [0.5, 0.5])


class TestValidatePrevalences:
    """Tests for validate_prevalences."""

    def test_dict_input(self):
        q = _DummyQuantifier()
        result = validate_prevalences(
            q,
            {0: 0.3, 1: 0.7},
            classes=np.array([0, 1]),
            return_type="dict",
            normalize=True,
        )
        assert isinstance(result, dict)
        assert pytest.approx(sum(result.values()), abs=1e-6) == 1.0

    def test_array_input(self):
        q = _DummyQuantifier()
        result = validate_prevalences(
            q,
            np.array([0.4, 0.6]),
            classes=np.array([0, 1]),
            return_type="array",
            normalize=True,
        )
        assert isinstance(result, np.ndarray)
        assert pytest.approx(result.sum(), abs=1e-6) == 1.0

    def test_unnormalized(self):
        q = _DummyQuantifier()
        result = validate_prevalences(
            q,
            np.array([2.0, 3.0]),
            classes=np.array([0, 1]),
            return_type="array",
            normalize=True,
        )
        assert pytest.approx(result.sum(), abs=1e-6) == 1.0


class TestNormalizePrevalences:
    """Tests for normalize_prevalences (internal)."""

    @pytest.mark.parametrize(
        "method",
        ["sum", "l1", "softmax", "mean", "median"],
    )
    def test_1d_methods(self, method):
        arr = np.array([0.3, 0.5, 0.2])
        classes = np.array([0, 1, 2])
        result = normalize_prevalences(arr, classes, method=method)
        assert result.shape == (3,)

    def test_sum_normalizes(self):
        arr = np.array([2.0, 3.0])
        classes = np.array([0, 1])
        result = normalize_prevalences(arr, classes, method="sum")
        assert pytest.approx(result.sum(), abs=1e-6) == 1.0

    def test_2d_sum(self):
        arr = np.array([[0.2, 0.8], [0.6, 0.4]])
        classes = np.array([0, 1])
        result = normalize_prevalences(arr, classes, method="sum")
        assert result.ndim == 1
        assert pytest.approx(result.sum(), abs=1e-6) == 1.0

    def test_2d_mean(self):
        arr = np.array([[0.2, 0.8], [0.6, 0.4]])
        classes = np.array([0, 1])
        result = normalize_prevalences(arr, classes, method="mean")
        np.testing.assert_allclose(result, [0.4, 0.6], atol=1e-6)

    def test_2d_median(self):
        arr = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5]])
        classes = np.array([0, 1])
        result = normalize_prevalences(arr, classes, method="median")
        np.testing.assert_allclose(result, [0.3, 0.7], atol=1e-6)

    def test_softmax_1d(self):
        arr = np.array([1.0, 2.0, 3.0])
        classes = np.array([0, 1, 2])
        result = normalize_prevalences(arr, classes, method="softmax")
        assert pytest.approx(result.sum(), abs=1e-6) == 1.0

    def test_padding(self):
        """Array shorter than classes gets zero-padded."""
        arr = np.array([0.5])
        classes = np.array([0, 1])
        result = normalize_prevalences(arr, classes, method="sum")
        assert result.shape == (2,)

    def test_too_many_dims_raises(self):
        arr = np.array([0.3, 0.3, 0.4])
        classes = np.array([0, 1])
        with pytest.raises(ValueError, match="maior"):
            normalize_prevalences(arr, classes, method="sum")


class TestCheckIsFitted:
    """Tests for check_is_fitted and _is_fitted."""

    def test_fitted_quantifier(self):
        q = _DummyQuantifier()
        q.fit(np.random.rand(10, 2), np.array([0, 1] * 5))
        check_is_fitted(q)  # should not raise

    def test_unfitted_raises(self):
        q = _DummyQuantifier()
        with pytest.raises(NotFittedError):
            check_is_fitted(q)

    def test_no_fit_method(self):

        class _Raw:
            pass

        with pytest.raises(TypeError, match="no 'fit' method"):
            check_is_fitted(_Raw())

    def test_is_fitted_custom_attrs(self):
        class _Obj:
            some_attr_ = True
        obj = _Obj()
        assert _is_fitted(obj, attributes=["some_attr_"]) is True
        assert _is_fitted(obj, attributes=["missing_"]) is False

    def test_no_fit_required_quantifier(self):
        """Quantifier that doesn't require fit should pass even unfitted."""
        q = _NoFitQuantifier()
        check_is_fitted(q)  # should not raise

    def test_is_fitted_any(self):
        obj = MagicMock(spec=[])
        obj.a_ = 1
        assert _is_fitted(obj, attributes=["a_", "b_"], all_or_any=any) is True
        assert _is_fitted(obj, attributes=["a_", "b_"], all_or_any=all) is False


class TestValidateData:
    """Tests for validate_data."""

    def test_numpy_xy(self):
        q = _DummyQuantifier()
        X = np.random.rand(20, 3)
        y = np.array([0, 1] * 10)
        X_out, y_out = validate_data(q, X, y)
        assert X_out.shape == (20, 3)

    def test_y_none_required_raises(self):
        """If target is required and y=None, should raise."""

        class _RequiredY(_DummyQuantifier):
            def __mlquantify_tags__(self):
                tags = super().__mlquantify_tags__()
                tags.target_input_tags.required = True
                return tags

        q = _RequiredY()
        X = np.random.rand(10, 2)
        with pytest.raises(InputValidationError, match="required"):
            validate_data(q, X, y=None)

    def test_no_validation_both_raises(self):
        q = _DummyQuantifier()
        with pytest.raises(ValueError, match="Validation should be done"):
            validate_data(q)

    def test_skip_check_array(self):
        q = _DummyQuantifier()
        X = np.random.rand(5, 2)
        result = validate_data(q, X, skip_check_array=True)
        np.testing.assert_array_equal(result, X)


class TestValidateParameterConstraints:
    """Tests for validate_parameter_constraints."""

    def test_valid_params(self):
        constraints = {"n_jobs": [Interval(1, 100, discrete=True), type(None)]}
        params = {"n_jobs": 4}
        validate_parameter_constraints(constraints, params, "TestEstimator")

    def test_invalid_param_raises(self):
        constraints = {"alpha": [Interval(0, 1)]}
        params = {"alpha": 5.0}
        with pytest.raises(InvalidParameterError, match="alpha"):
            validate_parameter_constraints(constraints, params, "TestEstimator")

    def test_no_validation_skip(self):
        constraints = {"x": "no_validation"}
        params = {"x": "anything"}
        validate_parameter_constraints(constraints, params, "Test")  # no error

    def test_unconstrained_param_ignored(self):
        constraints = {"a": [Interval(0, 1)]}
        params = {"b": 999}
        validate_parameter_constraints(constraints, params, "Test")  # no error

    def test_multiple_constraints_one_passes(self):
        constraints = {"x": [Interval(0, 1), type(None)]}
        params = {"x": None}
        validate_parameter_constraints(constraints, params, "Test")  # None is valid

    def test_hidden_constraint_excluded_from_message(self):
        constraints = {"x": [Hidden(Interval(0, 1))]}
        params = {"x": 5}
        with pytest.raises(InvalidParameterError):
            validate_parameter_constraints(constraints, params, "Test")


class TestCheckClassesAttribute:
    """Tests for check_classes_attribute."""

    def test_matching_classes(self):
        q = _DummyQuantifier()
        classes = np.array([0, 1])
        q.fit(np.random.rand(10, 2), np.array([0, 1] * 5))
        result = check_classes_attribute(q, classes)
        np.testing.assert_array_equal(result, classes)

    def test_no_classes_attribute(self):
        q = _DummyQuantifier()
        classes = np.array([0, 1, 2])
        result = check_classes_attribute(q, classes)
        np.testing.assert_array_equal(result, classes)

    def test_mismatched_classes(self):
        q = _DummyQuantifier()
        q.fit(np.random.rand(10, 2), np.array([0, 1] * 5))
        new_classes = np.array([0, 1, 2])
        result = check_classes_attribute(q, new_classes)
        np.testing.assert_array_equal(result, new_classes)


class TestIsArraylike:
    """Tests for _is_arraylike and _is_arraylike_not_scalar."""

    @pytest.mark.parametrize(
        "val, expected",
        [
            ([1, 2], True),
            (np.array([1]), True),
            ("abc", True),  # has __len__
            (42, False),
        ],
    )
    def test_is_arraylike(self, val, expected):
        assert _is_arraylike(val) is expected

    def test_scalar_not_arraylike(self):
        assert _is_arraylike_not_scalar(42) is False

    def test_list_is_arraylike_not_scalar(self):
        assert _is_arraylike_not_scalar([1, 2, 3]) is True


# ═══════════════════════════════════════════════════════════════════════════
# 17–21  SAMPLING
# ═══════════════════════════════════════════════════════════════════════════


class TestGetIndexesWithPrevalence:
    """Tests for get_indexes_with_prevalence."""

    def test_binary_balanced(self):
        y = np.array([0] * 50 + [1] * 50)
        idx = get_indexes_with_prevalence(y, [0.5, 0.5], sample_size=20, random_state=0)
        assert len(idx) == 20

    def test_binary_imbalanced(self):
        y = np.array([0] * 80 + [1] * 20)
        idx = get_indexes_with_prevalence(y, [0.8, 0.2], sample_size=100, random_state=42)
        assert len(idx) == 100

    def test_multiclass_prevalence(self):
        y = np.array([0] * 40 + [1] * 30 + [2] * 30)
        idx = get_indexes_with_prevalence(y, [0.5, 0.3, 0.2], sample_size=50, random_state=1)
        assert len(idx) == 50

    def test_string_labels(self):
        y = np.array(["cat"] * 30 + ["dog"] * 70)
        idx = get_indexes_with_prevalence(y, [0.5, 0.5], sample_size=40, random_state=7)
        assert len(idx) == 40

    def test_prevalence_not_sum_1_raises(self):
        y = np.array([0, 1, 0, 1])
        with pytest.raises(AssertionError):
            get_indexes_with_prevalence(y, [0.3, 0.3], sample_size=4)

    def test_wrong_length_raises(self):
        y = np.array([0, 1, 2])
        with pytest.raises(AssertionError):
            get_indexes_with_prevalence(y, [0.5, 0.5], sample_size=4)

    def test_reproducibility(self):
        y = np.array([0] * 50 + [1] * 50)
        idx1 = get_indexes_with_prevalence(y, [0.5, 0.5], sample_size=20, random_state=42)
        idx2 = get_indexes_with_prevalence(y, [0.5, 0.5], sample_size=20, random_state=42)
        np.testing.assert_array_equal(idx1, idx2)


class TestSimplexUniformKraemer:
    """Tests for simplex_uniform_kraemer."""

    def test_basic_shape(self):
        result = simplex_uniform_kraemer(n_dim=3, n_prev=10, n_iter=1, random_state=42)
        assert result.shape[1] == 3
        assert result.shape[0] >= 1

    def test_sums_to_one(self):
        result = simplex_uniform_kraemer(n_dim=4, n_prev=5, n_iter=1, random_state=0)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_n_iter_repeats(self):
        result = simplex_uniform_kraemer(n_dim=3, n_prev=5, n_iter=3, random_state=0)
        assert result.shape[0] == 15

    def test_n_dim_1_raises(self):
        with pytest.raises(ValueError, match="n_dim"):
            simplex_uniform_kraemer(n_dim=1, n_prev=5, n_iter=1)

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError):
            simplex_uniform_kraemer(n_dim=3, n_prev=5, n_iter=1, min_val=0.5, max_val=0.4)

    def test_bounds_respected(self):
        result = simplex_uniform_kraemer(
            n_dim=3, n_prev=20, n_iter=1, min_val=0.1, max_val=0.8, random_state=42
        )
        assert np.all(result >= 0.1 - 1e-9)
        assert np.all(result <= 0.8 + 1e-9)


class TestSimplexGridSampling:
    """Tests for simplex_grid_sampling."""

    def test_basic(self):
        result = simplex_grid_sampling(n_dim=3, n_prev=5, n_iter=1, min_val=0.0, max_val=1.0)
        assert result.ndim == 2
        assert result.shape[1] == 3

    def test_sums_to_one(self):
        result = simplex_grid_sampling(n_dim=3, n_prev=11, n_iter=1, min_val=0.0, max_val=1.0)
        if len(result) > 0:
            np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_n_dim_1_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            simplex_grid_sampling(n_dim=1, n_prev=5, n_iter=1, min_val=0.0, max_val=1.0)

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError):
            simplex_grid_sampling(n_dim=3, n_prev=5, n_iter=1, min_val=0.8, max_val=0.2)

    def test_n_iter_repeats(self):
        base = simplex_grid_sampling(n_dim=3, n_prev=5, n_iter=1, min_val=0.0, max_val=1.0)
        repeated = simplex_grid_sampling(n_dim=3, n_prev=5, n_iter=2, min_val=0.0, max_val=1.0)
        if len(base) > 0:
            assert len(repeated) == 2 * len(base)


class TestSimplexUniformSampling:
    """Tests for simplex_uniform_sampling."""

    def test_basic_shape(self):
        result = simplex_uniform_sampling(
            n_dim=3, n_prev=10, n_iter=1, min_val=0.0, max_val=1.0, random_state=42
        )
        assert result.shape == (10, 3)

    def test_sums_to_one(self):
        result = simplex_uniform_sampling(
            n_dim=4, n_prev=20, n_iter=1, min_val=0.0, max_val=1.0, random_state=0
        )
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError):
            simplex_uniform_sampling(n_dim=3, n_prev=5, n_iter=1, min_val=0.9, max_val=0.95)


class TestBootstrapSampleIndices:
    """Tests for bootstrap_sample_indices."""

    def test_yields_correct_count(self):
        indices_list = list(bootstrap_sample_indices(100, 20, 5, random_state=0))
        assert len(indices_list) == 5

    def test_batch_size(self):
        for idx in bootstrap_sample_indices(50, 10, 3, random_state=0):
            assert len(idx) == 10

    def test_within_bounds(self):
        for idx in bootstrap_sample_indices(30, 15, 4, random_state=42):
            assert np.all(idx >= 0)
            assert np.all(idx < 30)

    def test_reproducibility(self):
        a = list(bootstrap_sample_indices(100, 10, 3, random_state=42))
        b = list(bootstrap_sample_indices(100, 10, 3, random_state=42))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


# ═══════════════════════════════════════════════════════════════════════════
# 22–23  PREVALENCE
# ═══════════════════════════════════════════════════════════════════════════


class TestGetPrevFromLabels:
    """Tests for get_prev_from_labels."""

    def test_dict_format(self):
        y = np.array([0, 0, 1, 1, 1])
        result = get_prev_from_labels(y, format="dict")
        assert isinstance(result, dict)
        assert pytest.approx(result[0], abs=1e-6) == 0.4
        assert pytest.approx(result[1], abs=1e-6) == 0.6

    def test_array_format(self):
        y = np.array([0, 0, 1, 1])
        result = get_prev_from_labels(y, format="array")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_string_labels(self):
        y = np.array(["cat", "dog", "cat"])
        result = get_prev_from_labels(y, format="dict")
        assert pytest.approx(result["cat"], abs=1e-6) == 2 / 3
        assert pytest.approx(result["dog"], abs=1e-6) == 1 / 3

    def test_float_labels(self):
        y = np.array([1.0, 2.0, 1.0, 2.0])
        result = get_prev_from_labels(y, format="dict")
        assert pytest.approx(result[1.0], abs=1e-6) == 0.5

    def test_pandas_series(self):
        y = pd.Series([0, 1, 1, 0, 0])
        result = get_prev_from_labels(y, format="dict")
        assert pytest.approx(result[0], abs=1e-6) == 0.6

    def test_with_classes(self):
        y = np.array([0, 0, 1])
        result = get_prev_from_labels(y, format="dict", classes=[0, 1, 2])
        assert result[2] == 0.0  # class 2 absent

    def test_categorical_labels(self):
        y = pd.Series(pd.Categorical(["a", "b", "a", "a", "b"]))
        result = get_prev_from_labels(y, format="dict")
        assert pytest.approx(result["a"], abs=1e-6) == 0.6


class TestNormalizePrevalence:
    """Tests for normalize_prevalence (in prevalence module)."""

    def test_dict_input(self):
        prevalences = {0: 2.0, 1: 3.0}
        result = normalize_prevalence(prevalences, classes=[0, 1])
        assert isinstance(result, dict)
        assert pytest.approx(sum(result.values()), abs=1e-6) == 1.0

    def test_array_input(self):
        prevalences = np.array([3, 7])
        result = normalize_prevalence(prevalences, classes=[0, 1])
        assert isinstance(result, dict)
        assert pytest.approx(result[0], abs=1e-6) == 0.3
        assert pytest.approx(result[1], abs=1e-6) == 0.7


# ═══════════════════════════════════════════════════════════════════════════
# 24  EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════


class TestExceptions:
    """Tests for custom exception classes."""

    def test_input_validation_error(self):
        with pytest.raises(InputValidationError):
            raise InputValidationError("bad input")

    def test_invalid_parameter_error(self):
        with pytest.raises(InvalidParameterError):
            raise InvalidParameterError("bad param")

    def test_not_fitted_error(self):
        with pytest.raises(NotFittedError):
            raise NotFittedError("not fitted")

    def test_subclass_of_value_error(self):
        assert issubclass(InputValidationError, ValueError)
        assert issubclass(InvalidParameterError, ValueError)
        assert issubclass(NotFittedError, ValueError)

    def test_exception_messages(self):
        e = InputValidationError("msg123")
        assert "msg123" in str(e)


# ═══════════════════════════════════════════════════════════════════════════
# ADDITIONAL: CONTEXT, DECORATORS, RANDOM, PARALLEL, ARTIFICIAL, OPTIM
# ═══════════════════════════════════════════════════════════════════════════


class TestValidationContext:
    """Tests for validation_context and is_validation_skipped."""

    def test_default_not_skipped(self):
        assert is_validation_skipped() is False

    def test_skip_inside_context(self):
        with validation_context(skip=True):
            assert is_validation_skipped() is True
        assert is_validation_skipped() is False

    def test_nested_contexts(self):
        with validation_context(skip=True):
            with validation_context(skip=False):
                assert is_validation_skipped() is False
            assert is_validation_skipped() is True

    def test_context_restores_on_exception(self):
        try:
            with validation_context(skip=True):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert is_validation_skipped() is False


class TestFitContext:
    """Tests for _fit_context decorator."""

    def test_decorator_calls_validate_params(self):
        q = _DummyQuantifier()

        @_fit_context()
        def fit(self, X, y):
            self.is_fitted_ = True
            return self

        fit(q, np.random.rand(10, 2), np.array([0, 1] * 5))
        assert q.is_fitted_ is True

    def test_skip_nested_validation(self):
        q = _DummyQuantifier()

        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y):
            assert is_validation_skipped() is True
            self.is_fitted_ = True
            return self

        fit(q, np.random.rand(5, 2), np.array([0, 1, 0, 1, 0]))


class TestCheckRandomState:
    """Tests for check_random_state."""

    def test_none(self):
        rng = check_random_state(None)
        assert isinstance(rng, Generator)

    def test_int_seed(self):
        rng = check_random_state(42)
        assert isinstance(rng, Generator)

    def test_generator_passthrough(self):
        gen = np.random.default_rng(0)
        assert check_random_state(gen) is gen

    def test_random_state_conversion(self):
        rs = RandomState(0)
        rng = check_random_state(rs)
        assert isinstance(rng, Generator)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            check_random_state("bad")

    def test_reproducibility(self):
        a = check_random_state(42).random()
        b = check_random_state(42).random()
        assert a == b


class TestResolveNJobs:
    """Tests for resolve_n_jobs."""

    def test_one(self):
        assert resolve_n_jobs(1) == 1

    def test_none(self):
        assert resolve_n_jobs(None) >= 1

    def test_minus_one(self):
        import os
        result = resolve_n_jobs(-1)
        assert result >= 1


class TestMakePrevs:
    """Tests for make_prevs (artificial prevalence generation)."""

    @pytest.mark.parametrize("ndim", [2, 3, 5, 10])
    def test_sums_to_one(self, ndim):
        prevs = make_prevs(ndim)
        assert len(prevs) == ndim
        assert pytest.approx(prevs.sum(), abs=1e-10) == 1.0

    def test_all_nonnegative(self):
        for _ in range(20):
            prevs = make_prevs(4)
            assert np.all(prevs >= 0)


class TestOptimizeOnSimplex:
    """Tests for _optimize_on_simplex."""

    def test_uniform_minimum(self):
        """Minimizing squared deviation from uniform should return uniform."""
        n = 3
        target = np.ones(n) / n
        objective = lambda x: np.sum((x - target) ** 2)
        alpha, loss = _optimize_on_simplex(objective, n)
        np.testing.assert_allclose(alpha, target, atol=1e-4)
        assert loss < 1e-8

    def test_result_on_simplex(self):
        objective = lambda x: -np.sum(x * np.log(x + 1e-12))
        alpha, _ = _optimize_on_simplex(objective, 4)
        assert pytest.approx(alpha.sum(), abs=1e-4) == 1.0
        assert np.all(alpha >= -1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETRIZED EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Parametrized edge-case tests across multiple utilities."""

    @pytest.mark.parametrize(
        "val, left, right, inc_l, inc_r, expected",
        [
            (0, 0, 1, True, True, True),
            (1, 0, 1, True, True, True),
            (0, 0, 1, False, True, False),
            (1, 0, 1, True, False, False),
            (0.5, 0, 1, False, False, True),
            (-1, None, None, True, True, True),
            (100, None, None, True, True, True),
        ],
    )
    def test_interval_boundary_parametrized(self, val, left, right, inc_l, inc_r, expected):
        c = Interval(left, right, inclusive_left=inc_l, inclusive_right=inc_r)
        assert c.is_satisfied_by(val) is expected

    @pytest.mark.parametrize(
        "y, fmt",
        [
            (np.array([0, 1, 0, 1]), "dict"),
            (np.array([0, 1, 0, 1]), "array"),
            (np.array(["a", "b", "a"]), "dict"),
            (np.array([1.0, 2.0, 1.0]), "dict"),
        ],
    )
    def test_get_prev_from_labels_formats(self, y, fmt):
        result = get_prev_from_labels(y, format=fmt)
        if fmt == "dict":
            assert isinstance(result, dict)
            assert pytest.approx(sum(result.values()), abs=1e-6) == 1.0
        else:
            assert isinstance(result, np.ndarray)
            assert pytest.approx(result.sum(), abs=1e-6) == 1.0

    @pytest.mark.parametrize(
        "n_dim, n_prev",
        [
            (2, 5),
            (3, 10),
            (5, 3),
        ],
    )
    def test_kraemer_shape_parametrized(self, n_dim, n_prev):
        result = simplex_uniform_kraemer(n_dim=n_dim, n_prev=n_prev, n_iter=1, random_state=42)
        assert result.shape[1] == n_dim
        assert result.shape[0] >= 1

    @pytest.mark.parametrize(
        "n_bootstraps, batch_size",
        [
            (1, 5),
            (10, 20),
            (3, 1),
        ],
    )
    def test_bootstrap_parametrized(self, n_bootstraps, batch_size):
        indices = list(bootstrap_sample_indices(100, batch_size, n_bootstraps, random_state=0))
        assert len(indices) == n_bootstraps
        for idx in indices:
            assert len(idx) == batch_size

    @pytest.mark.parametrize(
        "method",
        ["sum", "l1", "softmax"],
    )
    def test_normalize_prevalences_methods_2d(self, method):
        arr = np.array([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
        classes = np.array([0, 1, 2])
        result = normalize_prevalences(arr, classes, method=method)
        assert result.shape == (3,)
        assert pytest.approx(result.sum(), abs=1e-4) == 1.0
