import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlquantify._config import config_context
from mlquantify.counting import CC, PCC, GACC, GPACC, TAC, TX, TMAX, FM
from mlquantify.utils._exceptions import InvalidParameterError


COUNTING_QUANTIFIERS = [CC, PCC, GACC, GPACC, TAC, TX, TMAX, FM]


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("quantifier_class", COUNTING_QUANTIFIERS)
def test_counting_methods_fit_predict_binary(quantifier_class, binary_dataset):
    X, y = binary_dataset
    q = quantifier_class(estimator=LogisticRegression(max_iter=1000, random_state=42))

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)


@pytest.mark.parametrize("quantifier_class", [CC, PCC, GACC, GPACC, FM])
def test_counting_methods_fit_predict_multiclass(quantifier_class, multiclass_dataset):
    X, y = multiclass_dataset
    q = quantifier_class(estimator=LogisticRegression(max_iter=1000, random_state=42))

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)


def test_counting_respects_dict_output_config(binary_dataset):
    X, y = binary_dataset
    q = CC(estimator=LogisticRegression(max_iter=1000, random_state=42))

    q.fit(X, y)

    with config_context(prevalence_return_type="dict"):
        prevalence = q.predict(X)

    assert set(prevalence) == {0, 1}
    assert sum(prevalence.values()) == pytest.approx(1.0)


def test_counting_accepts_common_input_formats(binary_dataset_formats):
    X, y = binary_dataset_formats
    q = PCC(estimator=LogisticRegression(max_iter=1000, random_state=42))

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)


def test_cc_threshold_parameter_validation(binary_dataset):
    X, y = binary_dataset

    CC(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        threshold=0.8,
    ).fit(X, y)

    with pytest.raises(InvalidParameterError):
        CC(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            threshold=1.5,
        ).fit(X, y)
