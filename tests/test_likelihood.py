import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlquantify._config import config_context
from mlquantify.likelihood import CDE, EMQ, MLPE


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("quantifier_class", [EMQ, CDE, MLPE])
def test_likelihood_methods_fit_predict_binary(quantifier_class, binary_dataset):
    X, y = binary_dataset
    q = quantifier_class(learner=LogisticRegression(max_iter=1000, random_state=42))

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)


def test_emq_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    q = EMQ(learner=LogisticRegression(max_iter=1000, random_state=42))

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)


@pytest.mark.parametrize("calib_function", ["bcts", "ts", "vs", "nbvs", None])
def test_emq_calibration(binary_dataset, calib_function):
    X, y = binary_dataset
    q = EMQ(
        learner=LogisticRegression(max_iter=1000, random_state=42),
        calib_function=calib_function,
    )

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)


def test_likelihood_dict_output_config(binary_dataset):
    X, y = binary_dataset
    q = EMQ(learner=LogisticRegression(max_iter=1000, random_state=42))

    q.fit(X, y)

    with config_context(prevalence_return_type="dict"):
        prevalence = q.predict(X)

    assert set(prevalence) == {0, 1}
    assert sum(prevalence.values()) == pytest.approx(1.0)
