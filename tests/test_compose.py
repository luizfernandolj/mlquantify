# tests/compose/test_compose_quantifier.py

import numpy as np
import pytest

from sklearn.linear_model import LogisticRegression


pytest.importorskip("qunfold")


def _as_array(prevalences):
    if isinstance(prevalences, dict):
        return np.asarray(list(prevalences.values()), dtype=float)

    return np.asarray(prevalences, dtype=float)


def _assert_valid_prevalence(prevalences, n_classes):
    prevalences = _as_array(prevalences)

    assert prevalences.shape == (n_classes,)
    assert np.all(prevalences >= 0)
    assert np.all(prevalences <= 1)
    assert prevalences.sum() == pytest.approx(1.0, abs=1e-5)


def _make_qunfold_components(method_name, learner=None):
    from qunfold.sklearn import CVClassifier
    from qunfold.methods.linear.losses import LeastSquaresLoss, HellingerSurrogateLoss
    from qunfold.methods.linear.representations import (
        ClassRepresentation,
        HistogramRepresentation,
    )

    if method_name == "ACC":
        return {
            "representation": ClassRepresentation(
                CVClassifier(learner, random_state=42),
                is_probabilistic=False,
            ),
            "loss": LeastSquaresLoss(),
        }

    if method_name == "PACC":
        return {
            "representation": ClassRepresentation(
                CVClassifier(learner, random_state=42),
                is_probabilistic=True,
            ),
            "loss": LeastSquaresLoss(),
        }

    if method_name == "HDx":
        return {
            "representation": HistogramRepresentation(
                n_bins=8,
                unit_scale=False,
            ),
            "loss": HellingerSurrogateLoss(),
        }

    if method_name == "HDy":
        return {
            "representation": HistogramRepresentation(
                n_bins=8,
                preprocessor=ClassRepresentation(
                    CVClassifier(learner, random_state=42),
                    is_probabilistic=True,
                ),
                unit_scale=False,
            ),
            "loss": HellingerSurrogateLoss(),
        }

    raise ValueError(f"Unknown method: {method_name}")


def _make_qunfold_native(method_name, learner=None):
    from qunfold import HDx, HDy, ACC, PACC
    from qunfold.sklearn import CVClassifier

    learner = CVClassifier(learner, random_state=42) if learner is not None else None

    if method_name == "ACC":
        return ACC(learner)
    if method_name == "PACC":
        return PACC(learner)
    if method_name == "HDx":
        return HDx(n_bins=8, seed=42)
    if method_name == "HDy":
        return HDy(learner, n_bins=8, seed=42)

    raise ValueError(f"Unknown method: {method_name}")


@pytest.mark.parametrize("method_name", ["ACC", "PACC", "HDx", "HDy"])
def test_compose_quantifier_matches_qunfold_native_binary(
    method_name,
    binary_dataset,
):
    from mlquantify.compose import ComposeQuantifier

    X, y = binary_dataset

    learner_native = LogisticRegression(max_iter=1000, random_state=42)
    learner_compose = LogisticRegression(max_iter=1000, random_state=42)

    native = _make_qunfold_native(
        method_name,
        learner=learner_native,
    )

    components = _make_qunfold_components(
        method_name,
        learner=learner_compose,
    )

    composed = ComposeQuantifier(
        representation=components["representation"],
        loss=components["loss"],
        seed=42,
    )

    native.fit(X, y)
    composed.fit(X, y)

    native_prevs = _as_array(native.predict(X))
    composed_prevs = _as_array(composed.predict(X))

    _assert_valid_prevalence(composed_prevs, n_classes=2)

    np.testing.assert_allclose(
        composed_prevs,
        native_prevs,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("method_name", ["ACC", "PACC", "HDx", "HDy"])
def test_compose_quantifier_matches_qunfold_native_multiclass(
    method_name,
    multiclass_dataset,
):
    from mlquantify.compose import ComposeQuantifier

    X, y = multiclass_dataset

    learner_native = LogisticRegression(max_iter=1000, random_state=42)
    learner_compose = LogisticRegression(max_iter=1000, random_state=42)

    native = _make_qunfold_native(
        method_name,
        learner=learner_native,
    )

    components = _make_qunfold_components(
        method_name,
        learner=learner_compose,
    )

    composed = ComposeQuantifier(
        representation=components["representation"],
        loss=components["loss"],
        seed=42,
    )

    native.fit(X, y)
    composed.fit(X, y)

    native_prevs = _as_array(native.predict(X))
    composed_prevs = _as_array(composed.predict(X))

    _assert_valid_prevalence(composed_prevs, n_classes=3)

    np.testing.assert_allclose(
        composed_prevs,
        native_prevs,
        atol=1e-5,
        rtol=1e-5,
    )


def test_compose_quantifier_dys_valid_prevalence(binary_dataset):
    from qunfold.sklearn import CVClassifier
    from qunfold.methods.linear.representations import (
        ClassRepresentation,
        HistogramRepresentation,
    )

    from mlquantify.compose import ComposeQuantifier
    from mlquantify.metrics import topsoe_jax

    X, y = binary_dataset

    learner = LogisticRegression(max_iter=1000, random_state=42)

    representation = HistogramRepresentation(
        n_bins=8,
        preprocessor=ClassRepresentation(
            CVClassifier(learner, random_state=42),
            is_probabilistic=True,
        ),
        unit_scale=False,
    )

    q = ComposeQuantifier(
        representation=representation,
        loss=topsoe_jax,
        seed=42,
    )

    q.fit(X, y)
    prevalences = q.predict(X)
    train_prevalence = np.bincount(y) / len(y)
    print(prevalences)
    print(train_prevalence)

    _assert_valid_prevalence(prevalences, n_classes=2)