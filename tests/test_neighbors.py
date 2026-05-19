
import pytest
import numpy as np
from mlquantify.neighbors import PWK


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


def test_pwk_fit_predict(binary_dataset):
    X, y = binary_dataset
    q = PWK(n_neighbors=10)
    q.fit(X, y)
    preds = q.predict(X)
    _assert_valid_prevalence(preds, n_classes=2)

def test_pwk_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    q = PWK(n_neighbors=10)
    q.fit(X, y)
    preds = q.predict(X)
    _assert_valid_prevalence(preds, n_classes=3)

def test_pwk_params(binary_dataset):
    X, y = binary_dataset
    q = PWK(n_neighbors=5, alpha=2.0)
    q.fit(X, y)
    preds = q.predict(X)
    _assert_valid_prevalence(preds, n_classes=2)

    # Test invalid n_neighbors
    with pytest.raises(ValueError):
         PWK(n_neighbors=-1).fit(X, y)
    
    # Test high neighbors count
    q = PWK(n_neighbors=len(X) - 1).fit(X, y)
