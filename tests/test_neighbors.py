
import pytest
import numpy as np
from mlquantify.neighbors import PWK

def test_pwk_fit_predict(binary_dataset):
    X, y = binary_dataset
    q = PWK(n_neighbors=10)
    q.fit(X, y)
    preds = q.predict(X)
    assert isinstance(preds, dict)
    assert sum(preds.values()) == pytest.approx(1.0)

def test_pwk_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    q = PWK(n_neighbors=10)
    q.fit(X, y)
    preds = q.predict(X)
    assert len(preds) == 3
    assert sum(preds.values()) == pytest.approx(1.0)

def test_pwk_params(binary_dataset):
    X, y = binary_dataset
    q = PWK(n_neighbors=5, alpha=2.0)
    q.fit(X, y)
    preds = q.predict(X)
    assert sum(preds.values()) == pytest.approx(1.0)

    # Test invalid n_neighbors
    with pytest.raises(ValueError):
         PWK(n_neighbors=-1).fit(X, y)
    
    # Test high neighbors count
    q = PWK(n_neighbors=len(X) - 1).fit(X, y)
