import pytest
import numpy as np
from mlquantify.neighbors import PWK
from sklearn.neighbors import KNeighborsClassifier

def test_PWK_binary(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    # PWK builds its own learner internally if passed params, 
    # but the docstring example suggests simple init.
    # The __init__ creates a PWKCLF.
    q = PWK(n_neighbors=5)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0

def test_PWK_classify(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = PWK(n_neighbors=5)
    q.fit(X_train, y_train)
    labels = q.classify(X_test)
    assert len(labels) == len(X_test)
    assert set(np.unique(labels)).issubset(set(np.unique(y_train)))
