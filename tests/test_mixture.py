import pytest
import numpy as np
from mlquantify.mixture import (
    DyS,
    HDy,
    SMM,
    SORD,
    HDx,
    MMD_RKHS
)
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------------
# Test Aggregative Mixture Models (DyS, HDy, SMM, SORD)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("Quantifier", [DyS, HDy, SMM, SORD])
def test_aggregative_mixture_binary(Quantifier, binary_classifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=binary_classifier)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0

# -------------------------------------------------------------------------
# Test Non-aggregative Mixture Models (HDx, MMD_RKHS)
# -------------------------------------------------------------------------

def test_HDx_binary(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = HDx()
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0

def test_MMD_RKHS_binary(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = MMD_RKHS(kernel='rbf')
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0
