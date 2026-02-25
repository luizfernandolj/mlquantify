import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from mlquantify.likelihood import EMQ

def test_emq_fit_predict(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    q = EMQ(learner=learner)
    q.fit(X, y)
    preds = q.predict(X)
    assert isinstance(preds, dict)
    assert sum(preds.values()) == pytest.approx(1.0)

def test_emq_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    learner = LogisticRegression()
    q = EMQ(learner=learner)
    q.fit(X, y)
    preds = q.predict(X)
    assert len(preds) == 3
    assert sum(preds.values()) == pytest.approx(1.0)

@pytest.mark.parametrize("calib_function", ["bcts", "ts", "vs", "nbvs", None])
def test_emq_calibration(binary_dataset, calib_function):
    X, y = binary_dataset
    learner = LogisticRegression()
    q = EMQ(learner=learner, calib_function=calib_function)
    q.fit(X, y)
    preds = q.predict(X)
    assert sum(preds.values()) == pytest.approx(1.0)

def test_emq_convergence_params(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    
    q = EMQ(learner=learner, max_iter=50)
    q.fit(X, y)
    preds = q.predict(X)
    assert sum(preds.values()) == pytest.approx(1.0)
    
    q = EMQ(learner=learner, max_iter=200)
    q.fit(X, y)
    assert q.max_iter <= 200
