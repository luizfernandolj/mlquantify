
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

def test_emq_calibration(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    
    # Test valid calibration
    # 'ts' calls TempScaling, assume implementation exists and runs
    q = EMQ(learner=learner, calib_function='ts')
    q.fit(X, y)
    preds = q.predict(X)
    assert sum(preds.values()) == pytest.approx(1.0)

def test_emq_convergence_params(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    
    # Test tolerance and max_iter
    q = EMQ(learner=learner, max_iter=50) # Coarse convergence
    q.fit(X, y)
    preds = q.predict(X)
    assert sum(preds.values()) == pytest.approx(1.0)
    
    # Test very strict
    q = EMQ(learner=learner, max_iter=200)
    q.fit(X, y)
    assert q.n_iter_ <= 200 # Check attribute if exposed, or just run without error

def test_emq_recal_train(binary_dataset):
     X, y = binary_dataset
     learner = LogisticRegression()
     q = EMQ(learner=learner, recal_train=True)
     q.fit(X, y)
     # Just check execution path
     preds = q.predict(X)
     assert isinstance(preds, dict)

