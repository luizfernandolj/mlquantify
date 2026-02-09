import pytest
import numpy as np
from mlquantify.adjust_counting import (
    CC,
    PCC,
    AC,
    PAC,
    FM,
    TAC,
    TX,
    TMAX,
    CDE
)

# -------------------------------------------------------------------------
# Test CC (Classify and Count)
# -------------------------------------------------------------------------

def test_CC_binary(binary_classifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    # Test with learner
    q = CC(learner=binary_classifier)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0

def test_CC_aggregate_binary():
    # Test aggregate directly
    y_pred = np.array([0, 0, 1, 1, 1])
    q = CC()
    prev = q.aggregate(y_pred)
    assert prev[0] == 0.4
    assert prev[1] == 0.6

# -------------------------------------------------------------------------
# Test PCC (Probabilistic Classify and Count)
# -------------------------------------------------------------------------

def test_PCC_binary(binary_classifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = PCC(learner=binary_classifier)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0

def test_PCC_aggregate_binary():
    # Probs: 5 samples, 2 classes.
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
        [0.4, 0.6]
    ]) # Mean: [0.48, 0.52]
    q = PCC()
    prev = q.aggregate(probs)
    assert prev[0] == pytest.approx(0.48)
    assert prev[1] == pytest.approx(0.52)

# -------------------------------------------------------------------------
# Test Matrix Adjustments (AC, PAC, FM)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("Quantifier", [AC, PAC, FM])
def test_matrix_adjustment_binary(Quantifier, binary_classifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=binary_classifier)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0

@pytest.mark.parametrize("Quantifier", [AC, PAC, FM])
def test_matrix_adjustment_multiclass(Quantifier, multiclass_classifier, multiclass_dataset):
    X_train, X_test, y_train, y_test = multiclass_dataset
    q = Quantifier(learner=multiclass_classifier)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 3
    assert pytest.approx(sum(prev.values())) == 1.0

# -------------------------------------------------------------------------
# Test Threshold Adjustments (TAC, TX, TMAX)
# -------------------------------------------------------------------------

@pytest.mark.parametrize("Quantifier", [TAC, TX, TMAX])
def test_threshold_adjustment_binary(Quantifier, binary_classifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=binary_classifier)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0
    
# -------------------------------------------------------------------------
# Test CDE (CDE-Iterate)
# -------------------------------------------------------------------------

def test_CDE_binary(binary_classifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = CDE(learner=binary_classifier)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0
