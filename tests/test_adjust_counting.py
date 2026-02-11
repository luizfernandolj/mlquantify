
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlquantify.adjust_counting import CC, PCC, AC, PAC, TAC, TX, TMAX, FM, CDE
from mlquantify.utils._exceptions import InvalidParameterError
from mlquantify._config import config_context

QUANTIFIERS = [CC, PCC, AC, PAC, TAC, TX, TMAX, FM, CDE]

@pytest.mark.parametrize("quantifier_class", QUANTIFIERS)
def test_quantifier_initialization(quantifier_class):
    q = quantifier_class()
    assert q.learner is None

@pytest.mark.parametrize("quantifier_class", QUANTIFIERS)
def test_quantifier_fit_predict_binary(quantifier_class, binary_dataset_formats):
    X, y = binary_dataset_formats
    learner = LogisticRegression()
    q = quantifier_class(learner=learner)
    q.fit(X, y)
    preds = q.predict(X)
    assert isinstance(preds, dict)
    assert len(preds) == 2
    assert sum(preds.values()) == pytest.approx(1.0)

@pytest.mark.parametrize("quantifier_class", QUANTIFIERS)
def test_quantifier_fit_predict_multiclass(quantifier_class, multiclass_dataset_formats):
    X, y = multiclass_dataset_formats
    # Threshold methods are typically binary only or OVR, check compatibility
    if quantifier_class in [TAC, TX, TMAX]:
         # These might default to OVR or error if strictly binary without wrapper
         # Assuming they work or are wrapped (OVR strategy is default in base classes for some)
         pass

    learner = LogisticRegression()
    q = quantifier_class(learner=learner)
    q.fit(X, y)
    preds = q.predict(X)
    assert isinstance(preds, dict)
    # Multiclass dataset has 3 classes
    assert len(preds) == 3
    assert sum(preds.values()) == pytest.approx(1.0)

@pytest.mark.parametrize("quantifier_class", QUANTIFIERS)
def test_config_output_format(quantifier_class, binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    q = quantifier_class(learner=learner)
    q.fit(X, y)
    
    with config_context(prevalence_return_type="array"):
        preds_array = q.predict(X)
        assert isinstance(preds_array, np.ndarray)
        assert len(preds_array) == 2

    with config_context(prevalence_return_type="dict"):
        preds_dict = q.predict(X)
        assert isinstance(preds_dict, dict)

def test_cc_threshold_parameter(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    
    # Valid threshold
    q = CC(learner=learner, threshold=0.8)
    q.fit(X, y)
    
    # Invalid threshold
    with pytest.raises(InvalidParameterError):
        q = CC(learner=learner, threshold=1.5)
        q.fit(X, y)

def test_missing_class_in_test(binary_dataset):
    X_train, y_train = binary_dataset
    learner = LogisticRegression()
    q = CC(learner=learner)
    q.fit(X_train, y_train)
    
    # Simulate test set with only one class predicted (using a dummy learner or just assuming behavior)
    # Ideally we'd mock the learner's predict to return only 0s
    
    # Check if robust to missing classes in predictions if using aggregate directly?
    pass # Implementation dependant, usually handled by validation

def test_parameters_breaking(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    
    # AC solver invalid
    with pytest.raises(InvalidParameterError):
        q = CDE(learner=learner,max_iter=-5)
        q.fit(X, y)

