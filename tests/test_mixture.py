
import pytest
import numpy as np
import mlquantify as mq
from sklearn.linear_model import LogisticRegression
from mlquantify.mixture import DyS, HDy, SMM, SORD, HDx, MMD_RKHS

MIXTURE_QUANTIFIERS = [DyS, HDy, SMM, SORD, HDx, MMD_RKHS]

@pytest.mark.parametrize("quantifier_class", MIXTURE_QUANTIFIERS)
def test_mixture_fit_predict(quantifier_class, binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    if quantifier_class == HDx or quantifier_class == MMD_RKHS:
        q = quantifier_class()
    else:
        q = quantifier_class(learner=learner)
    q.fit(X, y)
    preds = q.predict(X)
    assert isinstance(preds, dict)
    assert sum(preds.values()) == pytest.approx(1.0)

def test_dys_measures(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    for measure in ["hellinger", "topsoe", "probsymm"]:
        q = DyS(learner=learner, measure=measure)
        q.fit(X, y)
        preds = q.predict(X)
        assert sum(preds.values()) == pytest.approx(1.0)
    
    with pytest.raises(ValueError):
        q = DyS(learner=learner, measure="invalid")
        q.fit(X, y) # Validation might happen here

def test_bin_sizes(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    # HDy typically uses bins
    q = DyS(learner=learner, bins_size=[10, 20, 30, 50, 70, 100])
    q.fit(X, y)
    preds = q.predict(X)
    assert sum(preds.values()) == pytest.approx(1.0)

def test_sord_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    learner = LogisticRegression()
    q = SORD(learner=learner)
    q.fit(X, y)
    
    # SORD uses OvR by default which doesn't guarantee sum=1 without normalization
    with mq.config_context(prevalence_normalization="sum"):
        preds = q.predict(X)
        assert len(preds) == 3
        assert sum(preds.values()) == pytest.approx(1.0)

