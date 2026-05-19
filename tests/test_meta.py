
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from mlquantify.meta import EnsembleQ, AggregativeBootstrap, QuaDapt
from mlquantify.counting import CC, PCC, FM
from mlquantify.likelihood import EMQ
from mlquantify.matching import DyS


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)

def test_ensembleq_fit_predict(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    # Use simple quantifier for speed
    base_q = CC(learner=learner)
    # Smaller size for speed
    meta_q = EnsembleQ(quantifier=base_q, size=5, n_jobs=-1)
    meta_q.fit(X, y)
    preds = meta_q.predict(X)
    _assert_valid_prevalence(preds, n_classes=2)

@pytest.mark.parametrize("protocol", ["artificial", "natural", "uniform"])
def test_ensembleq_protocols(protocol, binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    base_q = CC(learner=learner)
    meta_q = EnsembleQ(quantifier=base_q, size=2, protocol=protocol)
    meta_q.fit(X, y)
    preds = meta_q.predict(X)
    _assert_valid_prevalence(preds, n_classes=2)

def test_aggregative_bootstrap(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    base_q = CC(learner=learner)
    meta_q = AggregativeBootstrap(quantifier=base_q, n_train_bootstraps=2, n_test_bootstraps=2)
    meta_q.fit(X, y)
    preds = meta_q.predict(X)
    _assert_valid_prevalence(preds, n_classes=2)

def test_quadapt_fit_predict(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    # QuaDapt requires soft predictions
    base_q = DyS(learner=learner) 
    meta_q = QuaDapt(quantifier=base_q)
    meta_q.fit(X, y)
    preds = meta_q.predict(X)
    _assert_valid_prevalence(preds, n_classes=2)

def test_quadapt_raises_hard_quantifier(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    # CC is hard (crisp) quantifier
    base_q = CC(learner=learner)
    meta_q = QuaDapt(quantifier=base_q)
    with pytest.raises(ValueError, match="not a soft"):
        meta_q.fit(X, y)
