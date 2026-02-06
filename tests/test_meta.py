import pytest
import numpy as np
from mlquantify.meta import EnsembleQ, AggregativeBootstrap, QuaDapt
from mlquantify.adjust_counting import TAC
from sklearn.linear_model import LogisticRegression

def test_EnsembleQ_binary(binary_classifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = EnsembleQ(
        quantifier=TAC(learner=binary_classifier),
        size=5,
        n_jobs=1
    )
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0

def test_AggregativeBootstrap_binary(binary_classifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = AggregativeBootstrap(
        quantifier=TAC(learner=binary_classifier),
        n_train_bootstraps=2,
        n_test_bootstraps=2
    )
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0

def test_QuaDapt_binary(binary_classifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    # QuaDapt requires soft predictions, ensure learner gives them
    # TAC usually works with hard predictions but uses predict_proba internally if learner has it?
    # Actually QuaDapt explicitly checks uses_soft_predictions and requires predict_proba
    # The learner passed to TAC must be a classifier. 
    # But QuaDapt wraps a quantifier.
    # The docstring example shows wrapping TAC.
    
    q = QuaDapt(
        quantifier=TAC(learner=binary_classifier),
        merging_factors=[0.5],
        measure='topsoe'
    )
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0
