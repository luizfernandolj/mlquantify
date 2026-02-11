
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from mlquantify.model_selection import GridSearchQ
from mlquantify.adjust_counting import CC
from mlquantify.metrics import MAE

class MockQuantifier(CC):
    def __init__(self, learner=None, threshold=0.5):
        super().__init__(learner=learner, threshold=threshold)

def test_gridsearchq_fit_predict(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    param_grid = {'threshold': [0.4, 0.6]}
    
    gs = GridSearchQ(
        quantifier=lambda: MockQuantifier(learner=learner), # Factory or class
        param_grid=param_grid,
        protocol='app',
        samples_sizes=50,
        n_repetitions=2,
        scoring=MAE
    )
    gs.fit(X, y)
    
    assert gs.best_params['threshold'] in [0.4, 0.6]
    preds = gs.predict(X)
    assert isinstance(preds, dict)
    assert sum(preds.values()) == pytest.approx(1.0)

def test_gridsearchq_random_state(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    param_grid = {'threshold': [0.5]}
    
    gs1 = GridSearchQ(
        quantifier=lambda: MockQuantifier(learner=learner),
        param_grid=param_grid,
        random_seed=42,
        n_repetitions=5
    )
    gs2 = GridSearchQ(
        quantifier=lambda: MockQuantifier(learner=learner),
        param_grid=param_grid,
        random_seed=42,
        n_repetitions=5
    )
    
    gs1.fit(X, y)
    gs2.fit(X, y)
    
    assert gs1.best_score == gs2.best_score

def test_gridsearchq_learner_params(binary_dataset):
    X, y = binary_dataset
    # Test if can set learner params via grid if exposed or wrappers used
    # GridSearchQ uses set_params on the quantifier instance.
    # If the quantifier exposes learner params (e.g. via sklearn delegation), checks this.
    # CC doesn't typically expose learner params directly as its own, unless through specific design.
    # Assuming user wants to check if it CAN run.
    pass

def test_protocols(binary_dataset):
    X, y = binary_dataset
    learner = LogisticRegression()
    param_grid = {'threshold': [0.5]}
    
    for protocol in ['app', 'npp', 'upp']:
        gs = GridSearchQ(
            quantifier=lambda: MockQuantifier(learner=learner),
            param_grid=param_grid,
            protocol=protocol,
            n_repetitions=2
        )
        gs.fit(X, y)
        assert gs.best_score is not None

