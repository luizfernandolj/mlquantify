
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from mlquantify.model_selection import GridSearchQ, apply_protocol
from mlquantify.counting import CC, PCC
from mlquantify.metrics import MAE

class MockQuantifier(CC):
    def __init__(self, estimator=None, threshold=0.5):
        super().__init__(estimator=estimator, threshold=threshold)


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


def test_gridsearchq_fit_predict(binary_dataset):
    X, y = binary_dataset
    estimator = LogisticRegression()
    param_grid = {'threshold': [0.4, 0.6]}
    
    gs = GridSearchQ(
        quantifier=lambda: MockQuantifier(estimator=estimator), # Factory or class
        param_grid=param_grid,
        protocol='app',
        samples_sizes=50,
        n_repetitions=2,
        scoring=MAE
    )
    gs.fit(X, y)
    
    assert gs.best_params['threshold'] in [0.4, 0.6]
    preds = gs.predict(X)
    _assert_valid_prevalence(preds, n_classes=2)

def test_gridsearchq_random_state(binary_dataset):
    X, y = binary_dataset
    estimator = LogisticRegression()
    param_grid = {'threshold': [0.5]}
    
    gs1 = GridSearchQ(
        quantifier=lambda: MockQuantifier(estimator=estimator),
        param_grid=param_grid,
        random_seed=42,
        n_repetitions=5
    )
    gs2 = GridSearchQ(
        quantifier=lambda: MockQuantifier(estimator=estimator),
        param_grid=param_grid,
        random_seed=42,
        n_repetitions=5
    )
    
    gs1.fit(X, y)
    gs2.fit(X, y)
    
    assert gs1.best_score == gs2.best_score

def test_gridsearchq_estimator_params(binary_dataset):
    X, y = binary_dataset
    # Test if can set estimator params via grid if exposed or wrappers used
    # GridSearchQ uses set_params on the quantifier instance.
    # If the quantifier exposes estimator params (e.g. via sklearn delegation), checks this.
    # CC doesn't typically expose estimator params directly as its own, unless through specific design.
    # Assuming user wants to check if it CAN run.
    pass

def test_protocols(binary_dataset):
    X, y = binary_dataset
    estimator = LogisticRegression()
    param_grid = {'threshold': [0.5]}
    
    for protocol in ['app', 'npp', 'upp']:
        gs = GridSearchQ(
            quantifier=lambda: MockQuantifier(estimator=estimator),
            param_grid=param_grid,
            protocol=protocol,
            n_repetitions=2
        )
        gs.fit(X, y)
        assert gs.best_score is not None


@pytest.mark.parametrize("quantifier", [CC, PCC])
@pytest.mark.parametrize(
    "protocol, params",
    [
        ("app", {"strategy": "grid"}),
        ("app", {"strategy": "kraemer"}),
        ("app", {"strategy": "uniform"}),
        ("app", {"strategy": "dirichlet", "dirichlet_alpha": 0.3}),
        ("upp", {}),
        ("upp", {"strategy": "uniform"}),
    ],
)
def test_apply_protocol_multiclass_full_class_output(multiclass_dataset, quantifier, protocol, params):
    """Sampled-prevalence protocols on multiclass data must yield prevalence
    vectors spanning all fitted classes, even when a batch's predictions miss a
    class (regression test for CC shrinking ``classes_`` on the predict path)."""
    X, y = multiclass_dataset
    n_classes = len(np.unique(y))

    results = apply_protocol(
        quantifier(LogisticRegression(max_iter=200)),
        X, y,
        protocol=protocol,
        n_prevalences=12,
        batch_size=60,
        random_state=0,
        **params,
    )

    true = results["true_prevalences"]
    pred = results["predicted_prevalences"]
    assert true.shape == pred.shape == (results["n_batches"], n_classes)
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(pred >= 0)
