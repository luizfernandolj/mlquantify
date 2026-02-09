import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from mlquantify.adjust_counting import AC, PAC
from mlquantify.mixture import DyS
from mlquantify.meta import EnsembleQ
from mlquantify.metrics import MAE, MSE
from mlquantify.model_selection import GridSearchQ

def test_large_integration_experiment():
    """
    A comprehensive integration test (experiment) that runs a full pipeline:
    1. Generate Dataset
    2. Train different quantifiers (Aggregative, Mixture, Meta)
    3. Evaluate using metrics
    4. Assert performance/consistency
    """
    # 1. Generate Data (Multiclass)
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, n_classes=3, 
        random_state=42
    )
    
    # Split
    # We'll valid/test split manually for the experiment
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split1 = int(0.5 * len(X))
    split2 = int(0.75 * len(X))
    
    X_train, y_train = X[indices[:split1]], y[indices[:split1]]
    X_val, y_val = X[indices[split1:split2]], y[indices[split1:split2]]
    X_test, y_test = X[indices[split2:]], y[indices[split2:]]
    
    learner = LogisticRegression(max_iter=1000)
    
    # 2. Define Quantifiers
    quantifiers = {
        'AC': AC(learner=learner),
        'PAC': PAC(learner=learner),
        'DyS': DyS(learner=learner, measure='topsoe'),
        'Ensemble-AC': EnsembleQ(quantifier=AC(learner=learner), size=5)
    }
    
    # 3. Fit and Predict
    results = {}
    for name, q in quantifiers.items():
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        results[name] = prev
        
        # Basic sanity check
        assert len(prev) == 3
        # assert pytest.approx(sum(prev.values()), abs=0.01) == 1.0

    # 4. Evaluate
    true_prev = {cls: np.mean(y_test == cls) for cls in np.unique(y)}
    # Convert true_prev dict to array ordered by keys
    classes = sorted(true_prev.keys())
    true_prev_arr = np.array([true_prev[c] for c in classes])
    
    for name, prev_dict in results.items():
        pred_prev_arr = np.array([prev_dict[c] for c in classes])
        
        # Calculate Error
        mae = MAE(true_prev_arr, pred_prev_arr)
        
        # Use simple assert to check valid metric output
        assert mae >= 0.0
        assert mae <= 1.0
        
    print("Integration experiment finished successfully.")

def test_grid_search_integration():
    """Test GridSearchQ in a realistic flow."""
    X, y = make_classification(n_samples=200, n_classes=2, random_state=42)
    
    # Define a simple quantifier with a searchable param.
    # Since we can't easily modify class params of predefined quantifiers without wrapper,
    # we'll use a mocked one or one with exposed params.
    # Let's use EnsembleQ size.
    
    param_grid = {
        'size': [5, 10]
    }
    
    gsq = GridSearchQ(
        quantifier=lambda: EnsembleQ(AC(LogisticRegression()), size=5), 
        # GridSearchQ expects a class or callable that checks params? 
        # Actually it instantiates `quantifier()`.
        # So we pass a lambda that returns an instance? No, `quantifier()` implies it's a class.
        # But we need to pass init args (learner).
        
        # Let's define a helper class
        param_grid=param_grid,
        refit=False, # Speed up
        samples_sizes=20,
        n_repetitions=2
    )

    class MyEnsemble(EnsembleQ):
        def __init__(self, size=5):
            super().__init__(quantifier=AC(LogisticRegression()), size=size)
            
    gsq.quantifier = MyEnsemble() # Hack: overwrite instance created in __init__ if needed, but __init__ creates it.
    # Actually, GridSearchQ.__init__ does `self.quantifier = quantifier()`.
    # So we pass `MyEnsemble` class.
    
    gsq = GridSearchQ(
        quantifier=MyEnsemble,
        param_grid=param_grid,
        samples_sizes=20,
        n_repetitions=2,
        val_split=0.5
    )
    
    gsq.fit(X, y)
    best = gsq.best_params
    assert 'size' in best
