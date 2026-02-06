import pytest
import numpy as np
from mlquantify.model_selection import GridSearchQ
from mlquantify.adjust_counting import TAC
from mlquantify.metrics import MAE
from sklearn.linear_model import LogisticRegression

def test_GridSearchQ_binary(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    base_q = TAC(learner=LogisticRegression())
    
    # We need to grid search over parameters of the quantifier.
    # TAC doesn't have many parameters itself, but we can search valid ones if any.
    # Or we can search over the learner's parameters if we can TACess them.
    # GridSearchQ sets parameters on the quantifier instance.
    # For TAC, parameters are mostly on the learner.
    # But GridSearchQ assumes set_params works.
    
    # Let's try searching over a fake param or use a Quantifier that has params.
    # GridSearchQ uses `set_params`.
    
    # Workaround: The learner is inside TAC. 
    # If we want to tune the learner, we might need `learner__C`.
    
    param_grid = {
        'learner__C': [0.1, 1.0]
    }
    
    gsq = GridSearchQ(
        quantifier=TAC, # Pass class, not instance
        param_grid=param_grid,
        samples_sizes=10,
        n_repetitions=2,
        scoring=MAE,
        refit=True,
        n_jobs=1
    )
    
    # GridSearchQ expects an unfitted instance in __init__ ?
    # The code says: self.quantifier = quantifier()
    # So we pass the class. 
    # But then how do we pass the learner? Default is None.
    # We might need a partial or a class that defaults the learner.
    
    class MyTAC(TAC):
        def __init__(self, learner=None):
            if learner is None:
                learner = LogisticRegression()
            super().__init__(learner=learner)
            
    gsq = GridSearchQ(
        quantifier=MyTAC,
        param_grid=param_grid,
        samples_sizes=50,
        n_repetitions=2
    )

    gsq.fit(X_train, y_train)
    prev = gsq.predict(X_test)
    assert len(prev) == 2
    assert gsq.best_params is not None
