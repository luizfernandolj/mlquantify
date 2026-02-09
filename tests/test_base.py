import pytest
import numpy as np
from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import AggregationMixin, SoftLearnerQMixin
from sklearn.linear_model import LogisticRegression

class MinimalQuantifier(BaseQuantifier):
    def __init__(self, param1=1):
        self.param1 = param1
    def fit(self, X, y):
        return self
    def predict(self, X):
        return {}

def test_base_quantifier_params():
    q = MinimalQuantifier(param1=10)
    assert q.get_params() == {'param1': 10}

def test_aggregation_mixin_params():
    class MyAggQ(AggregationMixin, BaseQuantifier):
        def __init__(self, learner=None):
            self.learner = learner
            
    learner = LogisticRegression(C=0.5)
    q = MyAggQ(learner=learner)
    
    # Test getting params (should include learner's params if sklearn compatible, 
    # but AggregationMixin doesn't auto-expose learner params in get_params unless 
    # we implement get_params to do so, OR if we rely on sklearn's BaseEstimator 
    # which inspects __init__.
    # Since MyAggQ inherits BaseEstimator, get_params returns 'learner'.
    
    params = q.get_params()
    assert 'learner' in params
    assert params['learner'].C == 0.5
    
    # Test setting params
    q.set_params(learner__C=0.1)
    assert q.learner.C == 0.1
