
import pytest
import numpy as np
from mlquantify.base import BaseQuantifier, MetaquantifierMixin, ProtocolMixin
from mlquantify.utils._constraints import Interval

class MockQuantifier(BaseQuantifier):
    _parameter_constraints = {
        "param1": [Interval(0, 10)],
    }
    
    def __init__(self, param1=5):
        self.param1 = param1

    def fit(self, X, y):
        self._validate_params()
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.array([0.5, 0.5])

def test_base_quantifier_inheritance():
    q = MockQuantifier()
    assert isinstance(q, BaseQuantifier)

def test_base_quantifier_parameter_validation(binary_dataset):
    X, y = binary_dataset
    
    # Valid parameter
    q = MockQuantifier(param1=5)
    q.fit(X, y)
    
    # Invalid parameter
    q = MockQuantifier(param1=20)
    with pytest.raises(ValueError):
        q.fit(X, y)

def test_metaquantifier_mixin():
    class MetaQ(MetaquantifierMixin, BaseQuantifier):
        pass
    
    q = MetaQ()
    assert isinstance(q, MetaquantifierMixin)

def test_protocol_mixin():
    class ProtocolQ(ProtocolMixin, BaseQuantifier):
        pass

    q = ProtocolQ()
    assert isinstance(q, ProtocolMixin)
    tags = q.__mlquantify_tags__()
    assert tags.estimation_type == "sample"
    assert tags.requires_fit is False
