
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import torch
    from mlquantify.neural import QuaNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class MockEmbedder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Return fake embeddings
        return np.random.rand(len(X), 10)
    def fit_transform(self, X, y=None):
        return self.transform(X)
    def predict_proba(self, X):
         # Return fake probabilities
         return np.random.rand(len(X), 2)
         
    @property
    def classes_(self):
        return np.array([0, 1])

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not installed")
def test_quanet_fit_predict(binary_dataset):
    X, y = binary_dataset
    # QuaNet requires a learner that outputs embeddings AND probabilities
    # We can mock this or use a pipeline if supported, but typically it expects specific methods
    # For now, let's use a MockEmbedder
    
    learner = MockEmbedder()
    
    # Use very small parameters for speed
    q = QuaNet(
        learner=learner,
        epoch_pre=1,
        epoch_opt=1,
        patience=1,
        lstm_hidden_size=4,
        ff_layers=[4],
        device='cpu',
        verbose=False
    )
    
    # QuaNet fit might require validation split internally or just takes X, y
    q.fit(X, y)
    
    preds = q.predict(X)
    assert isinstance(preds, dict)
    assert sum(preds.values()) == pytest.approx(1.0)
