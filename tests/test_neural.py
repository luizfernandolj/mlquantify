
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import torch
    import torch.nn as nn
    from mlquantify.neural import QuaNet, HistNetQBags, GMNetBags, PrevalenceBagMixin
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
    # QuaNet requires a estimator that outputs embeddings AND probabilities
    # We can mock this or use a pipeline if supported, but typically it expects specific methods
    # For now, let's use a MockEmbedder
    
    estimator = MockEmbedder()

    # Use very small parameters for speed
    q = QuaNet(
        estimator=estimator,
        fit_estimator=False,
        n_epochs=1,
        tr_iter=3,
        va_iter=2,
        sample_size=10,
        patience=1,
        lstm_hidden_size=4,
        ff_layers=[4],
        device='cpu',
    )

    # QuaNet fit splits X, y internally and trains on sampled bags
    q.fit(X, y)

    preds = q.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.sum() == pytest.approx(1.0)


def _make_prevalence_bags(rng, n_bags, bag_size, n_features=6):
    """Build prevalence-labelled bags from two Gaussian class clusters."""
    c0 = rng.normal(-1.0, 1.0, (2000, n_features))
    c1 = rng.normal(1.0, 1.0, (2000, n_features))
    Xs, ps = [], []
    for _ in range(n_bags):
        p1 = rng.uniform(0.0, 1.0)
        n1 = int(round(p1 * bag_size))
        n0 = bag_size - n1
        bag = np.vstack([c0[rng.integers(0, len(c0), n0)],
                         c1[rng.integers(0, len(c1), n1)]]).astype(np.float32)
        rng.shuffle(bag)
        Xs.append(bag)
        ps.append([1 - n1 / bag_size, n1 / bag_size])
    return Xs, np.asarray(ps)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not installed")
@pytest.mark.parametrize("as_array", [False, True])
def test_histnetq_bags_fit_predict(as_array):
    rng = np.random.default_rng(0)
    Xs, ps = _make_prevalence_bags(rng, n_bags=120, bag_size=80)
    Xs_in = np.stack(Xs) if as_array else Xs

    torch.manual_seed(0)
    fe = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 6))
    q = HistNetQBags(
        feature_extractor=fe, n_latent_features=6, n_bins=12, ff_layers=(16,),
        bag_size=80, n_bags=80, val_bags=20, batch_size=8,
        n_epochs=6, patience=6, random_state=0, device="cpu",
    )
    q.fit(Xs_in, ps)

    assert isinstance(q, PrevalenceBagMixin)
    assert list(q.classes_) == [0, 1]
    pred = q.predict(Xs[0])
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (2,)
    assert pred.sum() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not installed")
def test_gmnet_bags_and_count_prevalences():
    rng = np.random.default_rng(1)
    Xs, ps = _make_prevalence_bags(rng, n_bags=100, bag_size=80)

    torch.manual_seed(0)
    fe = nn.Sequential(nn.Linear(6, 16), nn.ReLU())
    g = GMNetBags(
        feature_extractor=fe, n_input_features=6, latent_dim=4, n_gaussians=8,
        n_latent=2, ff_layers=(16,), cka_lambda=0.01, bag_size=80,
        n_bags=60, val_bags=20, batch_size=8, n_epochs=6, patience=6,
        random_state=0, device="cpu",
    )
    # unnormalised prevalences (counts) must be renormalised internally
    g.fit(Xs, ps * 80)
    pred = g.predict(Xs[0])
    assert pred.shape == (2,)
    assert pred.sum() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not installed")
def test_bags_unequal_size_requires_bag_size():
    rng = np.random.default_rng(2)
    bags = [rng.normal(size=(rng.integers(60, 100), 6)).astype(np.float32)
            for _ in range(20)]
    ps = rng.dirichlet([1, 1], size=20)
    fe = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 6))
    q = HistNetQBags(feature_extractor=fe, n_latent_features=6, n_bins=8,
                     bag_size=None, n_epochs=1, n_bags=8, val_bags=4,
                     random_state=0, device="cpu")
    with pytest.raises(ValueError, match="same size"):
        q.fit(bags, ps)
