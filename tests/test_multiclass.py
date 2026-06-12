"""Tests for the multiclass strategy registry (OvR / OvO / custom)."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from mlquantify.multiclass import (
    available_strategies,
    get_strategy,
    register_strategy,
    OvRStrategy,
    OvOStrategy,
    _STRATEGIES,
)
from mlquantify.matching import DyS


def _prev_array(p, classes):
    if isinstance(p, dict):
        return np.array([p[c] for c in classes], dtype=float)
    return np.asarray(p, dtype=float)


def test_registry_contents():
    assert available_strategies() == ["ovo", "ovr"]
    assert isinstance(get_strategy("ovr"), OvRStrategy)
    assert isinstance(get_strategy("ovo"), OvOStrategy)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown multiclass strategy"):
        get_strategy("does_not_exist")


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_multiclass_strategy_valid_prevalence(strategy):
    X, y = make_classification(n_samples=600, n_classes=4, n_informative=8,
                               n_redundant=0, random_state=0)
    q = DyS(LogisticRegression(max_iter=300))
    q.strategy = strategy
    q.fit(X, y)

    classes = np.unique(y)
    p = _prev_array(q.predict(X), classes)
    assert p.shape == (4,)
    assert np.all(p >= 0)
    assert abs(p.sum() - 1) < 1e-6


def test_ovr_ovo_differ_but_both_valid():
    X, y = make_classification(n_samples=600, n_classes=3, n_informative=6,
                               n_redundant=0, random_state=3)
    classes = np.unique(y)

    qr = DyS(LogisticRegression(max_iter=300)); qr.strategy = "ovr"; qr.fit(X, y)
    qo = DyS(LogisticRegression(max_iter=300)); qo.strategy = "ovo"; qo.fit(X, y)
    pr = _prev_array(qr.predict(X), classes)
    po = _prev_array(qo.predict(X), classes)

    for p in (pr, po):
        assert np.all(p >= 0) and abs(p.sum() - 1) < 1e-6


def test_register_custom_strategy():
    """A new decomposition plugs in via the registry with no change to dispatch."""
    calls = {"predict": 0, "fit": 0}

    @register_strategy("ovr_clone")
    class _Clone(OvRStrategy):
        def fit(self, q, X, y, n_jobs=None, fit_args=None, fit_kwargs=None):
            calls["fit"] += 1
            return super().fit(q, X, y, n_jobs=n_jobs,
                               fit_args=fit_args, fit_kwargs=fit_kwargs)

        def predict(self, q, X, n_jobs=None):
            calls["predict"] += 1
            return super().predict(q, X, n_jobs=n_jobs)

    try:
        assert "ovr_clone" in available_strategies()

        X, y = make_classification(n_samples=600, n_classes=3, n_informative=6,
                                   n_redundant=0, random_state=1)
        classes = np.unique(y)

        base = DyS(LogisticRegression(max_iter=300)); base.strategy = "ovr"; base.fit(X, y)
        custom = DyS(LogisticRegression(max_iter=300)); custom.strategy = "ovr_clone"; custom.fit(X, y)

        a = _prev_array(base.predict(X), classes)
        b = _prev_array(custom.predict(X), classes)

        # the custom strategy (delegating to OvR) was actually dispatched ...
        assert calls["fit"] == 1
        assert calls["predict"] == 1
        # ... and yields the same result as the built-in OvR it wraps
        np.testing.assert_allclose(a, b)
    finally:
        _STRATEGIES.pop("ovr_clone", None)
