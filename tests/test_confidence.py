import numpy as np
import pytest

from mlquantify.confidence import (
    BaseConfidenceRegion,
    ConfidenceInterval,
    ConfidenceEllipseSimplex,
    ConfidenceEllipseCLR,
    construct_confidence_region,
)


@pytest.fixture
def boot_prevalences():
    """200 bootstrap prevalence estimates over 3 classes (rows sum to 1)."""
    rng = np.random.default_rng(0)
    return rng.dirichlet([2.0, 3.0, 5.0], size=200)


def _full_rank_cluster(center, scale=0.05, n=300, seed=0):
    """A tight, full-rank Gaussian cluster (non-singular covariance)."""
    rng = np.random.default_rng(seed)
    return np.asarray(center, dtype=float) + rng.normal(scale=scale, size=(n, len(center)))


# --- base class ---------------------------------------------------------

def test_base_is_abstract(boot_prevalences):
    # __init__ calls _compute_region(), which the base class leaves abstract.
    with pytest.raises(NotImplementedError):
        BaseConfidenceRegion(boot_prevalences)


# --- percentile intervals ----------------------------------------------

def test_interval_bounds_and_point_estimate(boot_prevalences):
    ci = ConfidenceInterval(boot_prevalences, confidence_level=0.9)
    low, high = ci.get_region()
    assert low.shape == high.shape == (3,)
    assert np.all(low <= high)
    # the point estimate is the mean of the bootstrap samples
    assert np.allclose(ci.get_point_estimate(), boot_prevalences.mean(axis=0))


def test_interval_contains(boot_prevalences):
    ci = ConfidenceInterval(boot_prevalences, confidence_level=0.9)
    assert bool(np.all(ci.contains(ci.get_point_estimate())))      # mean is inside
    assert not bool(np.all(ci.contains([0.99, 0.005, 0.005])))     # extreme corner is not


def test_interval_width_grows_with_confidence(boot_prevalences):
    narrow = ConfidenceInterval(boot_prevalences, confidence_level=0.80)
    wide = ConfidenceInterval(boot_prevalences, confidence_level=0.99)
    nlow, nhigh = narrow.get_region()
    wlow, whigh = wide.get_region()
    assert np.all((whigh - wlow) >= (nhigh - nlow) - 1e-12)


# --- chi-squared ellipse (simplex space) --------------------------------

def test_ellipse_region_and_contains():
    data = _full_rank_cluster([0.3, 0.4, 0.3], scale=0.05)
    ce = ConfidenceEllipseSimplex(data, confidence_level=0.95)
    mean_, precision, chi2_val = ce.get_region()
    assert mean_.shape == (3,)
    assert precision is not None and precision.shape == (3, 3)
    assert chi2_val > 0
    # the centre is always inside the ellipse (Mahalanobis distance 0)
    assert ce.contains(ce.get_point_estimate()) is True
    # a point far from the cluster is outside
    assert ce.contains(np.array([5.0, -5.0, 5.0])) is False


# --- chi-squared ellipse (CLR space) ------------------------------------

def test_ellipse_clr_runs(boot_prevalences):
    clr = ConfidenceEllipseCLR(boot_prevalences, confidence_level=0.9)
    assert clr.get_point_estimate().shape == (3,)
    # contains() must run and return a boolean for a valid prevalence vector
    result = clr.contains(np.array([0.2, 0.3, 0.5]))
    assert isinstance(result, (bool, np.bool_))


# --- factory ------------------------------------------------------------

@pytest.mark.parametrize("method, cls", [
    ("intervals", ConfidenceInterval),
    ("ellipse", ConfidenceEllipseSimplex),
    ("ellipse-clr", ConfidenceEllipseCLR),
    ("clr", ConfidenceEllipseCLR),
])
def test_factory_dispatch(boot_prevalences, method, cls):
    region = construct_confidence_region(boot_prevalences, 0.9, method=method)
    assert type(region) is cls


def test_factory_is_case_insensitive(boot_prevalences):
    assert type(construct_confidence_region(boot_prevalences, method="ELLIPSE")) is ConfidenceEllipseSimplex


def test_factory_unknown_method(boot_prevalences):
    with pytest.raises(NotImplementedError):
        construct_confidence_region(boot_prevalences, method="banana")
