
import pytest
import numpy as np
from sklearn.datasets import make_classification

from mlquantify._config import config_context
import mlquantify.readme
from mlquantify.readme import ReadMe

try:
    import torch  # noqa: F401
    from mlquantify.readme import ReadMe2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


def _shifted_bag(X, y, positive_prevalence, n=400, seed=0):
    rng = np.random.RandomState(seed)
    idx1, idx0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n1 = int(positive_prevalence * n)
    take = np.concatenate([
        rng.choice(idx0, n - n1, replace=True),
        rng.choice(idx1, n1, replace=True),
    ])
    return X[take], np.array([1 - positive_prevalence, positive_prevalence])


# ------------------------------------------------------------------- ReadMe

def test_readme_fit_predict_binary(binary_dataset):
    X, y = binary_dataset
    q = ReadMe(n_subsets=20, random_state=0)
    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=2)


def test_readme_fit_predict_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    q = ReadMe(n_subsets=20, random_state=0)
    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=3)


def test_readme_binary_input_without_binarization(binary_dataset):
    X, y = binary_dataset
    X_bin = (X > np.median(X, axis=0)).astype(int)
    q = ReadMe(n_subsets=10, binarize=False, random_state=0).fit(X_bin, y)
    assert q.thresholds_ is None
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X_bin)
    _assert_valid_prevalence(prevalence, n_classes=2)


def test_readme_rejects_continuous_when_binarize_false(binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError):
        ReadMe(binarize=False).fit(X, y)


def test_readme_recovers_shifted_prevalence():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               class_sep=1.5, random_state=0)
    q = ReadMe(n_subsets=30, subset_size=12, random_state=0).fit(X, y)
    for prev in (0.2, 0.8):
        X_bag, true = _shifted_bag(X, y, prev)
        with config_context(prevalence_return_type="array"):
            estimate = q.predict(X_bag)
        assert np.abs(estimate - true).mean() < 0.15


def test_readme_deterministic(binary_dataset):
    X, y = binary_dataset
    with config_context(prevalence_return_type="array"):
        p1 = ReadMe(n_subsets=10, random_state=7).fit(X, y).predict(X)
        p2 = ReadMe(n_subsets=10, random_state=7).fit(X, y).predict(X)
    np.testing.assert_allclose(p1, p2)


def test_readme_param_validation(binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError):
        ReadMe(subset_size=0).fit(X, y)
    with pytest.raises(ValueError):
        ReadMe(binarize="maybe").fit(X, y)


def test_readme_get_set_params_roundtrip():
    q = ReadMe(n_subsets=5, subset_size=8, random_state=3)
    params = q.get_params()
    q2 = ReadMe().set_params(**params)
    assert q2.get_params() == params


def test_readme_available_without_torch():
    assert "ReadMe" in mlquantify.readme.__all__


# ------------------------------------------------------------------ ReadMe2

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not installed")
def test_readme2_fit_predict_binary(binary_dataset):
    X, y = binary_dataset
    q = ReadMe2(n_boot=2, sgd_iters=50, n_boot_match=5, random_state=0)
    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not installed")
def test_readme2_fit_predict_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    q = ReadMe2(n_boot=2, sgd_iters=50, n_boot_match=5, random_state=0)
    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=3)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not installed")
def test_readme2_no_matching(binary_dataset):
    X, y = binary_dataset
    q = ReadMe2(n_boot=2, sgd_iters=50, matching=False, random_state=0)
    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not installed")
def test_readme2_recovers_shifted_prevalence():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               class_sep=1.5, random_state=0)
    q = ReadMe2(n_boot=3, sgd_iters=100, n_boot_match=10, random_state=0).fit(X, y)
    for prev in (0.2, 0.8):
        X_bag, true = _shifted_bag(X, y, prev)
        with config_context(prevalence_return_type="array"):
            estimate = q.predict(X_bag)
        assert np.abs(estimate - true).mean() < 0.15


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not installed")
def test_readme2_deterministic(binary_dataset):
    X, y = binary_dataset
    with config_context(prevalence_return_type="array"):
        p1 = ReadMe2(n_boot=2, sgd_iters=30, n_boot_match=3,
                     random_state=7).fit(X, y).predict(X)
        p2 = ReadMe2(n_boot=2, sgd_iters=30, n_boot_match=3,
                     random_state=7).fit(X, y).predict(X)
    np.testing.assert_allclose(p1, p2)
