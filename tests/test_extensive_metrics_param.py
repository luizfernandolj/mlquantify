import numpy as np
import pytest

from mlquantify.metrics import (
    AE,
    MAE,
    SE,
    MSE,
    KLD,
    NAE,
    NKLD,
    RAE,
    NRAE,
    NMD,
    RNOD,
    VSE,
    CvM_L1,
)


CORE_CASES = [
    (np.array([0.2, 0.8]), np.array([0.3, 0.7])),
    (np.array([0.6, 0.4]), np.array([0.4, 0.6])),
    (np.array([0.1, 0.9]), np.array([0.2, 0.8])),
    (np.array([0.5, 0.5]), np.array([0.55, 0.45])),
    (np.array([0.3, 0.7]), np.array([0.25, 0.75])),
    (np.array([0.15, 0.85]), np.array([0.2, 0.8])),
    (np.array([0.9, 0.1]), np.array([0.8, 0.2])),
    (np.array([0.7, 0.3]), np.array([0.65, 0.35])),
]

ARRAY_METRICS = [AE, KLD, NKLD]
SCALAR_METRICS = [MAE, SE, MSE, NAE, RAE, NRAE]


@pytest.mark.parametrize("metric", ARRAY_METRICS)
@pytest.mark.parametrize("prev_real, prev_pred", CORE_CASES)
def test_array_metrics_nonnegative(metric, prev_real, prev_pred):
    result = metric(prev_pred, prev_real)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(prev_real)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)


@pytest.mark.parametrize("metric", SCALAR_METRICS)
@pytest.mark.parametrize("prev_real, prev_pred", CORE_CASES)
def test_scalar_metrics_nonnegative(metric, prev_real, prev_pred):
    result = metric(prev_pred, prev_real)
    assert np.isfinite(result)
    assert result >= 0


@pytest.mark.parametrize("metric", [AE, MAE, KLD, NAE, RAE, NRAE])
def test_metrics_accept_dict(metric):
    prev_real = {0: 0.2, 1: 0.8}
    prev_pred = {0: 0.3, 1: 0.7}
    result = metric(prev_pred, prev_real)
    if metric is AE:
        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1}
    else:
        assert np.all(np.isfinite(np.asarray(result)))


def test_ae_padding_length_mismatch():
    prev_real = np.array([0.5, 0.4, 0.1])
    prev_pred = np.array([0.5, 0.5])
    result = AE(prev_pred, prev_real)
    assert result.shape[0] == len(prev_real)


ORDINAL_CASES = [
    (np.array([0.2, 0.3, 0.5]), np.array([0.3, 0.2, 0.5])),
    (np.array([0.1, 0.2, 0.7]), np.array([0.2, 0.1, 0.7])),
    (np.array([0.4, 0.4, 0.2]), np.array([0.3, 0.5, 0.2])),
    (np.array([0.6, 0.2, 0.2]), np.array([0.5, 0.3, 0.2])),
    (np.array([0.3, 0.3, 0.4]), np.array([0.2, 0.4, 0.4])),
    (np.array([0.2, 0.5, 0.3]), np.array([0.3, 0.4, 0.3])),
]


@pytest.mark.parametrize("prev_real, prev_pred", ORDINAL_CASES)
def test_nmd_nonnegative(prev_real, prev_pred):
    result = NMD(prev_pred, prev_real)
    assert np.isfinite(result)
    assert result >= 0


@pytest.mark.parametrize("prev_real, prev_pred", ORDINAL_CASES)
def test_rnod_nonnegative(prev_real, prev_pred):
    result = RNOD(prev_pred, prev_real)
    assert np.isfinite(result)
    assert result >= 0


def test_rnod_custom_distance():
    prev_real = np.array([0.2, 0.5, 0.3])
    prev_pred = np.array([0.3, 0.4, 0.3])
    distances = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    result = RNOD(prev_pred, prev_real, distances=distances)
    assert np.isfinite(result)
    assert result >= 0


REG_CASES = [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
    (np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]), np.array([0.5, 1.5, 2.5])),
    (np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
    (np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.5, 2.5]), np.array([0.0, 1.0, 2.0])),
    (np.array([-1.0, 0.0, 1.0]), np.array([-0.5, 0.5, 1.5]), np.array([-1.0, 0.0, 1.0])),
]


@pytest.mark.parametrize("prev_real, prev_pred, train_values", REG_CASES)
def test_vse_nonnegative_or_nan(prev_real, prev_pred, train_values):
    result = VSE(prev_pred, prev_real, train_values)
    if np.var(train_values, ddof=1) == 0:
        assert np.isnan(result)
    else:
        assert np.isfinite(result)
        assert np.all(result >= 0)


@pytest.mark.parametrize("prev_real, prev_pred, _", REG_CASES)
def test_cvm_l1_nonnegative(prev_real, prev_pred, _):
    result = CvM_L1(prev_pred, prev_real)
    assert np.isfinite(result)
    assert result >= 0


def test_cvm_l1_identical_zero():
    prev_real = np.array([1.0, 2.0, 3.0, 4.0])
    prev_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert CvM_L1(prev_pred, prev_real) == 0.0
