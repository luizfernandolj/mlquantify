import pytest
import numpy as np
from mlquantify.metrics import (
    AE,
    MAE,
    KLD,
    SE,
    MSE,
    NAE,
    NKLD,
    RAE,
    NRAE,
    NMD,
    RNOD,
    VSE,
    CvM_L1,
)

# -------------------------------------------------------------------------
# Test _slq.py (Standard Loss Quantification)
# -------------------------------------------------------------------------

def test_AE():
    true_prev = np.array([0.2, 0.8])
    pred_prev = np.array([0.3, 0.7])
    # Absolute errors: |0.3-0.2|=0.1, |0.7-0.8|=0.1
    expected = np.array([0.1, 0.1])
    np.testing.assert_array_almost_equal(AE(pred_prev, true_prev), expected)

def test_MAE():
    true_prev = np.array([0.2, 0.8])
    pred_prev = np.array([0.3, 0.7])
    # Mean of [0.1, 0.1] is 0.1
    assert MAE(pred_prev, true_prev) == pytest.approx(0.1)

def test_SE():
    true_prev = np.array([0.2, 0.8])
    pred_prev = np.array([0.3, 0.7])
    # Squared errors: (0.1)^2 = 0.01
    expected = np.array([0.01, 0.01])
    np.testing.assert_array_almost_equal(SE(pred_prev, true_prev), expected)

def test_MSE():
    true_prev = np.array([0.2, 0.8])
    pred_prev = np.array([0.3, 0.7])
    # Mean of [0.01, 0.01] is 0.01
    assert MSE(pred_prev, true_prev) == pytest.approx(0.01)

def test_KLD():
    # Avoid log(0) issues by using non-zero values
    true_prev = np.array([0.5, 0.5])
    pred_prev = np.array([0.5, 0.5])
    # KLD should be 0 for identical distributions
    # KLD element-wise is: p * log(p/q)
    # Here 0.5 * log(1) = 0
    np.testing.assert_array_almost_equal(KLD(pred_prev, true_prev), np.array([0.0, 0.0]))

    true_prev = np.array([0.4, 0.6])
    pred_prev = np.array([0.5, 0.5])
    # Element 0: 0.4 * log(0.4/0.5) = 0.4 * log(0.8) approx -0.089... -> abs gives positive contribution?
    # Wait, KLD implementation in _slq.py: prev_real * np.abs(np.log(prev_real / prev_pred))
    # Standard KL is sum(p * log(p/q)). Individual elements are p*log(p/q).
    # The implementation takes absolute value of log? That's unusual but we test behavior.
    
    val0 = 0.4 * np.abs(np.log(0.4/0.5))
    val1 = 0.6 * np.abs(np.log(0.6/0.5))
    expected = np.array([val0, val1])
    np.testing.assert_array_almost_equal(KLD(pred_prev, true_prev), expected)

def test_NKLD():
    # NKLD = 2 * (e^KLD / (e^KLD + 1)) - 1
    # If KLD sum (wait, implementation calls KLD which returns array)
    # NKLD calls KLD then does np.exp(kl_divergence).
    # Does NKLD sum the result?
    # Looking at code: euler = np.exp(kl_divergence), return 2*(...)-1.
    # It returns an array if KLD returns an array.
    true_prev = np.array([0.5, 0.5])
    pred_prev = np.array([0.5, 0.5])
    # KLD is [0, 0]. exp(0)=1. 2*(1/2)-1 = 0.
    np.testing.assert_array_almost_equal(NKLD(pred_prev, true_prev), np.array([0.0, 0.0]))

# -------------------------------------------------------------------------
# Test _oq.py (Ordinal Quantification)
# -------------------------------------------------------------------------

def test_NMD():
    # Normalized Match Distance (EMD)
    true_prev = np.array([0.5, 0.5])
    pred_prev = np.array([0.5, 0.5])
    # Zero distance for identical
    assert NMD(pred_prev, true_prev) == 0.0
    
    true_prev = np.array([1.0, 0.0, 0.0])
    pred_prev = np.array([0.0, 1.0, 0.0])
    # Distances default to 1. n_classes=3.
    # Cumulative real: [1, 1, 1] (last excluded in calculation usually? NMD uses cum_diffs[:-1])
    # Cumulative pred: [0, 1, 1]
    # Diff: [-1, 0]
    # Abs Diff: [1, 0]
    # Sum: 1. Div by (3-1)=2. Result 0.5.
    assert NMD(pred_prev, true_prev) == 0.5

def test_RNOD():
    # Root Normalised Order-aware Divergence
    # Test identical
    true_prev = np.array([0.2, 0.3, 0.5])
    pred_prev = np.array([0.2, 0.3, 0.5])
    assert RNOD(pred_prev, true_prev) == 0.0

# -------------------------------------------------------------------------
# Test _rq.py (Regression/Quantification)
# -------------------------------------------------------------------------

def test_CvM_L1():
    # Cramer-von Mises L1
    true_vals = np.array([1, 2, 3])
    pred_vals = np.array([1, 2, 3])
    # Should be 0
    assert CvM_L1(pred_vals, true_vals) == 0.0
    
    true_vals = np.array([1, 1, 1])
    pred_vals = np.array([2, 2, 2])
    # Distributions are distinct. 
    # Real cum: step at 1 to 1.0.
    # Pred cum: step at 2 to 1.0.
    # Between 1 and 2, Real=1.0, Pred=0.0. Diff=1.0.
    # Integration over domain... this function does discretized bins?
    # Implementation uses cumfreq with n_bins=100.
    # It returns mean absolute difference of cumulative frequencies.
    # Since they differ completely in range [1, 2], mean difference should be > 0.
    assert CvM_L1(pred_vals, true_vals) > 0.0
