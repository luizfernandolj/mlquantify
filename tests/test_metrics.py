
import pytest
import numpy as np
from mlquantify.metrics import MAE, MSE, KLD, NKLD, NMD, RNOD, AE, RAE

# All quantification metrics follow the scikit-learn convention: the true
# prevalence is the first argument, the prediction the second --  metric(y_true, y_pred).

def test_mae_calculation():
    y_true = np.array([0.2, 0.8])
    y_pred = np.array([0.3, 0.7])
    # |0.2-0.3| + |0.8-0.7| = 0.1 + 0.1 = 0.2 / 2 = 0.1
    assert MAE(y_true, y_pred) == pytest.approx(0.1)

def test_mse_calculation():
    y_true = np.array([0.2, 0.8])
    y_pred = np.array([0.4, 0.6])
    # (0.2-0.4)^2 + (0.8-0.6)^2 = 0.04 + 0.04 = 0.08 / 2 = 0.04
    assert MSE(y_true, y_pred) == pytest.approx(0.04)

def test_metrics_input_formats():
    y_true_dict = {'a': 0.2, 'b': 0.8}
    y_pred_dict = {'a': 0.3, 'b': 0.7}

    y_true_list = [0.2, 0.8]
    y_pred_list = [0.3, 0.7] # Intentionally same values as dict

    mae_dict = MAE(y_true_dict, y_pred_dict)
    mae_list = MAE(y_true_list, y_pred_list)

    assert mae_dict == pytest.approx(0.1)
    assert mae_list == pytest.approx(0.1)

def test_ae_per_class():
    y_true = np.array([0.2, 0.8])
    y_pred = np.array([0.3, 0.7])
    ae = AE(y_true, y_pred)
    assert np.allclose(ae, [0.1, 0.1])

    y_true_dict = {'a': 0.2, 'b': 0.8}
    y_pred_dict = {'a': 0.3, 'b': 0.7}
    ae_dict = AE(y_true_dict, y_pred_dict)
    assert ae_dict['a'] == pytest.approx(0.1)
    assert ae_dict['b'] == pytest.approx(0.1)

def test_mismatched_inputs():
    # Helper process_inputs usually pads with 0s
    y_true = np.array([0.2, 0.8])
    y_pred = np.array([0.3]) # Missing class

    # Implementation dependent: might pad or error.
    # Based on view_file: pads with 0s.
    # y_pred becomes [0.3, 0.0]
    # |0.2-0.3| + |0.8-0.0| = 0.1 + 0.8 = 0.9 / 2 = 0.45
    assert MAE(y_true, y_pred) == pytest.approx(0.45)

def test_ordinal_metrics():
    y_true = np.array([0.2, 0.5, 0.3])
    y_pred = np.array([0.2, 0.5, 0.3])
    assert NMD(y_true, y_pred) == 0.0
    assert RNOD(y_true, y_pred) == 0.0


def test_argument_order_is_true_pred():
    """Asymmetric metrics must read the FIRST argument as the truth.

    RAE normalises by the true prevalence, so swapping the arguments changes
    the result -- this guards the scikit-learn ``(y_true, y_pred)`` order
    against silent regressions.
    """
    y_true = np.array([0.25, 0.75])
    y_pred = np.array([0.5, 0.5])
    # AE = [0.25, 0.25]; RAE = mean([0.25/0.25, 0.25/0.75]) = mean([1.0, 1/3]) = 2/3
    assert RAE(y_true, y_pred, eps=0) == pytest.approx(2.0 / 3.0)
    # Swapping normalises by the prediction instead -> different value.
    assert RAE(y_pred, y_true, eps=0) != pytest.approx(2.0 / 3.0)


def test_rae_multiclass_per_class_normalisation():
    """RAE must normalise each class's error by that class's true prevalence."""
    y_true = np.array([0.2, 0.3, 0.5])
    y_pred = np.array([0.4, 0.3, 0.3])
    # AE = [0.2, 0.0, 0.2]; RAE = mean([0.2/0.2, 0, 0.2/0.5]) = mean([1, 0, 0.4])
    assert RAE(y_true, y_pred, eps=0) == pytest.approx(1.4 / 3.0)


def test_rae_zero_prevalence_is_finite():
    """Default smoothing keeps RAE finite when a class is absent from the truth."""
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.1, 0.9])
    result = RAE(y_true, y_pred)
    assert np.isfinite(result)
    # the literature convention when the sample size n is known
    result = RAE(y_true, y_pred, eps=1.0 / (2 * 100))
    assert np.isfinite(result)


def test_kld_zero_prevalence_is_finite():
    """Default smoothing keeps KLD/NKLD finite for absent or missed classes."""
    y_true = np.array([0.0, 1.0])       # class absent from the truth
    y_pred = np.array([0.1, 0.9])
    assert np.all(np.isfinite(KLD(y_true, y_pred)))
    assert np.all(np.isfinite(NKLD(y_true, y_pred)))

    y_pred_zero = np.array([1.0, 0.0])  # class missed by the prediction
    y_true_pos = np.array([0.5, 0.5])
    assert np.all(np.isfinite(KLD(y_true_pos, y_pred_zero)))
    assert np.all(np.isfinite(NKLD(y_true_pos, y_pred_zero)))


def test_kld_eps_zero_reproduces_unsmoothed_value():
    """eps=0 disables smoothing and matches the raw formula on positive input."""
    y_true = np.array([0.3, 0.7])
    y_pred = np.array([0.4, 0.6])
    expected = y_true * np.abs(np.log(y_true / y_pred))
    np.testing.assert_allclose(KLD(y_true, y_pred, eps=0), expected)


def test_degenerate_single_class_inputs_are_finite():
    """Normalised metrics guard their denominators in degenerate cases."""
    from mlquantify.metrics import NAE

    assert NAE(np.array([1.0]), np.array([1.0])) == 0.0
    assert NMD(np.array([1.0]), np.array([1.0])) == 0.0
    assert RNOD(np.array([1.0]), np.array([1.0])) == 0.0
    # all-zero truth (e.g. zero-padded input) must not divide by an empty support
    assert np.isfinite(RNOD(np.array([0.0, 0.0]), np.array([0.5, 0.5])))
