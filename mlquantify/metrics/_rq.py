import numpy as np
from scipy.stats import cumfreq
from mlquantify.metrics._slq import SE

from ._utils import process_inputs


def VSE(prev_real, prev_pred, train_values):
    r"""
    Compute the Variance-normalised Squared Error (VSE).

    Parameters
    ----------
    prev_real : array-like
        True regression values (from test set).

    prev_pred : array-like
        Predicted regression values (from test set).

    train_values : array-like
        True regression values from training set, used to compute variance normalization.

    Returns
    -------
    verror : float
        Variance-normalised squared error.
    """
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    if isinstance(train_values, dict):
        train_values = np.asarray(list(train_values.values()))
    var_train = np.var(train_values, ddof=1)
    if var_train == 0:
        return np.nan
    return SE(prev_real, prev_pred) / var_train


def CvM_L1(prev_real, prev_pred, n_bins=100):
    r"""
    Compute the L1 version of the Cramér–von Mises statistic (Xiao et al., 2006)
    between two cumulative distributions, as suggested by Bella et al. (2014).

    Parameters
    ----------
    prev_real : array-like
        True regression values.

    prev_pred : array-like
        Predicted regression values.

    n_bins : int, optional
        Number of bins used to estimate cumulative distributions (default=100).

    Returns
    -------
    statistic : float
        L1 Cramér–von Mises distance between cumulative distributions.
    """
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)

    # Compute empirical cumulative distributions
    min_val = min(np.min(prev_real), np.min(prev_pred))
    max_val = max(np.max(prev_real), np.max(prev_pred))

    real_cum = cumfreq(prev_real, numbins=n_bins, defaultreallimits=(min_val, max_val))
    pred_cum = cumfreq(prev_pred, numbins=n_bins, defaultreallimits=(min_val, max_val))

    # Normalize to [0, 1]
    F_real = real_cum.cumcount / real_cum.cumcount[-1]
    F_pred = pred_cum.cumcount / pred_cum.cumcount[-1]

    # L1 integral between cumulative distributions
    statistic = np.mean(np.abs(F_real - F_pred))
    return float(statistic)
