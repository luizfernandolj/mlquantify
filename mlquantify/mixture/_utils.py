import numpy as np


# =====================================================
# Utility functions
# =====================================================

def getHist(scores, nbins):
    r"""
    Calculate histogram-like bin probabilities for a given set of scores.

    This function divides the score range into equal bins and computes the proportion 
    of scores in each bin, normalized by the total count.

    Parameters
    ----------
    scores : np.ndarray
        A 1-dimensional array of scores.
    nbins : int
        Number of bins for dividing the score range.

    Returns
    -------
    np.ndarray
        An array containing the normalized bin probabilities.

    Notes
    -----
    - The bins are equally spaced between 0 and 1, with an additional upper boundary 
      to include the maximum score.
    - The returned probabilities are normalized to account for the total number of scores.
    """
    breaks = np.linspace(0, 1, int(nbins) + 1)
    breaks = np.delete(breaks, -1)
    breaks = np.append(breaks, 1.1)

    re = np.repeat(1 / (len(breaks) - 1), (len(breaks) - 1))
    for i in range(1, len(breaks)):
        re[i - 1] = (re[i - 1] + len(np.where((scores >= breaks[i - 1]) & (scores < breaks[i]))[0])) / (len(scores) + 1)

    return re


def ternary_search(left: float, right: float, func, tol: float = 1e-4) -> float:
    r"""
    Ternary search to find the minimum of a unimodal function in [left, right].

    Parameters
    ----------
    left : float
        Left bound.
    right : float
        Right bound.
    func : callable
        Function to minimize.
    tol : float, optional
        Tolerance for termination. Default is 1e-4.

    Returns
    -------
    float
        Approximate position of the minimum.
    """
    while right - left > tol:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        f1, f2 = func(m1), func(m2)
        if f1 < f2:
            right = m2
        else:
            left = m1
    return (left + right) / 2