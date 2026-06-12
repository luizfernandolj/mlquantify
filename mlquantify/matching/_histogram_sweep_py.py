"""Pure-Python reference for the histogram distribution-matching kernel.

This is the correctness oracle and the fallback used when the compiled
``_histogram_sweep`` extension is not available. It reproduces, in numpy, exactly
what the ``DistanceLoss(normalize=True)`` + ternary-search path computes:

    alpha* = argmin_alpha  distance( normalize((1-a)*neg + a*pos), normalize(test) )

with the same EPS/floor constants and the same ternary search as the Python
solver, so the compiled kernel can be checked against it bit-for-bit (up to the
search tolerance).
"""
import numpy as np

# metric name -> integer id shared with the compiled kernel
METRIC_IDS = {"hellinger": 0, "topsoe": 1, "probsymm": 2}

_EPS = 1e-12      # normalize_distribution clip (losses/_distances.py)
_FLOOR = 1e-20    # per-distance clip (metrics/_distances.py)


def _normalize(x):
    x = np.maximum(np.asarray(x, dtype=float), _EPS)
    total = x.sum()
    if total <= _EPS:
        return np.full(x.shape[0], 1.0 / x.shape[0])
    return x / total


def _distance(p, q, metric):
    p = np.maximum(p, _FLOOR)
    q = np.maximum(q, _FLOOR)
    if metric == 0:   # hellinger
        return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
    if metric == 1:   # topsoe
        return float(np.sum(p * np.log(2 * p / (p + q)) + q * np.log(2 * q / (p + q))))
    if metric == 2:   # probsymm
        return float(np.sum((p - q) * np.log(p / q)))
    raise ValueError(f"unknown metric id {metric}")


def match_sweep(neg, pos, test, metric=1, solver=0, grid_size=101, tol=1e-6):
    """Return the positive-class mixture weight alpha in ``[0, 1]``.

    Parameters
    ----------
    neg, pos : ndarray of shape (n_bins,)
        Class-conditional (negative / positive) representations.
    test : ndarray of shape (n_bins,)
        Test representation.
    metric : int, default=1
        Distance id (0 hellinger, 1 topsoe, 2 probsymm).
    solver : int, default=0
        Search mode: ``0`` ternary (unimodal), ``1`` exhaustive grid.
    grid_size : int, default=101
        Number of points for the grid search (``solver=1``).
    tol : float, default=1e-6
        Ternary-search tolerance on alpha (``solver=0``).
    """
    neg = np.asarray(neg, dtype=float)
    pos = np.asarray(pos, dtype=float)
    test_n = _normalize(test)

    def objective(a):
        mix = _normalize((1.0 - a) * neg + a * pos)
        return _distance(mix, test_n, metric)

    if solver == 1:  # grid (matches solvers.solve_binary grid)
        alphas = np.linspace(0.0, 1.0, int(grid_size))
        return float(alphas[int(np.argmin([objective(a) for a in alphas]))])

    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if objective(m1) < objective(m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)
