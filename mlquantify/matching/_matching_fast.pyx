# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Compiled histogram distribution-matching kernel.

Folds ``mixture -> normalize -> distance -> ternary-search-over-alpha`` into a
single ``nogil`` routine, eliminating the ~1100 tiny-array numpy calls/predict
of the pure-Python path. Output matches :mod:`mlquantify.matching._matching_py`.
"""
import numpy as np
from libc.math cimport sqrt, log

cdef double _EPS = 1e-12
cdef double _FLOOR = 1e-20


cdef inline double _distance(const double[::1] mix, double total,
                             const double[::1] test_n, Py_ssize_t n,
                             int metric) noexcept nogil:
    """distance(normalize(mix), test_n) with the library's clip constants."""
    cdef Py_ssize_t i
    cdef double s = 0.0, p, q, d
    if metric == 0:                                  # hellinger
        for i in range(n):
            p = mix[i] / total
            if p < _FLOOR: p = _FLOOR
            q = test_n[i]
            if q < _FLOOR: q = _FLOOR
            d = sqrt(p) - sqrt(q)
            s += d * d
        return sqrt(0.5 * s)
    elif metric == 1:                                # topsoe
        for i in range(n):
            p = mix[i] / total
            if p < _FLOOR: p = _FLOOR
            q = test_n[i]
            if q < _FLOOR: q = _FLOOR
            s += p * log(2.0 * p / (p + q)) + q * log(2.0 * q / (p + q))
        return s
    else:                                            # probsymm
        for i in range(n):
            p = mix[i] / total
            if p < _FLOOR: p = _FLOOR
            q = test_n[i]
            if q < _FLOOR: q = _FLOOR
            s += (p - q) * log(p / q)
        return s


cdef inline double _objective(const double[::1] neg, const double[::1] pos,
                              const double[::1] test_n, double a,
                              double[::1] mix, Py_ssize_t n,
                              int metric) noexcept nogil:
    """Build the alpha-mixture (clipped) into ``mix`` and score it."""
    cdef Py_ssize_t i
    cdef double total = 0.0, v
    for i in range(n):
        v = (1.0 - a) * neg[i] + a * pos[i]
        if v < _EPS: v = _EPS
        mix[i] = v
        total += v
    return _distance(mix, total, test_n, n, metric)


def match_sweep(const double[::1] neg, const double[::1] pos,
                const double[::1] test, int metric=1, int solver=0,
                int grid_size=101, double tol=1e-6):
    """Return the positive-class mixture weight alpha in ``[0, 1]``.

    ``solver`` selects the search: ``0`` ternary (unimodal), ``1`` exhaustive
    grid of ``grid_size`` points. See
    :func:`mlquantify.matching._matching_py.match_sweep` for semantics.
    """
    cdef Py_ssize_t i, k, n = neg.shape[0]
    cdef double[::1] test_n = np.empty(n, dtype=np.float64)
    cdef double[::1] mix = np.empty(n, dtype=np.float64)
    cdef double s = 0.0, lo = 0.0, hi = 1.0, m1, m2, f1, f2, v
    cdef double best, best_a = 0.0, a, f, result = 0.0

    with nogil:
        # normalize test once: clip to EPS, divide by sum (uniform if degenerate)
        for i in range(n):
            v = test[i]
            if v < _EPS: v = _EPS
            test_n[i] = v
            s += v
        if s <= _EPS:
            for i in range(n):
                test_n[i] = 1.0 / n
        else:
            for i in range(n):
                test_n[i] /= s

        if solver == 1:                              # exhaustive grid
            best = _objective(neg, pos, test_n, 0.0, mix, n, metric)
            for k in range(1, grid_size):
                a = <double> k / (grid_size - 1)
                f = _objective(neg, pos, test_n, a, mix, n, metric)
                if f < best:
                    best = f
                    best_a = a
            result = best_a
        else:                                        # ternary search
            while hi - lo > tol:
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                f1 = _objective(neg, pos, test_n, m1, mix, n, metric)
                f2 = _objective(neg, pos, test_n, m2, mix, n, metric)
                if f1 < f2:
                    hi = m2
                else:
                    lo = m1
            result = 0.5 * (lo + hi)

    return result
