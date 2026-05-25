import numpy as np
from scipy.optimize import lsq_linear


def solve_constrained_least_squares(
    M,
    q,
    bounds=(0.0, 1.0),
):
    r"""Solve a box-constrained least-squares problem for prevalence estimation.

    Finds the prevalence vector :math:`\hat{p}` that minimises
    :math:`\|M\hat{p} - q\|_2^2` subject to component-wise box constraints,
    and then projects the result onto the probability simplex by clipping to
    non-negative values and normalising.

    This is used by Adjusted Classify and Count (ACC) and related methods
    where ``M`` is the confusion-rate matrix and ``q`` is the vector of
    uncorrected class counts.

    Parameters
    ----------
    M : array-like of shape (n_components, n_classes)
        System matrix (e.g. confusion-rate matrix).
    q : array-like of shape (n_components,)
        Target vector (e.g. observed posterior mean vector).
    bounds : tuple of (float, float) or 2-tuple of arrays, default=(0.0, 1.0)
        Box constraints on each component of the solution, passed directly
        to :func:`scipy.optimize.lsq_linear`.

    Returns
    -------
    prevalence : ndarray of shape (n_classes,)
        Estimated prevalence vector summing to 1.
    residual : float
        :math:`\ell_2` residual norm :math:`\|M\hat{p} - q\|_2` at the
        solution.

    Examples
    --------
    >>> import numpy as np
    >>> from mlquantify.solvers._least_squares import solve_constrained_least_squares
    >>> M = np.array([[0.8, 0.2], [0.1, 0.9]])
    >>> q = np.array([0.45, 0.55])
    >>> prevalence, residual = solve_constrained_least_squares(M, q)
    >>> np.round(prevalence, 2)
    array([0.46, 0.54])
    """
    M = np.asarray(M, dtype=float)
    q = np.asarray(q, dtype=float)

    result = lsq_linear(
        M,
        q,
        bounds=bounds,
    )

    prevalence = np.asarray(result.x, dtype=float)

    prevalence = np.clip(prevalence, 0.0, None)

    total = prevalence.sum()

    if total > 0:
        prevalence /= total

    residual = float(np.linalg.norm(M @ prevalence - q))

    return prevalence, residual