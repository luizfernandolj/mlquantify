import numpy as np
from scipy.optimize import minimize_scalar


def ternary_search(
    left,
    right,
    objective,
    tol=1e-6,
):
    """Find the minimum of a unimodal function via ternary search.

    Iteratively narrows the interval ``[left, right]`` by comparing
    the objective at two interior probe points until the interval width
    falls below ``tol``.

    Parameters
    ----------
    left : float
        Left bound of the search interval.
    right : float
        Right bound of the search interval.
    objective : callable
        Scalar function to minimize over ``[left, right]``.  Must be
        unimodal on the interval.
    tol : float, default=1e-6
        Convergence criterion: the search stops when
        ``right - left <= tol``.

    Returns
    -------
    minimum : float
        Approximate location of the minimum.

    See Also
    --------
    solve_binary : Binary prevalence solver that can call this search.

    Examples
    --------
    >>> from mlquantify.solvers._binary import ternary_search
    >>> minimum = ternary_search(0.0, 1.0, lambda x: (x - 0.3) ** 2)
    >>> round(minimum, 4)
    0.3
    """
    while right - left > tol:
        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0

        f1 = objective(m1)
        f2 = objective(m2)

        if f1 < f2:
            right = m2
        else:
            left = m1

    return (left + right) / 2.0


def solve_binary(
    objective,
    solver="auto",
    grid_size=101,
    tol=1e-6,
):
    """Minimize a scalar objective over the binary prevalence space [0, 1].

    Supports three strategies: exhaustive grid search, ternary search
    (for unimodal objectives), and scipy's bounded scalar minimizer.

    Parameters
    ----------
    objective : callable
        Function ``f(alpha) -> float`` where ``alpha`` is the prevalence
        of the positive class in ``[0, 1]``.
    solver : {'auto', 'grid', 'ternary', 'bounded'}, default='auto'
        Optimization strategy.

        - ``'auto'`` selects ``'bounded'``.
        - ``'grid'`` evaluates ``objective`` at ``grid_size`` equally
          spaced points and returns the best.
        - ``'ternary'`` applies ternary search (assumes unimodal
          objective).
        - ``'bounded'`` uses :func:`scipy.optimize.minimize_scalar` with
          ``method='bounded'``.

    grid_size : int, default=101
        Number of candidate prevalence values for the ``'grid'`` solver.
    tol : float, default=1e-6
        Convergence tolerance for the ``'ternary'`` solver.

    Returns
    -------
    prevalence : ndarray of shape (2,)
        Estimated binary prevalence vector ``[1 - alpha, alpha]``.
    loss : float
        Objective value at the optimum.

    Raises
    ------
    ValueError
        If ``solver`` is not one of the recognised identifiers.

    See Also
    --------
    ternary_search : Unimodal line search used by the ``'ternary'`` strategy.
    minimize_prevalence : Binary/multiclass dispatcher built on this.

    Examples
    --------
    >>> from mlquantify.solvers._binary import solve_binary
    >>> prevalence, loss = solve_binary(lambda a: (a - 0.3) ** 2,
    ...                                 solver="bounded")
    >>> round(prevalence[1], 2)
    np.float64(0.3)
    """
    if solver == "auto":
        solver = "bounded"

    if solver == "grid":
        alphas = np.linspace(0.0, 1.0, int(grid_size))
        losses = np.asarray([objective(alpha) for alpha in alphas])

        best_idx = int(np.argmin(losses))

        alpha = float(alphas[best_idx])
        loss = float(losses[best_idx])

        return np.asarray([1.0 - alpha, alpha]), loss

    if solver == "ternary":
        alpha = ternary_search(
            0.0,
            1.0,
            objective,
            tol=tol,
        )

        loss = float(objective(alpha))

        return np.asarray([1.0 - alpha, alpha]), loss

    if solver == "bounded":
        result = minimize_scalar(
            objective,
            bounds=(0.0, 1.0),
            method="bounded",
        )

        alpha = float(result.x)
        loss = float(result.fun)

        return np.asarray([1.0 - alpha, alpha]), loss

    raise ValueError(f"Unknown binary solver: {solver}")
