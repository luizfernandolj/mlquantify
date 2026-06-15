from ._binary import solve_binary
from ._simplex import solve_simplex


def minimize_prevalence(
    objective,
    n_classes,
    solver="auto",
    grid_size=101,
    tol=1e-10,
    x0=None,
    random_state=None,
):
    """Minimize an objective function over the probability simplex.

    Dispatches to a binary solver when ``n_classes == 2`` and the solver
    is compatible, and to the SLSQP simplex solver otherwise.

    Parameters
    ----------
    objective : callable
        Objective function to minimize.  For binary problems it must
        accept a scalar ``alpha`` (prevalence of the positive class).
        For multi-class problems it must accept an array of shape
        ``(n_classes,)``.
    n_classes : int
        Number of classes.  Must be ``>= 2``.
    solver : {'auto', 'grid', 'ternary', 'bounded', 'slsqp'}, \
             default='auto'
        Solver to use.  ``'auto'`` selects ``'bounded'`` for binary
        problems and ``'slsqp'`` for multi-class problems.
    grid_size : int, default=101
        Number of evenly-spaced candidate values used by the ``'grid'``
        solver.
    tol : float, default=1e-10
        Convergence tolerance passed to the underlying solver.
    x0 : array-like of shape (n_classes,) or None, default=None
        Initial guess for ``'slsqp'``.  If ``None``, defaults to the
        uniform distribution (or a random simplex point when
        ``random_state`` is set).
    random_state : int, RandomState instance, or None, default=None
        Random seed used to generate a non-uniform starting point for
        ``'slsqp'`` when ``x0`` is ``None``.

    Returns
    -------
    prevalence : ndarray of shape (n_classes,)
        Estimated prevalence vector summing to 1.
    loss : float
        Objective value at the optimum.

    Raises
    ------
    ValueError
        If ``n_classes < 2`` or if ``solver`` is incompatible with the
        number of classes.

    See Also
    --------
    solve_binary : Binary backend used for two-class problems.
    solve_simplex : SLSQP backend used for multiclass problems.
    minimize_prevalence_blocks : Block-wise variant for histogram matching.

    Examples
    --------
    >>> import numpy as np
    >>> from mlquantify.solvers import minimize_prevalence
    >>> objective = lambda alpha: (alpha - 0.3) ** 2
    >>> prevalence, loss = minimize_prevalence(objective, n_classes=2,
    ...                                        solver="bounded")
    >>> round(prevalence[1], 2)
    np.float64(0.3)
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")

    if solver == "auto":
        solver = "bounded" if n_classes == 2 else "slsqp"

    if n_classes == 2 and solver in {
        "grid",
        "ternary",
        "bounded",
    }:
        return solve_binary(
            objective=objective,
            solver=solver,
            grid_size=grid_size,
            tol=tol,
        )

    if solver == "slsqp":
        return solve_simplex(
            objective=objective,
            n_classes=n_classes,
            x0=x0,
            tol=tol,
            random_state=random_state,
        )

    raise ValueError(
        f"Solver {solver!r} is incompatible with "
        f"{n_classes} classes."
    )
