import numpy as np
from scipy.optimize import minimize


def solve_simplex(
    objective,
    n_classes,
    x0=None,
    bounds=None,
    tol=1e-10,
    random_state=None,
):
    """Minimize a function over the probability simplex using SLSQP.

    Applies :func:`scipy.optimize.minimize` with the SLSQP method, an
    equality constraint ensuring the prevalence vector sums to 1, and
    box constraints bounding each component to ``[0, 1]``.  The result is
    clipped to non-negative values and re-normalized.

    Parameters
    ----------
    objective : callable
        Function ``f(p) -> float`` where ``p`` is an array of shape
        ``(n_classes,)``.
    n_classes : int
        Number of classes (determines the dimensionality of ``x0``).
    x0 : array-like of shape (n_classes,) or None, default=None
        Initial prevalence guess.  Defaults to the uniform distribution
        when ``random_state`` is ``None``, or to a random simplex point
        otherwise.
    bounds : list of (float, float) or None, default=None
        Per-component box constraints.  Defaults to ``[(0, 1)] * n_classes``.
    tol : float, default=1e-10
        Convergence tolerance passed to SLSQP.
    random_state : int, RandomState instance, or None, default=None
        Seed for generating a random starting point when ``x0`` is
        ``None``.

    Returns
    -------
    prevalence : ndarray of shape (n_classes,)
        Estimated prevalence vector summing to 1.
    loss : float
        Objective value at the optimum.

    Examples
    --------
    >>> import numpy as np
    >>> from mlquantify.solvers._simplex import solve_simplex
    >>> target = np.array([0.2, 0.5, 0.3])
    >>> objective = lambda p: np.sum((np.asarray(p) - target) ** 2)
    >>> prevalence, loss = solve_simplex(objective, n_classes=3)
    >>> np.round(prevalence, 2)
    array([0.2, 0.5, 0.3])
    """
    if x0 is None:
        if random_state is None:
            x0 = np.ones(n_classes) / n_classes
        else:
            x0 = _random_simplex_start(random_state, n_classes)

    if bounds is None:
        bounds = [(0.0, 1.0)] * n_classes

    constraints = {
        "type": "eq",
        "fun": lambda p: np.sum(p) - 1.0,
    }

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=tol,
    )

    prevalence = np.asarray(result.x, dtype=float)
    prevalence = np.clip(prevalence, 0.0, None)

    total = prevalence.sum()

    if total > 0:
        prevalence /= total

    loss = float(result.fun)

    return prevalence, loss


def _random_simplex_start(random_state, n_classes):
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")

    if hasattr(random_state, "rand"):
        latent = random_state.rand(n_classes - 1)
    else:
        latent = np.random.RandomState(random_state).rand(n_classes - 1)

    latent = latent * 2.0 - 1.0
    exp_latent = np.exp(latent)

    return np.concatenate(([1.0], exp_latent)) / (1.0 + exp_latent.sum())
