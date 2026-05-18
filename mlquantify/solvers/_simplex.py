import numpy as np
from scipy.optimize import minimize


def solve_simplex(
    objective,
    n_classes,
    x0=None,
    bounds=None,
    tol=1e-8,
):
    if x0 is None:
        x0 = np.ones(n_classes) / n_classes

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