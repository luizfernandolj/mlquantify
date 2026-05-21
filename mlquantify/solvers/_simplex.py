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
