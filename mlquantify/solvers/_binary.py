import numpy as np
from scipy.optimize import minimize_scalar


def ternary_search(
    left,
    right,
    objective,
    tol=1e-6,
):
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