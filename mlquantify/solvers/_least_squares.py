import numpy as np
from scipy.optimize import lsq_linear


def solve_constrained_least_squares(
    M,
    q,
    bounds=(0.0, 1.0),
):
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