import numpy as np

from mlquantify.metrics import (
    hellinger,
    topsoe,
    probsymm,
    sqEuclidean,
)


EPS = 1e-12


def normalize_distribution(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, EPS)

    total = x.sum()

    if total <= EPS:
        return np.ones_like(x) / len(x)

    return x / total


def get_loss(loss="hellinger", normalize=True):
    def objective(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if normalize:
            a = normalize_distribution(a)
            b = normalize_distribution(b)

        if a.shape != b.shape:
            raise ValueError(
                f"Representations must have the same shape. "
                f"Got {a.shape} and {b.shape}."
            )

        if loss == "hellinger":
            return float(hellinger(a, b))

        if loss == "topsoe":
            return float(topsoe(a, b))

        if loss == "probsymm":
            return float(probsymm(a, b))

        if loss == "sqEuclidean":
            return float(sqEuclidean(a, b))

        if loss == "euclidean":
            return float(np.sqrt(sqEuclidean(a, b)))

        raise ValueError(f"Unknown loss: {loss!r}")

    return objective