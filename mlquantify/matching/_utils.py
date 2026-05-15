import numpy as np
from math import pi
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances


EPS = 1e-12


def gaussian_kernel(X, Y=None, bandwidth=0.1):
    """Return the Gaussian KDE kernel matrix between two sample sets."""
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(Y)

    squared_distances = pairwise_distances(X, Y, metric="euclidean") ** 2
    n_features = X.shape[1]
    normalizer = (bandwidth**n_features) * ((2 * pi) ** (n_features / 2))
    return np.exp(-squared_distances / (2 * (bandwidth**2))) / (normalizer + EPS)


def negative_log_likelihood(likelihoods):
    """Return a numerically stable negative log-likelihood."""
    likelihoods = np.clip(likelihoods, EPS, None)
    return float(-np.sum(np.log(likelihoods)))


def get_histogram(values, n_bins, range=(0.0, 1.0)):
    """Return a smoothed normalized histogram."""
    values = np.asarray(values, dtype=float).ravel()
    hist, _ = np.histogram(values, bins=int(n_bins), range=range)
    hist = hist.astype(float) + 1.0
    return hist / hist.sum()


def get_feature_histogram(values, n_bins):
    """Return a smoothed histogram with data-dependent feature range."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return np.ones(int(n_bins), dtype=float) / int(n_bins)

    lower = np.min(values)
    upper = np.max(values)
    if lower == upper:
        lower -= 0.5
        upper += 0.5

    hist, _ = np.histogram(values, bins=int(n_bins), range=(lower, upper))
    hist = hist.astype(float) + 1.0
    return hist / hist.sum()


def ternary_search(left, right, func, tol=1e-4, max_iter=200):
    """Minimize a unimodal scalar function on an interval."""
    for _ in range(max_iter):
        if right - left <= tol:
            break
        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0
        if func(m1) < func(m2):
            right = m2
        else:
            left = m1
    return (left + right) / 2.0


def normalize_simplex(values):
    """Project a non-negative vector onto the probability simplex by clipping."""
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0.0)
    total = values.sum()
    if total <= 0:
        return np.ones_like(values) / len(values)
    return values / total


def optimize_on_simplex(objective, n_classes, x0=None):
    """Minimize an objective over the probability simplex using SLSQP."""
    if x0 is None:
        x0 = np.ones(n_classes) / n_classes

    constraints = {
        "type": "eq",
        "fun": lambda alpha: np.sum(alpha) - 1.0,
    }
    bounds = [(0.0, 1.0)] * n_classes

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[constraints],
        options={"maxiter": 200, "ftol": 1e-9},
    )

    alpha = normalize_simplex(getattr(result, "x", x0))
    loss = float(objective(alpha))
    return alpha, loss
