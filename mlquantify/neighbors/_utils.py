import numpy as np
from sklearn.metrics import pairwise_distances
from math import pi
from scipy.optimize import minimize


EPS = 1e-12

# ============================================================
# Utilitários
# ============================================================

def gaussian_kernel(X, Y, bandwidth):
    """Matriz de kernel gaussiano K(x,y) com largura 'bandwidth'."""
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(Y)
    sqd = pairwise_distances(X, Y, metric="euclidean") ** 2
    D = X.shape[1]
    norm = (bandwidth ** D) * ((2 * pi) ** (D / 2))
    return np.exp(-sqd / (2 * (bandwidth ** 2))) / (norm + EPS)


def negative_log_likelihood(mixture_likelihoods):
    """Retorna -∑ log(likelihoods) de forma numéricamente estável."""
    mixture_likelihoods = np.clip(mixture_likelihoods, EPS, None)
    return -np.sum(np.log(mixture_likelihoods))


def _simplex_constraints(n):
    """Restrições para simplex (∑ α_i = 1, α_i ≥ 0)."""
    cons = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    return cons, bounds


def _optimize_on_simplex(objective, n, x0=None):
    """Minimiza função objetivo sobre o simplex."""
    if x0 is None:
        x0 = np.ones(n) / n
    cons, bounds = _simplex_constraints(n)
    res = minimize(objective, x0, method="SLSQP", constraints=cons, bounds=bounds)
    x = np.clip(getattr(res, "x", x0), 0.0, None)
    s = np.sum(x)
    return x / s if s > 0 else np.ones(n) / n