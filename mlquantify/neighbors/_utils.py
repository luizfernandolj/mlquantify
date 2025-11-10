import numpy as np
from sklearn.metrics import pairwise_distances
from math import pi
from scipy.optimize import minimize


EPS = 1e-12

# ============================================================
# Utilitários
# ============================================================

def gaussian_kernel(X, Y, bandwidth):
    """
    Compute the Gaussian kernel matrix K(x, y) with specified bandwidth.

    This kernel matrix represents the similarity between each pair of points in X and Y,
    computed using the Gaussian (RBF) kernel function:

    \[
    K(x, y) = \frac{1}{(2 \pi)^{D/2} h^D} \exp\left(- \frac{\|x - y\|^2}{2 h^2}\right)
    \]

    where \( h \) is the bandwidth (smoothing parameter), and \( D \) is the dimensionality
    of the input feature space.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        Input data points.
    Y : array-like of shape (n_samples_Y, n_features) or None
        Input data points for kernel computation. If None, defaults to X.
    bandwidth : float
        Kernel bandwidth parameter \( h \).

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        Gaussian kernel matrix.
    """
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
    """
    Compute the negative log-likelihood of given mixture likelihoods in a numerically stable way.

    Given mixture likelihood values \( p_i \) for samples, the negative log-likelihood is:

    \[
    - \sum_i \log(p_i)
    \]

    Numerical stability is achieved by clipping likelihoods below a small epsilon.

    Parameters
    ----------
    mixture_likelihoods : array-like
        Likelihood values for the mixture distribution evaluated at samples.

    Returns
    -------
    nll : float
        Negative log-likelihood value.
    """
    mixture_likelihoods = np.clip(mixture_likelihoods, EPS, None)
    return -np.sum(np.log(mixture_likelihoods))


def _simplex_constraints(n):
    """
    Define constraints and bounds for optimization over the probability simplex.

    The simplex is defined as all vectors \( \alpha \in \mathbb{R}^n \) such that:

    \[
    \alpha_i \geq 0, \quad \sum_{i=1}^n \alpha_i = 1
    \]

    Parameters
    ----------
    n : int
        Dimensionality of the simplex (number of mixture components).

    Returns
    -------
    constraints : list of dict
        List containing equality constraint for sum of elements equaling 1.
    bounds : list of tuple
        Bounds for each element to lie between 0 and 1.
    """
    cons = [{"type": "eq", "fun": lambda a: np.sum(a) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    return cons, bounds


def _optimize_on_simplex(objective, n, x0=None):
    """
    Minimize an objective function over the probability simplex.

    This function solves for mixture weights \( \boldsymbol{\alpha} \) that minimize the 
    objective function under the constraints \(\alpha_i \geq 0\) and \(\sum_i \alpha_i = 1\).

    The optimization uses Sequential Least SQuares Programming (SLSQP).

    Parameters
    ----------
    objective : callable
        The objective function to minimize. It should accept a vector of length n and
        return a scalar loss.
    n : int
        Number of mixture components (dimension of \( \boldsymbol{\alpha} \)).
    x0 : array-like, optional
        Initial guess for \( \boldsymbol{\alpha} \). If None, defaults to uniform.

    Returns
    -------
    alpha_opt : ndarray of shape (n,)
        Optimized mixture weights summing to one.
    """
    if x0 is None:
        x0 = np.ones(n) / n
    cons, bounds = _simplex_constraints(n)
    res = minimize(objective, x0, method="SLSQP", constraints=cons, bounds=bounds)
    x = np.clip(getattr(res, "x", x0), 0.0, None)
    s = np.sum(x)
    return x / s if s > 0 else np.ones(n) / n