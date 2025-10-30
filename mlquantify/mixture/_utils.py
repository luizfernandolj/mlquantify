import numpy as np
from math import pi
from sklearn.metrics import pairwise_distances

EPS = 1e-12


def gaussian_kernel(X, Y, bandwidth):
    """Compute Gaussian kernel matrix between sets X and Y."""
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    sqd = pairwise_distances(X, Y, metric="euclidean") ** 2
    D = X.shape[1]
    norm = (bandwidth ** D) * ((2 * pi) ** (D / 2))
    return np.exp(-sqd / (2 * (bandwidth ** 2))) / (norm + EPS)


def hellinger_distance(p, q):
    """Compute Hellinger distance between two distributions."""
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return np.sqrt(1 - np.sum(np.sqrt(p * q)))


def cauchy_schwarz_divergence(p, q):
    """Compute Cauchy–Schwarz divergence between two distributions."""
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    num = np.sum(p * q)
    denom = np.sqrt(np.sum(p ** 2) * np.sum(q ** 2))
    return -np.log(num / (denom + EPS) + EPS)


def negative_log_likelihood(mixture_likelihood):
    """Return -sum(log(p(x))) with numerical stability."""
    mixture_likelihood = np.clip(mixture_likelihood, EPS, None)
    return -np.sum(np.log(mixture_likelihood))
