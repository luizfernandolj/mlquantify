import numpy as np
import scipy.optimize
from scipy.special import logsumexp


# ============================================================
# Utilities
# ============================================================

def _softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _nll_and_probs(logits, y):
    """
    Compute negative log-likelihood and softmax probabilities.

    Parameters
    ----------
    logits : ndarray (N, K)
    y : ndarray (N,) integer class labels

    Returns
    -------
    nll : float
    probs : ndarray (N, K)
    """
    log_sum_exp = logsumexp(logits, axis=1)
    true_logits = logits[np.arange(len(y)), y]
    nll = -np.mean(true_logits - log_sum_exp)

    probs = np.exp(logits - log_sum_exp[:, None])
    return nll, probs


# ============================================================
# 1) Temperature Scaling (TS)
# ============================================================

def temperature_scaling(logits, y):
    r"""Temperature Scaling (TS).

    Learns scalar temperature T > 0:

        p̂ = softmax(z / T)
    """

    N = len(y)

    def objective(x):
        T = x[0]
        scaled = logits / T

        nll, probs = _nll_and_probs(scaled, y)

        probs[np.arange(N), y] -= 1.0
        grad_T = -np.sum(logits * probs) / (N * T**2)

        return nll, np.array([grad_T])

    result = scipy.optimize.minimize(
        objective,
        x0=np.array([1.0]),
        bounds=[(1e-6, None)],
        method="L-BFGS-B",
        jac=True
    )

    T = result.x[0]
    return _softmax(logits / T)


# ============================================================
# 2) Bias-Corrected Temperature Scaling (BCTS)
# ============================================================

def bias_corrected_temperature_scaling(logits, y, beta=0.0):
    r"""Bias-Corrected Temperature Scaling.

        p̂ = softmax(z / T + b)
    """

    N, K = logits.shape

    def objective(x):
        T = x[0]
        b = x[1:]

        scaled = logits / T + b

        nll, probs = _nll_and_probs(scaled, y)
        nll += beta * np.sum(b**2)

        probs[np.arange(N), y] -= 1.0

        grad_T = -np.sum(logits * probs) / (N * T**2)
        grad_b = -np.mean(probs, axis=0) + 2 * beta * b

        return nll, np.concatenate([[grad_T], grad_b])

    result = scipy.optimize.minimize(
        objective,
        x0=np.concatenate([[1.0], np.zeros(K)]),
        bounds=[(1e-6, None)] + [(None, None)] * K,
        method="L-BFGS-B",
        jac=True
    )

    T = result.x[0]
    b = result.x[1:]

    return _softmax(logits / T + b)


# ============================================================
# 3) Vector Scaling (VS)
# ============================================================

def vector_scaling(logits, y):
    r"""Vector Scaling.

        p̂ = softmax(W ⊙ z + b)
    """

    N, K = logits.shape

    def objective(x):
        W = x[:K]
        b = x[K:]

        scaled = logits * W + b

        nll, probs = _nll_and_probs(scaled, y)

        probs[np.arange(N), y] -= 1.0

        grad_W = -np.mean(logits * probs, axis=0)
        grad_b = -np.mean(probs, axis=0)

        return nll, np.concatenate([grad_W, grad_b])

    result = scipy.optimize.minimize(
        objective,
        x0=np.concatenate([np.ones(K), np.zeros(K)]),
        bounds=[(None, None)] * (2 * K),
        method="L-BFGS-B",
        jac=True
    )

    W = result.x[:K]
    b = result.x[K:]

    return _softmax(logits * W + b)


# ============================================================
# 4) No-Bias Vector Scaling (NBVS)
# ============================================================

def no_bias_vector_scaling(logits, y):
    r"""No-Bias Vector Scaling.

        p̂ = softmax(W ⊙ z)
    """

    N, K = logits.shape

    def objective(W):
        scaled = logits * W

        nll, probs = _nll_and_probs(scaled, y)

        probs[np.arange(N), y] -= 1.0

        grad_W = -np.mean(logits * probs, axis=0)

        return nll, grad_W

    result = scipy.optimize.minimize(
        objective,
        x0=np.ones(K),
        bounds=[(None, None)] * K,
        method="L-BFGS-B",
        jac=True
    )

    W = result.x
    return _softmax(logits * W)