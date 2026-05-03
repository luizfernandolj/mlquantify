import numpy as np

def topsoe_backend(p, q, xp):
    p = xp.asarray(p)
    q = xp.asarray(q)

    p = xp.maximum(p, 1e-20)
    q = xp.maximum(q, 1e-20)

    return xp.sum(
        p * xp.log(2 * p / (p + q))
        + q * xp.log(2 * q / (p + q))
    )

def topsoe(p: np.ndarray, q: np.ndarray) -> float:
    r"""
    Topsoe distance between two probability distributions.

    .. math::
        D_T(p, q) = \sum \left( p \log \frac{2p}{p + q} + q \log \frac{2q}{p + q} \right)

    Parameters
    ----------
    p : np.ndarray
        First probability distribution.
    q : np.ndarray
        Second probability distribution.

    Returns
    -------
    float
        The Topsoe distance.
    """
    import numpy as np
    return topsoe_backend(p, q, np)

def topsoe_jax(p, q):
    import jax.numpy as jnp
    return topsoe_backend(p, q, jnp)


def probsymm_backend(p, q, xp):
    p = xp.maximum(p, 1e-20)
    q = xp.maximum(q, 1e-20)
    return xp.sum((p - q) * xp.log(p / q))


def probsymm(p: np.ndarray, q: np.ndarray) -> float:
    r"""
    Probabilistic Symmetric distance.

    .. math::
        D_{PS}(p, q) = \sum (p - q) \log \frac{p}{q}

    Parameters
    ----------
    p : np.ndarray
        First probability distribution.
    q : np.ndarray
        Second probability distribution.

    Returns
    -------
    float
        The Probabilistic Symmetric distance.
    """
    import numpy as np
    return probsymm_backend(p, q, np)

def probsymm_jax(p, q):
    import jax.numpy as jnp
    return probsymm_backend(p, q, jnp)


def hellinger_backend(p, q, xp):
    p = xp.maximum(p, 1e-20)
    q = xp.maximum(q, 1e-20)
    return xp.sqrt(0.5 * xp.sum((xp.sqrt(p) - xp.sqrt(q)) ** 2))


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    r"""
    Hellinger distance between two probability distributions.

    .. math::
        H(p, q) = \frac{1}{\sqrt{2}} \sqrt{\sum \left( \sqrt{p} - \sqrt{q} \right)^2}

    Parameters
    ----------
    p : np.ndarray
        First probability distribution.
    q : np.ndarray
        Second probability distribution.

    Returns
    -------
    float
        The Hellinger distance.
    """
    import numpy as np
    return hellinger_backend(p, q, np)

def hellinger_jax(p, q):
    import jax.numpy as jnp
    return hellinger_backend(p, q, jnp)


def sqEuclidean_backend(p, q, xp):
    p = xp.asarray(p)
    q = xp.asarray(q)
    return xp.sum((p - q) ** 2)


def sqEuclidean(p: np.ndarray, q: np.ndarray) -> float:
    r"""
    Squared Euclidean distance between two vectors.

    Parameters
    ----------
    p : np.ndarray
        First vector.
    q : np.ndarray
        Second vector.

    Returns
    -------
    float
        The squared Euclidean distance.
    """
    import numpy as np
    return sqEuclidean_backend(p, q, np)

def sqEuclidean_jax(p, q):
    import jax.numpy as jnp
    return sqEuclidean_backend(p, q, jnp)