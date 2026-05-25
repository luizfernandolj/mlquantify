# losses/_distances.py

import numpy as np

from mlquantify.losses._base import BaseLoss
from mlquantify.metrics import (
    hellinger,
    topsoe,
    probsymm,
    sqEuclidean,
)


EPS = 1e-12


def normalize_distribution(x):
    """Normalize an array to a valid probability distribution.

    Clips all values to a small positive epsilon to avoid zero-probability
    entries, then divides by the total sum.  If the sum is effectively zero
    after clipping, a uniform distribution is returned instead.

    Parameters
    ----------
    x : array-like of shape (n,)
        Input array representing un-normalized prevalences or likelihoods.

    Returns
    -------
    x_normalized : ndarray of shape (n,)
        Non-negative array that sums to 1.

    Examples
    --------
    >>> from mlquantify.losses._distances import normalize_distribution
    >>> normalize_distribution([0.1, 0.3, 0.6])
    array([0.1, 0.3, 0.6])
    >>> normalize_distribution([2, 3, 5])
    array([0.2, 0.3, 0.5])
    """
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, EPS)

    total = x.sum()

    if total <= EPS:
        return np.ones_like(x) / len(x)

    return x / total


class DistanceLoss(BaseLoss):
    r"""Generic distance-based loss between two probability distributions.

    Computes a symmetric distance measure between a mixture distribution
    and a target distribution.  Both arrays are optionally normalized to
    valid probability vectors before the distance is evaluated.

    Parameters
    ----------
    distance : {'hellinger', 'topsoe', 'probsymm', 'sqEuclidean', 'euclidean'}, \
               default='hellinger'
        The distance measure to apply.
    normalize : bool, default=True
        If ``True``, both ``mixture`` and ``target`` are passed through
        :func:`normalize_distribution` before comparison.

    Attributes
    ----------
    distance : str
        Name of the chosen distance measure.
    normalize : bool
        Whether inputs are normalized before computing the distance.

    Examples
    --------
    >>> from mlquantify.losses import get_loss
    >>> loss = get_loss("hellinger")
    >>> loss([0.4, 0.6], [0.5, 0.5])
    0.07612...
    """

    def __init__(self, distance="hellinger", normalize=True):
        self.distance = distance
        self.normalize = normalize

    def __call__(self, mixture, target):
        """Compute the distance between two distributions.

        Parameters
        ----------
        mixture : array-like of shape (n_classes,)
            Estimated prevalence or mixture vector.
        target : array-like of shape (n_classes,)
            Reference (true) prevalence or target vector.

        Returns
        -------
        loss : float
            Scalar distance value.

        Raises
        ------
        ValueError
            If ``mixture`` and ``target`` have different shapes, or if
            ``self.distance`` is not a recognised distance name.

        Examples
        --------
        >>> from mlquantify.losses import get_loss
        >>> loss = get_loss("topsoe")
        >>> round(loss([0.3, 0.7], [0.5, 0.5]), 4)
        0.0528
        """
        mixture = np.asarray(mixture, dtype=float)
        target = np.asarray(target, dtype=float)

        if self.normalize:
            mixture = normalize_distribution(mixture)
            target = normalize_distribution(target)

        if mixture.shape != target.shape:
            raise ValueError(
                f"mixture and target must have the same shape. "
                f"Got {mixture.shape} and {target.shape}."
            )

        if self.distance == "hellinger":
            return float(hellinger(mixture, target))

        if self.distance == "topsoe":
            return float(topsoe(mixture, target))

        if self.distance == "probsymm":
            return float(probsymm(mixture, target))

        if self.distance == "sqEuclidean":
            return float(sqEuclidean(mixture, target))

        if self.distance == "euclidean":
            return float(np.sqrt(sqEuclidean(mixture, target)))

        raise ValueError(f"Unknown distance: {self.distance!r}")


class LeastSquaresLoss(BaseLoss):
    """Squared Euclidean (least-squares) loss.

    Computes :math:`\\|target - M \\cdot mixture\\|_2^2`.  When no mixing
    matrix ``M`` is provided the loss reduces to
    :math:`\\|target - mixture\\|_2^2`.

    Examples
    --------
    >>> from mlquantify.losses import get_loss
    >>> loss = get_loss("least_squares")
    >>> loss([0.4, 0.6], [0.5, 0.5])
    0.02
    """

    def __call__(self, mixture, target, M=None):
        """Compute the squared Euclidean loss.

        Parameters
        ----------
        mixture : array-like of shape (n_classes,)
            Estimated prevalence vector.
        target : array-like of shape (n_components,)
            Target representation vector.
        M : array-like of shape (n_components, n_classes) or None, \
            default=None
            Optional mixing matrix.  When supplied, ``mixture`` is
            transformed as ``M @ mixture`` before computing the
            squared difference.

        Returns
        -------
        loss : float
            Scalar squared-norm value.

        Examples
        --------
        >>> from mlquantify.losses import get_loss
        >>> loss = get_loss("ls")
        >>> loss([0.3, 0.7], [0.5, 0.5])
        0.08
        """
        mixture = np.asarray(mixture, dtype=float)
        target = np.asarray(target, dtype=float)

        if M is not None:
            mixture = np.asarray(M, dtype=float) @ mixture

        diff = target - mixture
        return float(diff @ diff)


class HellingerSurrogateLoss(BaseLoss):
    r"""Optimization surrogate for the squared Hellinger distance.

    Minimizing :math:`-\sum_i \sqrt{p_i \cdot q_i}` is equivalent to
    minimizing the squared Hellinger distance
    :math:`H^2(p,q) = 1 - \sum_i \sqrt{p_i q_i}` and is numerically
    better behaved for gradient-free solvers.

    Parameters
    ----------
    normalize : bool, default=True
        If ``True``, both mixture and target are normalized to valid
        probability vectors before the surrogate is computed.

    Attributes
    ----------
    normalize : bool
        Whether inputs are normalized before computing the surrogate.

    Examples
    --------
    >>> from mlquantify.losses import get_loss
    >>> loss = get_loss("hellinger_surrogate")
    >>> loss([0.4, 0.6], [0.5, 0.5])
    -0.9899...
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, mixture, target, M=None):
        """Compute the Hellinger surrogate loss.

        Parameters
        ----------
        mixture : array-like of shape (n_classes,)
            Estimated prevalence vector.
        target : array-like of shape (n_components,)
            Target representation vector.
        M : array-like of shape (n_components, n_classes) or None, \
            default=None
            Optional mixing matrix applied as ``M @ mixture`` before
            computing the surrogate.

        Returns
        -------
        loss : float
            Negative sum of geometric means; lower (more negative) is better.

        Examples
        --------
        >>> from mlquantify.losses import get_loss
        >>> loss = get_loss("hd_surrogate")
        >>> round(loss([0.3, 0.7], [0.5, 0.5]), 4)
        -0.9747
        """
        mixture = np.asarray(mixture, dtype=float)
        target = np.asarray(target, dtype=float)

        if M is not None:
            mixture = np.asarray(M, dtype=float) @ mixture

        mixture = np.maximum(mixture, EPS)
        target = np.maximum(target, 0.0)

        if self.normalize:
            mixture = normalize_distribution(mixture)
            target = normalize_distribution(target)

        mask = target > 0

        return float(-np.sqrt(target[mask] * mixture[mask]).sum())


class EnergyLoss(BaseLoss):
    r"""Energy-distance loss for distribution matching.

    Computes the quadratic form

    .. math::

        \mathcal{L}(\hat{p}) = \hat{p}^\top (2\,q - M\,\hat{p})

    where :math:`\hat{p}` is the estimated prevalence vector, :math:`q`
    is a pre-computed cross-term vector, and :math:`M` is the pairwise
    energy-distance matrix between training classes.  Minimising this
    objective is equivalent to minimising the energy distance between the
    mixture distribution and the test distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from mlquantify.losses import get_loss
    >>> loss = get_loss("energy")
    >>> p = np.array([0.4, 0.6])
    >>> q = np.array([0.3, 0.7])
    >>> M = np.eye(2)
    >>> loss(p, q, M)
    0.04
    """

    def __call__(self, prevalence, q, M):
        """Compute the energy-distance loss.

        Parameters
        ----------
        prevalence : array-like of shape (n_classes,)
            Estimated class prevalence vector.
        q : array-like of shape (n_classes,)
            Cross-term vector equal to the mean energy distance between
            each training class and the test sample.
        M : array-like of shape (n_classes, n_classes)
            Pairwise energy-distance matrix between training classes.

        Returns
        -------
        loss : float
            Scalar energy-distance loss value.

        Examples
        --------
        >>> import numpy as np
        >>> from mlquantify.losses import get_loss
        >>> loss = get_loss("energy_distance")
        >>> p = np.array([0.5, 0.5])
        >>> q = np.array([0.4, 0.6])
        >>> M = np.array([[0.0, 1.0], [1.0, 0.0]])
        >>> loss(p, q, M)
        0.5
        """
        prevalence = np.asarray(prevalence, dtype=float)
        q = np.asarray(q, dtype=float)
        M = np.asarray(M, dtype=float)

        return float(prevalence @ (2.0 * q - M @ prevalence))
