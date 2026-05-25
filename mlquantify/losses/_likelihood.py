# losses/_likelihood.py

import numpy as np

from mlquantify.losses._base import BaseLoss


EPS = 1e-12


def _reduce_negative_log_likelihood(likelihood, reduction):
    likelihood = np.asarray(likelihood, dtype=float)
    likelihood = np.maximum(likelihood, EPS)
    values = -np.log(likelihood)

    if reduction == "mean":
        return float(values.mean())

    if reduction == "sum":
        return float(values.sum())

    raise ValueError("reduction must be 'mean' or 'sum'.")


def _mixture_likelihood(prevalences, class_likelihoods):
    prevalences = np.asarray(prevalences, dtype=float)
    class_likelihoods = np.asarray(class_likelihoods, dtype=float)

    if class_likelihoods.ndim != 2:
        raise ValueError("class_likelihoods must be a 2D array.")

    if class_likelihoods.shape[0] == prevalences.shape[0]:
        return prevalences @ class_likelihoods

    if class_likelihoods.shape[1] == prevalences.shape[0]:
        return class_likelihoods @ prevalences

    raise ValueError(
        "class_likelihoods must have one dimension matching the number "
        "of prevalences."
    )


class NegativeLogLikelihoodLoss(BaseLoss):
    """Negative log-likelihood loss for mixture likelihoods.

    Computes :math:`-\\log p(x)` element-wise and then reduces the resulting
    values by mean or sum.

    Parameters
    ----------
    reduction : {'mean', 'sum'}, default='mean'
        How to reduce the per-sample log-likelihood values.

    Attributes
    ----------
    reduction : str
        The configured reduction mode.

    Examples
    --------
    >>> from mlquantify.losses import get_loss
    >>> loss = get_loss("nll")
    >>> import numpy as np
    >>> loss(np.array([0.8, 0.6, 0.9]))  # doctest: +ELLIPSIS
    0.2576...
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, likelihood):
        """Compute the negative log-likelihood.

        Parameters
        ----------
        likelihood : array-like of shape (n_samples,)
            Per-sample likelihood values in the range ``(0, 1]``.

        Returns
        -------
        loss : float
            Reduced negative log-likelihood.

        Examples
        --------
        >>> from mlquantify.losses import get_loss
        >>> import numpy as np
        >>> loss = get_loss("nll")
        >>> round(loss(np.array([0.5, 0.5])), 4)
        0.6931
        """
        return _reduce_negative_log_likelihood(likelihood, self.reduction)


class MixtureNegativeLogLikelihoodLoss(BaseLoss):
    r"""Negative log-likelihood for class likelihood mixtures.

    Computes the mixture likelihood

    .. math::

        p(x_i) = \sum_c \hat{p}_c \cdot p(x_i \mid c)

    and then applies the negative log-likelihood reduction.  This is the
    standard loss used by the Expectation–Maximisation Quantifier (EMQ).

    Parameters
    ----------
    reduction : {'mean', 'sum'}, default='mean'
        How to reduce the per-sample log-likelihood values.

    Attributes
    ----------
    reduction : str
        The configured reduction mode.

    Examples
    --------
    >>> from mlquantify.losses import get_loss
    >>> import numpy as np
    >>> loss = get_loss("mixture_nll")
    >>> prev = np.array([0.4, 0.6])
    >>> lkl = np.array([[0.3, 0.7], [0.6, 0.4]])
    >>> loss(prev, lkl)  # doctest: +ELLIPSIS
    0.5...
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, prevalences, class_likelihoods):
        """Compute mixture negative log-likelihood.

        Parameters
        ----------
        prevalences : array-like of shape (n_classes,)
            Estimated class prevalence vector.
        class_likelihoods : array-like of shape (n_classes, n_samples) or \
            (n_samples, n_classes)
            Per-class likelihood for each test instance.  The axis whose
            length matches ``n_classes`` is contracted with ``prevalences``.

        Returns
        -------
        loss : float
            Reduced mixture negative log-likelihood.

        Examples
        --------
        >>> from mlquantify.losses import get_loss
        >>> import numpy as np
        >>> loss = get_loss("ml")
        >>> prev = np.array([0.5, 0.5])
        >>> lkl = np.array([[0.2, 0.8], [0.7, 0.3]])
        >>> round(loss(prev, lkl), 4)
        0.6931
        """
        mixture = _mixture_likelihood(prevalences, class_likelihoods)
        return _reduce_negative_log_likelihood(mixture, self.reduction)


class RegularizedMixtureNLLLoss(BaseLoss):
    r"""Mixture NLL with optional ordinal-smoothness regularization.

    Extends :class:`MixtureNegativeLogLikelihoodLoss` with first- and
    second-order penalty terms that encourage a smooth prevalence vector
    when the classes have an ordinal interpretation.

    The total loss is

    .. math::

        \mathcal{L} = \text{NLL}(\hat{p}) +
            \frac{\tau_0}{2} \sum_{c} (\hat{p}_{c+1} - \hat{p}_c)^2 +
            \frac{\tau_1}{2} \sum_{c} (-\hat{p}_{c} + 2\hat{p}_{c+1} - \hat{p}_{c+2})^2

    Parameters
    ----------
    tau_0 : float, default=0.0
        Weight of the first-order smoothness penalty.  Set to ``0`` to
        disable.
    tau_1 : float, default=0.0
        Weight of the second-order smoothness penalty.  Set to ``0`` to
        disable.
    reduction : {'mean', 'sum'}, default='mean'
        How to reduce the per-sample log-likelihood values.

    Attributes
    ----------
    tau_0 : float
        First-order regularization weight.
    tau_1 : float
        Second-order regularization weight.
    reduction : str
        The configured reduction mode.

    Examples
    --------
    >>> from mlquantify.losses import get_loss
    >>> import numpy as np
    >>> loss = get_loss("regularized_mixture_nll", tau_0=0.1)
    >>> prev = np.array([0.3, 0.4, 0.3])
    >>> lkl = np.array([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    >>> loss(prev, lkl)  # doctest: +ELLIPSIS
    1.1...
    """

    def __init__(self, tau_0=0.0, tau_1=0.0, reduction="mean"):
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.reduction = reduction

    def __call__(self, prevalences, class_likelihoods):
        """Compute the regularized mixture negative log-likelihood.

        Parameters
        ----------
        prevalences : array-like of shape (n_classes,)
            Estimated class prevalence vector.
        class_likelihoods : array-like of shape (n_classes, n_samples) or \
            (n_samples, n_classes)
            Per-class likelihood for each test instance.

        Returns
        -------
        loss : float
            Mixture NLL plus any active smoothness penalties.

        Examples
        --------
        >>> from mlquantify.losses import get_loss
        >>> import numpy as np
        >>> loss = get_loss("regularized_ml", tau_0=0.0)
        >>> prev = np.array([0.5, 0.5])
        >>> lkl = np.array([[0.2, 0.8], [0.7, 0.3]])
        >>> round(loss(prev, lkl), 4)
        0.6931
        """
        prevalences = np.asarray(prevalences, dtype=float)

        mixture = _mixture_likelihood(prevalences, class_likelihoods)
        loss = _reduce_negative_log_likelihood(mixture, self.reduction)

        if self.tau_0 > 0:
            loss += self.tau_0 * self._first_order_penalty(prevalences)

        if self.tau_1 > 0:
            loss += self.tau_1 * self._second_order_penalty(prevalences)

        return float(loss)

    @staticmethod
    def _first_order_penalty(prevalences):
        return np.sum((prevalences[1:] - prevalences[:-1]) ** 2) / 2.0

    @staticmethod
    def _second_order_penalty(prevalences):
        if len(prevalences) < 3:
            return 0.0

        return np.sum(
            (-prevalences[:-2] + 2.0 * prevalences[1:-1] - prevalences[2:]) ** 2
        ) / 2.0
