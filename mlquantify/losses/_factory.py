# losses/_factory.py

from mlquantify.losses._distances import (
    DistanceLoss,
    EnergyLoss,
    LeastSquaresLoss,
    HellingerSurrogateLoss,
)
from mlquantify.losses._likelihood import (
    NegativeLogLikelihoodLoss,
    MixtureNegativeLogLikelihoodLoss,
    RegularizedMixtureNLLLoss,
)


def get_loss(loss="hellinger", normalize=True, **kwargs):
    """Instantiate a loss object from a string identifier or return a callable.

    Provides a unified entry point for retrieving optimization loss objects
    used in distribution-matching quantifiers.  If ``loss`` is already
    callable it is returned unchanged.

    Parameters
    ----------
    loss : str or callable, default='hellinger'
        Loss identifier.  Accepted string values:

        - ``'hellinger'``, ``'topsoe'``, ``'probsymm'``, ``'sqEuclidean'``,
          ``'euclidean'`` — :class:`~mlquantify.losses.DistanceLoss`.
        - ``'least_squares'``, ``'least-squares'``, ``'least squares'``,
          ``'ls'``, ``'l2'`` — :class:`~mlquantify.losses.LeastSquaresLoss`.
        - ``'hellinger_surrogate'``, ``'hd_surrogate'`` —
          :class:`~mlquantify.losses.HellingerSurrogateLoss`.
        - ``'nll'``, ``'negative_log_likelihood'`` —
          :class:`~mlquantify.losses.NegativeLogLikelihoodLoss`.
        - ``'energy'``, ``'energy_distance'`` —
          :class:`~mlquantify.losses.EnergyLoss`.
        - ``'mixture_nll'``, ``'ml'`` —
          :class:`~mlquantify.losses.MixtureNegativeLogLikelihoodLoss`.
        - ``'regularized_mixture_nll'``, ``'regularized_ml'`` —
          :class:`~mlquantify.losses.RegularizedMixtureNLLLoss`.

        A callable is returned as-is.
    normalize : bool, default=True
        Passed to distance and surrogate losses to control whether inputs
        are normalized to valid probability vectors before evaluation.
    **kwargs
        Additional keyword arguments forwarded to the chosen loss class
        constructor.

    Returns
    -------
    loss_fn : BaseLoss or callable
        The configured loss object.

    Raises
    ------
    ValueError
        If ``loss`` is a string not matching any recognised identifier.

    Examples
    --------
    >>> from mlquantify.losses import get_loss
    >>> loss = get_loss("hellinger")
    >>> loss([0.4, 0.6], [0.5, 0.5])   # doctest: +ELLIPSIS
    0.076...
    >>> loss = get_loss("least_squares")
    >>> loss([0.4, 0.6], [0.5, 0.5])
    0.02
    """
    if callable(loss):
        return loss

    if loss in {"hellinger", "topsoe", "probsymm", "sqEuclidean", "euclidean"}:
        return DistanceLoss(distance=loss, normalize=normalize)

    if loss in {"least_squares", "least-squares", "least squares", "ls", "l2"}:
        return LeastSquaresLoss()

    if loss in {"hellinger_surrogate", "hd_surrogate"}:
        return HellingerSurrogateLoss(normalize=normalize)

    if loss in {"nll", "negative_log_likelihood"}:
        return NegativeLogLikelihoodLoss(**kwargs)
    
    if loss in ("energy", "energy_distance"):
        return EnergyLoss()

    if loss in {"mixture_nll", "ml"}:
        return MixtureNegativeLogLikelihoodLoss(**kwargs)

    if loss in {"regularized_mixture_nll", "regularized_ml"}:
        return RegularizedMixtureNLLLoss(**kwargs)

    raise ValueError(f"Unknown loss: {loss!r}")
