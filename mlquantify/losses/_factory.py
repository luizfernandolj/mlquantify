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
