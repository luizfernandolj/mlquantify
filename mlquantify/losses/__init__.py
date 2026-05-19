# losses/__init__.py

from mlquantify.losses._base import BaseLoss

from mlquantify.losses._distances import (
    DistanceLoss,
    LeastSquaresLoss,
    HellingerSurrogateLoss,
    EnergyLoss,
    normalize_distribution,
)

from mlquantify.losses._likelihood import (
    NegativeLogLikelihoodLoss,
    MixtureNegativeLogLikelihoodLoss,
    RegularizedMixtureNLLLoss,
)

from mlquantify.losses._factory import get_loss

__all__ = [
    "BaseLoss",
    "DistanceLoss",
    "LeastSquaresLoss",
    "HellingerSurrogateLoss",
    "EnergyLoss",
    "NegativeLogLikelihoodLoss",
    "MixtureNegativeLogLikelihoodLoss",
    "RegularizedMixtureNLLLoss",
    "normalize_distribution",
    "get_loss",
]