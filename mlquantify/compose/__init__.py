from ._base import BaseComposeQuantifier
from ._linear import LinearComposeQuantifier
from ._likelihood import LikelihoodComposeQuantifier
from mlquantify.losses import (
    BaseLoss,
    DistanceLoss,
    EnergyLoss,
    HellingerSurrogateLoss,
    LeastSquaresLoss,
    MixtureNegativeLogLikelihoodLoss,
    NegativeLogLikelihoodLoss,
    RegularizedMixtureNLLLoss,
    get_loss,
)

ComposeQuantifier = LinearComposeQuantifier

__all__ = [
    "BaseComposeQuantifier",
    "LinearComposeQuantifier",
    "LikelihoodComposeQuantifier",
    "ComposeQuantifier",
    "BaseLoss",
    "DistanceLoss",
    "EnergyLoss",
    "HellingerSurrogateLoss",
    "LeastSquaresLoss",
    "MixtureNegativeLogLikelihoodLoss",
    "NegativeLogLikelihoodLoss",
    "RegularizedMixtureNLLLoss",
    "get_loss",
]
