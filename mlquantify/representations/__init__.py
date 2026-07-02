from ._base import BaseRepresentation
from ._histogram import HistogramRepresentation
from ._density import KDERepresentation
from ._distance import DistanceRepresentation
from ._kernel import KernelMeanRepresentation
from ._prediction import (
    HardPredictionRepresentation,
    PredictionRepresentation,
    SoftPredictionRepresentation,
)

# New: torch-based representations (optional, requires PyTorch)
try:
    import torch as _torch
    from ._base import TorchRepresentation
    from ._torch_histogram import DifferentiableHistogramRepresentation
    from ._torch_gaussian import GaussianRepresentation

    _torch_repr_all = [
        "TorchRepresentation",
        "DifferentiableHistogramRepresentation",
        "GaussianRepresentation",
    ]
except ImportError:  # pragma: no cover - runtime-dependent
    _torch_repr_all = []

__all__ = [
    "BaseRepresentation",
    "HistogramRepresentation",
    "KDERepresentation",
    "DistanceRepresentation",
    "KernelMeanRepresentation",
    "PredictionRepresentation",
    "HardPredictionRepresentation",
    "SoftPredictionRepresentation",
] + _torch_repr_all
