"""Post-hoc calibration of classifier posteriors and quantifier outputs.

The scaling family of classifier calibrators (Temperature / Vector Scaling and
their bias-corrected variants) is provided by :class:`ClassifierCalibrator`.
"""

from ._base import Calibrator
from ._classifier import ClassifierCalibrator
from ._quantifier import QuantifierCalibrator

__all__ = [
    "Calibrator",
    "ClassifierCalibrator",
    "QuantifierCalibrator",
]
