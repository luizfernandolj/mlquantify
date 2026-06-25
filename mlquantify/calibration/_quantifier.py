"""Post-hoc calibration of quantifier outputs (not yet implemented)."""

from ._base import Calibrator

_NOT_IMPLEMENTED = (
    "QuantifierCalibrator is not implemented yet. To calibrate the classifier "
    "posteriors a quantifier consumes, use ClassifierCalibrator (temperature / "
    "vector scaling)."
)


class QuantifierCalibrator(Calibrator):
    """Post-hoc calibration of quantifier prevalence estimates.

    .. note::
       This class is a placeholder: its semantics are not yet defined and both
       methods raise :class:`NotImplementedError`. Use
       :class:`ClassifierCalibrator` to calibrate the posteriors a quantifier
       consumes.
    """

    def fit(self, y_true, y_pred):
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict(self, y_pred):
        raise NotImplementedError(_NOT_IMPLEMENTED)
