"""Base class and low-level helpers for post-hoc calibration."""

import numpy as np
from scipy.special import logsumexp


class Calibrator:
    r"""Base class for calibrators.

    A calibrator learns a post-hoc transformation of a model's outputs from a
    labelled held-out set with :meth:`fit`, then applies it to new outputs with
    :meth:`predict`. Subclasses implement both methods.

    Notes
    -----
    The argument order follows scikit-learn: :meth:`fit` takes the ground-truth
    labels first and the model output second -- ``fit(y_true, y_pred)``.

    Examples
    --------
    >>> from mlquantify.calibration import Calibrator
    >>> class IdentityCalibrator(Calibrator):
    ...     def fit(self, y_true, y_pred):
    ...         return self
    ...     def predict(self, y_pred):
    ...         return y_pred
    """

    def fit(self, y_true, y_pred):
        """Fit the calibration map from held-out labels and predictions."""
        raise NotImplementedError

    def predict(self, y_pred):
        """Apply the fitted calibration map to new predictions."""
        raise NotImplementedError

    def fit_predict(self, y_true, y_pred):
        """Convenience: ``fit(y_true, y_pred)`` then ``predict(y_pred)``."""
        return self.fit(y_true, y_pred).predict(y_pred)


def _to_logits(scores, eps=1e-12):
    r"""Map probability rows to centred logits (the inverse of softmax).

    ``softmax`` is invariant to a per-row additive constant, so its inverse is
    only defined up to that constant. We centre by the row mean of the logs,
    i.e. :math:`\log p_i - \overline{\log p}`.
    """
    logp = np.log(np.clip(np.asarray(scores, dtype=float), eps, None))
    return logp - logp.mean(axis=1, keepdims=True)


def _softmax(logits):
    """Row-wise softmax with the standard max-subtraction for stability."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits, onehot):
    """Mean negative log-likelihood of ``onehot`` under ``softmax(logits)``."""
    return float(np.mean(logsumexp(logits, axis=1) - np.sum(logits * onehot, axis=1)))
