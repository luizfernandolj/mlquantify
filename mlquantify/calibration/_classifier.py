"""Posterior calibration for classifiers: the scaling family.

Four post-hoc methods that rescale a classifier's logits to minimise the
negative log-likelihood (NLL) of a held-out set:

* **Temperature Scaling (TS)** -- one shared temperature ``T`` [1]_.
* **Bias-Corrected Temperature Scaling (BCTS)** -- ``T`` plus per-class
  biases [2]_.
* **Vector Scaling (VS)** -- per-class weights and biases [1]_.
* **No-Bias Vector Scaling (NBVS)** -- per-class weights only [2]_.

Each fits its parameters by minimising the NLL with L-BFGS-B. This is an
independent implementation written from the cited papers.

References
----------
.. [1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
   *On Calibration of Modern Neural Networks.* ICML.
.. [2] Alexandari, A., Kundaje, A., & Shrikumar, A. (2020).
   *Maximum Likelihood with Bias-Corrected Calibration is Hard-to-Beat.* ICML.
"""

import numpy as np
from scipy.optimize import minimize

from ._base import Calibrator, _to_logits, _softmax, _nll


def _fit_scaling(logits, onehot, per_class_weights, fit_biases):
    """Optimise scaling parameters by minimising the NLL with L-BFGS-B.

    Returns ``(weights, biases)`` of shape ``(n_classes,)`` such that the
    calibrated logits are ``logits * weights + biases``. For temperature
    scaling the weight is the shared ``1 / T`` broadcast across classes.
    """
    n, K = logits.shape

    def scaled(params):
        if per_class_weights:
            w = params[:K]
            b = params[K:] if fit_biases else np.zeros(K)
        else:  # one shared multiplicative factor (= 1 / temperature)
            w = np.full(K, params[0])
            b = params[1:] if fit_biases else np.zeros(K)
        return logits * w[None, :] + b[None, :]

    def objective(params):
        return _nll(scaled(params), onehot)

    if per_class_weights:
        x0 = np.concatenate([np.ones(K), np.zeros(K)]) if fit_biases else np.ones(K)
        bounds = [(0.0, None)] * K + ([(None, None)] * K if fit_biases else [])
    else:
        x0 = np.concatenate([[1.0], np.zeros(K)]) if fit_biases else np.array([1.0])
        bounds = [(1e-6, None)] + ([(None, None)] * K if fit_biases else [])

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, tol=1e-7)

    if per_class_weights:
        weights = res.x[:K]
        biases = res.x[K:] if fit_biases else np.zeros(K)
    else:
        weights = np.full(K, res.x[0])
        biases = res.x[1:] if fit_biases else np.zeros(K)
    return weights, biases


class ClassifierCalibrator(Calibrator):
    r"""Post-hoc calibration of classifier posteriors by logit scaling.

    Parameters
    ----------
    method : {'ts', 'bcts', 'vs', 'nbvs'}, default='bcts'
        Calibration map to fit:

        - ``'ts'``   -- Temperature Scaling (one temperature).
        - ``'bcts'`` -- Bias-Corrected Temperature Scaling (temperature + biases).
        - ``'vs'``   -- Vector Scaling (per-class weights + biases).
        - ``'nbvs'`` -- No-Bias Vector Scaling (per-class weights).
    input_type : {'proba', 'logits'}, default='proba'
        Whether ``y_pred`` holds probabilities (mapped to centred logits before
        scaling) or raw logits.

    Attributes
    ----------
    weights_ : ndarray of shape (n_classes,)
        Fitted multiplicative factors applied to the logits.
    biases_ : ndarray of shape (n_classes,)
        Fitted additive biases (zeros for ``'ts'`` and ``'nbvs'``).
    classes_ : ndarray
        Distinct labels seen in ``fit`` (only set for 1-D ``y_true``).
    n_features_in_ : int
        Number of classes (logit columns) seen in ``fit``.

    Notes
    -----
    Calibration must be fit on predictions held out from classifier training
    (e.g. a validation split or cross-validated predictions); fitting it on the
    classifier's own training predictions under-estimates the miscalibration.

    Examples
    --------
    >>> import numpy as np
    >>> from mlquantify.calibration import ClassifierCalibrator
    >>> proba = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
    >>> y = np.array([0, 1, 0])
    >>> cal = ClassifierCalibrator(method="ts").fit(y, proba)
    >>> calibrated = cal.predict(proba)
    >>> np.allclose(calibrated.sum(axis=1), 1.0)
    True
    """

    def __init__(self, method="bcts", input_type="proba"):
        self.method = method
        self.input_type = input_type

    # method -> (per_class_weights, fit_biases)
    _CONFIG = {
        "ts": (False, False),
        "bcts": (False, True),
        "vs": (True, True),
        "nbvs": (True, False),
    }

    def _as_logits(self, y_pred):
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim != 2:
            raise ValueError("y_pred must be 2-D of shape (n_samples, n_classes).")
        if self.input_type == "proba":
            return _to_logits(y_pred)
        if self.input_type == "logits":
            return y_pred
        raise ValueError(
            f"input_type must be 'proba' or 'logits', got {self.input_type!r}."
        )

    def _as_onehot(self, y_true, n_classes):
        y_true = np.asarray(y_true)
        if y_true.ndim == 2:
            return y_true.astype(float)
        classes = np.unique(y_true)
        self.classes_ = classes
        if len(classes) > n_classes:
            raise ValueError(
                f"y_true has {len(classes)} classes but y_pred has {n_classes} columns."
            )
        idx = np.searchsorted(classes, y_true)
        onehot = np.zeros((len(y_true), n_classes))
        onehot[np.arange(len(y_true)), idx] = 1.0
        return onehot

    def fit(self, y_true, y_pred):
        """Fit the scaling map on held-out labels ``y_true`` and outputs ``y_pred``."""
        if self.method not in self._CONFIG:
            raise ValueError(
                f"Unknown method {self.method!r}; choose from {sorted(self._CONFIG)}."
            )
        logits = self._as_logits(y_pred)
        K = logits.shape[1]
        onehot = self._as_onehot(y_true, K)
        per_class_weights, fit_biases = self._CONFIG[self.method]
        self.weights_, self.biases_ = _fit_scaling(
            logits, onehot, per_class_weights, fit_biases
        )
        self.n_features_in_ = K
        return self

    def predict(self, y_pred):
        """Return calibrated probabilities for ``y_pred``."""
        logits = self._as_logits(y_pred)
        scaled = logits * self.weights_[None, :] + self.biases_[None, :]
        return _softmax(scaled)
