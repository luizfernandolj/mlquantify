from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import AggregationMixin
import numpy as np
from mlquantify.base_aggregative import SoftLearnerQMixin
from mlquantify.metrics._slq import MAE
from mlquantify.utils import _fit_context
from mlquantify.utils._constraints import (
    Interval,
    CallableConstraint,
    Options
)

class EMQ(SoftLearnerQMixin, AggregationMixin, BaseQuantifier):
    r"""Expectation-Maximization Quantifier (EMQ).

    Estimates class prevalences under prior probability shift by alternating 
    between expectation **(E)** and maximization **(M)** steps on posterior probabilities. 

    E-step:
    .. math::
        p_i^{(s+1)}(x) = \frac{q_i^{(s)} p_i(x)}{\sum_j q_j^{(s)} p_j(x)}

    M-step:
    .. math::
        q_i^{(s+1)} = \frac{1}{N} \sum_{n=1}^N p_i^{(s+1)}(x_n)

    where 
    - :math:`p_i(x)` are posterior probabilities predicted by the classifier
    - :math:`q_i^{(s)}` are class prevalence estimates at iteration :math:`s`
    - :math:`N` is the number of test instances.

    Calibrations supported on posterior probabilities before **EM** iteration:

    Temperature Scaling (TS):
    .. math::
        \hat{p} = \text{softmax}\left(\frac{\log(p)}{T}\right)

    Bias-Corrected Temperature Scaling (BCTS):
    .. math::
        \hat{p} = \text{softmax}\left(\frac{\log(p)}{T} + b\right)

    Vector Scaling (VS):
    .. math::
        \hat{p}_i = \text{softmax}(W_i \cdot \log(p_i) + b_i)

    No-Bias Vector Scaling (NBVS):
    .. math::
        \hat{p}_i = \text{softmax}(W_i \cdot \log(p_i))

    Parameters
    ----------
    learner : estimator, optional
        Probabilistic classifier supporting predict_proba.
    tol : float, default=1e-4
        Convergence threshold.
    max_iter : int, default=100
        Maximum EM iterations.
    calib_function : str or callable, optional
        Calibration method:
        - 'ts': Temperature Scaling
        - 'bcts': Bias-Corrected Temperature Scaling
        - 'vs': Vector Scaling
        - 'nbvs': No-Bias Vector Scaling
        - callable: custom calibration function
    criteria : callable, default=MAE
        Convergence metric.

    References
    ----------
    .. [1] Saerens, M., Latinne, P., & Decaestecker, C. (2002).
        Adjusting the Outputs of a Classifier to New a Priori Probabilities.
        Neural Computation, 14(1), 2141-2156.
    .. [2] Esuli, A., Moreo, A., & Sebastiani, F. (2023). Learning to Quantify. Springer.
    """

    _parameter_constraints = {
        "tol": [Interval(0, None, inclusive_left=False)],
        "max_iter": [Interval(1, None, inclusive_left=True)],
        "calib_function": [
            Options(["bcts", "ts", "vs", "nbvs", None]),
        ],
        "criteria": [CallableConstraint()],
    }

    def __init__(self, 
                 learner=None, 
                 tol=1e-4, 
                 max_iter=100, 
                 calib_function=None,
                 criteria=MAE):
        self.learner = learner
        self.tol = tol
        self.max_iter = max_iter
        self.calib_function = calib_function
        self.criteria = criteria

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, X, y)
        validate_y(self, y)
        self.classes_ = np.unique(y)
        self.learner.fit(X, y)
        counts = np.array([np.count_nonzero(y == _class) for _class in self.classes_])
        self.priors = counts / len(y)
        self.y_train = y
                
        return self

    def predict(self, X):
        """Predict the prevalence of each class."""
        X = validate_data(self, X)
        estimator_function = _get_learner_function(self)
        predictions = getattr(self.learner, estimator_function)(X)
        prevalences = self.aggregate(predictions, self.y_train)
        return prevalences

    def aggregate(self, predictions, y_train):
        predictions = validate_predictions(self, predictions)
        self.classes_ = check_classes_attribute(self, np.unique(y_train))
        
        if not hasattr(self, 'priors') or len(self.priors) != len(self.classes_):
            counts = np.array([np.count_nonzero(y_train == _class) for _class in self.classes_])
            self.priors = counts / len(y_train)

        calibrated_predictions = self._apply_calibration(predictions)
        prevalences, _ = self.EM(
            posteriors=calibrated_predictions,
            priors=self.priors,
            tolerance=self.tol,
            max_iter=self.max_iter,
            criteria=self.criteria
        )

        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences
    

    @classmethod
    def EM(cls, posteriors, priors, tolerance=1e-6, max_iter=100, criteria=MAE):
        r"""Static method implementing the EM algorithm for quantification.

        Parameters
        ----------
        posteriors : ndarray of shape (n_samples, n_classes)
            Posterior probability predictions.
        priors : ndarray of shape (n_classes,)
            Training class prior probabilities.
        tolerance : float
            Convergence threshold based on difference between iterations.
        max_iter : int
            Max number of EM iterations.
        criteria : callable
            Metric to assess convergence, e.g., MAE.

        Returns
        -------
        qs : ndarray of shape (n_classes,)
            Estimated test set class prevalences.
        ps : ndarray of shape (n_samples, n_classes)
            Updated soft membership probabilities per instance.
        """
        
        Px = np.array(posteriors, dtype=np.float64)
        Ptr = np.array(priors, dtype=np.float64)
        
        

        if np.prod(Ptr) == 0:
            Ptr += tolerance
            Ptr /= Ptr.sum()

        qs = np.copy(Ptr)
        s, converged = 0, False
        qs_prev_ = None
        
        while not converged and s < max_iter:
            # E-step:
            ps_unnormalized = (qs / Ptr) * Px
            ps = ps_unnormalized / ps_unnormalized.sum(axis=1, keepdims=True)

            # M-step:
            qs = ps.mean(axis=0)

            if qs_prev_ is not None and criteria(qs_prev_, qs) < tolerance and s > 10:
                converged = True

            qs_prev_ = qs
            s += 1

        if not converged:
            print('[warning] the method has reached the maximum number of iterations; it might have not converged')

        return qs, ps


    def _apply_calibration(self, predictions):
        r"""Calibrate posterior predictions with specified calibration method.
        
        Parameters
        ----------
        predictions : ndarray
            Posterior predictions to calibrate.
        
        Returns
        -------
        calibrated_predictions : ndarray
            Calibrated posterior predictions.
        
        Raises
        ------
        ValueError
            If calib_function is unrecognized.
        """
        if self.calib_function is None:
            return predictions

        if isinstance(self.calib_function, str):
            method = self.calib_function.lower()
            if method == "ts":
                return self._temperature_scaling(predictions)
            elif method == "bcts":
                return self._bias_corrected_temperature_scaling(predictions)
            elif method == "vs":
                return self._vector_scaling(predictions)
            elif method == "nbvs":
                return self._no_bias_vector_scaling(predictions)

        elif callable(self.calib_function):
            return self.calib_function(predictions)

        raise ValueError(
            f"Invalid calib_function '{self.calib_function}'. Expected one of {{'bcts', 'ts', 'vs', 'nbvs', None, callable}}."
        )

    def _temperature_scaling(self, preds):
        """Temperature Scaling calibration applied to logits."""
        T = 1.0
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        scaled = logits / T
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

    def _bias_corrected_temperature_scaling(self, preds):
        """Bias-Corrected Temperature Scaling calibration."""
        T = 1.0
        bias = np.zeros(preds.shape[1])
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        logits = logits / T + bias
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _vector_scaling(self, preds):
        """Vector Scaling calibration."""
        W = np.ones(preds.shape[1])
        b = np.zeros(preds.shape[1])
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        scaled = logits * W + b
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

    def _no_bias_vector_scaling(self, preds):
        """No-Bias Vector Scaling calibration."""
        W = np.ones(preds.shape[1])
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        scaled = logits * W
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)