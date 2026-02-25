from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import AggregationMixin
import numpy as np
from mlquantify.base_aggregative import SoftLearnerQMixin, _get_learner_function
from mlquantify.metrics._slq import MAE
from mlquantify.utils import _fit_context, validate_data, check_classes_attribute, validate_predictions, validate_prevalences
from mlquantify.utils._constraints import (
    Interval,
    CallableConstraint,
    Options
)
from abstention.calibration import (
    NoBiasVectorScaling,
    TempScaling,
    VectorScaling
)

class EMQ(SoftLearnerQMixin, AggregationMixin, BaseQuantifier):
    r"""Expectation-Maximization Quantifier (EMQ).

    Estimates class prevalences under prior probability shift by alternating 
    between expectation **(E)** and maximization **(M)** steps on posterior probabilities. 

    .. dropdown:: Mathematical Formulation

        E-step:

        .. math::

            p_i^{(s+1)}(x) = \frac{q_i^{(s)} p_i(x)}{\sum_j q_j^{(s)} p_j(x)}

        M-step:

        .. math::

            q_i^{(s+1)} = \frac{1}{N} \sum_{n=1}^N p_i^{(s+1)}(x_n)

        where:

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

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=200, n_features=10, random_state=7)
    >>> q = EMQ(learner=LogisticRegression(max_iter=500), calib_function='ts')
    >>> q.fit(X[:150], y[:150])
    EMQ(...)
    >>> prev = q.predict(X[150:])
    >>> round(float(prev.sum()), 6)
    1.0
    >>> probs_train = q.learner.predict_proba(X[:150])
    >>> probs_test = q.learner.predict_proba(X[150:])
    >>> prev2 = q.aggregate(probs_test, probs_train, y[:150])
    >>> round(float(prev2.sum()), 6)
    1.0
    """

    _parameter_constraints = {
        "tol": [Interval(0, None, inclusive_left=False)],
        "max_iter": [Interval(1, None, inclusive_left=True)],
        "calib_function": [
            Options(["bcts", "ts", "vs", "nbvs", None]),
            CallableConstraint(),
        ],
        "criteria": [CallableConstraint()],
        "on_calib_error": [Options(["raise", "backup"])]
    }


    def __init__(self, 
                 learner=None, 
                 tol=1e-4, 
                 max_iter=100, 
                 calib_function=None,
                 criteria=MAE,
                 on_calib_error="backup"):
        self.learner = learner
        self.tol = tol
        self.max_iter = max_iter
        self.calib_function = calib_function
        self.criteria = criteria
        self.on_calib_error = on_calib_error

    def _resolve_calib_function(self):
        # Build calibrator factory once and avoid mutating user parameters.
        if self.calib_function is None:
            return None
        if callable(self.calib_function):
            return self.calib_function
        return {
            'nbvs': NoBiasVectorScaling(),
            'bcts': TempScaling(bias_positions='all'),
            'ts': TempScaling(),
            'vs': VectorScaling()
        }.get(self.calib_function, None)

    def _encode_targets(self, y_train):
        # Support arbitrary labels (numeric or not) for one-hot encoding.
        y_idx = np.searchsorted(self.classes_, y_train)
        return np.eye(len(self.classes_))[y_idx]
        

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)
        self.learner.fit(X, y)
        counts = np.array([np.count_nonzero(y == _class) for _class in self.classes_])
        self.priors = counts / len(y)
        self.y_train = y
        
        learner_function = _get_learner_function(self)
        self.train_predictions = getattr(self.learner, learner_function)(X)
                
        return self

    def predict(self, X):
        """Predict the prevalence of each class."""
        X = validate_data(self, X)
        estimator_function = _get_learner_function(self)
        predictions = getattr(self.learner, estimator_function)(X)
        prevalences = self.aggregate(predictions, self.train_predictions, self.y_train)
        return prevalences

    def aggregate(self, predictions, train_predictions, y_train):
        predictions = validate_predictions(self, predictions)
        self.classes_ = check_classes_attribute(self, np.unique(y_train))
        
        eps = 1e-6
        train_predictions = np.clip(train_predictions, eps, 1 - eps)
        logits = np.log(train_predictions)
        logits -= logits.mean(axis=1, keepdims=True)

        predictions = np.clip(predictions, eps, 1 - eps)
        logits_test = np.log(predictions)
        logits_test -= logits_test.mean(axis=1, keepdims=True)

        
        
        calibrated_predictions = predictions
        calib_factory = self._resolve_calib_function()
        
        if calib_factory is not None:
            try:
                self.calibrator = calib_factory(
                    logits, 
                    self._encode_targets(y_train),
                    posterior_supplied=False
                )
            except Exception as e:
                self.calibrator = self._catch_calib_error(e, "train")
            
        
        
        if not hasattr(self, 'priors') or len(self.priors) != len(self.classes_):
            counts = np.array([np.count_nonzero(y_train == _class) for _class in self.classes_])
            self.priors = counts / len(y_train)
            
        if calib_factory is not None:
            try:
                calibrated_predictions = self.calibrator(logits_test)
            except Exception as e:
                self._catch_calib_error(e, "test")
                calibrated_predictions = predictions

        prevalences, _ = self.EM(
            posteriors=calibrated_predictions,
            priors=self.priors,
            tolerance=self.tol,
            max_iter=self.max_iter,
            criteria=self.criteria
        )

        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences
    
    def _catch_calib_error(self, e, method):
        if self.on_calib_error == 'raise':
            raise RuntimeError(f'calibration {self.calib_function} failed at {method} time: {e}')
        elif self.on_calib_error == 'backup':
            if method == "train":
                return lambda P: P
            elif method == "test":
                return None
    

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
        
        Px = np.asarray(posteriors, dtype=np.float64)
        Ptr = np.asarray(priors, dtype=np.float64)

        # Avoid zero priors
        Ptr = np.clip(Ptr, tolerance, None)
        Ptr /= Ptr.sum()

        qs = Ptr.copy()
        qs_prev = None

        for s in range(max_iter):

            # ---- E-step ----
            ratio = qs / Ptr
            ps = Px * ratio
            ps /= ps.sum(axis=1, keepdims=True)

            # ---- M-step ----
            qs = ps.mean(axis=0)

            # ---- Convergence check ----
            if qs_prev is not None:
                if criteria(qs_prev, qs) < tolerance and s > 5:
                    break

            qs_prev = qs

        return qs, ps