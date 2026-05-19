from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import AggregationMixin
import numpy as np
from mlquantify.base_aggregative import (
    SoftLearnerQMixin,
    _get_learner_function,
    get_aggregation_requirements,
)
from mlquantify.metrics._slq import MAE
from mlquantify.utils import _fit_context, validate_data, check_classes_attribute, validate_predictions, validate_prevalences
from mlquantify.utils._constraints import (
    Interval,
    CallableConstraint,
    Options
)
from mlquantify.multiclass import binary_quantifier
from abstention.calibration import (
    NoBiasVectorScaling,
    TempScaling,
    VectorScaling
)



class BaseLikelihoodQuantifier(SoftLearnerQMixin, AggregationMixin, BaseQuantifier):
    r"""Base class for likelihood/prior-adjustment quantifiers."""

    def __init__(self, learner=None):
        self.learner = learner

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        X, y = validate_data(self, X, y)

        self.classes_ = np.unique(y)
        self.learner.fit(X, y)

        learner_function = _get_learner_function(self)

        self.train_predictions_ = getattr(self.learner, learner_function)(X)
        self.train_labels_ = np.asarray(y)
        self.priors_ = self._compute_priors(y)

        return self

    def predict(self, X):
        X = validate_data(self, X)

        learner_function = _get_learner_function(self)
        predictions = getattr(self.learner, learner_function)(X)

        requirements = get_aggregation_requirements(self)

        if (
            requirements.requires_train_proba
            and requirements.requires_train_labels
        ):
            return self.aggregate(
                predictions,
                self.train_predictions_,
                self.train_labels_,
            )

        if requirements.requires_train_labels:
            return self.aggregate(predictions, self.train_labels_)

        return self.aggregate(predictions)

    def _compute_priors(self, y):
        y = np.asarray(y)

        return np.asarray([
            np.mean(y == cls)
            for cls in self.classes_
        ], dtype=float)


class EMQ(BaseLikelihoodQuantifier):
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
        "on_calib_error": [Options(["raise", "backup"])],
    }

    def __init__(
        self,
        learner=None,
        tol=1e-4,
        max_iter=100,
        calib_function=None,
        criteria=MAE,
        on_calib_error="backup",
    ):
        self.learner = learner
        self.tol = tol
        self.max_iter = max_iter
        self.calib_function = calib_function
        self.criteria = criteria
        self.on_calib_error = on_calib_error

    
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = True
        tags.prediction_requirements.requires_train_labels = True
        return tags

    def aggregate(
        self,
        predictions,
        train_predictions=None,
        train_labels=None,
    ):
        predictions = validate_predictions(self, predictions)

        if train_predictions is None:
            train_predictions = self.train_predictions_

        if train_labels is None:
            train_labels = self.train_labels_

        self.classes_ = check_classes_attribute(self, np.unique(train_labels))
        self.priors_ = self._compute_priors(train_labels)

        calibrated_predictions = self._maybe_calibrate(
            predictions=predictions,
            train_predictions=train_predictions,
            train_labels=train_labels,
        )

        prevalences, _ = self.EM(
            posteriors=calibrated_predictions,
            priors=self.priors_,
            tolerance=self.tol,
            max_iter=self.max_iter,
            criteria=self.criteria,
        )

        return validate_prevalences(self, prevalences, self.classes_)

    def _resolve_calib_function(self):
        if self.calib_function is None:
            return None

        if callable(self.calib_function):
            return self.calib_function

        return {
            "nbvs": NoBiasVectorScaling(),
            "bcts": TempScaling(bias_positions="all"),
            "ts": TempScaling(),
            "vs": VectorScaling(),
        }.get(self.calib_function, None)

    def _maybe_calibrate(
        self,
        predictions,
        train_predictions,
        train_labels,
    ):
        calib_factory = self._resolve_calib_function()

        if calib_factory is None:
            return predictions

        eps = 1e-6

        train_predictions = np.clip(train_predictions, eps, 1.0 - eps)
        predictions = np.clip(predictions, eps, 1.0 - eps)

        train_logits = np.log(train_predictions)
        train_logits -= train_logits.mean(axis=1, keepdims=True)

        test_logits = np.log(predictions)
        test_logits -= test_logits.mean(axis=1, keepdims=True)

        try:
            calibrator = calib_factory(
                train_logits,
                self._encode_targets(train_labels),
                posterior_supplied=False,
            )
        except Exception as exc:
            calibrator = self._catch_calib_error(exc, "train")

        try:
            return calibrator(test_logits)
        except Exception as exc:
            self._catch_calib_error(exc, "test")
            return predictions

    def _encode_targets(self, y_train):
        y_idx = np.searchsorted(self.classes_, y_train)
        return np.eye(len(self.classes_))[y_idx]

    def _catch_calib_error(self, exc, method):
        if self.on_calib_error == "raise":
            raise RuntimeError(
                f"calibration {self.calib_function!r} failed at {method} time: {exc}"
            )

        if self.on_calib_error == "backup":
            if method == "train":
                return lambda P: P
            return None

        raise ValueError(f"Unknown on_calib_error={self.on_calib_error!r}")

    @classmethod
    def EM(
        cls,
        posteriors,
        priors,
        tolerance=1e-6,
        max_iter=100,
        criteria=MAE,
    ):
        Px = np.asarray(posteriors, dtype=np.float64)
        Ptr = np.asarray(priors, dtype=np.float64)

        Ptr = np.clip(Ptr, tolerance, None)
        Ptr /= Ptr.sum()

        qs = Ptr.copy()
        qs_prev = None

        for iteration in range(max_iter):
            ratio = qs / Ptr
            ps = Px * ratio
            ps /= ps.sum(axis=1, keepdims=True)

            qs = ps.mean(axis=0)

            if qs_prev is not None:
                if criteria(qs_prev, qs) < tolerance and iteration > 5:
                    break

            qs_prev = qs

        return qs, ps
    
    


@binary_quantifier(strategy_attr="strategy")
class CDE(BaseLikelihoodQuantifier):
    r"""CDE-Iterate for binary prevalence estimation."""

    _parameter_constraints = {
        "tol": [Interval(0, None, inclusive_left=False)],
        "max_iter": [Interval(1, None, inclusive_left=True)],
        "init_cfp": [Interval(0, None, inclusive_left=False)],
    }

    def __init__(
        self,
        learner=None,
        tol=1e-4,
        max_iter=100,
        init_cfp=1.0,
        strategy="ovr",
        n_jobs=None,
    ):
        super().__init__(learner=learner)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.init_cfp = float(init_cfp)
        self.strategy = strategy
        self.n_jobs = n_jobs

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_labels = True
        return tags

    def aggregate(
        self,
        predictions,
        train_predictions=None,
        train_labels=None,
    ):
        predictions = validate_predictions(self, predictions)

        if train_labels is None:
            train_labels = self.train_labels_

        self.classes_ = check_classes_attribute(self, np.unique(train_labels))

        if hasattr(self, "priors_"):
            priors = np.asarray(self.priors_, dtype=np.float64)
        else:
            priors = self._compute_priors(train_labels)

        prevalences = self._cde_iterate(
            predictions=predictions,
            priors=priors,
        )

        return validate_prevalences(self, prevalences, self.classes_)

    def _cde_iterate(self, predictions, priors):
        eps = 1e-12

        P = np.asarray(predictions, dtype=np.float64)
        P = np.clip(P, eps, 1.0)

        pL_neg = max(float(priors[0]), eps)
        pL_pos = max(float(priors[1]), eps)

        cFN = 1.0
        cFP = float(self.init_cfp)

        prev_prev_pos = None

        for _ in range(self.max_iter):
            tau = cFP / (cFP + cFN)

            pos_probs = P[:, 1]
            hard_pos = (pos_probs > tau).astype(float)

            prev_pos = float(hard_pos.mean())
            prev_neg = 1.0 - prev_pos

            prev_pos_safe = max(prev_pos, eps)
            prev_neg_safe = max(prev_neg, eps)

            cFP_new = (pL_pos / pL_neg) * (prev_neg_safe / prev_pos_safe) * cFN

            if prev_prev_pos is not None:
                if abs(prev_pos - prev_prev_pos) < self.tol:
                    break

            cFP = cFP_new
            prev_prev_pos = prev_pos

        return np.asarray([prev_neg, prev_pos], dtype=np.float64)
