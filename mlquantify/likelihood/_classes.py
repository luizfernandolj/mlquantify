from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import AggregativeMixin
import numpy as np
from mlquantify.base_aggregative import (
    SoftPredictionMixin,
    _get_estimator,
    _get_estimator_function,
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



class BaseLikelihoodQuantifier(SoftPredictionMixin, AggregativeMixin, BaseQuantifier):
    r"""Abstract base class for likelihood/prior-adjustment quantifiers.

    Provides ``fit`` and ``predict`` for quantifiers that adjust posterior
    probabilities to account for prior probability shift between training and
    test data. Subclasses must implement :meth:`aggregate`.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    train_predictions_ : ndarray of shape (n_samples, n_classes)
        Posterior probabilities on the training data.
    train_labels_ : ndarray of shape (n_samples,)
        Training labels.
    priors_ : ndarray of shape (n_classes,)
        Training class prevalences.

    Examples
    --------
    >>> from mlquantify.likelihood._classes import BaseLikelihoodQuantifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> class MyLikelihoodQ(BaseLikelihoodQuantifier):
    ...     def __init__(self, estimator=None):
    ...         super().__init__(estimator=estimator or LogisticRegression())
    ...     def aggregate(self, predictions, train_predictions=None, train_labels=None):
    ...         return np.mean(predictions, axis=0)
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> MyLikelihoodQ().fit(X, y).predict(X)
    {0: 0.49, 1: 0.51}
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        X, y = validate_data(self, X, y)

        self.classes_ = np.unique(y)
        estimator = _get_estimator(self)
        estimator.fit(X, y)
        self.estimator_ = estimator

        estimator_function = _get_estimator_function(self)

        self.train_predictions_ = getattr(estimator, estimator_function)(X)
        self.train_labels_ = np.asarray(y)
        self.priors_ = self._compute_priors(y)

        return self

    def predict(self, X):
        X = validate_data(self, X)

        estimator_function = _get_estimator_function(self)
        predictions = getattr(self.estimator_, estimator_function)(X)

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
    r"""Expectation-Maximization Quantifier (EMQ / SLD).

    Estimates class prevalences under prior probability shift by iterating
    between re-weighting posterior probabilities to reflect the current
    prevalence estimate (E-step) and updating the prevalence estimate as
    their average (M-step). Optionally applies a calibration step before
    the EM iteration to improve posterior quality.

    Supported calibration methods via ``calib_function``: Temperature Scaling
    (``'ts'``), Bias-Corrected Temperature Scaling (``'bcts'``), Vector
    Scaling (``'vs'``), and No-Bias Vector Scaling (``'nbvs'``).

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    tol : float, default=1e-4
        Convergence threshold on the prevalence change between iterations.
    max_iter : int, default=100
        Maximum number of EM iterations.
    calib_function : {'ts', 'bcts', 'vs', 'nbvs'} or callable or None, default=None
        Calibration applied to posteriors before EM. ``None`` skips calibration.
    criteria : callable, default=MAE
        Convergence criterion comparing successive prevalence estimates.
    on_calib_error : {'raise', 'backup'}, default='backup'
        Behaviour when calibration fails: ``'raise'`` re-raises the error;
        ``'backup'`` falls back to uncalibrated posteriors.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    priors_ : ndarray of shape (n_classes,)
        Training class prevalences.

    Examples
    --------
    >>> from mlquantify.likelihood import EMQ
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = EMQ(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}
    >>> # call aggregate with pre-computed posteriors
    >>> proba_train = q.estimator_.predict_proba(X)
    >>> proba_test = q.estimator_.predict_proba(X)
    >>> q.aggregate(proba_test, proba_train, y)
    {0: 0.49, 1: 0.51}

    References
    ----------
    .. dropdown:: References

        .. [1] Saerens, M., Latinne, P., & Decaestecker, C. (2002).
               Adjusting the Outputs of a Classifier to New a Priori Probabilities.
               *Neural Computation*, 14(1), 2141–2156.
        .. [2] Alexandari, A., Kundaje, A., & Shrikumar, A. (2020).
               Maximum Likelihood with Bias-Corrected Calibration is Hard-to-Beat
               at Label Shift Adaptation. *ICML*, pp. 222–232.
        .. [3] Esuli, A., Moreo, A., & Sebastiani, F. (2023).
               *Learning to Quantify*. Springer.
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
        estimator=None,
        tol=1e-4,
        max_iter=100,
        calib_function=None,
        criteria=MAE,
        on_calib_error="backup",
    ):
        self.estimator = estimator
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
    r"""CDE-Iterate quantifier.

    Estimates binary class prevalence by iteratively adjusting a decision
    threshold using class-cost ratios derived from training priors and the
    current prevalence estimate. At each iteration the threshold is updated
    until the predicted positive proportion stabilises.

    This is a **binary-only** method. Multiclass problems are handled with a
    one-vs-rest (OvR) strategy by default.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    tol : float, default=1e-4
        Convergence threshold on the positive prevalence change.
    max_iter : int, default=100
        Maximum number of iterations.
    init_cfp : float, default=1.0
        Initial cost of false positives.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.
    n_jobs : int or None, default=None
        Number of parallel jobs for multiclass decomposition.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    priors_ : ndarray of shape (n_classes,)
        Training class prevalences.

    Examples
    --------
    >>> from mlquantify.likelihood import CDE
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = CDE(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}
    >>> # call aggregate with pre-computed posteriors
    >>> proba_test = q.estimator_.predict_proba(X)
    >>> q.aggregate(proba_test, train_labels=y)
    {0: 0.49, 1: 0.51}

    References
    ----------
    .. dropdown:: References

        .. [1] Barranquero, J., Díez, J., & del Coz, J. J. (2015).
               Quantification-Oriented Learning Based on Reliable Classifiers.
               *Pattern Recognition*, 48(2), 591–604.
    """

    _parameter_constraints = {
        "tol": [Interval(0, None, inclusive_left=False)],
        "max_iter": [Interval(1, None, inclusive_left=True)],
        "init_cfp": [Interval(0, None, inclusive_left=False)],
    }

    def __init__(
        self,
        estimator=None,
        tol=1e-4,
        max_iter=100,
        init_cfp=1.0,
        strategy="ovr",
        n_jobs=None,
    ):
        super().__init__(estimator=estimator)
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
