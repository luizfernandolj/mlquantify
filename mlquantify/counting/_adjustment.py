from mlquantify.utils._validation import validate_prevalences
from mlquantify.base import BaseQuantifier
from mlquantify._config import config_context
import numpy as np     
from abc import abstractmethod
from scipy.optimize import minimize
import warnings
from sklearn.metrics import confusion_matrix
from mlquantify.counting._base import BaseAdjustCount
from mlquantify.counting._counting import CC, PCC
from mlquantify.utils import (
    _fit_context, 
    validate_data,
    validate_prevalences,
    validate_predictions,
    check_classes_attribute
)
from mlquantify.base_aggregative import (
    CrispPredictionMixin,
    SoftPredictionMixin,
    AggregativeMixin,
    uses_soft_predictions, 
    _get_estimator_function
)
from mlquantify.multiclass import binary_quantifier
from mlquantify.utils._optimization import _optimize_on_simplex
from mlquantify.counting._utils import evaluate_thresholds
from mlquantify.utils._constraints import Interval, Options


@binary_quantifier(strategy_attr="strategy")
class ThresholdAdjustment(SoftPredictionMixin, BaseAdjustCount):
    r"""Abstract base class for ROC-threshold adjustment quantifiers.

    Corrects the bias in :class:`CC` estimates by selecting a threshold
    on the ROC curve and adjusting the observed positive proportion using
    the corresponding TPR and FPR. Subclasses implement
    :meth:`get_best_threshold` to define the selection strategy.

    This is a **binary-only** method. When applied to multiclass problems,
    a one-vs-rest (OvR) strategy is applied automatically.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Default classification threshold; used directly by :class:`TAC`.
    strategy : {'ovr'}, default='ovr'
        Multiclass decomposition strategy.
    n_jobs : int or None, default=None
        Number of parallel jobs for multiclass decomposition.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    train_predictions : ndarray
        Cross-validated soft predictions used for TPR/FPR estimation.
    y_train : ndarray of shape (n_samples,)
        Training labels corresponding to ``train_predictions``.
        
    Examples
    --------
    >>> from mlquantify.counting import ThresholdAdjustment
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> class CustomTA(ThresholdAdjustment):
    ...     def get_best_threshold(self, thresholds, tprs, fprs):
    ...         idx = np.argmin(np.abs(tprs - 0.5))
    ...         return thresholds[idx], tprs[idx], fprs[idx]
    >>> X, y = make_classification(n_samples=100, n_classes=2, n_informative=5, random_state=42)
    >>> quantifier = CustomTA(estimator=RandomForestClassifier(random_state=42))
    >>> quantifier.fit(X, y)
    >>> quantifier.predict(X)
    array([[0.3, 0.7]])
    
    References
    ----------
    .. dropdown:: References

        .. [1] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate
               Classification. *ECML*, pp. 564–575.
        .. [2] Forman, G. (2008). Quantifying Counts and Costs via Classification.
               *Data Mining and Knowledge Discovery*, 17(2), 164–206.
    """

    _parameter_constraints = {
        "threshold": [
            Interval(0.0, 1.0),
            Interval(0, 1, discrete=True),
        ],
    }


    def __init__(self, estimator=None, threshold=0.5, strategy="ovr", n_jobs=None):
        super().__init__(estimator=estimator)
        self.threshold = threshold
        self.strategy = strategy
        self.n_jobs = n_jobs

    def _adjust(self, predictions, train_y_scores, y_train):
        r"""Internal adjustment computation based on selected ROC threshold."""
        positive_scores = train_y_scores[:, 1]
        
        thresholds, tprs, fprs = evaluate_thresholds(y_train, positive_scores)
        threshold, tpr, fpr = self.get_best_threshold(thresholds, tprs, fprs)

        with config_context(prevalence_return_type="array"):
            cc_predictions = CC(threshold=threshold).aggregate(predictions, y_train)
        cc_predictions = cc_predictions[1]

        if tpr - fpr == 0:
            prevalence = cc_predictions
        else:
            prevalence = np.clip((cc_predictions - fpr) / (tpr - fpr), 0, 1)
        
        return np.asarray([1 - prevalence, prevalence])
    
    @abstractmethod
    def get_best_threshold(self, thresholds, tprs, fprs):
        r"""Select the best threshold according to the specific method."""
        ...


class TAC(ThresholdAdjustment):
    r"""Threshold Adjusted Count (TAC).

    Applies the threshold adjustment correction at a fixed classification
    threshold. TPR and FPR are estimated from cross-validated predictions
    at the specified ``threshold`` value.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        The classification threshold at which TPR and FPR are evaluated.
        
    Examples
    --------
    >>> from mlquantify.counting import TAC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_classes=2, n_informative=5, random_state=42)
    >>> quantifier = TAC(threshold=0.3, estimator=RandomForestClassifier(random_state=42))
    >>> quantifier.fit(X, y)
    >>> quantifier.predict(X)
    array([[0.4, 0.6]])

    References
    ----------
    .. dropdown:: References

        .. [1] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate
               Classification. *ECML*, pp. 564–575.
    """

    def get_best_threshold(self, thresholds, tprs, fprs):
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)


class TX(ThresholdAdjustment):
    r"""Threshold X (TX).

    Selects the threshold where ``TPR + FPR = 1``, balancing the two
    error rates symmetrically around the ROC diagonal.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Unused by this subclass; kept for API consistency.
        
    Examples
    --------
    >>> from mlquantify.counting import TX
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_classes=2, n_informative=5, random_state=42)
    >>> quantifier = TX(estimator=RandomForestClassifier(random_state=42))
    >>> quantifier.fit(X, y)
    >>> quantifier.predict(X)
    array([[0.4, 0.6]])

    References
    ----------
    .. dropdown:: References

        .. [1] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate
               Classification. *ECML*, pp. 564–575.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmin(np.abs((1-tprs) - fprs))
        return thresholds[idx], tprs[idx], fprs[idx]


class TMAX(ThresholdAdjustment):
    r"""Threshold MAX (TMAX).

    Selects the threshold that maximizes ``|TPR - FPR|``, which corresponds
    to the most discriminative operating point on the ROC curve.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Unused by this subclass; kept for API consistency.
        
    Examples
    --------
    >>> from mlquantify.counting import TMAX
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_classes=2, n_informative=5, random_state=42)
    >>> quantifier = TMAX(estimator=RandomForestClassifier(random_state=42))
    >>> quantifier.fit(X, y)
    >>> quantifier.predict(X)
    array([[0.4, 0.6]])

    References
    ----------
    .. dropdown:: References

        .. [1] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate
               Classification. *ECML*, pp. 564–575.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmax(np.abs(tprs - fprs))
        return thresholds[idx], tprs[idx], fprs[idx]


class T50(ThresholdAdjustment):
    r"""T50 — threshold where TPR is closest to 0.5.

    Selects the classification threshold at which the true positive rate
    (TPR) is approximately 0.5, avoiding the extreme ends of the ROC
    curve where estimates tend to be unstable.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Unused by this subclass; kept for API consistency.
        
    Examples
    --------
    >>> from mlquantify.counting import T50
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_classes=2, n_informative=5, random_state=42)
    >>> quantifier = T50(estimator=RandomForestClassifier(random_state=42))
    >>> quantifier.fit(X, y)
    >>> quantifier.predict(X)
    array([[0.4, 0.6]])

    References
    ----------
    .. dropdown:: References

        .. [1] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate
               Classification. *ECML*, pp. 564–575.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmin(np.abs(tprs - 0.5))
        return thresholds[idx], tprs[idx], fprs[idx]


class MS(ThresholdAdjustment):
    r"""Median Sweep (MS).

    Applies the threshold adjustment formula at every threshold on the
    ROC curve and returns the median prevalence estimate. This reduces
    variance compared to any single-threshold method.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Unused by this subclass; kept for API consistency.

    Examples
    --------
    >>> from mlquantify.counting import MS
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_classes=2, n_informative=5, random_state=42)
    >>> quantifier = MS(estimator=RandomForestClassifier(random_state=42))
    >>> quantifier.fit(X, y)
    >>> quantifier.predict(X)
    array([[0.4, 0.6]])

    References
    ----------
    .. dropdown:: References

        .. [1] Forman, G. (2008). Quantifying Counts and Costs via Classification.
               *Data Mining and Knowledge Discovery*, 17(2), 164–206.
    """
    def _adjust(self, predictions, train_y_scores, y_train):
        positive_scores = train_y_scores[:, 1]
        
        thresholds, tprs, fprs = evaluate_thresholds(y_train, positive_scores)
        thresholds, tprs, fprs = self.get_best_threshold(thresholds, tprs, fprs)
        
        prevs = []
        for thr, tpr, fpr in zip(thresholds, tprs, fprs):
            with config_context(prevalence_return_type="array"):
                cc_predictions = CC(threshold=thr).aggregate(predictions, y_train)
            cc_predictions = cc_predictions[1]
            
            if tpr - fpr == 0:
                prevalence = cc_predictions
            else:
                prevalence = np.clip((cc_predictions - fpr) / (tpr - fpr), 0, 1)
                
            prevs.append(prevalence)
        prevalence = np.median(prevs)
        return np.asarray([1 - prevalence, prevalence])
    
    def get_best_threshold(self, thresholds, tprs, fprs):
        return thresholds, tprs, fprs


class MS2(MS):
    r"""Median Sweep 2 (MS2).

    A constrained variant of :class:`MS` that only sweeps thresholds
    where ``|TPR - FPR| > 0.25``, discarding ambiguous regions of the
    ROC curve. Falls back to all thresholds if no qualifying threshold
    exists.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Unused by this subclass; kept for API consistency.

    Warns
    -----
    UserWarning
        If all TPR or FPR values are zero, or if no threshold satisfies
        the ``|TPR - FPR| > 0.25`` constraint.
        
    Examples
    --------
    >>> from mlquantify.counting import MS2
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_classes=2, n_informative=5, random_state=42)
    >>> quantifier = MS2(estimator=RandomForestClassifier(random_state=42))
    >>> quantifier.fit(X, y)
    >>> quantifier.predict(X)
    array([[0.4, 0.6]])

    References
    ----------
    .. dropdown:: References

        .. [1] Forman, G. (2008). Quantifying Counts and Costs via Classification.
               *Data Mining and Knowledge Discovery*, 17(2), 164–206.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        if np.all(tprs == 0) or np.all(fprs == 0):
            warnings.warn("All TPR or FPR values are zero.")
        indices = np.where(np.abs(tprs - fprs) > 0.25)[0]
        if len(indices) == 0:
            warnings.warn("No cases satisfy |TPR - FPR| > 0.25.")
            indices = np.where(np.abs(tprs - fprs) >= 0)[0]
        return thresholds[indices], tprs[indices], fprs[indices]
