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
    r"""Base Class for Threshold-based adjustment methods for quantification.

    This is the base class for methods such as ACC, X, MAX, T50, MS, and MS2, 
    which adjust prevalence estimates based on the classifier's ROC curve, 
    as proposed by [1]_.

    These methods correct the bias in *Classify & Count (CC)* estimates caused 
    by differences in class distributions between the training and test datasets.
    
    The adjusted prevalence is calculated using the following formula:

    .. math::

        \hat{p} = \frac{p' - \text{FPR}}{\text{TPR} - \text{FPR}}

    where:
        - :math:`p'` is the observed positive proportion from CC,
        - :math:`\text{TPR} = P(\hat{y}=1|y=1)` is the True Positive Rate,
        - :math:`\text{FPR} = P(\hat{y}=1|y=0)` is the False Positive Rate.
    

    Parameters
    ----------
    estimator : estimator, optional
        A supervised learning model with `fit` and `predict_proba` methods.
    threshold : float, default=0.5
        Classification threshold in [0, 1].
    strategy : {'ovr'}, default='ovr'
        Strategy used for multiclass adaptation.

    Attributes
    ----------
    estimator : estimator
        The underlying classification model.
    classes : ndarray of shape (n_classes,)
        Unique class labels observed during training.

    Notes
    -----
    - Defined only for binary quantification tasks.
    - When applied to multiclass problems, the one-vs-rest strategy (`ovr`) 
    is used automatically.
    

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mlquantify.counting import ThresholdAdjustment
    >>> import numpy as np
    >>> class CustomThreshold(ThresholdAdjustment):
    ...     def get_best_threshold(self, thresholds, tprs, fprs):
    ...         idx = np.argmax(tprs - fprs)
    ...         return thresholds[idx], tprs[idx], fprs[idx]
    >>> X = np.random.randn(100, 4)
    >>> y = np.random.randint(0, 2, 100)
    >>> q = CustomThreshold(estimator=LogisticRegression())
    >>> q.fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}

    References
    ----------
    .. [1] Forman, G. (2005). "Counting Positives Accurately Despite Inaccurate 
        Classification", *Proceedings of ECML*, pp. 564-575.
    .. [2] Forman, G. (2008). "Quantifying Counts and Costs via Classification", 
        *Data Mining and Knowledge Discovery*, 17(2), 164-206.
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
        """Internal adjustment computation based on selected ROC threshold."""
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
        """Select the best threshold according to the specific method."""
        ...


class TAC(ThresholdAdjustment):
    r"""Threshold Adjusted Count (TAC) — baseline threshold correction.

    This method corrects the bias in class prevalence estimates caused by imperfect 
    classification accuracy, by adjusting the observed positive count using estimates 
    of the classifier's true positive rate (TPR) and false positive rate (FPR).

    It uses a fixed classification threshold and applies the formula:

    .. math::

        p = \frac{p' - \text{FPR}}{\text{TPR} - \text{FPR}}

    where :math:`p'` is the observed positive proportion from :class:`CC`,
    
    
    Parameters
    ----------
    estimator : estimator, optional
        A supervised learning model with `fit` and `predict_proba` methods.
    threshold : float, default=0.5
        Classification threshold in [0, 1] for applying in the :class:`CC` output.

    References
    ----------
    .. [1] Forman, G. (2005). "Counting Positives Accurately Despite Inaccurate Classification",
           *ECML*, pp. 564-575.
    """

    def get_best_threshold(self, thresholds, tprs, fprs):
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)


class TX(ThresholdAdjustment):
    r"""Threshold X method — threshold where :math:`\text{TPR} + \text{FPR} = 1`.

    This method selects the classification threshold at which the sum of the true positive
    rate (TPR) and false positive rate (FPR) equals one. This threshold choice balances 
    errors in a specific way improving quantification.


    Parameters
    ----------
    estimator : estimator, optional
        A supervised learning model with `fit` and `predict_proba` methods.
    threshold : float, default=0.5
        Classification threshold in [0, 1] for applying in the :class:`CC` output.

    References
    ----------
    .. [1] Forman, G. (2005). "Counting Positives Accurately Despite Inaccurate Classification",
           *ECML*, pp. 564-575.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmin(np.abs((1-tprs) - fprs))
        return thresholds[idx], tprs[idx], fprs[idx]


class TMAX(ThresholdAdjustment):
    r"""Threshold MAX method — threshold maximizing :math:`\text{TPR} - \text{FPR}`.

    This method selects the threshold that maximizes the difference between the true positive
    rate (TPR) and the false positive rate (FPR), effectively optimizing classification
    performance for quantification.


    Parameters
    ----------
    estimator : estimator, optional
        A supervised learning model with `fit` and `predict_proba` methods.
    threshold : float, default=0.5
        Classification threshold in [0, 1] for applying in the :class:`CC` output.


    References
    ----------
    .. [1] Forman, G. (2005). "Counting Positives Accurately Despite Inaccurate Classification",
           *ECML*, pp. 564-575.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmax(np.abs(tprs - fprs))
        return thresholds[idx], tprs[idx], fprs[idx]


class T50(ThresholdAdjustment):
    r"""T50 — selects threshold where :math:`\text{TPR} = 0.5`.

    This method chooses the classification threshold such that the true positive rate (TPR)
    equals 0.5, avoiding regions with unreliable estimates at extreme thresholds.


    Parameters
    ----------
    estimator : estimator, optional
        A supervised learning model with `fit` and `predict_proba` methods.
    threshold : float, default=0.5
        Classification threshold in [0, 1] for applying in the :class:`CC` output.


    References
    ----------
    .. [1] Forman, G. (2005). "Counting Positives Accurately Despite Inaccurate Classification",
           *ECML*, pp. 564-575.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmin(np.abs(tprs - 0.5))
        return thresholds[idx], tprs[idx], fprs[idx]


class MS(ThresholdAdjustment):
    r"""Median Sweep (MS) — median prevalence estimate across all thresholds.

    This method computes class prevalence estimates at multiple classification thresholds,
    using the adjusted count formula for each, then returns the median of these estimates,
    reducing variance caused by any single threshold selection.

    It thus leverages the strengths of bootstrap-like variance reduction without heavy
    computation.
    
    
    Parameters
    ----------
    estimator : estimator, optional
        A supervised learning model with `fit` and `predict_proba` methods.
    threshold : float, default=0.5
        Classification threshold in [0, 1] for applying in the :class:`CC` output.
    

    References
    ----------
    .. [1] Forman, G. (2008). "Quantifying Counts and Costs via Classification",
           *Data Mining and Knowledge Discovery*, 17(2), 164-206.
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
    r"""MS2 — Median Sweep variant constraining :math:`|\text{TPR} - \text{FPR}| > 0.25`.

    This variant of Median Sweep excludes thresholds where the absolute difference
    between true positive rate (TPR) and false positive rate (FPR) is below 0.25,
    improving stability by avoiding ambiguous threshold regions.


    Parameters
    ----------
    estimator : estimator, optional
        A supervised learning model with `fit` and `predict_proba` methods.
    threshold : float, default=0.5
        Classification threshold in [0, 1] for applying in the :class:`CC` output.


    Warnings
    --------
    - Warns if all TPR or FPR values are zero.
    - Warns if no thresholds satisfy the constraint.

    References
    ----------
    .. [1] Forman, G. (2008). "Quantifying Counts and Costs via Classification",
           *Data Mining and Knowledge Discovery*, 17(2), 164-206.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        if np.all(tprs == 0) or np.all(fprs == 0):
            warnings.warn("All TPR or FPR values are zero.")
        indices = np.where(np.abs(tprs - fprs) > 0.25)[0]
        if len(indices) == 0:
            warnings.warn("No cases satisfy |TPR - FPR| > 0.25.")
            indices = np.where(np.abs(tprs - fprs) >= 0)[0]
        return thresholds[indices], tprs[indices], fprs[indices]
