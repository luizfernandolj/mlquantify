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

    Targets prior probability shift. Corrects the bias in :class:`CC`
    estimates by selecting a threshold on the ROC curve and adjusting the
    observed positive proportion using the corresponding TPR and FPR.
    Subclasses implement :meth:`get_best_threshold` to define the selection
    strategy.

    This is a **binary-only** method. When applied to multiclass problems,
    a one-vs-rest (OvR) strategy is applied automatically.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Default classification threshold.
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
    train_predictions : ndarray
        Cross-validated soft predictions used for TPR/FPR estimation.
    y_train : ndarray of shape (n_samples,)
        Training labels corresponding to ``train_predictions``.

    See Also
    --------
    TAC : Adjusted count at the default threshold.
    TX : Threshold where FPR crosses 1 - TPR.
    TMAX : Threshold maximizing TPR - FPR.
    T50 : Threshold where TPR is 0.5.
    MS : Median Sweep over all thresholds.
    MS2 : Median Sweep restricted to reliable thresholds.

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


    def __init__(
        self,
        estimator=None,
        threshold=0.5,
        strategy="ovr",
        n_jobs=None,
        cv=5,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )
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
    r"""Threshold Adjusted Count (TAC) quantifier.

    Targets prior probability shift. A threshold-policy member of the
    Adjusted Count family that applies the adjusted-count correction at a
    fixed decision threshold, with TPR and FPR estimated from
    cross-validated predictions at that threshold. Binary-only; requires
    soft scores.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        The classification threshold at which TPR and FPR are evaluated.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    See Also
    --------
    ACC : Adjusted Classify and Count.
    TX, TMAX, T50, MS, MS2 : Other threshold-selection policies.

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

        .. [1] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate classification. *ECML*, pp. 564–575.
    """

    def get_best_threshold(self, thresholds, tprs, fprs):
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)


class TX(ThresholdAdjustment):
    r"""Threshold X (TX) quantifier.

    Targets prior probability shift. A threshold-policy member of the
    Adjusted Count family that selects the operating point where the
    false-positive rate crosses ``1 - TPR`` (i.e. ``FPR = 1 - TPR``), then
    applies the adjusted-count correction there. Binary-only; requires soft
    scores.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Unused by this subclass; kept for API consistency.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    The crossing point keeps both error rates moderate and ``TPR - FPR``
    comfortably away from zero, giving a stable correction denominator under
    class imbalance.

    See Also
    --------
    ACC : Adjusted Classify and Count.
    TAC, TMAX, T50, MS, MS2 : Other threshold-selection policies.

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

        .. [1] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate classification. *ECML*, pp. 564–575.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmin(np.abs((1-tprs) - fprs))
        return thresholds[idx], tprs[idx], fprs[idx]


class TMAX(ThresholdAdjustment):
    r"""Threshold MAX (TMAX) quantifier.

    Targets prior probability shift. A threshold-policy member of the
    Adjusted Count family that selects the operating point maximizing
    ``TPR - FPR`` (the most discriminative point on the ROC curve) before
    applying the adjusted-count correction. Binary-only; requires soft
    scores.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Unused by this subclass; kept for API consistency.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    Maximizing ``TPR - FPR`` yields the most stable correction denominator,
    but Forman (2008) reports that TMAX carries a systematic bias because the
    chosen rates often fail to transfer to the test set.

    See Also
    --------
    ACC : Adjusted Classify and Count.
    TAC, TX, T50, MS, MS2 : Other threshold-selection policies.

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

        .. [1] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate classification. *ECML*, pp. 564–575.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmax(np.abs(tprs - fprs))
        return thresholds[idx], tprs[idx], fprs[idx]


class T50(ThresholdAdjustment):
    r"""Threshold 50 (T50) quantifier.

    Targets prior probability shift. A threshold-policy member of the
    Adjusted Count family that selects the operating point where the
    true-positive rate is closest to 0.5, then applies the adjusted-count
    correction there. Binary-only; requires soft scores.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Unused by this subclass; kept for API consistency.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    Pinning TPR makes the policy insensitive to positive-class scarcity;
    Forman reports it stays robust with as few as roughly ten training
    positives.

    See Also
    --------
    ACC : Adjusted Classify and Count.
    TAC, TX, TMAX, MS, MS2 : Other threshold-selection policies.

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

        .. [1] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate classification. *ECML*, pp. 564–575.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmin(np.abs(tprs - 0.5))
        return thresholds[idx], tprs[idx], fprs[idx]


class MS(ThresholdAdjustment):
    r"""Median Sweep (MS) quantifier.

    Targets prior probability shift. A threshold-policy member of the
    Adjusted Count family that computes the adjusted-count correction at
    every ROC threshold and returns the median estimate, rather than
    committing to a single operating point. Binary-only; requires soft
    scores.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Unused by this subclass; kept for API consistency.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    The median is robust to the unreliable thresholds (small ``TPR - FPR``)
    that would destabilise a single-point adjusted count, which is why Forman
    found Median Sweep so robust across shifts.

    See Also
    --------
    MS2 : Median Sweep restricted to reliable thresholds.
    ACC, TAC, TX, TMAX, T50 : Other Adjusted Count policies.

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

        .. [1] Forman, G. (2008). Quantifying Counts and Costs via Classification. *Data Mining and Knowledge Discovery*, 17(2), 164–206.
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
    r"""Median Sweep 2 (MS2) quantifier.

    Targets prior probability shift. A constrained variant of :class:`MS`
    that sweeps only the thresholds where ``|TPR - FPR| > 0.25``, discarding
    low-separation operating points before taking the median adjusted count.
    Falls back to all thresholds if none qualify. Binary-only; requires soft
    scores.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    threshold : float, default=0.5
        Unused by this subclass; kept for API consistency.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Warns
    -----
    UserWarning
        If all TPR or FPR values are zero, or if no threshold satisfies
        the ``|TPR - FPR| > 0.25`` constraint.

    Notes
    -----
    Pruning low-separation thresholds removes the highest-variance terms of
    the sweep, usually improving on :class:`MS` when the classifier is weak in
    part of the score range.

    See Also
    --------
    MS : Unrestricted Median Sweep.
    ACC, TAC, TX, TMAX, T50 : Other Adjusted Count policies.

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

        .. [1] Forman, G. (2008). Quantifying Counts and Costs via Classification. *Data Mining and Knowledge Discovery*, 17(2), 164–206.
    """
    def get_best_threshold(self, thresholds, tprs, fprs):
        if np.all(tprs == 0) or np.all(fprs == 0):
            warnings.warn("All TPR or FPR values are zero.")
        indices = np.where(np.abs(tprs - fprs) > 0.25)[0]
        if len(indices) == 0:
            warnings.warn("No cases satisfy |TPR - FPR| > 0.25.")
            indices = np.where(np.abs(tprs - fprs) >= 0)[0]
        return thresholds[indices], tprs[indices], fprs[indices]


@binary_quantifier(strategy_attr="strategy")
class ACC(CrispPredictionMixin, BaseAdjustCount):
    r"""Adjusted Classify and Count (ACC) quantifier.

    Targets prior probability shift. Runs :class:`CC` and corrects the biased
    count using the classifier's true- and false-positive rates, which are
    estimated by cross-validation and assumed invariant across the shift. The
    count, TPR and FPR are all derived from the classifier's argmax hard
    prediction. This is a **binary-only** method; multiclass problems are
    handled with a one-vs-rest (OvR) strategy.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit``, ``predict_proba``, and ``predict`` methods.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

        - ``'ovr'`` : one-vs-rest, one binary adjusted count per class.
        - ``'ovo'`` : one-vs-one, one binary adjusted count per class pair.
    n_jobs : int or None, default=None
        Number of parallel jobs for multiclass decomposition.
    cv : int, default=5
        Cross-validation folds used when ``estimator_fitted=False``.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    ACC corrects the linear bias of :class:`CC` and is unbiased when the
    estimated TPR/FPR match the test set. When ``TPR - FPR`` is small (a weak
    classifier under heavy imbalance) the correction amplifies estimation
    error, trading bias for variance. The implementation follows the reference
    QoT formulation, deriving TPR, FPR and the count from argmax hard
    predictions rather than a soft-probability threshold.

    See Also
    --------
    CC : Classify and Count quantifier.
    TAC : Adjusted count at a fixed soft threshold.
    GACC : Multiclass generalisation via constrained regression.

    Examples
    --------
    >>> from mlquantify.counting import ACC
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
    >>> q = ACC(LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.30, 1: 0.70}
    >>> # aggregate() path with pre-computed hard predictions
    >>> preds = q.estimator_.predict(X)
    >>> q.aggregate(preds, y_train=y)
    {0: 0.30, 1: 0.70}

    References
    ----------
    .. dropdown:: References

        .. [1] Forman, G. (2005). Counting Positives Accurately Despite
               Inaccurate Classification. *ECML*, pp. 564–575.
        .. [2] Forman, G. (2008). Quantifying Counts and Costs via
               Classification. *Data Mining and Knowledge Discovery*,
               17(2), 164–206.
    """

    def __init__(
        self,
        estimator=None,
        strategy="ovr",
        n_jobs=None,
        cv=5,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )
        self.strategy = strategy
        self.n_jobs = n_jobs

    def _adjust(self, predictions, train_predictions, y_train):
        # Both predictions and train_predictions are 1-D arrays of 0/1 hard
        # binary labels coming from CrispPredictionMixin → estimator.predict().
        # With _OvRBinaryWrapper, predict() returns (argmax == cls_idx).
        pos_mask = (y_train == 1)
        neg_mask = (y_train == 0)

        tpr = train_predictions[pos_mask].mean() if pos_mask.any() else 0.0
        fpr = train_predictions[neg_mask].mean() if neg_mask.any() else 0.0

        # CC: fraction of test samples predicted as positive by argmax
        cc = predictions.mean()

        diff = tpr - fpr
        if abs(diff) < 1e-10:
            prevalence = cc
        else:
            prevalence = float(np.clip((cc - fpr) / diff, 0.0, 1.0))

        return np.asarray([1 - prevalence, prevalence])

    def get_best_threshold(self, thresholds, tprs, fprs):
        # Not used — _adjust is fully overridden and does not call this.
        pass
