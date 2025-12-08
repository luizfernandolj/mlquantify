import numpy as np     
from abc import abstractmethod
from scipy.optimize import minimize
import warnings
from sklearn.metrics import confusion_matrix

from mlquantify.adjust_counting._base import BaseAdjustCount
from mlquantify.adjust_counting._counting import CC, PCC
from mlquantify.base_aggregative import (
    CrispLearnerQMixin,
    SoftLearnerQMixin,
    uses_soft_predictions,
)
from mlquantify.multiclass import define_binary
from mlquantify.adjust_counting._utils import evaluate_thresholds
from mlquantify.utils._constraints import Interval, Options


@define_binary
class ThresholdAdjustment(SoftLearnerQMixin, BaseAdjustCount):
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
    learner : estimator, optional
        A supervised learning model with `fit` and `predict_proba` methods.
    threshold : float, default=0.5
        Classification threshold in [0, 1].
    strategy : {'ovr'}, default='ovr'
        Strategy used for multiclass adaptation.

    Attributes
    ----------
    learner : estimator
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
    >>> from mlquantify.adjust_counting import ThresholdAdjustment
    >>> import numpy as np
    >>> class CustomThreshold(ThresholdAdjustment):
    ...     def get_best_threshold(self, thresholds, tprs, fprs):
    ...         idx = np.argmax(tprs - fprs)
    ...         return thresholds[idx], tprs[idx], fprs[idx]
    >>> X = np.random.randn(100, 4)
    >>> y = np.random.randint(0, 2, 100)
    >>> q = CustomThreshold(learner=LogisticRegression())
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

    def __init__(self, learner=None, threshold=0.5, strategy="ovr"):
        super().__init__(learner=learner)
        self.threshold = threshold
        self.strategy = strategy

    def _adjust(self, predictions, train_y_scores, train_y_values):
        """Internal adjustment computation based on selected ROC threshold."""
        positive_scores = train_y_scores[:, 1]
        
        thresholds, tprs, fprs = evaluate_thresholds(train_y_values, positive_scores)
        threshold, tpr, fpr = self.get_best_threshold(thresholds, tprs, fprs)

        cc_predictions = CC(threshold=threshold).aggregate(predictions, train_y_values)
        cc_predictions = list(cc_predictions.values())[1]

        if tpr - fpr == 0:
            prevalence = cc_predictions
        else:
            prevalence = np.clip((cc_predictions - fpr) / (tpr - fpr), 0, 1)
        
        return np.asarray([1 - prevalence, prevalence])
    
    @abstractmethod
    def get_best_threshold(self, thresholds, tprs, fprs):
        """Select the best threshold according to the specific method."""
        ...


class MatrixAdjustment(BaseAdjustCount):
    r"""Base class for matrix-based quantification adjustments.

    This class implements the matrix correction model for quantification
    as formulated in Firat (2016) [1]_, which expresses the observed prevalences 
    as a linear combination of true prevalences through the confusion matrix.

    The system is modeled as:

    .. math::

        \mathbf{y} = \mathbf{C}\hat{\pi}_F + \varepsilon

    subject to the constraints:

    .. math::

        \hat{\pi}_F \ge 0, \quad \sum_k \hat{\pi}_{F,k} = 1

    where:
        - :math:`\mathbf{y}` is the vector of predicted prevalences in test set,
        - :math:`\mathbf{C}` is the confusion matrix,
        - :math:`\hat{\pi}_F` is the true class prevalence vector (unknown),
        - :math:`\varepsilon` is the residual error.

    The model can be solved via:

    - **Linear algebraic solution**: uses matrix inversion
    - **Constrained optimization**: quadratic or least-squares approach


    Parameters
    ----------
    learner : estimator, optional
        Classifier with `fit` and `predict` methods.
    solver : {'optim', 'linear'}, optional
        Solver for the adjustment system:
        
        - `'linear'`: uses matrix inversion (e.g., GAC, GPAC)
        - `'optim'`: uses optimization (e.g., FM)

    Attributes
    ----------
    CM : ndarray of shape (n_classes, n_classes)
        Confusion matrix used for correction.
    classes : ndarray
        Class labels observed in training.


    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mlquantify.adjust_counting import MatrixAdjustment
    >>> import numpy as np
    >>> class MyMatrix(MatrixAdjustment):
    ...     def _compute_confusion_matrix(self, preds, y):
    ...         cm = np.ones((2, 2))
    ...         return cm / cm.sum(axis=1, keepdims=True)
    >>> q = MyMatrix(learner=LogisticRegression(), solver='linear')
    >>> X = np.random.randn(50, 4)
    >>> y = np.random.randint(0, 2, 50)
    >>> q.fit(X, y)
    >>> q.predict(X)
    {0: 0.5, 1: 0.5}

    References
    ----------
    .. [1] Firat, A. (2016). "Unified Framework for Quantification", 
        *Proceedings of AAAI Conference on Artificial Intelligence*, 
        pp. 1-8.
    """


    _parameter_constraints = {"solver": Options(["optim", "linear"])}

    def __init__(self, learner=None, solver=None):
        super().__init__(learner=learner)
        self.solver = solver
    
    def _adjust(self, predictions, train_y_pred, train_y_values):
        n_class = len(np.unique(train_y_values))
        self.CM = np.zeros((n_class, n_class))

        if self.solver == 'optim':
            priors = np.array(list(CC().aggregate(train_y_pred, train_y_values).values()))
            self.CM = self._compute_confusion_matrix(train_y_pred, train_y_values, priors)
            prevs_estim = self._get_estimations(predictions > priors, train_y_values)
            prevalence = self._solve_optimization(prevs_estim, priors)
        else:
            self.CM = self._compute_confusion_matrix(train_y_pred, train_y_values)
            prevs_estim = self._get_estimations(predictions, train_y_values)
            prevalence = self._solve_linear(prevs_estim)
        
        return prevalence

    def _solve_linear(self, prevs_estim):
        r"""
        Solve the system using matrix inversion.
        """
        try:
            adjusted = np.linalg.solve(self.CM, prevs_estim)
            adjusted = np.clip(adjusted, 0, 1)
            adjusted /= adjusted.sum()
        except np.linalg.LinAlgError:
            adjusted = prevs_estim
        return adjusted

    def _solve_optimization(self, prevs_estim, priors):
        r"""Solve the system linearly.

        The solution is obtained by matrix inversion:

        .. math::

            \hat{\pi}_F = \mathbf{C}^{-1} \mathbf{p}

        where :math:`\mathbf{C}` is the confusion matrix and :math:`\mathbf{p}` 
        is the observed prevalence vector.

        Parameters
        ----------
        p : ndarray of shape (n_classes,)
            Observed prevalence vector from test set.

        Returns
        -------
        ndarray of shape (n_classes,)
            Adjusted prevalence estimates :math:`\hat{\pi}_F`.
        """
        def objective(prevs_pred):
            return np.linalg.norm(self.CM @ prevs_pred - prevs_estim)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        bounds = [(0, 1)] * self.CM.shape[1]
        init = np.full(self.CM.shape[1], 1 / self.CM.shape[1])
        result = minimize(objective, init, constraints=constraints, bounds=bounds)
        return result.x if result.success else priors

    def _get_estimations(self, predictions, train_y_values):
        """Return prevalence estimates using CC (crisp) or PCC (probabilistic)."""
        if uses_soft_predictions(self):
            return np.array(list(PCC().aggregate(predictions).values()))
        return np.array(list(CC().aggregate(predictions, train_y_values).values()))

    @abstractmethod
    def _compute_confusion_matrix(self, predictions, *args):
        ...


class FM(SoftLearnerQMixin, MatrixAdjustment):
    r"""Friedman Method for quantification adjustment.

    This class implements the Friedman (2015) matrix-based quantification adjustment, which formulates the quantification problem as a constrained optimization problem. It adjusts the estimated class prevalences by minimizing the difference between predicted and expected prevalences, subject to valid prevalence constraints.

    The confusion matrix is computed by applying estimated posterior probabilities
    over true labels, enabling accurate correction of prevalence estimates under
    concept drift.
    
    The confusion matrix is estimated for each class :math:`k` by:
    applying thresholding on posterior probabilities against prior prevalence,
    as described in the FM algorithm. This enables the correction using
    a quadratic optimization approach.

    The method solves:

    .. math::

        \min_{\hat{\pi}_F} \| \mathbf{C} \hat{\pi}_F - \mathbf{p} \|^2

    subject to constraints:

    .. math::

        \hat{\pi}_F \geq 0, \quad \sum_k \hat{\pi}_{F,k} = 1

    where :math:`\mathbf{C}` is the confusion matrix, :math:`\mathbf{p}` is the
    vector of predicted prevalences.
    

    Parameters
    ----------
    learner : estimator, optional
        Base classifier with `fit` and `predict_proba` methods.
        If None, a default estimator will be used.

    Attributes
    ----------
    CM : ndarray of shape (n_classes, n_classes)
        Confusion matrix used for correction.


    Examples
    --------
    >>> from mlquantify.adjust_counting import FM
    >>> import numpy as np
    >>> X = np.random.randn(50, 4)
    >>> y = np.random.randint(0, 2, 50)
    >>> fm = FM(learner=LogisticRegression())
    >>> fm.fit(X, y)
    >>> fm.predict(X)
    {0: 0.5, 1: 0.5}

    References
    ----------
    .. [1] Friedman, J. H., et al. (2015). "Detecting and Dealing with Concept Drift",
           *Proceedings of the IEEE*, 103(11), 1522-1541.
    """
    def __init__(self, learner=None):
        super().__init__(learner=learner, solver='optim')
    
    def _compute_confusion_matrix(self, posteriors, y_true, priors):
        for i, _class in enumerate(self.classes_):
            indices = (y_true == _class)
            self.CM[:, i] = self._get_estimations(posteriors[indices] > priors)
        return self.CM


class GAC(CrispLearnerQMixin, MatrixAdjustment):
    r"""Generalized Adjusted Count method.

    This class implements the Generalized Adjusted Count (GAC) algorithm for
    quantification adjustment as described in Firat (2016) [1]_. The method
    adjusts the estimated class prevalences by normalizing the confusion matrix
    based on prevalence estimates, providing a correction for bias caused by 
    distribution differences between training and test data.
    
    The confusion matrix is normalized by dividing each column by the prevalence 
    estimate of the corresponding class. For classes with zero estimated prevalence, 
    the diagonal element is set to 1 to avoid division by zero.

    This normalization ensures that the matrix best reflects the classifier's
    behavior relative to the estimated class distributions, improving quantification
    accuracy.

    Parameters
    ----------
    learner : estimator, optional
        Base classifier with `fit` and `predict` methods.

    Attributes
    ----------
    CM : ndarray of shape (n_classes, n_classes)
        Normalized confusion matrix used for adjusting predicted prevalences.
    classes_ : ndarray
        Array of class labels observed during training.


    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mlquantify.adjust_counting import GAC
    >>> import numpy as np
    >>> gac = GAC(learner=LogisticRegression())
    >>> X = np.random.randn(50, 4)
    >>> y = np.random.randint(0, 2, 50)
    >>> gac.fit(X, y)
    >>> gac.predict(X)
    {0: 0.5, 1: 0.5}

    References
    ----------
    .. [1] Firat, A. (2016). "Unified Framework for Quantification", 
           *Proceedings of AAAI Conference on Artificial Intelligence*, pp. 1-8.
    """
    def __init__(self, learner=None):
        super().__init__(learner=learner, solver='linear')
    
    def _compute_confusion_matrix(self, predictions, y_values):
        self.CM = confusion_matrix(y_values, predictions, labels=self.classes_).T
        self.CM = self.CM.astype(float)
        prev_estim = self.CM.sum(axis=0)

        for i, _ in enumerate(self.classes_):
            if prev_estim[i] == 0:
                self.CM[i, i] = 1
            else:
                self.CM[:, i] /= prev_estim[i]
        return self.CM


class GPAC(SoftLearnerQMixin, MatrixAdjustment):
    r"""Probabilistic Generalized Adjusted Count (GPAC) method.

    This class implements the probabilistic extension of the Generalized Adjusted Count method
    as presented in Firat (2016) [1]_. The GPAC method normalizes the confusion matrix by
    the estimated prevalences from posterior probabilities, enabling a probabilistic correction
    of class prevalences.

    The normalization divides each column of the confusion matrix by the estimated prevalence
    of the corresponding class. If a class has zero estimated prevalence, the diagonal element
    for that class is set to 1 to maintain matrix validity.
    
    GPAC extends the GAC approach by using soft probabilistic predictions (posterior probabilities)
    rather than crisp class labels, potentially improving quantification accuracy when 
    posterior probabilities are well calibrated.

    Parameters
    ----------
    learner : estimator, optional
        Base classifier with `fit` and `predict_proba` methods.

    Attributes
    ----------
    CM : ndarray of shape (n_classes, n_classes)
        Normalized confusion matrix used for adjustment.
    classes_ : ndarray
        Array of class labels observed during training.


    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mlquantify.adjust_counting import GPAC
    >>> import numpy as np
    >>> gpac = GPAC(learner=LogisticRegression())
    >>> X = np.random.randn(50, 4)
    >>> y = np.random.randint(0, 2, 50)
    >>> gpac.fit(X, y)
    >>> gpac.predict(X)
    {0: 0.5, 1: 0.5}

    References
    ----------
    .. [1] Firat, A. (2016). "Unified Framework for Quantification",
           *Proceedings of AAAI Conference on Artificial Intelligence*, pp. 1-8.
    """
    def __init__(self, learner=None):
        super().__init__(learner=learner, solver='linear')
    
    def _compute_confusion_matrix(self, posteriors, y_values):
        n_classes = len(self.classes_)
        confusion = np.eye(n_classes)

        for i, class_label in enumerate(self.classes_):
            indices = (y_values == class_label)
            if np.any(indices):
                confusion[i] = posteriors[indices].mean(axis=0)
        
        self.CM = confusion.T
        return self.CM


class ACC(ThresholdAdjustment):
    r"""Adjusted Count (ACC) — baseline threshold correction.

    This method corrects the bias in class prevalence estimates caused by imperfect 
    classification accuracy, by adjusting the observed positive count using estimates 
    of the classifier's true positive rate (TPR) and false positive rate (FPR).

    It uses a fixed classification threshold and applies the formula:

    .. math::

        p = \frac{p' - \text{FPR}}{\text{TPR} - \text{FPR}}

    where :math:`p'` is the observed positive proportion from :class:`CC`,
    
    
    Parameters
    ----------
    learner : estimator, optional
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


class X_method(ThresholdAdjustment):
    r"""X method — threshold where :math:`\text{TPR} + \text{FPR} = 1`.

    This method selects the classification threshold at which the sum of the true positive
    rate (TPR) and false positive rate (FPR) equals one. This threshold choice balances 
    errors in a specific way improving quantification.


    Parameters
    ----------
    learner : estimator, optional
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


class MAX(ThresholdAdjustment):
    r"""MAX method — threshold maximizing :math:`\text{TPR} - \text{FPR}`.

    This method selects the threshold that maximizes the difference between the true positive
    rate (TPR) and the false positive rate (FPR), effectively optimizing classification
    performance for quantification.


    Parameters
    ----------
    learner : estimator, optional
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
    learner : estimator, optional
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
    learner : estimator, optional
        A supervised learning model with `fit` and `predict_proba` methods.
    threshold : float, default=0.5
        Classification threshold in [0, 1] for applying in the :class:`CC` output.
    

    References
    ----------
    .. [1] Forman, G. (2008). "Quantifying Counts and Costs via Classification",
           *Data Mining and Knowledge Discovery*, 17(2), 164-206.
    """
    def _adjust(self, predictions, train_y_scores, train_y_values):
        positive_scores = train_y_scores[:, 1]
        
        thresholds, tprs, fprs = evaluate_thresholds(train_y_values, positive_scores)
        thresholds, tprs, fprs = self.get_best_threshold(thresholds, tprs, fprs)
        
        prevs = []
        for thr, tpr, fpr in zip(thresholds, tprs, fprs):
            cc_predictions = CC(threshold=thr).aggregate(predictions, train_y_values)
            cc_predictions = list(cc_predictions.values())[1]
            
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
    learner : estimator, optional
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
