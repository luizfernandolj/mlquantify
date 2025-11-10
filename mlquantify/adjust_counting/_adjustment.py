import numpy as np     
from abc import abstractmethod
from scipy.optimize import minimize
import warnings

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
    r"""
    Applies threshold-based adjustment methods for quantification.

    This is the base class for methods such as ACC, X, MAX, T50, MS, and MS2, 
    which adjust prevalence estimates based on the classifier’s ROC curve, as proposed by 
    Forman (2005, 2008).

    These methods correct the bias in *Classify & Count (CC)* estimates caused by differences
    in class distributions between the training and test datasets.

    Mathematical formulation

    Given:
    - \( p' \): observed positive proportion from CC,
    - \( \text{TPR} = P(\hat{y}=1|y=1) \),
    - \( \text{FPR} = P(\hat{y}=1|y=0) \),

    the adjusted prevalence is given by:

    \[
    \hat{p} = \frac{p' - \text{FPR}}{\text{TPR} - \text{FPR}}
    \]

    (Forman, *Counting Positives Accurately Despite Inaccurate Classification*, ECML 2005;
     *Quantifying Counts and Costs via Classification*, DMKD 2008).


    Notes
    -----
    - Defined only for binary quantification tasks.
    - When applied to multiclass problems, the one-vs-rest strategy (`ovr`) is used automatically.


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


    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mlquantify.adjust_counting import ThresholdAdjustment
    >>> import numpy as np
    >>> class CustomThreshold(ThresholdAdjustment):
    ...     def _get_best_threshold(self, thresholds, tprs, fprs):
    ...         idx = np.argmax(tprs - fprs)
    ...         return thresholds[idx], tprs[idx], fprs[idx]
    >>> X = np.random.randn(100, 4)
    >>> y = np.random.randint(0, 2, 100)
    >>> q = CustomThreshold(learner=LogisticRegression())
    >>> q.fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}
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
        
        thresholds, tprs, fprs = evaluate_thresholds(train_y_values, positive_scores, self.classes_)
        threshold, tpr, fpr = self._get_best_threshold(thresholds, tprs, fprs)

        cc_predictions = CC(threshold).aggregate(predictions)[1]

        if tpr - fpr == 0:
            prevalence = cc_predictions
        else:
            prevalence = np.clip((cc_predictions - fpr) / (tpr - fpr), 0, 1)
        
        return np.asarray([1 - prevalence, prevalence])
    
    @abstractmethod
    def _get_best_threshold(self, thresholds, tprs, fprs):
        """Select the best threshold according to the specific method."""
        ...


class MatrixAdjustment(BaseAdjustCount):
    r"""
    Base class for matrix-based quantification adjustments (FM, GAC, GPAC).

    This class implements the matrix correction model for quantification
    as formulated in Firat (2016), which expresses the observed prevalences as
    a linear combination of true prevalences through the confusion matrix.

    Mathematical model

    The system is given by:

    \[
    \mathbf{y} = \mathbf{C}\hat{\pi}_F + \varepsilon
    \]
    
    subject to:
    
    \[
    \hat{\pi}_F \ge 0, \quad \sum_k \hat{\pi}_{F,k} = 1
    \]

    where:
    - \( \mathbf{y} \): vector of predicted prevalences in test set,
    - \( \mathbf{C} \): confusion matrix,
    - \( \hat{\pi}_F \): true class prevalence vector (unknown),
    - \( \varepsilon \): residual error.

    The model can be solved either via:
    - Linear algebraic solution, or
    - Constrained optimization (quadratic or least-squares).


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


    References
    ----------
    - Firat, A. (2016). *Unified Framework for Quantification.* AAAI, pp. 1-8.


    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mlquantify.adjust_counting import MatrixAdjustment
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
    """

    _parameter_constraints = {"solver": Options(["optim", "linear"])}

    def __init__(self, learner=None, solver=None):
        super().__init__(learner=learner)
        self.solver = solver
    
    def _adjust(self, predictions, train_y_pred, train_y_values):
        n_class = len(np.unique(train_y_values))
        self.CM = np.zeros((n_class, n_class))

        if self.solver == 'optim':
            priors = np.array(list(CC().aggregate(train_y_pred).values()))
            self.CM = self._compute_confusion_matrix(train_y_pred, train_y_values, priors)
            prevs_estim = self._get_estimations(predictions > priors)
            prevalence = self._solve_optimization(prevs_estim, priors)
        else:
            self.CM = self._compute_confusion_matrix(train_y_pred)
            prevs_estim = self._get_estimations(predictions)
            prevalence = self._solve_linear(prevs_estim)
        
        return prevalence

    def _solve_linear(self, prevs_estim):
        r"""
        Solve the system linearly:

        \[
        \hat{\pi}_F = \mathbf{C}^{-1} \mathbf{p}
        \]
        """
        try:
            adjusted = np.linalg.solve(self.CM, prevs_estim)
            adjusted = np.clip(adjusted, 0, 1)
            adjusted /= adjusted.sum()
        except np.linalg.LinAlgError:
            adjusted = prevs_estim
        return adjusted

    def _solve_optimization(self, prevs_estim, priors):
        r"""
        Solve via constrained least squares:

        \[
        \min_{\hat{\pi}_F} \| \mathbf{C}\hat{\pi}_F - \mathbf{p} \|_2^2
        \quad \text{s.t. } \hat{\pi}_F \ge 0, \ \sum_k \hat{\pi}_{F,k} = 1
        \]
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

    def _get_estimations(self, predictions):
        """Return prevalence estimates using CC (crisp) or PCC (probabilistic)."""
        if uses_soft_predictions(self):
            return np.array(list(PCC().aggregate(predictions).values()))
        return np.array(list(CC().aggregate(predictions).values()))

    @abstractmethod
    def _compute_confusion_matrix(self, predictions, *args):
        ...


class FM(SoftLearnerQMixin, MatrixAdjustment):
    """Forman's Matrix Adjustment (FM) — solved via optimization."""
    def __init__(self, learner=None):
        super().__init__(learner=learner, solver='optim')
    
    def _compute_confusion_matrix(self, posteriors, y_true, priors):
        for i, _class in enumerate(self.classes_):
            indices = (y_true == _class)
            self.CM[:, i] = self._get_estimations(posteriors[indices] > priors)
        return self.CM


class GAC(CrispLearnerQMixin, MatrixAdjustment):
    """Gonzalez-Castro’s Generalized Adjusted Count (GAC) method."""
    def __init__(self, learner=None):
        super().__init__(learner=learner, solver='linear')
    
    def _compute_confusion_matrix(self, predictions):
        prev_estim = self._get_estimations(predictions)
        for i, _ in enumerate(self.classes_):
            if prev_estim[i] == 0:
                self.CM[i, i] = 1
            else:
                self.CM[:, i] /= prev_estim[i]
        return self.CM


class GPAC(SoftLearnerQMixin, MatrixAdjustment):
    """Probabilistic GAC (GPAC) — soft version using posterior probabilities."""
    def __init__(self, learner=None):
        super().__init__(learner=learner, solver='linear')
    
    def _compute_confusion_matrix(self, posteriors):
        prev_estim = self._get_estimations(posteriors)
        for i, _ in enumerate(self.classes_):
            if prev_estim[i] == 0:
                self.CM[i, i] = 1
            else:
                self.CM[:, i] /= prev_estim[i]
        return self.CM


class ACC(ThresholdAdjustment):
    """Adjusted Count (ACC) — baseline threshold correction."""
    def _get_best_threshold(self, thresholds, tprs, fprs):
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)


class X_method(ThresholdAdjustment):
    """X method — threshold where \( \text{TPR} + \text{FPR} = 1 \)."""
    def _get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmin(np.abs(1 - (tprs + fprs)))
        return thresholds[idx], tprs[idx], fprs[idx]


class MAX(ThresholdAdjustment):
    r"""MAX method — threshold maximizing \( \text{TPR} - \text{FPR} \)."""
    def _get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmax(np.abs(tprs - fprs))
        return thresholds[idx], tprs[idx], fprs[idx]


class T50(ThresholdAdjustment):
    r"""T50 — selects threshold where \( \text{TPR} = 0.5 \)."""
    def _get_best_threshold(self, thresholds, tprs, fprs):
        idx = np.argmin(np.abs(tprs - 0.5))
        return thresholds[idx], tprs[idx], fprs[idx]


class MS(ThresholdAdjustment):
    r"""Median Sweep (MS) — median prevalence across all thresholds."""
    def _adjust(self, predictions, train_y_scores, train_y_values):
        positive_scores = train_y_scores[:, 1]
        
        thresholds, tprs, fprs = evaluate_thresholds(train_y_values, positive_scores, self.classes_)
        thresholds, tprs, fprs = self._get_best_threshold(thresholds, tprs, fprs)
        
        prevs = []
        for thr, tpr, fpr in zip(thresholds, tprs, fprs):
            cc_predictions = CC(thr).aggregate(predictions)
            cc_predictions = cc_predictions[1]
            prevalence = cc_predictions if tpr - fpr == 0 else (cc_predictions - fpr) / (tpr - fpr)
            prevs.append(prevalence)
        prevalence = np.median(prevs)
        return np.asarray([1 - prevalence, prevalence])
    
    def _get_best_threshold(self, thresholds, tprs, fprs):
        return thresholds, tprs, fprs


class MS2(MS):
    r"""MS2 — Median Sweep variant with constraint \( |\text{TPR} - \text{FPR}| > 0.25 \)."""
    def _get_best_threshold(self, thresholds, tprs, fprs):
        if np.all(tprs == 0) or np.all(fprs == 0):
            warnings.warn("All TPR or FPR values are zero.")
        indices = np.where(np.abs(tprs - fprs) > 0.25)[0]
        if len(indices) == 0:
            warnings.warn("No cases satisfy |TPR - FPR| > 0.25.")
            indices = np.where(np.abs(tprs - fprs) >= 0)[0]
        return thresholds[indices], tprs[indices], fprs[indices]
