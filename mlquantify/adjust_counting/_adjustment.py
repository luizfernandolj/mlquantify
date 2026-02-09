from mlquantify.utils._validation import validate_prevalences
from mlquantify.base import BaseQuantifier
from mlquantify.config import config_context
import numpy as np     
from abc import abstractmethod
from scipy.optimize import minimize
import warnings
from sklearn.metrics import confusion_matrix
from mlquantify.adjust_counting._base import BaseAdjustCount
from mlquantify.adjust_counting._counting import CC, PCC
from mlquantify.utils import (
    _fit_context, 
    validate_data,
    validate_prevalences,
    validate_predictions,
    check_classes_attribute
)
from mlquantify.base_aggregative import (
    CrispLearnerQMixin,
    SoftLearnerQMixin,
    AggregationMixin,
    uses_soft_predictions, 
    _get_learner_function
)
from mlquantify.multiclass import define_binary
from mlquantify.utils._optimization import _optimize_on_simplex
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

    def __init__(self, learner=None, threshold=0.5, strategy="ovr", n_jobs=None):
        super().__init__(learner=learner)
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

    _parameter_constraints = {
        "solver": Options(["minimize", "linear", None]),
        "method": Options(["inverse", "invariant_ratio"])
    }

    def __init__(self, learner=None, solver=None, method='inverse'):
        super().__init__(learner=learner)
        self.solver = solver
        self.method = method
    
    def _adjust(self, predictions, train_predictions, y_train):
        n_class = len(self.classes_)
        self.CM = np.zeros((n_class, n_class))
        if self.solver == 'minimize':
            class_counts = np.array([np.count_nonzero(y_train == _class) for _class in self.classes_])
            priors = class_counts / len(y_train)
            self.CM = self._compute_confusion_matrix(train_predictions, y_train, priors)
            prevs_estim = self._get_estimations(predictions > priors, y_train)
        else:
            self.CM = self._compute_confusion_matrix(train_predictions, y_train)
            prevs_estim = self._get_estimations(predictions, y_train)

        try:
            prevalence = self._solve_adjustment(self.CM, prevs_estim)
        except:
            prevalence = prevs_estim

        prevalence = np.clip(prevalence, 0, 1)
        prevalence = validate_prevalences(self, prevalence, self.classes_)
        
        return prevalence

    def _solve_adjustment(self, confusion_matrix, prevs_estim):
        A = confusion_matrix
        B = prevs_estim
        
        if self.method == 'inverse':
            pass
        elif self.method == 'invariant_ratio':
            A[-1, :] = 1.0
            B[-1] = 1.0

        if self.solver == 'minimize':
            def objective(prevs_pred):
                return np.linalg.norm(A @ prevs_pred - B)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x}
            ]

            prevalence, self.loss_ = _optimize_on_simplex(objective, len(self.classes_), constraints)

            return prevalence
            
        elif self.solver == 'linear':
            try:
                adjusted = np.linalg.solve(A, B)
                return adjusted
            except np.linalg.LinAlgError:
                return B

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
        r"""Solve the system using constrained optimization.

        The solution is obtained by minimizing the discrepancy:

            || C @ \hat{\pi}_F - p ||

        subject to the constraints:

            \hat{\pi}_F \ge 0,  sum_k \hat{\pi}_{F,k} = 1

        where:
            - C is the confusion matrix,
            - p is the observed prevalence vector from the test set.

        Parameters
        ----------
        prevs_estim : ndarray of shape (n_classes,)
            Observed prevalence vector from the test set.
        priors : ndarray of shape (n_classes,)
            Fallback class prior vector used if optimization fails.

        Returns
        -------
        ndarray of shape (n_classes,)
            Adjusted prevalence estimates \hat{\pi}_F.
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

    def _get_estimations(self, predictions, y_train):
        """Return prevalence estimates using CC (crisp) or PCC (probabilistic)."""
        with config_context(prevalence_return_type="array"):
            if uses_soft_predictions(self):
                return np.asarray(PCC().aggregate(predictions))
            return np.asarray(CC().aggregate(predictions, y_train))

    @abstractmethod
    def _compute_confusion_matrix(self, predictions, *args):
        ...



@define_binary
class CDE(SoftLearnerQMixin, AggregationMixin, BaseQuantifier):
    r"""CDE-Iterate for binary classification prevalence estimation.

    Threshold :math:`\tau` from false positive and false negative costs:

    .. math::

        \tau = \frac{c_{FP}}{c_{FP} + c_{FN}}

    Hard classification by thresholding posterior probability :math:`p(+|x)` at :math:`\tau`:

    .. math::

        \hat{y}(x) = \mathbf{1}_{p(+|x) > \tau}

    Prevalence estimation via classify-and-count:

    .. math::

        \hat{p}_U(+) = \frac{1}{N} \sum_{n=1}^N \hat{y}(x_n)

    False positive cost update:
    
    .. math::

        c_{FP}^{new} = \frac{p_L(+)}{p_L(-)} \times \frac{\hat{p}_U(-)}{\hat{p}_U(+)} \times c_{FN}

    Parameters
    ----------
    learner : estimator, optional
        Wrapped classifier (unused).
    tol : float, default=1e-4
        Convergence tolerance.
    max_iter : int, default=100
        Max iterations.
    init_cfp : float, default=1.0
        Initial false positive cost.

    References
    ----------
    .. [1] Esuli, A., Moreo, A., & Sebastiani, F. (2023). Learning to Quantify. Springer.
    """

    _parameter_constraints = {
        "tol": [Interval(0, None, inclusive_left=False)],
        "max_iter": [Interval(1, None, inclusive_left=True)],
        "init_cfp": [Interval(0, None, inclusive_left=False)]
    }

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = False
        return tags


    def __init__(self, learner=None, tol=1e-4, max_iter=100, init_cfp=1.0, strategy="ovr", n_jobs=None):
        self.learner = learner
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.init_cfp = float(init_cfp) 
        self.strategy = strategy
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)
        self.learner.fit(X, y)
        counts = np.array([np.count_nonzero(y == _class) for _class in self.classes_])
        self.priors = counts / len(y)
        self.y_train = y
                
        return self


    def predict(self, X):
        """Predict class prevalences for the given data."""
        predictions = getattr(self.learner, _get_learner_function(self))(X)
        prevalences = self.aggregate(predictions, self.y_train)
        return prevalences


    def aggregate(self, predictions, y_train):
        """Aggregate predictions and apply matrix- or rate-based bias correction. 
        
        Parameters
        ----------
        predictions : ndarray of shape (n_samples, n_classes)
            Learner predictions on test data. Can be probabilities (n_samples, n_classes) or class labels (n_samples,).
        y_train : ndarray of shape (n_samples,)
            True class labels of the training data.
        
        Returns
        -------
        ndarray of shape (n_classes,)
            Class prevalence estimates.

        Examples
        --------
        >>> from mlquantify.adjust_counting import CDE
        >>> import numpy as np
        >>> q = CDE()
        >>> predictions = np.random.rand(200)
        >>> train_predictions = np.random.rand(200) # generated via cross-validation
        >>> y_train = np.random.randint(0, 2, 200)
        >>> q.aggregate(predictions, train_predictions, y_train)
        {0: 0.51, 1: 0.49}
        """

        self.classes_ = check_classes_attribute(self, np.unique(y_train))
        predictions = validate_predictions(self, predictions)

        if hasattr(self, 'priors'):
            Ptr = np.asarray(self.priors, dtype=np.float64)
        else:
            counts = np.array([np.count_nonzero(y_train == _class) for _class in self.classes_])
            Ptr = counts / len(y_train)

        P = np.asarray(predictions, dtype=np.float64)

        # ensure no zeros
        eps = 1e-12
        P = np.clip(P, eps, 1.0)

        # training priors pL(+), pL(-)
        # assume Ptr order matches columns of P; if Ptr sums to 1 but order unknown, user must match.
        pL_pos = Ptr[1]
        pL_neg = Ptr[0]
        if pL_pos <= 0 or pL_neg <= 0:
            # keep them positive to avoid divisions by zero
            pL_pos = max(pL_pos, eps)
            pL_neg = max(pL_neg, eps)

        # initialize costs
        cFN = 1.0
        cFP = float(self.init_cfp)

        prev_prev_pos = None
        s = 0

        # iterate: compute threshold from costs, classify, estimate prevalences via CC,
        # update cFP, repeat
        while s < self.max_iter:
            # decision threshold tau for positive class:
            # Derivation:
            # predict positive if cost_FP * p(-|x) < cost_FN * p(+|x)
            # => predict positive if p(+|x) / p(-|x) > cost_FP / cost_FN
            # since p(+|x) / p(-|x) = p(+|x) / (1 - p(+|x)):
            # p(+|x) > cost_FP / (cost_FP + cost_FN)
            tau = cFP / (cFP + cFN)

            # hard predictions for positive class using threshold on posterior for positive (col 1)
            pos_probs = P[:, 1]
            hard_pos = (pos_probs > tau).astype(float)

            # classify-and-count prevalence estimate on U
            prev_pos = hard_pos.mean()
            prev_neg = 1.0 - prev_pos

            # update cFP according to:
            # cFP_new = (pL_pos / pL_neg) * (pU_hat(neg) / pU_hat(pos)) * cFN
            # guard against zero prev_pos / prev_neg
            prev_pos_safe = max(prev_pos, eps)
            prev_neg_safe = max(prev_neg, eps)

            cFP_new = (pL_pos / pL_neg) * (prev_neg_safe / prev_pos_safe) * cFN

            # check convergence on prevalences (absolute change)
            if prev_prev_pos is not None and abs(prev_pos - prev_prev_pos) < self.tol:
                break

            # prepare next iter
            cFP = cFP_new
            prev_prev_pos = prev_pos
            s += 1

        # if didn't converge within max_iter we keep last estimate (lack of fisher consistency)
        if s >= self.max_iter:
            # optional: warning
            # print('[warning] CDE-Iterate reached max_iter without converging')
            pass

        prevalences = np.array([prev_neg, prev_pos], dtype=np.float64)
        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences


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
        If None, it is expected that user will use the `aggregate` method directly.

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
        super().__init__(learner=learner, solver='minimize')
    
    def _compute_confusion_matrix(self, predictions, y_true, priors):
        n_classes = len(self.classes_)
        self.CM = np.zeros((n_classes, n_classes))

        for i, _class in enumerate(self.classes_):
            indices = (y_true == _class)
            preds_sub = predictions[indices]

            mask = preds_sub > priors               # (n_i, n_classes)
            masked = np.where(mask, preds_sub, -np.inf)
            best_classes = np.argmax(masked, axis=1)

            hard_preds = np.zeros_like(preds_sub, dtype=bool)
            rows = np.arange(preds_sub.shape[0])
            hard_preds[rows, best_classes] = True

            self.CM[:, i] = self._get_estimations(hard_preds, y_true[indices])
        
        return self.CM



class AC(CrispLearnerQMixin, MatrixAdjustment):
    r"""Adjusted Count method.

    This class implements the Adjusted Count (AC) algorithm for
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
    >>> from mlquantify.adjust_counting import AC
    >>> import numpy as np
    >>> ac = AC(learner=LogisticRegression())
    >>> X = np.random.randn(50, 4)
    >>> y = np.random.randint(0, 2, 50)
    >>> ac.fit(X, y)
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


class PAC(SoftLearnerQMixin, MatrixAdjustment):
    r"""Probabilistic Adjusted Count (PAC) method.

    This class implements the probabilistic extension of the Adjusted Count method
    as presented in Firat (2016) [1]_. The PAC method normalizes the confusion matrix by
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
    >>> from mlquantify.adjust_counting import PAC
    >>> import numpy as np
    >>> pac = PAC(learner=LogisticRegression())
    >>> X = np.random.randn(50, 4)
    >>> y = np.random.randint(0, 2, 50)
    >>> pac.fit(X, y)
    >>> pac.predict(X)
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


class TX(ThresholdAdjustment):
    r"""Threshold X method — threshold where :math:`\text{TPR} + \text{FPR} = 1`.

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


class TMAX(ThresholdAdjustment):
    r"""Threshold MAX method — threshold maximizing :math:`\text{TPR} - \text{FPR}`.

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
