import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier

from mlquantify.base_aggregative import (
    AggregationMixin,
    _get_learner_function
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import check_classes_attribute, validate_predictions, validate_y, validate_data, validate_prevalences
from mlquantify.utils._get_scores import apply_cross_validation




class BaseCount(AggregationMixin, BaseQuantifier):
    r"""Base class for count-based quantifiers.

    Implements the foundation for *count-based quantification* methods,
    where class prevalences are estimated directly from classifier outputs
    without any correction.

    The method assumes a classifier :math:`f(x)` producing either hard or
    probabilistic predictions. The prevalence of each class :math:`c` in
    the unlabeled test set is estimated as:

    .. math::
        \hat{\pi}_c = \frac{1}{N} \sum_{i=1}^{N} I(f(x_i) = c)

    for *hard* classifiers, or equivalently as:

    .. math::
        \hat{\pi}_c = \frac{1}{N} \sum_{i=1}^{N} f_c(x_i)

    for *soft* classifiers where :math:`f_c(x)` denotes the posterior
    probability of class :math:`c`.

    This is the classical *Classify and Count (CC)* and *Probabilistic
    Classify and Count (PCC)* approach, introduced by Forman (2005, 2008)
    and unified in the constrained regression framework of Firat et al. (2016).

    Parameters
    ----------
    learner : object, optional
        A supervised learning model implementing `fit` and `predict`
        or `predict_proba`.

    Attributes
    ----------
    learner : object
        Underlying classification model.
    classes : ndarray of shape (n_classes,)
        Unique class labels observed during training.

    Examples
    --------
    >>> from mlquantify.base_count import BaseCount
    >>> from mlquantify.utils.validation import validate_prevalences
    >>> import numpy as np

    >>> class CC(CrispLearnerQMixin, BaseCount):
    ...     def __init__(self, learner=None, threshold=0.5):
    ...         super().__init__(learner)
    ...         self.threshold = threshold
    ...     def aggregate(self, predictions):
    ...         predictions = validate_predictions(self, predictions)
    ...         self.classes = self.classes if hasattr(self, 'classes') else np.unique(predictions)
    ...         counts = np.array([np.count_nonzero(predictions == c) for c in self.classes])
    ...         prevalences = counts / len(predictions)
    ...         return validate_prevalences(self, prevalences, self.classes)

    >>> from sklearn.linear_model import LogisticRegression
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> q = CC(learner=LogisticRegression())
    >>> q.fit(X, y)
    >>> q.predict(X).round(3)
    array([0.47, 0.53])

    References
    ----------
    [1] Forman, G. (2005). *Counting Positives Accurately Despite Inaccurate Classification.*
        ECML, pp. 564-575.
    [2] Forman, G. (2008). *Quantifying Counts and Costs via Classification.*
        Data Mining and Knowledge Discovery, 17(2), 164-206.
    """

    @abstractmethod
    def __init__(self, learner=None):
        self.learner = learner
        
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = False
        tags.prediction_requirements.requires_train_labels = False
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, *args, **kwargs):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, X, y)
        validate_y(self, y)
        self.classes_ = np.unique(y)
        if not learner_fitted:
            self.learner.fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X):
        """Predict class prevalences for the given data."""
        estimator_function = _get_learner_function(self)
        predictions = getattr(self.learner, estimator_function)(X)
        prevalences = self.aggregate(predictions)
        return prevalences
    
    @abstractmethod
    def aggregate(self, predictions):
        """Aggregate predictions into class prevalence estimates."""
        ...


class BaseAdjustCount(AggregationMixin, BaseQuantifier):
    r"""Base class for adjustment-based quantifiers.

    This class generalizes *adjusted count* quantification methods,
    providing a framework for correcting bias in raw classifier outputs
    based on estimated confusion matrices or rate statistics.

    Following Forman (2005, 2008), in the binary case the correction
    uses true positive (TPR) and false positive (FPR) rates to adjust
    the observed positive proportion :math:`\hat{p}'_{+}`:

    .. math::
        \hat{p}_{+} = \frac{\hat{p}'_{+} - \text{FPR}}{\text{TPR} - \text{FPR}}

    In the multiclass extension (Firat et al., 2016), the same principle
    can be expressed using matrix algebra. Let :math:`C` denote the
    normalized confusion matrix where :math:`C_{ij} = P(\hat{y}=i|y=j)`
    estimated via cross-validation. Then, given the observed distribution
    of predictions :math:`\hat{\pi}'`, the corrected prevalence vector
    :math:`\hat{\pi}` is obtained as:

    .. math::
        \hat{\pi}' = C \hat{\pi}
        \quad \Rightarrow \quad
        \hat{\pi} = C^{-1} \hat{\pi}'

    subject to non-negativity and unit-sum constraints:

    .. math::
        \hat{\pi}_c \ge 0, \quad \sum_c \hat{\pi}_c = 1

    This formulation can be solved via constrained least squares
    (L2), least absolute deviation (L1), or Hellinger divergence
    minimization, as discussed by Firat et al. (2016).

    Parameters
    ----------
    learner : object, optional
        Supervised learner implementing `fit`, `predict`, or `predict_proba`.

    Attributes
    ----------
    learner : object
        Underlying classification model.
    train_predictions : ndarray of shape (n_samples_train, n_classes)
        Predictions on training data from cross-validation.
    train_y_values : ndarray of shape (n_samples_train,)
        True labels corresponding to training predictions.
    classes : ndarray of shape (n_classes,)
        Unique class labels.

    Examples
    --------
    >>> from mlquantify.base_count import BaseAdjustCount
    >>> import numpy as np

    >>> class ACC(CrispLearnerQMixin, BaseAdjustCount):
    ...     def _adjust(self, preds, train_preds, y_train):
    ...         tpr = np.mean(train_preds[y_train == 1])
    ...         fpr = np.mean(train_preds[y_train == 0])
    ...         p_obs = np.mean(preds)
    ...         p_adj = (p_obs - fpr) / (tpr - fpr)
    ...         return np.clip([1 - p_adj, p_adj], 0, 1)

    >>> from sklearn.linear_model import LogisticRegression
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> q = ACC(learner=LogisticRegression())
    >>> q.fit(X, y)
    >>> q.predict(X).round(3)
    array([0.52, 0.48])

    References
    ----------
    [1] Forman, G. (2005). *Counting Positives Accurately Despite Inaccurate Classification.*
        ECML 2005, LNAI 3720, pp. 564-575.
    [2] Forman, G. (2008). *Quantifying Counts and Costs via Classification.*
        Data Mining and Knowledge Discovery, 17(2), 164-206.
    [3] Firat, A. (2016). *Unified Framework for Quantification.*
        Proceedings of the AAAI Conference on Artificial Intelligence, Sections 3.2-3.3.
    """

    @abstractmethod
    def __init__(self, learner=None):
        self.learner = learner

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, X, y)
        validate_y(self, y)
        self.classes_ = np.unique(y)
        learner_function = _get_learner_function(self)
        
        if learner_fitted:
            train_predictions = getattr(self.learner, learner_function)(X)
            y_train_labels = y
        else:
            train_predictions, y_train_labels = apply_cross_validation(
                self.learner,
                X,
                y,
                function=learner_function,
                cv=5,
                stratified=True,
                random_state=None,
                shuffle=True
            )
        
        self.train_predictions = train_predictions
        self.train_y_values = y_train_labels
        return self
    
    def predict(self, X):
        """Predict class prevalences for the given data."""
        predictions = getattr(self.learner, _get_learner_function(self))(X)
        prevalences = self.aggregate(predictions, self.train_predictions, self.train_y_values)
        return prevalences

    def aggregate(self, predictions, train_predictions, y_train_values):
        """Aggregate predictions and apply matrix- or rate-based bias correction."""
        self.classes_ = check_classes_attribute(self, np.unique(y_train_values))
        predictions = validate_predictions(self, train_predictions)
        prevalences = self._adjust(predictions, train_predictions, y_train_values)
        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences
