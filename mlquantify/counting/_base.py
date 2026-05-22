import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier

from mlquantify.base_aggregative import (
    AggregativeMixin,
    _get_estimator,
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import check_classes_attribute, validate_predictions, validate_y, validate_data, validate_prevalences




class BaseCount(AggregativeMixin, BaseQuantifier):
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
    estimator : object, optional
        A supervised learning model implementing `fit` and `predict`
        or `predict_proba`.

    Attributes
    ----------
    estimator : object
        Underlying classification model.
    classes : ndarray of shape (n_classes,)
        Unique class labels observed during training.

    Examples
    --------
    >>> from mlquantify.base_count import BaseCount
    >>> from mlquantify.utils.validation import validate_prevalences
    >>> import numpy as np

    >>> class CC(CrispPredictionMixin, BaseCount):
    ...     def __init__(self, estimator=None, threshold=0.5):
    ...         super().__init__(estimator)
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
    >>> q = CC(estimator=LogisticRegression())
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
    def __init__(self, estimator=None):
        self.estimator = estimator
        
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = False
        tags.prediction_requirements.requires_train_labels = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, estimator_fitted=False, *args, **kwargs):
        """Fit the quantifier using the provided data and estimator."""
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)
        estimator = _get_estimator(self)
        if not estimator_fitted:
            estimator.fit(X, y, *args, **kwargs)
        self.estimator_ = estimator
        return self
    
    def predict(self, X):
        """Predict class prevalences for the given data."""
        predictions = self._predict_estimator(X)
        prevalences = self.aggregate(predictions)
        return prevalences
    
    @abstractmethod
    def aggregate(self, predictions):
        ...


class BaseAdjustCount(AggregativeMixin, BaseQuantifier):
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
    estimator : object, optional
        Supervised estimator implementing `fit` and (`predict` or `predict_proba`) depending on the quantifier.

    Attributes
    ----------
    estimator : object
        Underlying classification model.
    train_predictions : ndarray of shape (n_samples_train, n_classes)
        Predictions on training data from cross-validation.
    y_train : ndarray of shape (n_samples_train,)
        True labels corresponding to training predictions.
    classes : ndarray of shape (n_classes,)
        Unique class labels.

    Examples
    --------
    >>> from mlquantify.base_count import BaseAdjustCount
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> class ACC(CrispPredictionMixin, BaseAdjustCount):
    ...     def _adjust(self, preds, train_preds, y_train):
    ...         tpr = np.mean(train_preds[y_train == 1])
    ...         fpr = np.mean(train_preds[y_train == 0])
    ...         p_obs = np.mean(preds)
    ...         p_adj = (p_obs - fpr) / (tpr - fpr)
    ...         return np.clip([1 - p_adj, p_adj], 0, 1)
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> q = ACC(estimator=LogisticRegression())
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
    def __init__(self, estimator=None, cv=5, stratified=True, shuffle=False, random_state=None):
        self.estimator = estimator
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, estimator_fitted=False, cv_prediction="refit"):
        """Fit the quantifier using the provided data and estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            True labels.
        estimator_fitted : bool, optional
            If True, the estimator is already fitted, by default False.
        cv : int, optional
            Number of cross-validation folds, by default 5.
        stratified : bool, optional
            Whether to stratify the cross-validation, by default True.
        random_state : int, optional
            Random state for reproducibility, by default None.
        shuffle : bool, optional
            Whether to shuffle the data, by default False.
        
        Returns
        -------
        self : BaseAdjustCount
            Fitted quantifier.
        
        """
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)
        
        train_predictions, y_train = self._fit_estimator_predictions(
            X,
            y,
            estimator_fitted=estimator_fitted,
            cv=self.cv,
            stratified=self.stratified,
            random_state=self.random_state,
            shuffle=self.shuffle,
            cv_prediction=cv_prediction,
        )
        
        self.train_predictions = train_predictions
        self.y_train = y_train
        return self
    
    def predict(self, X):
        """Predict class prevalences for the given data."""
        X = validate_data(self, X)
        
        predictions = self._predict_estimator(X)
        
        prevalences = self.aggregate(predictions, self.train_predictions, self.y_train)
        return prevalences

    def aggregate(self, predictions, train_predictions, y_train):
        """Aggregate predictions and apply matrix- or rate-based bias correction. 
        
        Parameters
        ----------
        predictions : ndarray of shape (n_samples, n_classes)
            Estimator predictions on test data. Can be probabilities (n_samples, n_classes) or class labels (n_samples,).
        train_predictions : ndarray of shape (n_samples, n_classes)
            Estimator predictions on training data. Can be probabilities (n_samples, n_classes) or class labels (n_samples,).
        y_train : ndarray of shape (n_samples,)
            True class labels of the training data.
        
        Returns
        -------
        ndarray of shape (n_classes,)
            Class prevalence estimates.

        Examples
        --------
        >>> from mlquantify.counting import AC
        >>> import numpy as np
        >>> q = FM()
        >>> predictions = np.random.rand(200)
        >>> train_predictions = np.random.rand(200) # generated via cross-validation
        >>> y_train = np.random.randint(0, 2, 200)
        >>> q.aggregate(predictions, train_predictions, y_train)
        {0: 0.51, 1: 0.49}
        """
        self.classes_ = check_classes_attribute(self, np.unique(y_train))
        
        predictions = validate_predictions(self, predictions)
        train_predictions = validate_predictions(self, train_predictions)
        
        prevalences = self._adjust(predictions, train_predictions, y_train)
        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences
