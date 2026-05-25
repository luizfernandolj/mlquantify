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

    Provides ``fit`` and ``predict`` for :class:`~mlquantify.counting.CC` and
    :class:`~mlquantify.counting.PCC`. Subclasses must implement :meth:`aggregate`.

    Parameters
    ----------
    estimator : estimator, optional
        A supervised learning model with ``fit`` and ``predict`` or
        ``predict_proba``, depending on the subclass.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying estimator.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.counting._base import BaseCount
    >>> from mlquantify.base_aggregative import CrispPredictionMixin
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> class MyCC(CrispPredictionMixin, BaseCount):
    ...     def __init__(self, estimator=None):
    ...         self.estimator = estimator or LogisticRegression()
    ...     def aggregate(self, predictions, y_train=None):
    ...         classes, counts = np.unique(predictions, return_counts=True)
    ...         return dict(zip(classes, counts / len(predictions)))
    >>> X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
    >>> MyCC().fit(X, y).predict(X)
    {0: 0.47, 1: 0.53}
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
        r"""Fit the quantifier using the provided data and estimator."""
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)
        estimator = _get_estimator(self)
        if not estimator_fitted:
            estimator.fit(X, y, *args, **kwargs)
        self.estimator_ = estimator
        return self
    
    def predict(self, X):
        r"""Predict class prevalences for the given data."""
        predictions = self._predict_estimator(X)
        prevalences = self.aggregate(predictions)
        return prevalences
    
    @abstractmethod
    def aggregate(self, predictions):
        ...


class BaseAdjustCount(AggregativeMixin, BaseQuantifier):
    r"""Base class for adjustment-based quantifiers.

    Fits the underlying estimator with cross-validation to produce
    out-of-fold predictions (``train_predictions``) used for bias
    correction in :meth:`aggregate`. Subclasses must implement
    :meth:`_adjust`.

    Parameters
    ----------
    estimator : estimator, optional
        A supervised learning model.
    cv : int, default=5
        Number of cross-validation folds.
    stratified : bool, default=True
        Whether to use stratified CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying estimator.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    train_predictions : ndarray
        Cross-validated predictions on the training data.
    y_train : ndarray of shape (n_samples,)
        Labels corresponding to ``train_predictions``.

    Examples
    --------
    >>> from mlquantify.counting._base import BaseAdjustCount
    >>> from mlquantify.base_aggregative import CrispPredictionMixin
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> class MyAdjusted(CrispPredictionMixin, BaseAdjustCount):
    ...     def __init__(self, estimator=None):
    ...         self.estimator = estimator or LogisticRegression()
    ...         self.cv = 5
    ...         self.stratified = True
    ...         self.shuffle = False
    ...         self.random_state = None
    ...     def _adjust(self, predictions, train_predictions, y_train):
    ...         tpr = np.mean(train_predictions[y_train == 1] > 0.5)
    ...         fpr = np.mean(train_predictions[y_train == 0] > 0.5)
    ...         p = np.mean(predictions > 0.5)
    ...         p_adj = np.clip((p - fpr) / max(tpr - fpr, 1e-8), 0, 1)
    ...         return np.array([1 - p_adj, p_adj])
    >>> X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
    >>> MyAdjusted().fit(X, y).predict(X)
    {0: 0.48, 1: 0.52}
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
        r"""Fit the quantifier using the provided data and estimator.
        
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
        r"""Predict class prevalences for the given data."""
        X = validate_data(self, X)
        
        predictions = self._predict_estimator(X)
        
        prevalences = self.aggregate(predictions, self.train_predictions, self.y_train)
        return prevalences

    def aggregate(self, predictions, train_predictions, y_train):
        r"""Aggregate predictions and apply matrix- or rate-based bias correction. 
        
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
