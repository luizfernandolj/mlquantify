from copy import deepcopy
import numpy as np
from abc import abstractmethod
from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import get_aggregation_requirements
from mlquantify.utils._decorators import _fit_context
from mlquantify.base import BaseQuantifier, MetaquantifierMixin
from mlquantify.utils._validation import validate_prevalences, check_has_method


from copy import deepcopy
from itertools import combinations
import numpy as np
from abc import abstractmethod
from mlquantify.base import BaseQuantifier, MetaquantifierMixin
from mlquantify.base_aggregative import get_aggregation_requirements
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_prevalences, check_has_method


# ============================================================
# Decorator for enabling binary quantification behavior
# ============================================================
def define_binary(cls):
    """Decorator to enable binary quantification extensions (One-vs-Rest or One-vs-One).

    This decorator dynamically extends a quantifier class to handle multiclass
    quantification tasks by decomposing them into multiple binary subproblems,
    following either the One-vs-Rest (OvR) or One-vs-One (OvO) strategy.

    It automatically replaces the class methods `fit`, `predict`, and `aggregate`
    with binary-aware versions from `BinaryQuantifier`, while preserving access
    to the original implementations via `_original_fit`, `_original_predict`, 
    and `_original_aggregate`.

    Parameters
    ----------
    cls : class
        A subclass of `BaseQuantifier` implementing standard binary quantification
        methods (`fit`, `predict`, and `aggregate`).

    Returns
    -------
    class
        The same class with binary quantification capabilities added.

    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier
    >>> from mlquantify.binary import define_binary

    >>> @define_binary
    ... class MyQuantifier(BaseQuantifier):
    ...     def fit(self, X, y):
    ...         # Custom binary training logic
    ...         self.classes_ = np.unique(y)
    ...         return self
    ...
    ...     def predict(self, X):
    ...         # Return dummy prevalences
    ...         return np.array([0.4, 0.6])
    ...
    ...     def aggregate(self, preds, y_train):
    ...         # Example aggregation method
    ...         return np.mean(preds, axis=0)

    >>> qtf = MyQuantifier()
    >>> qtf.strategy = 'ovr'  # or 'ovo'
    >>> X = np.random.randn(10, 5)
    >>> y = np.random.randint(0, 3, 10)
    >>> qtf.fit(X, y)
    MyQuantifier(...)
    >>> qtf.predict(X)
    array([...])
    """
    if check_has_method(cls, "fit"):
        cls._original_fit = cls.fit
    if check_has_method(cls, "predict"):
        cls._original_predict = cls.predict
    if check_has_method(cls, "aggregate"):
        cls._original_aggregate = cls.aggregate

    cls.fit = BinaryQuantifier.fit
    cls.predict = BinaryQuantifier.predict
    cls.aggregate = BinaryQuantifier.aggregate

    return cls


# ============================================================
# Fitting strategies
# ============================================================
def _fit_ovr(quantifier, X, y):
    """Fit using One-vs-Rest (OvR) strategy.

    Creates a binary quantifier for each class, trained to distinguish that class
    versus all others.

    Parameters
    ----------
    quantifier : BaseQuantifier
        The quantifier instance being trained.
    X : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y : array-like of shape (n_samples,)
        Class labels.

    Returns
    -------
    dict
        A mapping from class label to fitted binary quantifier.
    """
    quantifiers = {}
    for cls in np.unique(y):
        qtf = deepcopy(quantifier)
        y_bin = (y == cls).astype(int)
        qtf._original_fit(X, y_bin)
        quantifiers[cls] = qtf
    return quantifiers


def _fit_ovo(quantifier, X, y):
    """Fit using One-vs-One (OvO) strategy.

    Creates a binary quantifier for every pair of classes, trained to distinguish
    one class from another.

    Parameters
    ----------
    quantifier : BaseQuantifier
        The quantifier instance being trained.
    X : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y : array-like of shape (n_samples,)
        Class labels.

    Returns
    -------
    dict
        A mapping from (class1, class2) tuples to fitted binary quantifiers.
    """
    quantifiers = {}
    for cls1, cls2 in combinations(np.unique(y), 2):
        qtf = deepcopy(quantifier)
        mask = (y == cls1) | (y == cls2)
        y_bin = (y[mask] == cls1).astype(int)
        qtf._original_fit(X[mask], y_bin)
        quantifiers[(cls1, cls2)] = qtf
    return quantifiers


# ============================================================
# Prediction strategies
# ============================================================
def _predict_ovr(quantifier, X):
    """Predict using One-vs-Rest (OvR) strategy.

    Each binary quantifier produces a prevalence estimate for its corresponding class.

    Parameters
    ----------
    quantifier : BinaryQuantifier
        Fitted quantifier containing binary models.
    X : array-like of shape (n_samples, n_features)
        Test feature matrix.

    Returns
    -------
    np.ndarray
        Predicted prevalences for each class.
    """
    preds = np.zeros(len(quantifier.qtfs_))
    for i, qtf in enumerate(quantifier.qtfs_.values()):
        preds[i] = qtf._original_predict(X)[1]
    return preds


def _predict_ovo(quantifier, X):
    """Predict using One-vs-One (OvO) strategy.

    Each binary quantifier outputs a prevalence estimate for the pair of classes it was trained on.

    Parameters
    ----------
    quantifier : BinaryQuantifier
        Fitted quantifier containing binary models.
    X : array-like of shape (n_samples, n_features)
        Test feature matrix.

    Returns
    -------
    np.ndarray
        Pairwise prevalence predictions.
    """
    preds = np.zeros(len(quantifier.qtfs_))
    for i, (cls1, cls2) in enumerate(combinations(quantifier.qtfs_.keys(), 2)):
        qtf = quantifier.qtfs_[(cls1, cls2)]
        preds[i] = qtf._original_predict(X)[1]
    return preds


# ============================================================
# Aggregation strategies
# ============================================================
def _aggregate_ovr(quantifier, preds, y_train, train_preds=None):
    """Aggregate binary predictions using One-vs-Rest (OvR).

    Parameters
    ----------
    quantifier : BinaryQuantifier
        Quantifier performing the aggregation.
    preds : ndarray of shape (n_samples, n_classes)
        Model predictions.
    y_train : ndarray of shape (n_samples,)
        Training labels.
    train_preds : ndarray of shape (n_samples, n_classes), optional
        Predictions on the training set.

    Returns
    -------
    dict
        Class-wise prevalence estimates.
    """
    prevalences = {}
    for i, cls in enumerate(np.unique(y_train)):
        bin_preds = np.column_stack([1 - preds[:, i], preds[:, i]])
        y_bin = (y_train == cls).astype(int)
        args = [bin_preds]

        if train_preds is not None:
            bin_train_preds = np.column_stack([1 - train_preds[:, i], train_preds[:, i]])
            args.append(bin_train_preds)

        args.append(y_bin)
        prevalences[cls] = quantifier._original_aggregate(*args)[1]
    return prevalences


def _aggregate_ovo(quantifier, preds, y_train, train_preds=None):
    """Aggregate binary predictions using One-vs-One (OvO).

    Parameters
    ----------
    quantifier : BinaryQuantifier
        Quantifier performing the aggregation.
    preds : ndarray
        Model predictions.
    y_train : ndarray
        Training labels.
    train_preds : ndarray, optional
        Predictions on the training set.

    Returns
    -------
    dict
        Pairwise prevalence estimates.
    """
    prevalences = {}
    for cls1, cls2 in combinations(np.unique(y_train), 2):
        bin_preds = np.column_stack([1 - preds[:, (cls1, cls2)], preds[:, (cls1, cls2)]])
        mask = (y_train == cls1) | (y_train == cls2)
        y_bin = (y_train[mask] == cls1).astype(int)

        args = [bin_preds]
        if train_preds is not None:
            bin_train_preds = np.column_stack([1 - train_preds[:, (cls1, cls2)], train_preds[:, (cls1, cls2)]])
            args.append(bin_train_preds)

        args.append(y_bin)
        prevalences[(cls1, cls2)] = quantifier._original_aggregate(*args)[1]
    return prevalences


# ============================================================
# Main Class
# ============================================================
class BinaryQuantifier(MetaquantifierMixin, BaseQuantifier):
    """Meta-quantifier enabling One-vs-Rest and One-vs-One strategies.

    This class extends a base quantifier to handle multiclass problems by 
    decomposing them into binary subproblems. It automatically delegates fitting, 
    prediction, and aggregation operations to the appropriate binary quantifiers.

    Attributes
    ----------
    qtfs_ : dict
        Dictionary mapping class labels or label pairs to fitted binary quantifiers.
    strategy : {'ovr', 'ovo'}
        Defines how multiclass quantification is decomposed.
    """

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(qtf, X, y):
        """Fit the quantifier under a binary decomposition strategy."""
        if len(np.unique(y)) <= 2:
            qtf.binary = True
            return qtf._original_fit(X, y)

        qtf.strategy = getattr(qtf, "strategy", "ovr")

        if qtf.strategy == "ovr":
            qtf.qtfs_ = _fit_ovr(qtf, X, y)
        elif qtf.strategy == "ovo":
            qtf.qtfs_ = _fit_ovo(qtf, X, y)
        else:
            raise ValueError("Strategy must be 'ovr' or 'ovo'")

        return qtf

    def predict(qtf, X):
        """Predict class prevalences using the trained binary quantifiers."""
        if hasattr(qtf, "binary") and qtf.binary:
            return qtf._original_predict(X)
        else:
            if qtf.strategy == "ovr":
                preds = _predict_ovr(qtf, X)
            elif qtf.strategy == "ovo":
                preds = _predict_ovo(qtf, X)
            else:
                raise ValueError("Strategy must be 'ovr' or 'ovo'")

        return validate_prevalences(qtf, preds, qtf.qtfs_.keys())

    def aggregate(qtf, *args):
        """Aggregate binary predictions to obtain multiclass prevalence estimates."""
        requirements = get_aggregation_requirements(qtf)

        if requirements.requires_train_proba and requirements.requires_train_labels:
            preds, train_preds, y_train = args
            args_dict = dict(preds=preds, train_preds=train_preds, y_train=y_train)
        elif requirements.requires_train_labels:
            preds, y_train = args
            args_dict = dict(preds=preds, y_train=y_train)
        else:
            raise ValueError("Binary aggregation requires at least train labels")

        classes = np.unique(args_dict["y_train"])
        qtf.strategy = getattr(qtf, "strategy", "ovr")

        if hasattr(qtf, "binary") and qtf.binary:
            return qtf._original_aggregate(*args_dict.values())

        if qtf.strategy == "ovr":
            prevalences = _aggregate_ovr(qtf, **args_dict)
        elif qtf.strategy == "ovo":
            prevalences = _aggregate_ovo(qtf, **args_dict)
        else:
            raise ValueError("Strategy must be 'ovr' or 'ovo'")

        return validate_prevalences(qtf, prevalences, classes)
