from copy import deepcopy
import numpy as np
import functools
from inspect import signature
from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import get_aggregation_requirements
from mlquantify.utils._decorators import _fit_context
from mlquantify.base import BaseQuantifier, MetaquantifierMixin
from mlquantify.utils._validation import validate_prevalences, check_has_method


from copy import deepcopy
from itertools import combinations
import numpy as np
from abc import abstractmethod
from joblib import Parallel, delayed
from mlquantify.base import BaseQuantifier, MetaquantifierMixin
from mlquantify.base_aggregative import get_aggregation_requirements
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_prevalences, check_has_method


# ============================================================
# OvR binary wrapper for pre-fitted multiclass estimators
# ============================================================
class _OvRBinaryWrapper:
    """Wraps a multiclass estimator to present a binary interface for one OvR class.

    When a pre-fitted multiclass estimator is reused across OvR sub-problems,
    ``predict_proba`` returns ``n_classes`` columns.  This wrapper selects the
    column for the target class and returns a proper 2-column binary output, so
    all downstream quantifiers (histogram, threshold, etc.) work correctly.

    Parameters
    ----------
    estimator : fitted sklearn-compatible classifier
        The original multiclass estimator.
    cls_idx : int
        Index of the target class in ``estimator.classes_`` (= column in
        ``predict_proba``).
    """

    def __init__(self, estimator, cls_idx):
        self.estimator = estimator
        self.cls_idx = cls_idx
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)
        pos = proba[:, self.cls_idx]
        return np.column_stack([1 - pos, pos])

    def predict(self, X):
        # A sample is positive only if cls is the argmax class,
        # consistent with how a multiclass classifier makes hard predictions.
        proba = self.estimator.predict_proba(X)
        return (np.argmax(proba, axis=1) == self.cls_idx).astype(int)


# ============================================================
# Decorator for enabling binary quantification behavior
# ============================================================
def binary_quantifier(_cls=None, *, strategy_attr="strategy"):
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
    strategy_attr : str, default="strategy"
        Name of the attribute that stores the multiclass decomposition strategy.

    Returns
    -------
    class
        The same class with binary quantification capabilities added.

    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier
    >>> from mlquantify.multiclass import binary_quantifier

    >>> @binary_quantifier(strategy_attr="strategy")
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
    {0: 0.33333333333333337, 1: 0.33333333333333337, 2: 0.33333333333333337}
    """
    def decorate(cls):
        cls._binary_strategy_attr = strategy_attr

        if check_has_method(cls, "fit"):
            cls._original_fit = cls.fit
            cls.fit = functools.wraps(cls._original_fit)(BinaryQuantifier.fit)
            cls.fit.__doc__ = cls._original_fit.__doc__
            cls.fit.__signature__ = signature(cls._original_fit)

        if check_has_method(cls, "predict"):
            cls._original_predict = cls.predict
            cls.predict = functools.wraps(cls._original_predict)(BinaryQuantifier.predict)
            cls.predict.__doc__ = cls._original_predict.__doc__
            cls.predict.__signature__ = signature(cls._original_predict)

        if check_has_method(cls, "aggregate"):
            cls._original_aggregate = cls.aggregate
            cls.aggregate = functools.wraps(cls._original_aggregate)(BinaryQuantifier.aggregate)
            cls.aggregate.__doc__ = cls._original_aggregate.__doc__
            cls.aggregate.__signature__ = signature(cls._original_aggregate)

        cls.fit_predict = BinaryQuantifier.fit_predict
        return cls

    if _cls is not None:
        return decorate(_cls)
    return decorate


def define_binary(cls=None):
    """Backward-compatible alias for ``binary_quantifier(strategy_attr="strategy")``."""
    return binary_quantifier(cls, strategy_attr="strategy")


def _get_binary_strategy(quantifier):
    strategy_attr = getattr(quantifier, "_binary_strategy_attr", "strategy")
    strategy = getattr(quantifier, strategy_attr, "ovr")
    setattr(quantifier, strategy_attr, strategy)
    return strategy


# ============================================================
# Fitting strategies
# ============================================================
# ============================================================
# Fitting strategies
# ============================================================
def _fit_binary_ovr(quantifier, X, y, cls, fit_args=None, fit_kwargs=None):
    """Helper function to fit a single binary quantifier for OvR."""
    if fit_args is None:
        fit_args = ()
    if fit_kwargs is None:
        fit_kwargs = {}

    qtf = deepcopy(quantifier)
    y_bin = (y == cls).astype(int)

    # When the estimator is pre-fitted as a multiclass classifier,
    # predict_proba returns n_classes columns.  Wrap it so every downstream
    # quantifier sees a proper 2-column binary interface for this OvR class.
    fit_kwargs = dict(fit_kwargs)  # don't mutate the caller's dict
    if fit_kwargs.get("estimator_fitted", False) and hasattr(qtf, "estimator"):
        cls_idx = int(np.searchsorted(np.unique(y), cls))
        qtf.estimator = _OvRBinaryWrapper(qtf.estimator, cls_idx)

    qtf._original_fit(X, y_bin, *fit_args, **fit_kwargs)
    return cls, qtf


def _fit_ovr(quantifier, X, y, n_jobs=None, fit_args=None, fit_kwargs=None):
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
    n_jobs : int, optional
        The number of jobs to run in parallel. None means 1.

    Returns
    -------
    dict
        A mapping from class label to fitted binary quantifier.
    """
    # Parallel execution for each class
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_binary_ovr)(
            quantifier,
            X,
            y,
            cls,
            fit_args=fit_args,
            fit_kwargs=fit_kwargs,
        )
        for cls in np.unique(y)
    )
    return dict(results)


def _fit_binary_ovo(quantifier, X, y, cls1, cls2, fit_args=None, fit_kwargs=None):
    """Helper function to fit a single binary quantifier for OvO."""
    if fit_args is None:
        fit_args = ()
    if fit_kwargs is None:
        fit_kwargs = {}

    qtf = deepcopy(quantifier)
    mask = (y == cls1) | (y == cls2)
    y_bin = (y[mask] == cls1).astype(int)

    fit_kwargs = dict(fit_kwargs)
    sample_weight = fit_kwargs.get("sample_weight")
    if sample_weight is not None and len(sample_weight) == len(y):
        fit_kwargs["sample_weight"] = np.asarray(sample_weight)[mask]

    qtf._original_fit(X[mask], y_bin, *fit_args, **fit_kwargs)
    return (cls1, cls2), qtf


def _fit_ovo(quantifier, X, y, n_jobs=None, fit_args=None, fit_kwargs=None):
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
    n_jobs : int, optional
        The number of jobs to run in parallel. None means 1.

    Returns
    -------
    dict
        A mapping from (class1, class2) tuples to fitted binary quantifiers.
    """
    # Parallel execution for each pair of classes
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_binary_ovo)(
            quantifier,
            X,
            y,
            cls1,
            cls2,
            fit_args=fit_args,
            fit_kwargs=fit_kwargs,
        )
        for cls1, cls2 in combinations(np.unique(y), 2)
    )
    return dict(results)


# ============================================================
# Prediction strategies
# ============================================================

# ============================================================
# Prediction strategies
# ============================================================
def _predict_binary_ovr(qtf, i, X):
    """Helper function to predict using a single binary quantifier for OvR."""
    return i, qtf._original_predict(X)[1]


def _predict_ovr(quantifier, X, n_jobs=None):
    """Predict using One-vs-Rest (OvR) strategy.

    Each binary quantifier produces a prevalence estimate for its corresponding class.

    Parameters
    ----------
    quantifier : BinaryQuantifier
        Fitted quantifier containing binary models.
    X : array-like of shape (n_samples, n_features)
        Test feature matrix.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    np.ndarray
        Predicted prevalences for each class.
    """
    qtfs = list(quantifier.qtfs_.values())
    results = Parallel(n_jobs=n_jobs)(
        delayed(_predict_binary_ovr)(qtf, i, X) 
        for i, qtf in enumerate(qtfs)
    )
    
    # Sort results by index to ensure correct order
    results.sort(key=lambda x: x[0])
    preds = np.array([res[1] for res in results])
    return preds


def _predict_binary_ovo(qtf, i, X):
    """Helper function to predict using a single binary quantifier for OvO."""
    return i, qtf._original_predict(X)[1]


def _predict_ovo(quantifier, X, n_jobs=None):
    """Predict using One-vs-One (OvO) strategy.

    Each binary quantifier outputs a prevalence estimate for the pair of classes it was trained on.

    Parameters
    ----------
    quantifier : BinaryQuantifier
        Fitted quantifier containing binary models.
    X : array-like of shape (n_samples, n_features)
        Test feature matrix.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    np.ndarray
        Pairwise prevalence predictions.
    """
    qtfs = []
    classes = getattr(quantifier, 'classes_', None)
    if classes is None:
         # Fallback if classes_ not set (should not happen if fitted)
         # Try to infer from keys
         flat_keys = [item for sublist in quantifier.qtfs_.keys() for item in sublist]
         classes = np.unique(flat_keys)

    pairs = list(combinations(classes, 2))
    tasks = []
    for i, (cls1, cls2) in enumerate(pairs):
        qtf = quantifier.qtfs_.get((cls1, cls2))
        if qtf is None:
             qtf = quantifier.qtfs_.get((cls2, cls1))
        
        if qtf:
            tasks.append((i, qtf))
        else:
            # Should not happen if fitted correctly
            pass

    results = Parallel(n_jobs=n_jobs)(
        delayed(_predict_binary_ovo)(qtf, i, X) 
        for i, qtf in tasks
    )
    
    results.sort(key=lambda x: x[0])
    preds = np.array([res[1] for res in results])
    return preds


def _ovo_pairwise_to_class_prevalences(pair_preds, classes):
    class_sums = {cls: 0.0 for cls in classes}
    class_counts = {cls: 0 for cls in classes}

    for (cls1, cls2), prev_cls1 in pair_preds.items():
        class_sums[cls1] += float(prev_cls1)
        class_counts[cls1] += 1
        class_sums[cls2] += float(1.0 - prev_cls1)
        class_counts[cls2] += 1

    return {
        cls: (class_sums[cls] / class_counts[cls]) if class_counts[cls] else 0.0
        for cls in classes
    }

# ============================================================
# Aggregation strategies
# ============================================================


def _aggregate_binary_ovr(qtf, i, cls, preds, train_preds, y_train):
    """Helper for OvR aggregation."""
    bin_preds = np.column_stack([1 - preds[:, i], preds[:, i]])
    y_bin = (y_train == cls).astype(int)
    args = [bin_preds]

    if train_preds is not None:
        bin_train_preds = np.column_stack([1 - train_preds[:, i], train_preds[:, i]])
        args.append(bin_train_preds)

    args.append(y_bin)
    return cls, qtf._original_aggregate(*args)[1]


def _aggregate_ovr(quantifier, preds, y_train, train_preds=None, n_jobs=None):
    """Aggregate binary predictions using One-vs-Rest (OvR).
    
    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs.
    """
    classes = np.unique(y_train)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_aggregate_binary_ovr)(
            quantifier.qtfs_[cls], i, cls, preds, train_preds, y_train
        )
        for i, cls in enumerate(classes)
    )
    
    return dict(results)


def _aggregate_binary_ovo(qtf, cls1, cls2, preds, train_preds, y_train):
    """Helper for OvO aggregation."""
    bin_preds = np.column_stack([1 - preds[:, (cls1, cls2)], preds[:, (cls1, cls2)]])
    mask = (y_train == cls1) | (y_train == cls2)
    y_bin = (y_train[mask] == cls1).astype(int)

    args = [bin_preds]
    if train_preds is not None:
        bin_train_preds = np.column_stack([1 - train_preds[:, (cls1, cls2)], train_preds[:, (cls1, cls2)]])
        args.append(bin_train_preds)

    args.append(y_bin)
    return (cls1, cls2), qtf._original_aggregate(*args)[1]


def _aggregate_ovo(quantifier, preds, y_train, train_preds=None, n_jobs=None):
    """Aggregate binary predictions using One-vs-One (OvO).
    
    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(_aggregate_binary_ovo)(
            quantifier.qtfs_[(cls1, cls2)], cls1, cls2, preds, train_preds, y_train
        )
        for cls1, cls2 in combinations(np.unique(y_train), 2)
    )
    return dict(results)



# ============================================================
# Fit-Predict strategies (Per-core fit & predict, no save)
# ============================================================
def _fit_predict_binary_ovr(quantifier, X, y, cls, X_test):
    """Helper for OvR fit_predict."""
    qtf = deepcopy(quantifier)
    y_bin = (y == cls).astype(int)
    qtf._original_fit(X, y_bin)
    
    pred = qtf._original_predict(X_test)[1]
    return cls, pred


def _fit_predict_binary_ovo(quantifier, X, y, cls1, cls2, X_test):
    """Helper for OvO fit_predict."""
    qtf = deepcopy(quantifier)
    mask = (y == cls1) | (y == cls2)
    y_bin = (y[mask] == cls1).astype(int)
    qtf._original_fit(X[mask], y_bin)
    
    pred = qtf._original_predict(X_test)[1]
    return (cls1, cls2), pred


def _fit_predict_ovr(quantifier, X, y, X_test, n_jobs=None):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_binary_ovr)(quantifier, X, y, cls, X_test) 
        for cls in np.unique(y)
    )
    # Sort results to match classes order for array construction
    return dict(results)


def _fit_predict_ovo(quantifier, X, y, X_test, n_jobs=None):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_binary_ovo)(quantifier, X, y, cls1, cls2, X_test) 
        for cls1, cls2 in combinations(np.unique(y), 2)
    )
    return dict(results)


# ============================================================
# Strategy registry
# ============================================================
class MulticlassStrategy:
    """Base class for multiclass decomposition strategies.

    A strategy turns a binary quantifier into a multiclass one: it decomposes the
    problem, fits/predicts the binary sub-quantifiers, and recombines their
    outputs into per-class prevalences. Register new strategies (e.g. ECOC,
    hierarchical) with :func:`register_strategy` and they become usable through
    the quantifier's ``strategy`` attribute with no change to the dispatch.

    Subclasses implement :meth:`fit`, :meth:`predict`, :meth:`aggregate` and
    :meth:`fit_predict`, returning prevalences *before* the shared
    ``validate_prevalences`` normalization applied by :class:`BinaryQuantifier`.
    """

    name = None

    def fit(self, quantifier, X, y, n_jobs=None, fit_args=None, fit_kwargs=None):
        """Return ``{key: fitted_binary_quantifier}`` for the decomposition."""
        raise NotImplementedError

    def predict(self, quantifier, X, n_jobs=None):
        """Return per-class prevalences (dict or array) for ``X``."""
        raise NotImplementedError

    def aggregate(self, quantifier, classes, args_dict, n_jobs=None):
        """Return per-class prevalences from pre-computed predictions."""
        raise NotImplementedError

    def fit_predict(self, quantifier, X, y, X_test, classes, n_jobs=None):
        """Fit on ``(X, y)`` and return per-class prevalences for ``X_test``."""
        raise NotImplementedError


_STRATEGIES = {}


def register_strategy(name):
    """Register a :class:`MulticlassStrategy` subclass under ``name``.

    The decorated class is instantiated once and stored in the registry, so it
    can be selected via a quantifier's ``strategy`` attribute (e.g.
    ``strategy='ovr'``).
    """
    def decorator(cls):
        cls.name = name
        _STRATEGIES[name] = cls()
        return cls
    return decorator


def get_strategy(name):
    """Return the registered :class:`MulticlassStrategy` instance for ``name``."""
    try:
        return _STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown multiclass strategy {name!r}. "
            f"Available strategies: {sorted(_STRATEGIES)}."
        ) from None


def available_strategies():
    """Return the sorted names of all registered multiclass strategies."""
    return sorted(_STRATEGIES)


@register_strategy("ovr")
class OvRStrategy(MulticlassStrategy):
    """One-vs-Rest: one binary quantifier per class, recombined by class."""

    def fit(self, quantifier, X, y, n_jobs=None, fit_args=None, fit_kwargs=None):
        return _fit_ovr(quantifier, X, y, n_jobs=n_jobs,
                        fit_args=fit_args, fit_kwargs=fit_kwargs)

    def predict(self, quantifier, X, n_jobs=None):
        return _predict_ovr(quantifier, X, n_jobs=n_jobs)

    def aggregate(self, quantifier, classes, args_dict, n_jobs=None):
        return _aggregate_ovr(quantifier, n_jobs=n_jobs, **args_dict)

    def fit_predict(self, quantifier, X, y, X_test, classes, n_jobs=None):
        return _fit_predict_ovr(quantifier, X, y, X_test, n_jobs=n_jobs)


@register_strategy("ovo")
class OvOStrategy(MulticlassStrategy):
    """One-vs-One: one binary quantifier per class pair, recombined pairwise."""

    def fit(self, quantifier, X, y, n_jobs=None, fit_args=None, fit_kwargs=None):
        return _fit_ovo(quantifier, X, y, n_jobs=n_jobs,
                        fit_args=fit_args, fit_kwargs=fit_kwargs)

    def predict(self, quantifier, X, n_jobs=None):
        preds = _predict_ovo(quantifier, X, n_jobs=n_jobs)
        pairs = list(combinations(quantifier.classes_, 2))
        pair_preds = dict(zip(pairs, preds))
        return _ovo_pairwise_to_class_prevalences(pair_preds, quantifier.classes_)

    def aggregate(self, quantifier, classes, args_dict, n_jobs=None):
        pair_prev = _aggregate_ovo(quantifier, n_jobs=n_jobs, **args_dict)
        return _ovo_pairwise_to_class_prevalences(pair_prev, classes)

    def fit_predict(self, quantifier, X, y, X_test, classes, n_jobs=None):
        preds_dict = _fit_predict_ovo(quantifier, X, y, X_test, n_jobs=n_jobs)
        return _ovo_pairwise_to_class_prevalences(preds_dict, classes)


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
    def fit(qtf, X, y, *args, **kwargs):
        """Fit the quantifier under a binary decomposition strategy."""
        if len(np.unique(y)) <= 2:
            qtf.binary = True
            return qtf._original_fit(X, y, *args, **kwargs)

        strategy = get_strategy(_get_binary_strategy(qtf))
        n_jobs = getattr(qtf, "n_jobs", None)

        qtf.binary = False
        qtf.classes_ = np.unique(y)
        qtf.qtfs_ = strategy.fit(
            qtf, X, y, n_jobs=n_jobs, fit_args=args, fit_kwargs=kwargs,
        )

        return qtf

    def predict(qtf, X):
        """Predict class prevalences using the trained binary quantifiers."""
        if hasattr(qtf, "binary") and qtf.binary:
            return qtf._original_predict(X)

        strategy = get_strategy(_get_binary_strategy(qtf))
        n_jobs = getattr(qtf, "n_jobs", None)
        preds = strategy.predict(qtf, X, n_jobs=n_jobs)

        return validate_prevalences(qtf, preds, qtf.classes_)

    def aggregate(qtf, *args, classes=None):
        """Aggregate binary predictions to obtain multiclass prevalence estimates."""
        requirements = get_aggregation_requirements(qtf)

        if requirements.requires_train_proba and requirements.requires_train_labels:
            preds, train_preds, y_train = args
            args_dict = dict(preds=preds, train_preds=train_preds, y_train=y_train)
        elif requirements.requires_train_labels:
            preds, y_train = args
            args_dict = dict(preds=preds, y_train=y_train)
        else:
            # CC-style aggregation: only the test predictions are needed.
            (preds,) = args
            args_dict = dict(preds=preds)

        if classes is not None:
            resolved = np.unique(classes)
        elif "y_train" in args_dict:
            resolved = np.unique(args_dict["y_train"])
        else:
            preds_arr = np.asarray(args_dict["preds"])
            resolved = (
                np.arange(preds_arr.shape[1])
                if preds_arr.ndim == 2
                else np.unique(preds_arr)
            )
        n_jobs = getattr(qtf, "n_jobs", None)

        if (hasattr(qtf, "binary") and qtf.binary) or len(resolved) <= 2:
            return qtf._original_aggregate(*args_dict.values(), classes=classes)

        if "y_train" not in args_dict:
            raise ValueError(
                "Aggregating more than two classes from predictions alone is "
                "ambiguous for a binary quantifier; pass the training labels "
                "or use predict()."
            )

        strategy = get_strategy(_get_binary_strategy(qtf))
        prevalences = strategy.aggregate(qtf, resolved, args_dict, n_jobs=n_jobs)

        return validate_prevalences(qtf, prevalences, resolved)

    def fit_predict(qtf, X, y, X_test):
        """Fit and predict class prevalences without storing models."""
        classes = np.unique(y)
        if len(classes) <= 2:
            # For binary case, standard fit+predict logic (or call underlying if supported)
            # Default to sequential fit then predict since logic is internal
            qtf.fit(X, y)
            return qtf.predict(X_test)

        strategy = get_strategy(_get_binary_strategy(qtf))
        n_jobs = getattr(qtf, "n_jobs", None)

        # Set classes_ attribute required by validate_prevalences
        qtf.classes_ = classes

        prevalences = strategy.fit_predict(qtf, X, y, X_test, classes, n_jobs=n_jobs)
        return validate_prevalences(qtf, prevalences, classes)
