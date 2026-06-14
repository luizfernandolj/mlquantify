import numpy as np

from mlquantify.base_aggregative import (
    SoftPredictionMixin,
    CrispPredictionMixin
)

from mlquantify.counting._base import BaseCount
from mlquantify.utils._validation import validate_predictions, validate_prevalences, check_classes_attribute, resolve_aggregate_classes
from mlquantify.utils._constraints import Interval
        


class CC(CrispPredictionMixin, BaseCount):
    """Classify and Count (CC) quantifier.

    Targets prior probability shift. Estimates class prevalences by
    classifying each test instance with a hard classifier and counting the
    fraction assigned to each class. It is the simplest quantification
    baseline and is systematically biased whenever the test class
    distribution differs from the training one.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict`` methods.
        If ``None``, call :meth:`aggregate` directly with pre-computed predictions.
    threshold : float, default=0.5
        Decision threshold applied to soft scores to produce hard labels.
        Must be in ``[0.0, 1.0]``.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    CC applies no correction for classifier error, so its absolute error grows
    roughly linearly as the test prevalence departs from the training
    prevalence. Contrast with :class:`PCC`, which averages soft posteriors, and
    :class:`ACC`, which corrects for the true- and false-positive rates.

    See Also
    --------
    PCC : Probabilistic Classify and Count quantifier.
    ACC : Adjusted Classify and Count quantifier.

    Examples
    --------
    >>> from mlquantify.counting import CC
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
    >>> q = CC(LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.47, 1: 0.53}
    >>> # call aggregate directly with pre-computed hard predictions
    >>> preds = q.estimator_.predict(X)
    >>> q.aggregate(preds, classes=[0, 1])
    {0: 0.47, 1: 0.53}

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


    def __init__(self, estimator=None, threshold=0.5):
        super().__init__(estimator=estimator)
        self.threshold = threshold

    def aggregate(self, predictions, classes=None):
        """Aggregate predictions into class prevalence estimates.

        Parameters
        ----------
        predictions : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Estimator predictions on test data. Can be probabilities
            (n_samples, n_classes) or class labels (n_samples,).
        classes : array-like of shape (n_classes,) or None, default=None
            Class labels the output must report, in order. When given, every
            class appears in the result even if absent from ``predictions``
            (with prevalence 0). When ``None``, the classes seen during ``fit``
            are used; if the quantifier is unfitted, they are inferred from the
            predictions.

        Returns
        -------
        ndarray of shape (n_classes,)
            Class prevalence estimates.

        Examples
        --------
        >>> from mlquantify.counting import CC
        >>> import numpy as np
        >>> q = CC()
        >>> predictions = np.random.rand(200)
        >>> q.aggregate(predictions)
        {0: 0.51, 1: 0.49}
        """
        # Convert soft scores to crisp labels using the known class set (the
        # requested ``classes`` or the fitted ``classes_``); only then infer the
        # output classes, falling back to the crisp predictions when unfitted.
        label_source = classes if classes is not None else getattr(self, "classes_", None)
        predictions = validate_predictions(self, predictions, self.threshold, label_source)
        self.classes_ = resolve_aggregate_classes(
            self, classes, getattr(self, "classes_", None), predictions
        )
        class_counts = np.array([np.count_nonzero(predictions == _class) for _class in self.classes_])
        prevalences = class_counts / len(predictions)

        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences


class PCC(SoftPredictionMixin, BaseCount):
    """Probabilistic Classify and Count (PCC) quantifier.

    Targets prior probability shift. Estimates class prevalences by averaging
    the posterior probabilities returned by a probabilistic classifier over
    all test instances. Generally less biased than :class:`CC`, but, lacking
    any error correction, it still drifts under strong shift and assumes a
    well-calibrated classifier.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
        If ``None``, call :meth:`aggregate` directly.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    PCC is unbiased under prior shift only when the posteriors are well
    calibrated on the test distribution; calibration drift re-introduces bias.
    It applies no explicit correction for classifier error.

    See Also
    --------
    CC : Classify and Count quantifier.
    ACC : Adjusted Classify and Count quantifier.
    GPACC : Multiclass, corrected generalisation of PCC.

    Examples
    --------
    >>> from mlquantify.counting import PCC
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
    >>> q = PCC(LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.48, 1: 0.52}
    >>> # call aggregate directly with pre-computed posterior probabilities
    >>> proba = q.estimator_.predict_proba(X)
    >>> q.aggregate(proba)
    {0: 0.48, 1: 0.52}

    References
    ----------
    .. dropdown:: References

        .. [1] Bella, A., Ferri, C., Hernández-Orallo, J., & Ramírez-Quintana, M. J. (2010).
               Quantification via Probability Estimators. *ICDM*, pp. 737–742.
        .. [2] Forman, G. (2005). Counting Positives Accurately Despite Inaccurate Classification. *ECML*, pp. 564–575.
    """


    def __init__(self, estimator=None):
        super().__init__(estimator=estimator)

    def aggregate(self, predictions, classes=None):
        """Aggregate predictions into class prevalence estimates.

        Parameters
        ----------
        predictions : ndarray of shape (n_samples, n_classes)
            Estimator predictions on test data. Can be probabilities
            (n_samples, n_classes) or class labels (n_samples,).
        classes : array-like of shape (n_classes,) or None, default=None
            Class labels the output must report, in order. When given, every
            class appears in the result even if absent from ``predictions``
            (with prevalence 0). When ``None``, the classes seen during ``fit``
            are used; if the quantifier is unfitted, they are inferred from the
            predictions (categorical) or their width (probabilities).

        Returns
        -------
        ndarray of shape (n_classes,)
            Class prevalence estimates.

        Examples
        --------
        >>> from mlquantify.counting import PCC
        >>> import numpy as np
        >>> q = PCC()
        >>> predictions = np.random.rand(200, 2)
        >>> q.aggregate(predictions)
        {0: 0.50, 1: 0.50}
        """
        predictions = validate_predictions(self, predictions)

        # Handle categorical predictions (1D array with class labels)
        if predictions.ndim == 1 and not np.issubdtype(predictions.dtype, (np.floating, np.integer)):
            self.classes_ = resolve_aggregate_classes(
                self, classes, getattr(self, "classes_", None), predictions
            )
            class_counts = np.array([np.count_nonzero(predictions == _class) for _class in self.classes_])
            prevalences = class_counts / len(predictions)
        else:
            # Handle probability predictions (2D array or 1D probabilities).
            # The column count fixes the number of classes; ``classes`` only
            # relabels them.
            width = predictions.shape[1] if predictions.ndim == 2 else 2
            self.classes_ = resolve_aggregate_classes(
                self, classes, getattr(self, "classes_", None), np.arange(width)
            )
            prevalences = np.mean(predictions, axis=0) if predictions.ndim == 2 else predictions.mean()
            if predictions.ndim == 1:
                prevalences = np.array([1-prevalences, prevalences])

        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences
