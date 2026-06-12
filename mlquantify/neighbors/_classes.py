import numpy as np
from mlquantify.utils._constraints import Interval, Options
from mlquantify.neighbors._classification import PWKCLF
from mlquantify.base import BaseQuantifier
from mlquantify.utils._decorators import _fit_context
from mlquantify.counting import CC
from mlquantify.utils import validate_data
from mlquantify.utils._validation import validate_prevalences


class PWK(BaseQuantifier):
    r"""Probabilistic Weighted k-Nearest Neighbour (PWK) quantifier.

    Targets prior probability shift. A classify-and-count quantifier built on a
    nearest-neighbour classifier whose votes are re-weighted to undo training
    class imbalance: each neighbour's vote is scaled by a class-specific weight
    (controlled by ``alpha``), so the count is not dominated by the majority
    class. Native multiclass; classifier-free in the sense of needing no
    external estimator.

    Parameters
    ----------
    alpha : float, default=1
        Imbalance-correction exponent. ``1`` applies the standard inverse-size
        weighting; higher values further amplify minority-class neighbours.
    n_neighbors : int, default=10
        Number of nearest neighbours considered for each test instance.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Neighbour-search algorithm.

        - ``'auto'`` : pick the best of the below from the fitted data (default).
        - ``'ball_tree'`` : ball-tree index; good in higher dimensions.
        - ``'kd_tree'`` : k-d tree index; fast in low dimensions.
        - ``'brute'`` : exhaustive search; exact, best for small data.
    metric : str, default='euclidean'
        Distance metric for the neighbour search.
    leaf_size : int, default=30
        Leaf size for the tree-based algorithms (speed/memory trade-off).
    p : int, default=2
        Power parameter for the Minkowski metric (``1`` = Manhattan, ``2`` = Euclidean).
    metric_params : dict or None, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int or None, default=None
        Number of parallel jobs for the neighbour search.

    Attributes
    ----------
    estimator : PWKCLF
        Fitted underlying weighted k-NN classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    PWK is a classify-and-count method whose only quantification-specific
    ingredient is the imbalance re-weighting; it needs no separate scorer and
    handles multiclass directly, but inherits k-NN's sensitivity to feature
    scaling and dimensionality.

    See Also
    --------
    CC : Plain classify-and-count baseline.
    ACC : Adjusted count for binary prior shift.

    Examples
    --------
    >>> from mlquantify.neighbors import PWK
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = PWK(alpha=1.5, n_neighbors=5).fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}

    References
    ----------
    .. dropdown:: References

        .. [1] Barranquero, J., Díez, J., & del Coz, J. J. (2013).
               Quantification-Oriented Learning Based on Reliable Classifiers.
               *Pattern Recognition*, 48(2), 591–604.
    """
    
    _parameter_constraints = {
        "alpha": [Interval(1, None, inclusive_right=False)],
        "n_neighbors": [Interval(1, None, inclusive_right=False)],
        "algorithm": [Options(["auto", "ball_tree", "kd_tree", "brute"])],
        "metric": [str],
        "leaf_size": [Interval(1, None, inclusive_right=False)],
        "p": [Interval(1, None, inclusive_right=False)],
        "metric_params": [dict, type(None)],
        "n_jobs": [Interval(-1, None, inclusive_right=False), type(None)],
    }
    
    def __init__(self,
                 alpha=1,
                 n_neighbors=10,
                 algorithm="auto",
                 metric="euclidean",
                 leaf_size=30,
                 p=2,
                 metric_params=None,
                 n_jobs=None):
        estimator = PWKCLF(alpha=alpha,
                         n_neighbors=n_neighbors,
                         algorithm=algorithm,
                         metric=metric,
                         leaf_size=leaf_size,
                         p=p,
                         metric_params=metric_params,
                         n_jobs=n_jobs)
        self.algorithm = algorithm
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.leaf_size = leaf_size
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.estimator = estimator
        
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the PWK quantifier to the training data.

        Builds a :class:`~mlquantify.counting.CC` quantifier around the
        underlying weighted k-NN classifier and fits it on the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self : PWK
            The fitted quantifier.

        Examples
        --------
        >>> from mlquantify.neighbors import PWK
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> q = PWK(alpha=1.5, n_neighbors=5).fit(X, y)
        """
        X, y = validate_data(self, X, y, ensure_2d=True, ensure_min_samples=2)
        self.classes_ = np.unique(y)
        self.cc = CC(self.estimator)
        return self.cc.fit(X, y)

    def predict(self, X):
        """Predict class prevalences for the given test data.

        Classifies each test instance with the fitted weighted k-NN classifier
        and counts the fraction assigned to each class (Classify and Count).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        prevalences : dict or ndarray of shape (n_classes,)
            Estimated class prevalences summing to 1.

        Examples
        --------
        >>> from mlquantify.neighbors import PWK
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> q = PWK(alpha=1.5, n_neighbors=5).fit(X, y)
        >>> q.predict(X)
        {0: 0.49, 1: 0.51}
        """
        X = validate_data(self, X, ensure_2d=True)
        prevalences = self.cc.predict(X)
        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences

    def classify(self, X):
        """Classify test instances using the underlying weighted k-NN estimator.

        Returns hard class labels produced by :class:`PWKCLF` without any
        prevalence-level aggregation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class label for each test instance.

        Examples
        --------
        >>> from mlquantify.neighbors import PWK
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> q = PWK(alpha=1.5, n_neighbors=5).fit(X, y)
        >>> q.classify(X[:5])
        array([0, 1, 0, 1, 0])
        """
        return self.estimator.predict(X)
        