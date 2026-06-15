import numpy as np
from mlquantify.utils._constraints import Interval, Options
from mlquantify.neighbors._classification import PWKCLF
from mlquantify.utils import validate_data
from mlquantify.counting import CC


class PWK(CC):
    r"""Probabilistic Weighted k-Nearest Neighbour (PWK) quantifier.

    Targets prior probability shift. PWK is an **aggregative** Classify-and-Count
    quantifier — it shares the standard ``fit`` / ``predict`` / ``aggregate``
    interface of :class:`~mlquantify.counting.CC` — but its classifier is a
    k-nearest-neighbour rule *modified for quantification* (:class:`PWKCLF`):
    each neighbour's vote is re-weighted by a class-specific factor (controlled
    by ``alpha``) so the count is not dominated by the majority class. Unlike the
    other aggregative quantifiers, PWK therefore takes **no external estimator
    parameter**: the modified k-NN is intrinsic to the method.

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
        The underlying weighted k-NN classifier (built from the parameters
        above; not an argument).
    estimator_ : PWKCLF
        The fitted classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    PWK is a classify-and-count method whose only quantification-specific
    ingredient is the imbalance re-weighting; it needs no separate scorer and
    handles multiclass directly, but inherits k-NN's sensitivity to feature
    scaling and dimensionality. Because it subclasses :class:`CC`, ``aggregate``
    (with its optional ``classes`` argument) is available too.

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
    {0: ..., 1: ...}

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
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.leaf_size = leaf_size
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        # PWK is aggregative, but its estimator is fixed to the quantification
        # k-NN rather than supplied by the user.
        super().__init__(
            estimator=PWKCLF(alpha=alpha,
                             n_neighbors=n_neighbors,
                             algorithm=algorithm,
                             metric=metric,
                             leaf_size=leaf_size,
                             p=p,
                             metric_params=metric_params,
                             n_jobs=n_jobs),
        )

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
        >>> labels = q.classify(X[:5])
        """
        X = validate_data(self, X, ensure_2d=True)
        return self.estimator_.predict(X)
