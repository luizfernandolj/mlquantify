import numpy as np
from scipy.spatial.distance import cdist

from mlquantify.representations._base import BaseRepresentation


class DistanceRepresentation(BaseRepresentation):
    """Distance-based representation for quantification.

    Summarises a collection of instances as the vector of mean pairwise
    distances to each training class.  The representation of a set of
    test instances is the column-wise mean of the per-instance distance
    vectors, yielding a single ``(n_classes,)`` descriptor.

    This is used by the Energy Distance Quantifier (EDy), where the
    distance between the test and each class centroid forms the basis of
    the prevalence estimation objective.

    Parameters
    ----------
    metric : str, default='euclidean'
        Distance metric passed to :func:`scipy.spatial.distance.cdist`.

    Attributes
    ----------
    X_train_ : ndarray of shape (n_samples, n_features)
        Training feature matrix stored at fit time.
    y_train_ : ndarray of shape (n_samples,)
        Training labels stored at fit time.
    class_representations_ : ndarray of shape (n_classes,)
        Mean pairwise distance from each training class to the full
        training set.
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during fit.

    See Also
    --------
    KernelMeanRepresentation : RKHS mean-embedding representation (MMD).
    EDy : Energy-distance quantifier built on this representation.

    Examples
    --------
    >>> from mlquantify.representations._distance import DistanceRepresentation
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> rep = DistanceRepresentation().fit(X, y)
    >>> rep.transform(X[:10]).shape
    (2,)
    """

    def __init__(self, metric="euclidean"):
        self.metric = metric

    def transform(self, X):
        """Compute mean pairwise distances to each training class.

        For every test instance the mean distance to all training samples
        of each class is computed.  The returned vector is the column-wise
        mean across all test instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        representation : ndarray of shape (n_classes,)
            Mean distance from the test set to each training class.

        Examples
        --------
        >>> from mlquantify.representations._distance import DistanceRepresentation
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> X = rng.standard_normal((80, 2))
        >>> y = (X[:, 0] > 0).astype(int)
        >>> rep = DistanceRepresentation().fit(X, y)
        >>> dist = rep.transform(X[:5])
        >>> dist.shape
        (2,)
        """
        X = np.asarray(X, dtype=float)

        values = np.zeros((X.shape[0], len(self.classes_)), dtype=float)

        for class_idx, cls in enumerate(self.classes_):
            X_cls = self.X_train_[self.y_train_ == cls]

            if len(X_cls) > 0:
                values[:, class_idx] = cdist(
                    X,
                    X_cls,
                    metric=self.metric,
                ).mean(axis=1)

        return values.mean(axis=0)

    def _fit(self, X, y, sample_weight=None):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)

        self.class_representations_ = np.asarray([
            self.transform(self.X_train_[self.y_train_ == cls])
            for cls in self.classes_
        ])
