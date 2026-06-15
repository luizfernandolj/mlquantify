# representations/_kernel.py

import numpy as np

from sklearn.metrics.pairwise import pairwise_kernels

from ._base import BaseRepresentation


class KernelMeanRepresentation(BaseRepresentation):
    r"""Kernel mean embedding representation.

    Represents a sample of instances by its kernel mean embedding in a
    reproducing-kernel Hilbert space (the mean feature map), so that matching
    distributions reduces to matching mean embeddings. Exact under a linear
    kernel and an approximation under non-linear kernels.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel defining the RKHS feature map (``'rbf'``, ``'linear'``,
        ``'poly'``, ``'sigmoid'``, ...); should be universal for consistency.
    gamma : float or None, default=None
        Kernel coefficient for ``'rbf'``/``'poly'``/``'sigmoid'``; ``None``
        uses ``1 / n_features``.
    degree : int, default=3
        Polynomial degree for the ``'poly'`` kernel.
    coef0 : float, default=0.0
        Independent term for the ``'poly'`` and ``'sigmoid'`` kernels.

    See Also
    --------
    DistanceRepresentation : Pairwise-distance representation (energy distance).
    MMD_RKHS : Quantifier built on this representation.
    """

    def __init__(
        self,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=0.0,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def transform(self, X):
        """Compute the empirical mean embedding of a set of instances.

        Returns the column-wise mean of the feature matrix, which is the
        kernel mean embedding under a linear kernel and an approximation
        under non-linear kernels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        embedding : ndarray of shape (n_features,)
            Mean feature vector.

        Examples
        --------
        >>> from mlquantify.representations._kernel import KernelMeanRepresentation
        >>> import numpy as np
        >>> rep = KernelMeanRepresentation()
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> rep.transform(X)
        array(...)
        """
        X = np.asarray(X, dtype=float)

        return X.mean(axis=0)

    def _fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)

        self.class_representations_ = np.asarray([
            self.transform(X[y == cls])
            for cls in self.classes_
        ])

    def pairwise(self, X, Y):
        """Compute a pairwise kernel matrix between two arrays.

        Dispatches to :func:`sklearn.metrics.pairwise.pairwise_kernels`
        using the kernel and hyperparameters configured at construction.

        Parameters
        ----------
        X : array-like of shape (n_x, n_features)
            First set of samples.
        Y : array-like of shape (n_y, n_features)
            Second set of samples.

        Returns
        -------
        K : ndarray of shape (n_x, n_y)
            Kernel matrix where ``K[i, j] = k(X[i], Y[j])``.

        Examples
        --------
        >>> from mlquantify.representations._kernel import KernelMeanRepresentation
        >>> import numpy as np
        >>> rep = KernelMeanRepresentation(kernel="rbf", gamma=1.0)
        >>> X = np.array([[0.0], [1.0]])
        >>> Y = np.array([[0.5]])
        >>> rep.pairwise(X, Y).shape
        (2, 1)
        """
        params = {}

        if self.kernel in {"rbf", "poly", "sigmoid"} and self.gamma is not None:
            params["gamma"] = self.gamma

        if self.kernel == "poly":
            params["degree"] = self.degree
            params["coef0"] = self.coef0

        if self.kernel == "sigmoid":
            params["coef0"] = self.coef0

        return pairwise_kernels(
            X,
            Y,
            metric=self.kernel,
            **params,
        )
