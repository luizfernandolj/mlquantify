# representations/_density.py

import numpy as np

from sklearn.neighbors import KernelDensity

from ._base import BaseRepresentation


class KDERepresentation(BaseRepresentation):
    r"""Kernel density estimation representation."""

    def __init__(
        self,
        bandwidth=0.1,
        kernel="gaussian",
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def transform(self, X):
        """Return the input as a float array (identity transform).

        The KDE representation uses raw feature vectors as the test-time
        representation; density evaluation happens in
        :meth:`class_likelihoods`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Input cast to float64.

        Examples
        --------
        >>> from mlquantify.representations._density import KDERepresentation
        >>> import numpy as np
        >>> rep = KDERepresentation()
        >>> X = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> rep.transform(X).shape
        (2, 2)
        """
        return np.asarray(X, dtype=float)

    def _fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)

        self.class_kdes_ = [
            KernelDensity(
                bandwidth=self.bandwidth,
                kernel=self.kernel,
            ).fit(X[y == cls])
            for cls in self.classes_
        ]

        self.class_representations_ = np.asarray(
            self.class_kdes_,
            dtype=object,
        )

    def class_likelihoods(self, X):
        """Evaluate per-class kernel density likelihoods for test instances.

        For each class KDE fitted during :meth:`fit`, scores every test
        sample and exponentiates the log-density to obtain raw likelihood
        values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        likelihoods : ndarray of shape (n_classes, n_samples)
            Per-class likelihood for each test instance, where
            ``likelihoods[c, i]`` is the density of class ``c`` at
            sample ``i``.

        Examples
        --------
        >>> from mlquantify.representations._density import KDERepresentation
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((100, 2))
        >>> y = (X[:, 0] > 0).astype(int)
        >>> rep = KDERepresentation().fit(X, y)
        >>> lkl = rep.class_likelihoods(X[:5])
        >>> lkl.shape
        (2, 5)
        """
        X = np.asarray(X, dtype=float)

        return np.asarray([
            np.exp(kde.score_samples(X))
            for kde in self.class_kdes_
        ])