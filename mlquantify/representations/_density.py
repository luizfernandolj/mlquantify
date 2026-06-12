# representations/_density.py

import numpy as np

from sklearn.neighbors import KernelDensity

from ._base import BaseRepresentation


class KDERepresentation(BaseRepresentation):
    r"""Kernel density estimation representation.
    
    This representation fits a kernel density estimator (KDE) to the training data for each class, using the specified bandwidth and kernel. The test-time representation is simply the raw feature vector, and per-class likelihoods can be obtained by evaluating the fitted KDEs on the test samples. This allows for a non-parametric density-based representation of the data, which can capture complex class distributions without assuming a specific parametric form.

    Parameters
    ----------
    bandwidth : float, default=0.1
        The bandwidth parameter for the KDE, controlling the smoothness of the density estimate. Smaller values lead to a more flexible fit that can capture finer details of the data distribution, while larger values produce a smoother estimate that may overlook small-scale structure.
    kernel : str, default="gaussian"
        The kernel to use for the KDE. Options are:
        - "gaussian": Gaussian kernel (the default), which produces a smooth density estimate.
        - "tophat": Uniform kernel, which gives equal weight to all points within the bandwidth radius and zero weight to points outside.
        - "epanechnikov": Epanechnikov kernel, which is a parabolic kernel that gives more weight to points closer to the center of the bandwidth and less weight to points farther away, with zero weight beyond the bandwidth radius.
        - "exponential": Exponential kernel, which gives more weight to points closer to the center of the bandwidth and decays exponentially with distance, without a hard cutoff.
        - "linear": Linear kernel, which gives weight that decreases linearly with distance from the center of the bandwidth, with zero weight beyond the bandwidth radius.
        - "cosine": Cosine kernel, which gives weight that follows a cosine function of the distance from the center of the bandwidth, with zero weight beyond the bandwidth radius

    See Also
    --------
    HistogramRepresentation : Discrete (binned) alternative.
    KernelMeanRepresentation : RKHS mean-embedding representation.

    Examples
    --------
    >>> from mlquantify.representations._density import KDERepresentation
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 2))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> rep = KDERepresentation(bandwidth=0.2, kernel="gaussian").fit(X, y)
    >>> rep.class_representations_[0]
    KernelDensity(bandwidth=0.2, kernel='gaussian')
    
    """

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