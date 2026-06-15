import numpy as np

from mlquantify.losses import LeastSquaresLoss
from mlquantify.matching._base import BaseMatchingQuantifier
from mlquantify.representations import KernelMeanRepresentation
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._validation import validate_data


class MatchingKernelQuantifier(BaseMatchingQuantifier):
    r"""Abstract base class for kernel mean matching quantifiers.

    Estimates class prevalences by minimising the distance between the kernel
    mean embedding of the test data and the mixture of class-conditional kernel
    mean embeddings computed on training data.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel function to use. One of ``'rbf'``, ``'linear'``, ``'poly'``,
        ``'sigmoid'``, ``'cosine'``.
    gamma : float or None, default=None
        Kernel coefficient for ``'rbf'``, ``'poly'``, and ``'sigmoid'``.
        If ``None``, uses ``1 / n_features``.
    degree : int, default=3
        Polynomial degree for the ``'poly'`` kernel.
    coef0 : float, default=0.0
        Independent term for ``'poly'`` and ``'sigmoid'`` kernels.
    solver : str, default='slsqp'
        Optimization solver.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.matching._kernel import MatchingKernelQuantifier
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> class MyKernelQ(MatchingKernelQuantifier):
    ...     def fit(self, X, y):
    ...         self.classes_ = np.unique(y)
    ...         return self._fit(X, y)
    ...     def predict(self, X):
    ...         return self._predict(X)
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> MyKernelQ().fit(X, y).predict(X)
    {0: ..., 1: ...}
    """

    _parameter_constraints = {
        "kernel": [Options(["rbf", "linear", "poly", "sigmoid", "cosine"])],
        "gamma": [Interval(0, None, inclusive_left=False), Options([None])],
        "degree": [Interval(1, None, inclusive_left=True)],
        "coef0": [Interval(0, None, inclusive_left=True)],
        "solver": [Options(["auto", "slsqp"])],
    }

    def __init__(
        self,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=0.0,
        solver="slsqp",
    ):
        super().__init__(
            representation=KernelMeanRepresentation(
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
            ),
            normalize=False,
        )
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.solver = solver

    def _solve_prevalence(self, test_representation, train_representations):
        solver = "slsqp" if self.solver == "auto" else self.solver

        class_means = np.asarray(train_representations, dtype=float)
        test_mean = np.asarray(test_representation, dtype=float)
        loss_function = LeastSquaresLoss()

        def objective(prevalence):
            mixture_mean = prevalence @ class_means
            return loss_function(mixture_mean, test_mean)

        prevalence, loss = minimize_prevalence(
            objective=objective,
            n_classes=len(self.classes_),
            solver=solver,
        )

        return prevalence, loss

class MMD_RKHS(MatchingKernelQuantifier):
    r"""Maximum Mean Discrepancy in RKHS (MMD-RKHS) quantifier.

    Targets prior probability shift. Matches distributions by their mean
    embeddings in a reproducing-kernel Hilbert space: it finds the prevalence
    vector whose mixture of per-class mean embeddings is closest to the test
    mean embedding, a convex quadratic program on the simplex. Native
    multiclass, classifier-free, and provably consistent under a universal
    kernel.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel defining the RKHS feature map; should be universal for
        consistency.

        - ``'rbf'`` : Gaussian radial-basis kernel; universal (recommended).
        - ``'linear'`` : inner product; matches only first moments.
        - ``'poly'`` : polynomial kernel; matches moments up to ``degree``.
        - ``'sigmoid'`` : hyperbolic-tangent kernel.
        - ``'cosine'`` : cosine-similarity kernel.
    gamma : float or None, default=None
        Kernel coefficient for ``'rbf'``, ``'poly'`` and ``'sigmoid'``.
        ``None`` uses ``1 / n_features``. Smaller over-smooths the embedding;
        larger over-fits it.
    degree : int, default=3
        Polynomial degree for the ``'poly'`` kernel (highest moment matched).
    coef0 : float, default=0.0
        Independent term for ``'poly'`` and ``'sigmoid'`` kernels.
    solver : str, default='slsqp'
        Constrained optimizer for the convex QP on the simplex (see
        :mod:`mlquantify.solvers`).

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    Statistically consistent under a universal kernel; its error shrinks when
    classes are well separated and the data spread is small. Kernel choice and
    bandwidth matter and can be tuned.

    See Also
    --------
    EDy : Energy-distance sample matching using classifier predictions.
    KDEyML : Multivariate-density multiclass matching.

    Examples
    --------
    >>> from mlquantify.matching import MMD_RKHS
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = MMD_RKHS().fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Zhang, K., Schölkopf, B., Muandet, K., & Wang, Z. (2013).
               Domain Adaptation under Target and Conditional Shift.
               *ICML*, pp. 819–827.
        .. [2] Iyer, A., Nath, S., & Sarawagi, S. (2014). Maximum Mean
               Discrepancy for Class Ratio Estimation: Convergence Bounds and
               Kernel Selection. *ICML*, 32.
    """

    def __init__(
        self,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=0.0,
        solver="slsqp",
    ):
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            solver=solver,
        )
        
    def fit(self, X, y, sample_weight=None):
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        return self._fit(X, y, sample_weight)

    def predict(self, X):
        X = validate_data(self, X)
        return self._predict(X)
