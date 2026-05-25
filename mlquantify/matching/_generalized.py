# matching/_generalized.py

from mlquantify.compose import (
    LinearComposeQuantifier,
    LikelihoodComposeQuantifier,
)
from mlquantify.base_aggregative import AggregativeMixin, SoftPredictionMixin
from mlquantify.losses import EnergyLoss
from mlquantify.representations import (
    DistanceRepresentation,
    HistogramRepresentation,
    KDERepresentation,
    PredictionRepresentation,
)


class GHDy(SoftPredictionMixin, AggregativeMixin, LinearComposeQuantifier):
    r"""Generalized HDy (GHDy) quantifier.

    Extends :class:`HDy` to the multiclass setting using the linear-composition
    framework. Builds histogram representations over posterior probabilities for
    each class and estimates prevalences by minimising the Hellinger distance.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    bins : int, default=10
        Number of histogram bins.
    loss : str, default='hellinger'
        Loss function for solving the linear system.
    solver : str, default='slsqp'
        Optimization solver.
    cv : int or None, default=None
        Cross-validation folds for computing training scores.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.matching import GHDy
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = GHDy(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.33, 1: 0.34, 2: 0.33}

    References
    ----------
    .. dropdown:: References

        .. [1] González-Castro, V., Alaiz-Rodriguez, R., & Alegre, E. (2013).
               Class Distribution Estimation Based on the Hellinger Distance.
               *Information Sciences*, 218, 146–164.
    """

    def __init__(
        self,
        estimator=None,
        bins=10,
        loss="hellinger",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            representation=PredictionRepresentation(
                method="soft",
                average=False,
                representation=HistogramRepresentation(
                    bins=bins,
                    mode="onehot",
                    bin_edges="auto",
                ),
            ),
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=True,
            random_state=random_state,
        )
        self.bins = bins
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state


class GHDx(LinearComposeQuantifier):
    r"""Generalized HDx (GHDx) quantifier.

    Extends :class:`HDx` to the multiclass setting using the linear-composition
    framework. Builds histogram representations over input features (no classifier
    needed) and estimates prevalences by minimising the Hellinger distance.

    Parameters
    ----------
    bins : int, default=10
        Number of histogram bins per feature.
    loss : str, default='hellinger'
        Loss function for solving the linear system.
    solver : str, default='slsqp'
        Optimization solver.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.matching import GHDx
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = GHDx().fit(X, y)
    >>> q.predict(X)
    {0: 0.33, 1: 0.34, 2: 0.33}

    References
    ----------
    .. dropdown:: References

        .. [1] González-Castro, V., Alaiz-Rodriguez, R., & Alegre, E. (2013).
               Class Distribution Estimation Based on the Hellinger Distance.
               *Information Sciences*, 218, 146–164.
    """

    def __init__(
        self,
        bins=10,
        loss="hellinger",
        solver="slsqp",
        random_state=None,
    ):
        super().__init__(
            estimator=None,
            representation=HistogramRepresentation(
                bins=bins,
                mode="onehot",
                bin_edges="auto",
            ),
            loss=loss,
            solver=solver,
            aggregative=False,
            normalize=True,
            random_state=random_state,
        )
        self.bins = bins
        self.random_state = random_state


class GKDEyML(SoftPredictionMixin, AggregativeMixin, LikelihoodComposeQuantifier):
    r"""Generalized KDEy Maximum Likelihood (GKDEyML) quantifier.

    Multiclass extension of KDEy using the likelihood-composition framework.
    Fits KDE densities over classifier posterior probabilities for each class
    and estimates prevalences by maximising the mixture log-likelihood.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    bandwidth : float, default=0.1
        Bandwidth of the kernel density estimator.
    kernel : str, default='gaussian'
        Kernel type for the KDE.
    solver : str, default='slsqp'
        Optimization solver.
    cv : int or None, default=None
        Cross-validation folds for computing training scores.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.matching import GKDEyML
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = GKDEyML(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.33, 1: 0.34, 2: 0.33}

    References
    ----------
    .. dropdown:: References

        .. [1] Moreo, A., González, P., & del Coz, J. J. (2024).
               Kernel Density Estimation for Multiclass Quantification.
               *Machine Learning*, 113, 3075–3107.
    """

    def __init__(
        self,
        estimator=None,
        bandwidth=0.1,
        kernel="gaussian",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            representation=PredictionRepresentation(
                method="soft",
                average=False,
                representation=KDERepresentation(
                    bandwidth=bandwidth,
                    kernel=kernel,
                ),
            ),
            solver=solver,
            aggregative=True,
            random_state=random_state,
        )

        self.bandwidth = bandwidth
        self.kernel = kernel
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state


KDEyML = GKDEyML


class EDy(SoftPredictionMixin, AggregativeMixin, LinearComposeQuantifier):
    r"""Energy Distance y (EDy) quantifier.

    Estimates class prevalences by minimising the energy distance between the
    test posterior-probability distribution and the mixture of class-conditional
    distributions, using the linear-composition framework.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    metric : str, default='euclidean'
        Distance metric used for the energy distance.
    solver : str, default='slsqp'
        Optimization solver.
    cv : int or None, default=None
        Cross-validation folds for computing training scores.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.matching import EDy
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = EDy(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.33, 1: 0.34, 2: 0.33}

    References
    ----------
    .. dropdown:: References

        .. [1] Esuli, A., Moreo, A., & Sebastiani, F. (2023).
               *Learning to Quantify*. Springer.
    """

    def __init__(
        self,
        estimator=None,
        metric="euclidean",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            representation=PredictionRepresentation(
                method="soft",
                representation=DistanceRepresentation(metric=metric),
                average=False,
            ),
            loss=EnergyLoss(),
            solver=solver,
            aggregative=True,
            normalize=False,
            random_state=random_state,
        )
        self.cv = cv
        self.metric = metric
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state


class EDx(LinearComposeQuantifier):
    r"""Energy Distance x (EDx) quantifier.

    Estimates class prevalences by minimising the energy distance between the
    test feature distribution and the mixture of class-conditional feature
    distributions, using the linear-composition framework (no classifier needed).

    Parameters
    ----------
    metric : str, default='euclidean'
        Distance metric used for the energy distance.
    solver : str, default='slsqp'
        Optimization solver.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.matching import EDx
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = EDx().fit(X, y)
    >>> q.predict(X)
    {0: 0.33, 1: 0.34, 2: 0.33}

    References
    ----------
    .. dropdown:: References

        .. [1] Esuli, A., Moreo, A., & Sebastiani, F. (2023).
               *Learning to Quantify*. Springer.
    """

    def __init__(
        self,
        metric="euclidean",
        solver="slsqp",
        random_state=None,
    ):
        super().__init__(
            representation=DistanceRepresentation(metric=metric),
            loss=EnergyLoss(),
            solver=solver,
            aggregative=False,
            normalize=False,
            random_state=random_state,
        )
        self.metric = metric
        self.random_state = random_state
