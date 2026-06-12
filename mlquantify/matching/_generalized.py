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

    Targets prior probability shift. The multiclass, constrained-regression
    form of :class:`HDy`: it builds per-class posterior histograms and solves
    on the probability simplex for the prevalence that matches the test
    histogram under a Hellinger objective. Native multiclass; requires soft
    posteriors.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    bins : int, default=10
        Number of histogram bins over the posterior scores; controls resolution.
    loss : str, default='hellinger'
        Distribution dissimilarity minimised on the simplex (Hellinger for GHDy).
    solver : str, default='slsqp'
        Constrained optimizer over the simplex (see :mod:`mlquantify.solvers`).
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

    Notes
    -----
    Reduces to binary :class:`HDy` for two classes; preferred over running HDy
    one-vs-all for genuine multiclass problems.

    See Also
    --------
    HDy : Binary histogram mixture model.
    GHDx : Classifier-free, feature-space variant.
    GACC : Constrained-regression counting sibling.

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

    Targets prior probability shift. The multiclass, classifier-free form of
    :class:`HDx`: it builds per-feature histograms and solves on the
    probability simplex for the prevalence matching the test feature
    histograms under a Hellinger objective. Needs only raw features.

    Parameters
    ----------
    bins : int, default=10
        Number of histogram bins per feature; controls resolution.
    loss : str, default='hellinger'
        Distribution dissimilarity minimised on the simplex (Hellinger for GHDx).
    solver : str, default='slsqp'
        Constrained optimizer over the simplex (see :mod:`mlquantify.solvers`).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    Classifier-free, so it avoids calibration issues but cannot exploit a
    learned representation; it weakens with many noisy or correlated features.

    See Also
    --------
    HDx : Binary feature-space mixture model.
    GHDy : Score-space variant using a classifier.

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

    Targets prior probability shift. The compose-framework rendering of
    :class:`KDEyML`: per-class kernel densities over the posteriors feed a
    mixture log-likelihood maximised on the probability simplex. Native
    multiclass; requires soft posteriors.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    bandwidth : float, default=0.1
        Bandwidth of the kernel density estimator; controls density smoothness.
    kernel : str, default='gaussian'
        Kernel shape placed on each training point (``'gaussian'``,
        ``'tophat'``, ``'epanechnikov'``, ``'exponential'``, ``'linear'``,
        ``'cosine'``).
    solver : str, default='slsqp'
        Constrained optimizer over the simplex (see :mod:`mlquantify.solvers`).
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

    See Also
    --------
    KDEyML : Direct maximum-likelihood KDE quantifier.
    GHDy : Histogram constrained-regression sibling.

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

    Targets prior probability shift. A distribution-matching quantifier that
    represents each class by the full set of its classifier predictions and
    minimises the energy distance between the test set and the
    prevalence-weighted mixture of class sets. Native multiclass and strong in
    the multiclass regime.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    metric : str, default='euclidean'
        Ground distance between predictions used in the energy term.

        - ``'euclidean'`` : L2 distance between prediction vectors.
        - ``'cityblock'`` : Manhattan / L1 distance (the paper's EDy choice).
        - ``'cosine'`` : 1 minus cosine similarity.
    solver : str, default='slsqp'
        Constrained optimizer over the simplex (see :mod:`mlquantify.solvers`).
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

    Notes
    -----
    The energy objective is the quadratic form ``p^T (2q - M p)``; class and
    test distributions are estimated via cross-validation. EDy uses classifier
    predictions, while :class:`EDx` uses raw features.

    See Also
    --------
    EDx : Classifier-free, feature-space energy-distance variant.
    MMD_RKHS : Kernel-mean-embedding sample matching.

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

    Targets prior probability shift. The classifier-free energy-distance
    quantifier: it computes the energy distance directly on raw feature vectors
    between the test set and the prevalence mixture of per-class sets. Native
    multiclass; needs no scorer.

    Parameters
    ----------
    metric : str, default='euclidean'
        Ground distance on features used in the energy term.

        - ``'euclidean'`` : L2 distance between feature vectors.
        - ``'cityblock'`` : Manhattan / L1 distance.
        - ``'cosine'`` : 1 minus cosine similarity.
    solver : str, default='slsqp'
        Constrained optimizer over the simplex (see :mod:`mlquantify.solvers`).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    Feature-space version of :class:`EDy`; it avoids classifier calibration but
    ignores any learned representation and is sensitive to feature scaling.

    See Also
    --------
    EDy : Energy distance on classifier predictions.

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
