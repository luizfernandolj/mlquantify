import numpy as np

from mlquantify.base_aggregative import (
    AggregativeMixin,
    CrispPredictionMixin,
    SoftPredictionMixin,
)
from mlquantify.compose import LinearComposeQuantifier
from mlquantify.representations import PredictionRepresentation


class GACC(CrispPredictionMixin, AggregativeMixin, LinearComposeQuantifier):
    r"""Generalized Adjusted Classify and Count (GACC).

    Targets prior probability shift. Extends confusion-matrix adjustment to
    the multiclass setting through the unified constrained-regression
    framework: a per-class matrix of mean hard (crisp) predictions is
    estimated by cross-validation, and the test prevalence is recovered by
    minimizing the chosen loss on the probability simplex. Native multiclass,
    so no One-vs-All decomposition is needed.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict`` methods.
    loss : str, default='ls'
        Loss minimized over the simplex when solving the linear system.

        - ``'ls'`` : least squares (L2); the standard adjusted-count objective.
        - ``'l1'`` : least absolute deviation (L1); more robust to outliers.
    solver : str, default='slsqp'
        Constrained optimizer used on the simplex; ``'slsqp'`` is the
        sequential least-squares programming backend (see :mod:`mlquantify.solvers`).
    cv : int or None, default=None
        Number of cross-validation folds. Defaults to 5 if ``None``.
    stratified : bool, default=True
        Whether to use stratified CV splits.
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
    GACC is the multiclass generalisation of :class:`ACC`: the per-class
    matrix is the training confusion matrix and solving it on the simplex
    inverts that matrix while guaranteeing a valid distribution.

    See Also
    --------
    ACC : Binary adjusted count.
    GPACC : Soft-prediction variant.
    FM : Friedman's prior-threshold variant.

    Examples
    --------
    >>> from mlquantify.counting import GACC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = GACC(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ..., 2: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Firat, A. (2016). Unified Framework for Quantification. *AAAI*.
        .. [2] Esuli, A., Moreo, A., & Sebastiani, F. (2023). *Learning to Quantify*. Springer.
    """


    def __init__(
        self,
        estimator=None,
        loss="ls",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            representation=PredictionRepresentation(
                method="hard",
                average=True,
            ),
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=False,
            random_state=random_state,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state


class GPACC(SoftPredictionMixin, AggregativeMixin, LinearComposeQuantifier):
    r"""Generalized Probabilistic Adjusted Classify and Count (GPACC).

    Targets prior probability shift. Like :class:`GACC`, but builds the
    per-class matrix from soft posterior probabilities (``predict_proba``)
    instead of hard predictions, so it exploits classifier confidence and is
    better behaved when the posteriors are well calibrated. Native multiclass;
    requires a probabilistic classifier.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    loss : str, default='ls'
        Loss minimized over the simplex when solving the linear system.

        - ``'ls'`` : least squares (L2); the standard objective.
        - ``'l1'`` : least absolute deviation (L1); more robust to outliers.
    solver : str, default='slsqp'
        Constrained optimizer used on the simplex (see :mod:`mlquantify.solvers`).
    cv : int or None, default=None
        Number of cross-validation folds. Defaults to 5 if ``None``.
    stratified : bool, default=True
        Whether to use stratified CV splits.
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
    GPACC is the multiclass, corrected generalisation of :class:`PCC` (also
    known as PACC): the soft-count matrix is inverted on the simplex instead
    of being read off directly, removing PCC's residual bias.

    See Also
    --------
    PCC : Probabilistic Classify and Count.
    GACC : Hard-prediction variant.
    FM : Friedman's prior-threshold variant.

    Examples
    --------
    >>> from mlquantify.counting import GPACC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = GPACC(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ..., 2: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Firat, A. (2016). Unified Framework for Quantification. *AAAI*.
        .. [2] Esuli, A., Moreo, A., & Sebastiani, F. (2023). *Learning to Quantify*. Springer.
    """

    def __init__(
        self,
        estimator=None,
        loss="ls",
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
                average=True,
            ),
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=False,
            random_state=random_state,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state


class FM(SoftPredictionMixin, AggregativeMixin, LinearComposeQuantifier):
    r"""Friedman Method (FM) quantifier.

    Targets prior probability shift. A constrained-regression quantifier
    whose feature transform indicates, for each class, whether the posterior
    exceeds that class's training prior, rather than a fixed 0.5 threshold.
    Thresholding at the class's own prior minimizes the variance of the
    proportion estimate, making FM robust to skewed class distributions.
    Native multiclass; requires a probabilistic classifier.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    loss : str, default='ls'
        Loss minimized over the simplex when solving the linear system.

        - ``'ls'`` : least squares (L2); the standard objective.
        - ``'l1'`` : least absolute deviation (L1); more robust to outliers.
    solver : str, default='slsqp'
        Constrained optimizer used on the simplex (see :mod:`mlquantify.solvers`).
    cv : int or None, default=None
        Number of cross-validation folds. Defaults to 5 if ``None``.
    stratified : bool, default=True
        Whether to use stratified CV splits.
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
    GACC : Hard-prediction adjusted count.
    GPACC : Soft-prediction adjusted count.

    Examples
    --------
    >>> from mlquantify.counting import FM
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = FM(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ..., 2: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Friedman, J. (2015). Detecting and Dealing with Concept Drift.
               Technical Report.
        .. [2] Tasche, D. (2024). Comments on Friedman's Method for Class Distribution Estimation. *LQ 2024 Workshop Proceedings*.
    """

    def __init__(
        self,
        estimator=None,
        loss="ls",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            representation=PredictionRepresentation(
                func=self._friedman_prediction,
                average=True,
            ),
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=False,
            random_state=random_state,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

    @staticmethod
    def _friedman_prediction(X, representation):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = np.column_stack((1.0 - X, X))

        return (X >= representation.priors_).astype(float)
