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

    Extends confusion-matrix adjustment to the multiclass setting by
    solving a constrained linear system. A confusion matrix is estimated
    via cross-validation from hard (crisp) classifier predictions, and
    the test prevalence is recovered by minimizing the chosen loss subject
    to simplex constraints.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict`` methods.
    loss : str, default='ls'
        Loss function for solving the linear system (e.g. ``'ls'``, ``'l1'``).
    solver : str, default='slsqp'
        Optimization solver for the constrained problem.
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

    Examples
    --------
    >>> from mlquantify.counting import GACC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = GACC(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.33, 1: 0.34, 2: 0.33}

    References
    ----------
    .. dropdown:: References

        .. [1] Firat, A. (2016). Unified Framework for Quantification. *AAAI*.
        .. [2] Esuli, A., Moreo, A., & Sebastiani, F. (2023).
               *Learning to Quantify*. Springer.
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

    Same as :class:`GACC` but uses soft (probabilistic) predictions from
    ``predict_proba`` to build the confusion matrix, making it better
    calibrated when posterior probabilities are reliable.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    loss : str, default='ls'
        Loss function for solving the linear system.
    solver : str, default='slsqp'
        Optimization solver for the constrained problem.
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

    Examples
    --------
    >>> from mlquantify.counting import GPACC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = GPACC(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.33, 1: 0.34, 2: 0.33}

    References
    ----------
    .. dropdown:: References

        .. [1] Firat, A. (2016). Unified Framework for Quantification. *AAAI*.
        .. [2] Esuli, A., Moreo, A., & Sebastiani, F. (2023).
               *Learning to Quantify*. Springer.
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
    r"""Friedman Method (FM).

    A soft adjustment method that binarizes posterior probabilities by
    comparing them against class priors learned during training, rather
    than a fixed 0.5 threshold. This makes the method more robust to
    skewed class distributions.

    Uses the same linear-system framework as :class:`GPACC` but applies
    a class-prior-aware binarization transform before aggregation.

    Parameters
    ----------
    estimator : estimator, optional
        A classifier with ``fit`` and ``predict_proba`` methods.
    loss : str, default='ls'
        Loss function for the constrained system.
    solver : str, default='slsqp'
        Optimization solver.
    cv : int or None, default=None
        Number of cross-validation folds.
    stratified : bool, default=True
        Stratified CV flag.
    shuffle : bool, default=False
        Shuffle flag.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.counting import FM
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_classes=3, n_informative=5,
    ...                            n_redundant=0, random_state=42)
    >>> q = FM(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.33, 1: 0.34, 2: 0.33}

    References
    ----------
    .. dropdown:: References

        .. [1] Friedman, J. (2015). Detecting and Dealing with Concept Drift.
               Technical Report.
        .. [2] Tasche, D. (2024). Comments on Friedman's Method for Class
               Distribution Estimation. *LQ 2024 Workshop Proceedings*.
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
