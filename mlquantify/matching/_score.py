from abc import abstractmethod

import numpy as np

from mlquantify.base_aggregative import (
    AggregativeMixin,
    SoftPredictionMixin,
)
from mlquantify.representations import PredictionRepresentation
from mlquantify.matching._base import BaseMatchingQuantifier
from mlquantify.multiclass import binary_quantifier
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_data, resolve_aggregate_classes


EPS = 1e-12


@binary_quantifier(strategy_attr="strategy")
class MatchingScoreQuantifier(BaseMatchingQuantifier):
    r"""Abstract base class for binary score-based distribution matching.

    Subclasses estimate the positive-class prevalence by comparing the test
    score distribution with a mixture of positive and negative training scores.

    This is a **binary-only** method. When applied to multiclass problems,
    a one-vs-rest (OvR) strategy is applied automatically.

    Parameters
    ----------
    solver : str, default='auto'
        Optimization solver for prevalence estimation.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.matching._score import MatchingScoreQuantifier
    >>> from mlquantify.base_aggregative import SoftPredictionMixin, AggregativeMixin
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> class MyScoreQ(SoftPredictionMixin, AggregativeMixin, MatchingScoreQuantifier):
    ...     def __init__(self, estimator=None):
    ...         super().__init__(solver='bounded')
    ...         self.estimator = estimator or LogisticRegression()
    ...     def _solve_prevalence(self, test_representation, train_representations):
    ...         alpha = np.clip(np.mean(test_representation), 0, 1)
    ...         return np.array([1 - alpha, alpha]), None
    >>> q = MyScoreQ()   # subclasses implement _solve_prevalence
    """

    _parameter_constraints = {
        "solver": [Options(["auto", "grid", "ternary", "bounded"])],
        "strategy": [Options(["ovr", "ovo"])],
    }

    def __init__(
        self,
        solver="auto",
        strategy="ovr",
    ):
        super().__init__(
            representation=PredictionRepresentation(
                func=self._positive_scores,
                average=False,
            ),
            normalize=False,
        )
        self.solver = solver
        self.strategy = strategy

    @abstractmethod
    def _solve_prevalence(self, test_representation):
        """Estimate binary prevalence from test scores."""

    @staticmethod
    def _positive_scores(X, representation):
        X = np.asarray(X, dtype=float)

        if X.ndim == 2:
            return X[:, -1]

        return X.ravel()


@binary_quantifier(strategy_attr="strategy")
class SORD(SoftPredictionMixin, AggregativeMixin, MatchingScoreQuantifier):
    r"""Sample Ordinal Distance (SORD) quantifier.

    Targets prior probability shift. A parameter-free mixture model: instead of
    binning scores it compares the raw weighted score samples with an ordinal
    (earth-mover-like) distance, searching the mixture proportion that best
    blends the positive and negative training scores to match the test sample.
    Binary-only, with no bin count to tune; multiclass via one-vs-rest.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    n_grid : int, default=101
        Number of grid points for the prevalence (alpha) search.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

        - ``'ovr'`` : one-vs-rest, one binary quantifier per class.
        - ``'ovo'`` : one-vs-one, one binary quantifier per class pair.
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
    SORD is parameter-free (no bin count) and competitive with a tuned
    Topsoe-:class:`DyS`. Its cost grows roughly as ``O(n log n)`` in the
    combined sample size, so training scores may be sub-sampled.

    See Also
    --------
    DyS : Histogram mixture model with a selectable distance.
    SMM : Mean-matching member of the same framework.

    Examples
    --------
    >>> from mlquantify.matching import SORD
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = SORD(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}
    >>> # call aggregate with pre-computed posterior scores
    >>> scores = q.estimator_.predict_proba(X)
    >>> q.aggregate(scores, scores, y)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Esuli, A., Moreo, A., & Sebastiani, F. (2023).
               *Learning to Quantify*. Springer.
    """


    _parameter_constraints = {
        "n_grid": [Interval(2, None)],
        "strategy": [Options(["ovr", "ovo"])],
        "cv": [Interval(2, None), None],
        "stratified": [bool],
        "shuffle": [bool],
        "random_state": ["random_state", None],
    }

    def __init__(
        self,
        estimator=None,
        n_grid=101,
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        self.n_grid = n_grid

        super().__init__(
            solver="grid",
            strategy=strategy,
        )
        self.estimator = estimator
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        estimator_fitted=False,
        sample_weight=None,
        cv_prediction="refit",
    ):

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        X, y = self._fit_estimator_predictions(
            X,
            y,
            estimator_fitted=estimator_fitted,
            cv_prediction=cv_prediction,
        )

        return self._fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        X = validate_data(self, X, ensure_2d=True)
        test_scores = self._predict_estimator(X)
        return self._predict(test_scores)
    
    def aggregate(self, test_scores, train_scores, y_train, classes=None):
        if not getattr(self, "_precomputed", False):
            self._fit(train_scores, y_train)
        if classes is not None:
            self.classes_ = resolve_aggregate_classes(self, classes, self.classes_)
        return self._predict(test_scores)

    def _solve_prevalence(self, test_representation, train_representations):
        test_scores = np.asarray(test_representation, dtype=float).ravel()

        neg_repr = train_representations[0]
        pos_repr = train_representations[1]

        n_pos = len(pos_repr)
        n_neg = len(neg_repr)
        n_test = len(test_scores)

        if n_pos == 0 or n_neg == 0 or n_test == 0:
            raise ValueError(
                "SORD requires non-empty positive, negative, and test samples."
            )

        pos_scores = np.asarray(pos_repr, dtype=float)
        neg_scores = np.asarray(neg_repr, dtype=float)
        scores = np.concatenate([pos_scores, neg_scores, test_scores])

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        gaps = np.diff(sorted_scores)

        def objective(alpha):
            weights = np.concatenate(
                [
                    np.full(n_pos, alpha / n_pos),
                    np.full(n_neg, (1.0 - alpha) / n_neg),
                    np.full(n_test, -1.0 / n_test),
                ]
            )

            sorted_weights = weights[order]
            cumulative_weights = np.cumsum(sorted_weights)[:-1]

            return float(np.sum(np.abs(gaps * cumulative_weights)))

        prevalences, distance = minimize_prevalence(
            objective=objective,
            n_classes=2,
            solver="grid",
            grid_size=self.n_grid,
        )

        return prevalences, distance


@binary_quantifier(strategy_attr="strategy")
class SMM(SoftPredictionMixin, AggregativeMixin, MatchingScoreQuantifier):
    r"""Sample Mean Matching (SMM) quantifier.

    Targets prior probability shift. The lightest mixture model: it summarises
    each class by its mean classifier score and solves analytically, in closed
    form, for the mixture proportion that makes the blended mean equal the test
    mean score. Binary-only, with no search and no bins; multiclass via
    one-vs-rest.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    moment : int, default=1
        Score moment used for matching (``1`` = mean).
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

        - ``'ovr'`` : one-vs-rest, one binary quantifier per class.
        - ``'ovo'`` : one-vs-one, one binary quantifier per class pair.
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
    SMM is extremely fast (closed form) but uses only the first moment of the
    score distribution, so it is less accurate than :class:`DyS`/:class:`SORD`
    when the class scores overlap.

    See Also
    --------
    DyS : Full-histogram mixture model.
    SORD : Bin-free ordinal-distance member of the same framework.

    Examples
    --------
    >>> from mlquantify.matching import SMM
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = SMM(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}
    >>> # call aggregate with pre-computed posterior scores
    >>> scores = q.estimator_.predict_proba(X)
    >>> q.aggregate(scores, scores, y)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Esuli, A., Moreo, A., & Sebastiani, F. (2023).
               *Learning to Quantify*. Springer.
    """


    _parameter_constraints = {
        "moment": [Interval(1, None)],
        "strategy": [Options(["ovr", "ovo"])],
        "cv": [Interval(2, None), None],
        "stratified": [bool],
        "shuffle": [bool],
        "random_state": ["random_state", None],
    }

    def __init__(
        self,
        estimator=None,
        moment=1,
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        self.moment = moment

        super().__init__(
            solver="bounded",
            strategy=strategy,
        )
        self.estimator = estimator
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        estimator_fitted=False,
        sample_weight=None,
        cv_prediction="refit",
    ):

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        X, y = self._fit_estimator_predictions(
            X,
            y,
            estimator_fitted=estimator_fitted,
            cv_prediction=cv_prediction,
        )

        return self._fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        X = validate_data(self, X, ensure_2d=True)
        test_scores = self._predict_estimator(X)
        return self._predict(test_scores)

    def aggregate(self, test_scores, train_scores, y_train, classes=None):
        if not getattr(self, "_precomputed", False):
            self._fit(train_scores, y_train)
        if classes is not None:
            self.classes_ = resolve_aggregate_classes(self, classes, self.classes_)
        return self._predict(test_scores)

    def _solve_prevalence(self, test_representation, train_representations):
        test_scores = np.asarray(test_representation, dtype=float).ravel()
        neg_repr = train_representations[0]
        pos_repr = train_representations[1]

        mean_pos = np.mean(pos_repr)
        mean_neg = np.mean(neg_repr)
        mean_test = np.mean(test_scores)

        if mean_pos - mean_neg == 0:
            alpha = mean_test
        else:
            alpha = np.clip((mean_test - mean_neg) / (mean_pos - mean_neg), 0, 1)

        prevalences = np.array([1 - alpha, alpha])

        return prevalences, None
