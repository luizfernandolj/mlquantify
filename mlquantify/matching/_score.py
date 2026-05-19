from abc import abstractmethod

import numpy as np

from mlquantify.base_aggregative import (
    AggregationMixin,
    SoftLearnerQMixin,
)
from mlquantify.representations import PredictionRepresentation
from mlquantify.matching._base import BaseMatchingQuantifier
from mlquantify.multiclass import binary_quantifier
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_data


EPS = 1e-12


@binary_quantifier(strategy_attr="strategy")
class MatchingScoreQuantifier(BaseMatchingQuantifier):
    r"""Base class for binary score-based matching quantifiers.

    Subclasses estimate the positive-class prevalence by comparing the test
    score distribution with a mixture of positive and negative training scores.
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


class SORD(SoftLearnerQMixin, AggregationMixin, MatchingScoreQuantifier):
    r"""Sample Ordinal Distance (SORD) quantification method.

    Estimates prevalence by minimizing the weighted sum of absolute score differences
    between test data and training classes. The method creates weighted score 
    vectors for positive, negative, and test samples, sorts them, and computes
    a cumulative absolute difference as the distance measure.

    Parameters
    ----------
    learner : estimator, optional
        Base probabilistic classifier.

    References
    ----------
    .. [2] Esuli et al. (2023). Learning to Quantify. Springer.
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
        learner=None,
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
        self.learner = learner
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, sample_weight=None):

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        X, y = self._fit_learner_predictions(
            X,
            y,
            learner_fitted=learner_fitted,
        )

        return self._fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        X = validate_data(self, X, ensure_2d=True)
        test_scores = self._predict_learner(X)
        return self._predict(test_scores)
    
    def aggregate(self, test_scores, train_scores, y_train):
        if not getattr(self, "_precomputed", False):
            self._fit(train_scores, y_train)
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


class SMM(SoftLearnerQMixin, AggregationMixin, MatchingScoreQuantifier):
    r"""Sample Mean Matching (SMM) quantification method.

    Estimates class prevalence by matching the mean score of the test samples 
    to a convex combination of positive and negative training scores. The mixture 
    weight :math:`\alpha` is computed as:

    .. math::

        \alpha = \frac{\bar{s}_{test} - \bar{s}_{neg}}{\bar{s}_{pos} - \bar{s}_{neg}}

    where :math:`\bar{s}` denotes the sample mean.

    Parameters
    ----------
    learner : estimator, optional
        Base probabilistic classifier.

    References
    ----------
    .. [2] Esuli et al. (2023). Learning to Quantify. Springer.
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
        learner=None,
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
        self.learner = learner
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, sample_weight=None):

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        X, y = self._fit_learner_predictions(
            X,
            y,
            learner_fitted=learner_fitted,
        )

        return self._fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        X = validate_data(self, X, ensure_2d=True)
        test_scores = self._predict_learner(X)
        return self._predict(test_scores)

    def aggregate(self, test_scores, train_scores, y_train):
        if not getattr(self, "_precomputed", False):
            self._fit(train_scores, y_train)
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
