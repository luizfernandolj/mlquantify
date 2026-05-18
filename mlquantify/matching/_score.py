from abc import abstractmethod

import numpy as np

from mlquantify.base_aggregative import (
    AggregationMixin,
    SoftLearnerQMixin,
)
from mlquantify.representations import ScoreRepresentation
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
            representation=ScoreRepresentation(),
            distance="sqEuclidean",
            solver=solver,
            normalize=False,
        )
        self.strategy = strategy

    @abstractmethod
    def _solve_prevalence(self, test_representation):
        """Estimate binary prevalence from test scores."""


class SORD(SoftLearnerQMixin, AggregationMixin, MatchingScoreQuantifier):
    r"""Sample Ordinal Distance quantifier.

    SORD estimates prevalence by comparing the ordered score distributions of
    the test sample and a weighted mixture of the positive and negative training
    score samples.
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
        if not self._precomputed:
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
    r"""Score Moment Matching quantifier.

    SMM estimates prevalence by matching moments of the score distribution.
    For the first moment, this corresponds to matching the average score of the
    test sample with a convex combination of the average scores of the positive
    and negative training samples.
    """

    _parameter_constraints = {
        "moment": [Interval(1, None)],
        "solver": [Options(["auto", "grid", "ternary", "bounded"])],
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
        solver="bounded",
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        self.moment = moment

        super().__init__(
            learner=learner,
            solver=solver,
            strategy=strategy,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )

    def _solve_prevalence(self, test_representation, train_representations):
        test_scores = np.asarray(test_representation, dtype=float).ravel()
        neg_repr = train_representations[0]
        pos_repr = train_representations[1]

        moment = int(self.moment)

        pos_scores = np.asarray(pos_repr, dtype=float)
        neg_scores = np.asarray(neg_repr, dtype=float)
        
        pos_moment = np.mean(pos_scores ** moment)
        neg_moment = np.mean(neg_scores ** moment)
        test_moment = np.mean(test_scores ** moment)

        def objective(alpha):
            mixture_moment = alpha * pos_moment + (1.0 - alpha) * neg_moment
            return float((mixture_moment - test_moment) ** 2)

        prevalences, distance = minimize_prevalence(
            objective=objective,
            n_classes=2,
            solver=self.solver,
        )

        return prevalences, distance
