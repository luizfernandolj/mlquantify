from abc import abstractmethod

import numpy as np

from mlquantify.base_aggregative import (
    AggregationMixin,
    SoftLearnerQMixin,
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.matching._base import BaseMatchingQuantifier
from mlquantify.matching._utils import ternary_search
from mlquantify.representations import HistogramRepresentation
from mlquantify.multiclass import binary_quantifier
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._get_scores import apply_cross_validation
from mlquantify.utils._validation import validate_data
from scipy.optimize import minimize, minimize_scalar
from mlquantify.solvers import minimize_prevalence


@binary_quantifier(strategy_attr="strategy")
class MatchingHistogramQuantifier(BaseMatchingQuantifier):
    r"""Abstract base class for histogram-based distribution matching.

    Subclasses learn class-conditional histogram representations from training
    data and estimate the test prevalence by finding the mixture of those
    histograms that best matches the test histogram.
    """

    def __init__(
        self,
        bins_size,
        distance="hellinger",
        solver="auto",
        strategy="ovr",
    ):
        self.bins_size = bins_size
        self.distance = distance
        self.solver = solver
        self.strategy = strategy
        super().__init__(
            representation=HistogramRepresentation(bins=bins_size, mode="histogram"),
            distance=distance, 
            solver=solver, 
        )

    def _solve_prevalence(self, test_representation, train_representations):
        solver = self.solver

        if solver == "auto":
            solver = "ternary" if self.distance in ["hellinger", "topsoe", "probsymm"] else "grid"

        neg_repr = train_representations[0]
        pos_repr = train_representations[1]

        def objective(alpha):
            mix_representation = self._mixture([pos_repr, neg_repr], [1 - alpha, alpha])
            return self.get_distance(
                mix_representation,
                test_representation,
                distance=self.distance,
            )

        return minimize_prevalence(
            objective=objective,
            n_classes=2,
            solver=solver
        )

    
        

class DyS(SoftLearnerQMixin, AggregationMixin, MatchingHistogramQuantifier):
    r"""Distribution y-Similarity with histogram score matching."""

    def __init__(
        self,
        learner=None,
        bins_size=None,
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            bins_size=bins_size,
            distance="hellinger",
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

    




class HDy(SoftLearnerQMixin, AggregationMixin, MatchingHistogramQuantifier):
    r"""Distribution y-Similarity with histogram score matching."""

    def __init__(
        self,
        learner=None,
        bins_size=None,
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            bins_size=bins_size,
            distance="hellinger",
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


class HDx(MatchingHistogramQuantifier):
    r"""Distribution y-Similarity with histogram score matching."""

    _parameter_constraints = {
        "bins_size": ["array-like", None],
        "strategy": [Options(["ovr", "ovo"])],
    }

    def __init__(
        self,
        bins_size=None,
        strategy="ovr"
    ):
        if bins_size is None:
            bins_size = np.append(np.linspace(2, 20, 10), 30).astype(int)

        super().__init__(
            bins_size=bins_size,
            distance="hellinger",
            solver="ternary",
            strategy=strategy,
        )

    def fit(self, X, y, sample_weight=None):

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        return self._fit(X, y, sample_weight)

    def predict(self, X):
        X = validate_data(self, X)
        return self._predict(X)

