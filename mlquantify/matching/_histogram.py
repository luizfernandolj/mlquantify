from abc import abstractmethod

import numpy as np

from mlquantify.base_aggregative import (
    _get_learner_function,
    AggregativeQuantifierMixin,
    SoftLearnerMixin,
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.matching._base import BaseMatchingQuantifier
from mlquantify.matching._utils import ternary_search
from mlquantify.compose.representations import HistogramRepresentation
from mlquantify.multiclass import binary_quantifier
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._get_scores import apply_cross_validation
from mlquantify.utils._validation import validate_data
from scipy.optimize import minimize, minimize_scalar


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
        super().__init__(
            representation=HistogramRepresentation(bins=bins_size, mode="histogram"),
            distance=distance, 
            solver=solver, 
        )
        self.strategy = strategy

    def _solve_prevalence(self, test_representation):
        solver = self.solver

        if solver == "auto":
            solver = "ternary" if self.distance in ["hellinger", "topsoe", "probsymm"] else "grid"

        neg_repr = self.tr_representation_[0]
        pos_repr = self.tr_representation_[1]

        def objective(alpha):
            mix_representation = self._mix(pos_repr, neg_repr, alpha)
            return self.get_distance(
                mix_representation,
                test_representation,
                distance=self.distance,
            )

        if solver == "ternary":
            alpha = ternary_search(0.0, 1.0, objective)
            distance = objective(alpha)

        elif solver == "grid":
            alphas = np.linspace(0.0, 1.0, 101)
            distances = np.array([objective(alpha) for alpha in alphas])
            best_idx = int(np.argmin(distances))
            alpha = float(alphas[best_idx])
            distance = float(distances[best_idx])
            self.distances_ = distances

        elif solver == "bounded":
            res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
            alpha = float(res.x)
            distance = float(res.fun)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        prevalence = np.array([1.0 - alpha, alpha], dtype=float)
        return prevalence, distance
    
    def _mix(self, pos_representation, neg_representation, alpha):
        return alpha * pos_representation + (1 - alpha) * neg_representation
    


class AggregateMatchingHistogramQuantifier(SoftLearnerMixin, AggregativeQuantifierMixin, MatchingHistogramQuantifier):

    _parameter_constraints = {
        "bins_size": ["array-like", None],
        "distance": [Options(["hellinger", "topsoe", "probsymm", "sqEuclidean", "euclidean"])],
        "solver": [Options(["auto", "grid", "ternary", "bounded"])],
        "cv": [Interval(2, None), None],
        "stratified": [bool],
        "shuffle": [bool],
        "random_state": ["random_state", None],
    }

    def __init__(
        self,
        learner=None,
        bins_size=None,
        distance="hellinger",
        solver="auto",
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        if bins_size is None:
            bins_size = np.append(np.linspace(2, 20, 10), 30).astype(int)
        self.learner = learner
        self.distance = distance
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        super().__init__(
            bins_size=bins_size,
            distance=distance,
            solver=solver,
            strategy=strategy,
        )


    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, sample_weight=None):

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        learner_function = _get_learner_function(self)

        if learner_fitted:
            train_scores = getattr(self.learner, learner_function)(X)
            y_train = y
        else:
            train_scores, y_train = apply_cross_validation(
                self.learner,
                X,
                y,
                function= learner_function,
                cv= self.cv,
                stratified= self.stratified,
                random_state= self.random_state,
                shuffle= self.shuffle
            )
            self.learner.fit(X, y)
        
        return self._fit(train_scores, y_train, sample_weight)

    def predict(self, X):
        X = validate_data(self, X, ensure_2d=True)
        test_scores = self.learner.predict_proba(X)
        return self._predict(test_scores)
    
    def aggregate(self, test_scores, train_scores, y_train):
        if not self._precomputed:
            self._fit(train_scores, y_train)
        return self._predict(test_scores)
        

class DyS(AggregateMatchingHistogramQuantifier):
    r"""Distribution y-Similarity with histogram score matching."""

    def __init__(
        self,
        learner=None,
        bins_size=None,
        distance="topsoe",
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            learner=learner,
            bins_size=bins_size,
            distance=distance,
            solver="ternary",
            strategy=strategy,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )

    




class HDy(AggregateMatchingHistogramQuantifier):
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
            learner=learner,
            bins_size=bins_size,
            distance="hellinger",
            solver="grid",
            strategy=strategy,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )


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

