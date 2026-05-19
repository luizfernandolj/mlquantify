import numpy as np

from mlquantify.base_aggregative import (
    AggregationMixin,
    CrispLearnerQMixin,
    SoftLearnerQMixin,
)
from mlquantify.compose import LinearComposeQuantifier
from mlquantify.representations import PredictionRepresentation


class GAC(CrispLearnerQMixin, AggregationMixin, LinearComposeQuantifier):
    r"""Generalized Adjusted Count."""

    def __init__(
        self,
        learner=None,
        loss="ls",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            learner=learner,
            representation=PredictionRepresentation(
                method="hard",
                average=True,
            ),
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=False,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state


class GPAC(SoftLearnerQMixin, AggregationMixin, LinearComposeQuantifier):
    r"""Generalized Probabilistic Adjusted Count."""

    def __init__(
        self,
        learner=None,
        loss="ls",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            learner=learner,
            representation=PredictionRepresentation(
                method="soft",
                average=True,
            ),
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=False,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state


class FM(SoftLearnerQMixin, AggregationMixin, LinearComposeQuantifier):
    r"""Friedman Method."""

    def __init__(
        self,
        learner=None,
        loss="ls",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            learner=learner,
            representation=PredictionRepresentation(
                func=self._friedman_prediction,
                average=True,
            ),
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=False,
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
