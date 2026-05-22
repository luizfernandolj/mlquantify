import numpy as np

from mlquantify.base_aggregative import (
    AggregativeMixin,
    CrispPredictionMixin,
    SoftPredictionMixin,
)
from mlquantify.compose import LinearComposeQuantifier
from mlquantify.representations import PredictionRepresentation


class GACC(CrispPredictionMixin, AggregativeMixin, LinearComposeQuantifier):
    r"""Generalized Adjusted Classify and Count.
    
    This class implements a generalized version of the Adjusted Classify and Count (ACC) method for quantification.
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
    r"""Generalized Probabilistic Adjusted Count."""

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
    r"""Friedman Method."""

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
