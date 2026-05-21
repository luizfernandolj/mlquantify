# matching/_generalized.py

from mlquantify.compose import (
    LinearComposeQuantifier,
    LikelihoodComposeQuantifier,
)
from mlquantify.base_aggregative import AggregationMixin, SoftLearnerQMixin
from mlquantify.losses import EnergyLoss
from mlquantify.representations import (
    DistanceRepresentation,
    HistogramRepresentation,
    KDERepresentation,
    PredictionRepresentation,
)


class GHDy(SoftLearnerQMixin, AggregationMixin, LinearComposeQuantifier):
    r"""Generalized HDy using histogram representations over posterior probabilities."""

    def __init__(
        self,
        learner=None,
        bins=10,
        loss="hellinger",
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
    r"""Generalized HDx using histogram representations over input features."""

    def __init__(
        self,
        bins=10,
        loss="hellinger",
        solver="slsqp",
        random_state=None,
    ):
        super().__init__(
            learner=None,
            representation=HistogramRepresentation(
                bins=bins,
                mode="onehot",
            ),
            loss=loss,
            solver=solver,
            aggregative=False,
            normalize=True,
            random_state=random_state,
        )
        self.bins = bins
        self.random_state = random_state


class GKDEyML(SoftLearnerQMixin, AggregationMixin, LikelihoodComposeQuantifier):
    r"""KDEy with maximum-likelihood estimation."""

    def __init__(
        self,
        learner=None,
        bandwidth=0.1,
        kernel="gaussian",
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


class EDy(SoftLearnerQMixin, AggregationMixin, LinearComposeQuantifier):
    r"""Energy distance over posterior probabilities."""

    def __init__(
        self,
        learner=None,
        metric="euclidean",
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
    r"""Energy distance over input features."""

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
