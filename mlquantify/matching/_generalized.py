from mlquantify.compose import LinearComposeQuantifier
from mlquantify.representations import HistogramRepresentation


class GHDy(LinearComposeQuantifier):
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
        self.bins = bins
        super().__init__(
            learner=learner,
            representation=HistogramRepresentation(
                bins=bins,
                mode="onehot",
            ),
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=True,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )


class GHDx(LinearComposeQuantifier):
    def __init__(
        self,
        bins=10,
        loss="hellinger",
        solver="slsqp",
    ):
        self.bins = bins
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
        )
