from mlquantify.compose import LinearComposeQuantifier
from mlquantify.representations import HardPredictionRepresentation, SoftPredictionRepresentation


class GAC(LinearComposeQuantifier):
    def __init__(
        self,
        learner=None,
        loss="sqEuclidean",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            learner=learner,
            representation=HardPredictionRepresentation(),
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=False,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )


class GPAC(LinearComposeQuantifier):
    def __init__(
        self,
        learner=None,
        loss="sqEuclidean",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            learner=learner,
            representation=SoftPredictionRepresentation(),
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=False,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )