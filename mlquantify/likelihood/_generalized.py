# likelihood/_generalized.py

from mlquantify.base_aggregative import AggregationMixin, SoftLearnerQMixin
from mlquantify.compose import LikelihoodComposeQuantifier
from mlquantify.representations import PredictionRepresentation


class MLPE(SoftLearnerQMixin, AggregationMixin, LikelihoodComposeQuantifier):
    r"""Maximum Likelihood Prior Estimation.

    This corresponds to a likelihood maximization method over adjusted
    posterior probabilities.
    """

    def __init__(
        self,
        learner=None,
        solver="slsqp",
        tau_0=0.0,
        tau_1=0.0,
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
            ),
            solver=solver,
            aggregative=True,
            tau_0=tau_0,
            tau_1=tau_1,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
