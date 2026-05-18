from mlquantify.compose import LikelihoodComposeQuantifier


class MLPE(LikelihoodComposeQuantifier):
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
            representation=None,
            solver=solver,
            aggregative=True,
            tau_0=tau_0,
            tau_1=tau_1,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )