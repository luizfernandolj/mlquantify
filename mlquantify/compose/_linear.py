from mlquantify.compose._base import BaseComposeQuantifier
from mlquantify.compose._utils import (
    class_representations_to_matrix,
    validate_matrix_problem,
)
from mlquantify.losses import EnergyLoss, get_loss
from mlquantify.solvers import minimize_prevalence


class LinearComposeQuantifier(BaseComposeQuantifier):
    r"""Compose quantifier for linear representation matching.

    Solves problems of the form:

        q ≈ M p

    where p is the prevalence vector.
    """

    def __init__(
        self,
        representation,
        learner=None,
        loss="hellinger",
        solver="slsqp",
        aggregative=True,
        normalize=None,
        random_state=None,
    ):
        super().__init__(
            representation=representation,
            learner=learner,
            solver=solver,
            aggregative=aggregative,
        )

        self.loss = loss
        self.normalize = normalize
        self.random_state = random_state

    def _solve_prevalence(self, test_representation, train_representation):
        M = class_representations_to_matrix(train_representation)
        q = test_representation

        M, q = validate_matrix_problem(M, q)

        normalize = self.normalize

        if normalize is None and isinstance(self.loss, str):
            normalize = self.loss in {
                "hellinger",
                "topsoe",
                "probsymm",
            }
        elif normalize is None:
            normalize = False

        loss_function = get_loss(
            loss=self.loss,
            normalize=normalize,
        )

        def objective(prevalences):
            if isinstance(loss_function, EnergyLoss):
                return loss_function(prevalences, q, M)

            mixture = M @ prevalences
            return loss_function(mixture, q)

        return minimize_prevalence(
            objective=objective,
            n_classes=len(self.classes_),
            solver=self.solver,
            random_state=self.random_state,
        )
