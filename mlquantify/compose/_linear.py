from mlquantify.compose._base import BaseComposeQuantifier
from mlquantify.compose._losses import get_loss
from mlquantify.compose._utils import (
    class_representations_to_matrix,
    validate_matrix_problem,
)
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
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            representation=representation,
            learner=learner,
            solver=solver,
            aggregative=aggregative,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )

        self.loss = loss
        self.normalize = normalize

    def _solve_prevalence(self, test_representation, train_representation):
        M = class_representations_to_matrix(train_representation)
        q = test_representation

        M, q = validate_matrix_problem(M, q)

        normalize = self.normalize

        if normalize is None:
            normalize = self.loss in {
                "hellinger",
                "topsoe",
                "probsymm",
            }

        loss_function = get_loss(
            loss=self.loss,
            normalize=normalize,
        )

        def objective(prevalences):
            mixture = M @ prevalences
            return loss_function(mixture, q)

        return minimize_prevalence(
            objective=objective,
            n_classes=len(self.classes_),
            solver=self.solver,
        )