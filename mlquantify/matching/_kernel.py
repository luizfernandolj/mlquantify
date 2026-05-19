import numpy as np

from mlquantify.losses import LeastSquaresLoss
from mlquantify.matching._base import BaseMatchingQuantifier
from mlquantify.representations import KernelMeanRepresentation
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._validation import validate_data


class MatchingKernelQuantifier(BaseMatchingQuantifier):
    r"""Abstract base class for kernel mean matching quantifiers."""

    _parameter_constraints = {
        "kernel": [Options(["rbf", "linear", "poly", "sigmoid", "cosine"])],
        "gamma": [Interval(0, None, inclusive_left=False), Options([None])],
        "degree": [Interval(1, None, inclusive_left=True)],
        "coef0": [Interval(0, None, inclusive_left=True)],
        "solver": [Options(["auto", "slsqp"])],
    }

    def __init__(
        self,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=0.0,
        solver="slsqp",
    ):
        super().__init__(
            representation=KernelMeanRepresentation(
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
            ),
            normalize=False,
        )
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.solver = solver

    def _solve_prevalence(self, test_representation, train_representations):
        solver = "slsqp" if self.solver == "auto" else self.solver

        class_means = np.asarray(train_representations, dtype=float)
        test_mean = np.asarray(test_representation, dtype=float)
        loss_function = LeastSquaresLoss()

        def objective(prevalence):
            mixture_mean = prevalence @ class_means
            return loss_function(mixture_mean, test_mean)

        prevalence, loss = minimize_prevalence(
            objective=objective,
            n_classes=len(self.classes_),
            solver=solver,
        )

        return prevalence, loss

class MMD_RKHS(MatchingKernelQuantifier):
    r"""Maximum Mean Discrepancy in RKHS for class-ratio estimation."""

    def __init__(
        self,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=0.0,
        solver="slsqp",
    ):
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            solver=solver,
        )
        
    def fit(self, X, y, sample_weight=None):
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        return self._fit(X, y, sample_weight)

    def predict(self, X):
        X = validate_data(self, X)
        return self._predict(X)
