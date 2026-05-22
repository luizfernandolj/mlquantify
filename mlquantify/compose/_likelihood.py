import numpy as np

from mlquantify.compose._base import BaseComposeQuantifier
from mlquantify.losses import RegularizedMixtureNLLLoss
from mlquantify.representations import PredictionRepresentation
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_data, validate_prevalences


class LikelihoodComposeQuantifier(BaseComposeQuantifier):
    def __init__(
        self,
        representation=None,
        estimator=None,
        solver="slsqp",
        aggregative=True,
        tau_0=0.0,
        tau_1=0.0,
        random_state=None,
    ):
        super().__init__(
            representation=representation,
            estimator=estimator,
            solver=solver,
            aggregative=aggregative,
        )
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        estimator_fitted=False,
        sample_weight=None,
        cv_prediction="refit",
    ):
        if self.representation is not None:
            return super().fit(
                X,
                y,
                estimator_fitted=estimator_fitted,
                sample_weight=sample_weight,
                cv_prediction=cv_prediction,
            )

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        if self._uses_estimator_predictions():
            X_rep, y_rep = self._fit_estimator_predictions(
                X,
                y,
                estimator_fitted=estimator_fitted,
                cv_prediction=cv_prediction,
            )
        else:
            X_rep, y_rep = X, y

        self.train_priors_ = np.asarray([
            np.mean(y_rep == cls)
            for cls in self.classes_
        ])
        self.train_representation_ = None
        self.best_distance_ = None
        self.distances_ = None

        return self

    def predict(self, X):
        if self.representation is not None:
            return super().predict(X)

        X = validate_data(self, X)
        if self._uses_estimator_predictions():
            test_representation = self._predict_estimator(X)
        else:
            test_representation = X

        prevalences, distance = self._solve_prevalence(
            test_representation=test_representation,
            train_representation=None,
        )

        self.best_distance_ = distance

        return validate_prevalences(self, prevalences, self.classes_)

    def _solve_prevalence(self, test_representation, train_representation):
        class_likelihoods = self._class_likelihoods(test_representation)
        loss_function = RegularizedMixtureNLLLoss(
            tau_0=self.tau_0,
            tau_1=self.tau_1,
        )

        def objective(prevalences):
            return loss_function(prevalences, class_likelihoods)

        return minimize_prevalence(
            objective=objective,
            n_classes=len(self.classes_),
            solver=self.solver,
            random_state=self.random_state,
        )

    def _class_likelihoods(self, test_representation):
        representation = self.representation

        if isinstance(representation, PredictionRepresentation):
            nested_representation = representation.representation

            if nested_representation is not None and hasattr(representation, "class_likelihoods"):
                return representation.class_likelihoods(test_representation)

        elif representation is not None and hasattr(representation, "class_likelihoods"):
            return self.representation.class_likelihoods(test_representation)

        priors = np.asarray(self.train_priors_, dtype=float)
        pxy = np.asarray(test_representation, dtype=float) / np.maximum(priors, 1e-12)
        pxy = pxy / np.maximum(pxy.sum(axis=1, keepdims=True), 1e-12)
        return pxy.T

    def _regularization(self, prevalences):
        p = np.asarray(prevalences, dtype=float)

        xi_0 = np.sum((p[1:] - p[:-1]) ** 2) / 2.0
        xi_1 = np.sum((-p[:-2] + 2 * p[1:-1] - p[2:]) ** 2) / 2.0

        return self.tau_0 * xi_0 + self.tau_1 * xi_1
