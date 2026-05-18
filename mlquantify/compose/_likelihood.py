import numpy as np

from mlquantify.compose._base import BaseComposeQuantifier
from mlquantify.matching._utils import negative_log_likelihood
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_data, validate_prevalences


EPS = 1e-12


class LikelihoodComposeQuantifier(BaseComposeQuantifier):
    def __init__(
        self,
        representation=None,
        learner=None,
        solver="slsqp",
        aggregative=True,
        tau_0=0.0,
        tau_1=0.0,
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
        self.tau_0 = tau_0
        self.tau_1 = tau_1

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, sample_weight=None):
        if self.representation is not None:
            return super().fit(
                X,
                y,
                learner_fitted=learner_fitted,
                sample_weight=sample_weight,
            )

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        X_rep, y_rep = self._fit_learner_predictions(
            X,
            y,
            learner_fitted=learner_fitted,
        )

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
        test_representation = self._predict_learner(X)

        prevalences, distance = self._solve_prevalence(
            test_representation=test_representation,
            train_representation=None,
        )

        self.best_distance_ = distance

        return validate_prevalences(self, prevalences, self.classes_)

    def _solve_prevalence(self, test_representation, train_representation):
        likelihoods = self._class_likelihoods(test_representation)

        def objective(prevalences):
            mixture = likelihoods @ prevalences
            loss = -np.log(np.maximum(mixture, 1e-12)).mean()
            loss += self._regularization(prevalences)
            return float(loss)

        return minimize_prevalence(
            objective=objective,
            n_classes=len(self.classes_),
            solver=self.solver,
        )

    def _class_likelihoods(self, test_representation):
        if self.representation is not None and hasattr(self.representation, "class_likelihoods"):
            return self.representation.class_likelihoods(test_representation).T

        priors = np.asarray(self.train_priors_, dtype=float)
        pxy = np.asarray(test_representation, dtype=float) / np.maximum(priors, 1e-12)
        pxy = pxy / np.maximum(pxy.sum(axis=1, keepdims=True), 1e-12)
        return pxy

    def _regularization(self, prevalences):
        p = np.asarray(prevalences, dtype=float)

        xi_0 = np.sum((p[1:] - p[:-1]) ** 2) / 2.0
        xi_1 = np.sum((-p[:-2] + 2 * p[1:-1] - p[2:]) ** 2) / 2.0

        return self.tau_0 * xi_0 + self.tau_1 * xi_1
