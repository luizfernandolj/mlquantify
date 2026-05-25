import numpy as np

from mlquantify.compose._base import BaseComposeQuantifier
from mlquantify.losses import RegularizedMixtureNLLLoss
from mlquantify.representations import PredictionRepresentation
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_data, validate_prevalences


class LikelihoodComposeQuantifier(BaseComposeQuantifier):
    r"""Compose quantifier based on mixture negative log-likelihood.

    Minimizes the (optionally regularized) mixture negative log-likelihood
    to estimate class prevalences.  When a nested representation is
    provided, per-class likelihoods are obtained from that representation;
    otherwise they are derived from the test posteriors and training
    priors.

    This class is the backbone of the KDEy quantifier family.

    Parameters
    ----------
    representation : BaseRepresentation or None, default=None
        Optional representation (e.g. :class:`KDERepresentation`) that
        exposes ``class_likelihoods``.  When ``None``, the estimator's
        posterior outputs are used directly.
    estimator : classifier or None, default=None
        Base classifier.  Required when ``aggregative=True``.
    solver : str, default='slsqp'
        Prevalence solver passed to
        :func:`~mlquantify.solvers.minimize_prevalence`.
    aggregative : bool, default=True
        If ``True``, the estimator is used to obtain predictions before
        computing the representation.
    tau_0 : float, default=0.0
        First-order ordinal smoothness regularization weight.
    tau_1 : float, default=0.0
        Second-order ordinal smoothness regularization weight.
    random_state : int, RandomState instance, or None, default=None
        Random seed for the SLSQP starting point.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during fit.
    train_priors_ : ndarray of shape (n_classes,)
        Empirical class proportions in the training data.
    best_distance_ : float or None
        Objective value at the optimum after the last ``predict`` call.
    """

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
        """Fit the likelihood compose quantifier.

        When ``representation`` is provided the fitting is delegated to
        the parent :class:`BaseComposeQuantifier`.  Otherwise, only the
        estimator is trained (the likelihood is evaluated directly from
        posteriors at prediction time).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training class labels.
        estimator_fitted : bool, default=False
            Skip fitting the estimator if already fitted.
        sample_weight : array-like of shape (n_samples,) or None, \
            default=None
            Per-sample weights.
        cv_prediction : {'refit', 'ensemble'}, default='refit'
            Cross-validation strategy.

        Returns
        -------
        self : LikelihoodComposeQuantifier
            The fitted quantifier.
        """
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
        """Predict class prevalences by minimizing mixture NLL.

        When ``representation`` is set, delegates to the parent class.
        Otherwise computes per-class likelihoods from the estimator's
        posteriors and minimizes the (regularized) mixture NLL.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        prevalences : dict or ndarray of shape (n_classes,)
            Estimated class prevalences.
        """
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
        """Solve for prevalences by minimizing mixture NLL.

        Parameters
        ----------
        test_representation : array-like of shape (n_samples, n_classes)
            Per-class posteriors or likelihoods for each test instance.
        train_representation : ignored
            Not used; present for API compatibility.

        Returns
        -------
        prevalence : ndarray of shape (n_classes,)
            Estimated prevalence vector.
        loss : float
            Minimum NLL objective value.
        """
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
