from abc import abstractmethod

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import KernelDensity

from mlquantify.base_aggregative import _get_learner_function
from mlquantify.matching._base import BaseMatchingQuantifier
from mlquantify.matching._utils import (
    EPS,
    gaussian_kernel,
    negative_log_likelihood,
    normalize_simplex,
    optimize_on_simplex,
)
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._get_scores import apply_cross_validation
from mlquantify.utils._validation import validate_data, validate_prevalences


class KernelQuantifier(BaseMatchingQuantifier):
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
        super().__init__(distance="sqEuclidean", solver=solver)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.X_train_ = None
        self.y_train_ = None
        self.class_means_ = None
        self.K_train_ = None

    def _fit_representation(self, X, y, sample_weight=None):
        self.X_train_ = X
        self.y_train_ = y
        self.class_means_, self.K_train_ = self._compute_class_means(
            X,
            y,
            sample_weight=sample_weight,
        )
        return self.class_means_

    def _transform_representation(self, X):
        return self._compute_unlabeled_mean(X)

    def _kernel_kwargs(self):
        params = {}
        if self.kernel == "rbf" and self.gamma is not None:
            params["gamma"] = self.gamma
        if self.kernel == "poly":
            params["degree"] = self.degree
            params["coef0"] = self.coef0
        if self.kernel == "sigmoid":
            if self.gamma is not None:
                params["gamma"] = self.gamma
            params["coef0"] = self.coef0
        return params

    def _compute_class_means(self, X, y, sample_weight=None):
        K = pairwise_kernels(X, X, metric=self.kernel, **self._kernel_kwargs())

        means = []
        sample_weight = None if sample_weight is None else np.asarray(sample_weight)
        for cls in self.classes_:
            mask = y == cls
            if sample_weight is None:
                means.append(K[mask].mean(axis=0))
            else:
                weights = sample_weight[mask]
                means.append(np.average(K[mask], axis=0, weights=weights))

        return np.vstack(means), K

    def _compute_unlabeled_mean(self, X):
        K_test_train = pairwise_kernels(
            X,
            self.X_train_,
            metric=self.kernel,
            **self._kernel_kwargs(),
        )
        return K_test_train.mean(axis=0)

    def _build_qp_matrices(self, class_means, test_mean):
        G = class_means @ class_means.T
        h = class_means @ test_mean
        return G, h

    def _solve_prevalence(self, test_representation):
        prevalence, distance = self.best_mixture(test_representation)
        return prevalence, distance

    @abstractmethod
    def best_mixture(self, test_representation):
        """Return prevalences and objective value for a test representation."""


class MMD_RKHS(KernelQuantifier):
    r"""Maximum Mean Discrepancy in RKHS for class-ratio estimation."""

    def best_mixture(self, test_representation):
        class_means = self.class_means_
        test_mean = np.asarray(test_representation, dtype=float)
        G, h = self._build_qp_matrices(class_means, test_mean)

        def objective(theta):
            theta = normalize_simplex(theta)
            return float(theta @ G @ theta - 2.0 * (h @ theta))

        prevalence, loss = optimize_on_simplex(objective, len(self.classes_))
        self.best_distance_ = loss
        return prevalence, loss


class KDEyQuantifier(BaseMatchingQuantifier):
    r"""Abstract base class for KDEy score-distribution matching quantifiers.

    KDEy methods represent the class-conditional distributions of posterior
    probabilities with kernel density estimators and solve for the prevalence
    vector that best explains the test posterior distribution.
    """

    _parameter_constraints = {
        "bandwidth": [Interval(0, None, inclusive_left=False)],
        "kernel": [Options(["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"])],
    }

    def __init__(self, learner=None, bandwidth=0.1, kernel="gaussian"):
        super().__init__(distance="hellinger", solver="slsqp")
        self.learner = learner
        self.bandwidth = bandwidth
        self.kernel = kernel

        self.train_predictions_ = None
        self.y_train_ = None
        self.class_kdes_ = None

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.has_estimator = True
        tags.estimator_function = "predict_proba"
        tags.estimator_type = "soft"
        tags.prediction_requirements.requires_train_proba = True
        tags.prediction_requirements.requires_train_labels = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        learner_fitted=False,
        cv=5,
        stratified=True,
        shuffle=False,
        sample_weight=None,
    ):
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        learner_function = _get_learner_function(self)
        if learner_fitted:
            train_predictions = getattr(self.learner, learner_function)(X)
            y_train = y
        else:
            train_predictions, y_train = apply_cross_validation(
                self.learner,
                X,
                y,
                function=learner_function,
                cv=cv,
                stratified=stratified,
                shuffle=shuffle,
            )
            self.learner.fit(X, y)

        self.train_predictions_ = np.asarray(train_predictions, dtype=float)
        self.y_train_ = np.asarray(y_train)
        self.tr_representation_ = self._fit_representation(
            self.train_predictions_,
            self.y_train_,
            sample_weight=sample_weight,
        )
        self._precomputed = True
        return self

    def predict(self, X):
        X = validate_data(self, X)
        predictions = getattr(self.learner, _get_learner_function(self))(X)
        test_representation = self._transform_representation(predictions)
        prevalence, distance = self._solve_prevalence(test_representation)
        self.best_distance_ = distance
        return validate_prevalences(self, prevalence, self.classes_)

    def _fit_representation(self, X, y, sample_weight=None):
        X = self._as_2d_samples(X)
        self._fit_kde_models(X, y, sample_weight=sample_weight)
        return {
            "predictions": np.asarray(X, dtype=float),
            "y": np.asarray(y),
        }

    def _transform_representation(self, X):
        return self._as_2d_samples(X)

    @staticmethod
    def _as_2d_samples(values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return values.reshape(-1, 1)
        return np.atleast_2d(values)

    def _fit_kde_models(self, predictions, y, sample_weight=None):
        predictions = self._as_2d_samples(predictions)
        y = np.asarray(y)
        sample_weight = None if sample_weight is None else np.asarray(sample_weight)

        self.class_kdes_ = []
        for cls in self.classes_:
            mask = y == cls
            cls_predictions = predictions[mask]
            weights = None if sample_weight is None else sample_weight[mask]
            if cls_predictions.shape[0] == 0:
                cls_predictions = np.ones((1, predictions.shape[1])) / predictions.shape[1]
                weights = None

            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(cls_predictions, sample_weight=weights)
            self.class_kdes_.append(kde)

    def _solve_prevalence(self, test_representation):
        prevalence, distance = self.best_mixture(test_representation)
        return prevalence, distance

    @abstractmethod
    def best_mixture(self, test_representation):
        """Return prevalences and objective value for a test representation."""


class KDEyML(KDEyQuantifier):
    r"""KDEy with maximum-likelihood matching."""

    def best_mixture(self, test_representation):
        test_representation = self._as_2d_samples(test_representation)
        class_likelihoods = np.array(
            [
                np.exp(kde.score_samples(test_representation)) + EPS
                for kde in self.class_kdes_
            ]
        )

        def objective(alpha):
            mixture_likelihoods = alpha @ class_likelihoods
            return negative_log_likelihood(mixture_likelihoods)

        prevalence, loss = optimize_on_simplex(objective, len(self.classes_))
        self.best_distance_ = loss
        return prevalence, loss


class KDEyHD(KDEyQuantifier):
    r"""KDEy with Monte Carlo Hellinger-distance matching."""

    _parameter_constraints = {
        "bandwidth": [Interval(0, None, inclusive_left=False)],
        "kernel": [Options(["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"])],
        "montecarlo_trials": [Interval(1, None, discrete=True)],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        learner=None,
        bandwidth=0.1,
        kernel="gaussian",
        montecarlo_trials=10000,
        random_state=None,
    ):
        super().__init__(learner=learner, bandwidth=bandwidth, kernel=kernel)
        self.montecarlo_trials = montecarlo_trials
        self.random_state = random_state

    def _fit_representation(self, X, y, sample_weight=None):
        representation = super()._fit_representation(X, y, sample_weight=sample_weight)

        n_classes = len(self.classes_)
        samples_per_class = max(1, int(np.ceil(self.montecarlo_trials / n_classes)))
        samples = []
        for class_idx, kde in enumerate(self.class_kdes_):
            random_state = self.random_state
            if isinstance(random_state, (int, np.integer)):
                random_state = int(random_state) + class_idx
            samples.append(kde.sample(samples_per_class, random_state=random_state))

        self.reference_samples_ = np.vstack(samples)
        self.reference_classwise_density_ = np.array(
            [np.exp(kde.score_samples(self.reference_samples_)) for kde in self.class_kdes_]
        )
        self.reference_density_ = np.mean(self.reference_classwise_density_, axis=0) + EPS
        return representation

    def best_mixture(self, test_representation):
        test_representation = self._as_2d_samples(test_representation)
        test_kde = KernelDensity(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
        ).fit(test_representation)

        test_density = np.exp(test_kde.score_samples(self.reference_samples_)) + EPS
        importance_weights = test_density / self.reference_density_
        class_ratios = self.reference_classwise_density_ / test_density

        def objective(alpha):
            density_ratio = alpha @ class_ratios
            return float(np.mean(((np.sqrt(density_ratio) - 1.0) ** 2) * importance_weights))

        prevalence, loss = optimize_on_simplex(objective, len(self.classes_))
        self.best_distance_ = loss
        return prevalence, loss


class KDEyCS(KDEyQuantifier):
    r"""KDEy with closed-form Cauchy-Schwarz divergence matching."""

    _parameter_constraints = {
        "bandwidth": [Interval(0, None, inclusive_left=False)],
        "kernel": [Options(["gaussian"])],
    }

    def _fit_representation(self, X, y, sample_weight=None):
        if self.kernel != "gaussian":
            raise ValueError("KDEyCS supports only the gaussian KDE kernel.")

        predictions = self._as_2d_samples(X)
        y = np.asarray(y)

        self.centers_ = [predictions[y == cls] for cls in self.classes_]
        self.class_counts_ = np.array([max(1, len(centers)) for centers in self.centers_], dtype=float)
        self.effective_bandwidth_ = np.sqrt(2.0) * self.bandwidth

        gram = np.zeros((len(self.classes_), len(self.classes_)))
        for i, left in enumerate(self.centers_):
            for j, right in enumerate(self.centers_[i:], start=i):
                value = np.sum(
                    gaussian_kernel(
                        left,
                        right,
                        bandwidth=self.effective_bandwidth_,
                    )
                )
                gram[i, j] = gram[j, i] = value

        self.train_gram_ = gram
        return {
            "predictions": predictions,
            "y": y,
        }

    def best_mixture(self, test_representation):
        test_representation = self._as_2d_samples(test_representation)
        train_test_kernel_sums = np.array(
            [
                np.sum(
                    gaussian_kernel(
                        centers,
                        test_representation,
                        bandwidth=self.effective_bandwidth_,
                    )
                )
                for centers in self.centers_
            ]
        )
        test_weight = 1.0 / max(1, test_representation.shape[0])

        def objective(alpha):
            alpha = normalize_simplex(alpha)
            weighted_alpha = alpha / (self.class_counts_ + EPS)
            train_test_term = np.dot(weighted_alpha, train_test_kernel_sums) * test_weight
            train_train_term = weighted_alpha @ ((self.train_gram_ + EPS) @ weighted_alpha)
            return float(-np.log(train_test_term + EPS) + 0.5 * np.log(train_train_term + EPS))

        prevalence, loss = optimize_on_simplex(objective, len(self.classes_))
        self.best_distance_ = loss
        return prevalence, loss
