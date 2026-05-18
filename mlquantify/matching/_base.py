import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier
from mlquantify.utils._constraints import (
    Options,
)


from mlquantify.utils._validation import validate_data, validate_prevalences
from mlquantify.metrics import (
    hellinger,
    topsoe,
    probsymm,
    sqEuclidean
)


EPS = 1e-12


class BaseMatchingQuantifier(BaseQuantifier):
    r"""Base class for distribution matching quantifiers.

    Distribution matching quantifiers represent the training class-conditional
    distributions and the test distribution in a common space, then estimate
    class prevalences by matching the test representation with a convex
    combination of the training representations.
    """

    _parameter_constraints = {
        "distance": [Options(["hellinger", "topsoe", "probsymm", "sqEuclidean", "euclidean"])],
        "solver": [Options(["auto", "grid", "ternary", "slsqp"])],
        "normalize": ["boolean", None],
    }

    def __init__(
        self, 
        representation, 
        distance="hellinger", 
        solver="auto",
        normalize=None
        ):
        if normalize is None:
            normalize = distance in {
                "hellinger",
                "topsoe",
                "probsymm",
            }
        self.representation = representation
        self.distance = distance
        self.solver = solver
        self.normalize = normalize

    def _fit(self, X, y, sample_weight=None):
        """Fit the quantifier to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features.

        y : array-like of shape (n_samples,)
            Training data labels.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for fitting. If None, all samples are given equal weight.
        """
        self.representation.fit(X, y, sample_weight=sample_weight)
        
        if not hasattr(self.representation, "class_representations_"):
            raise AttributeError(
                "representation must define class_representations_ after fit()."
            )

        self.classes_ = getattr(self.representation, "classes_", np.unique(y))
        self.tr_representation_ = self.representation.class_representations_

        self._precomputed = True
        self.best_distance_ = None
        self.distances_ = None

        return self

    def _predict(self, X):
        if not self._precomputed:
            raise ValueError("This quantifier is not fitted yet.")

        X = validate_data(self, X)
        ts_representation = self.representation.transform(X)

        prevalences, distance = self._solve_prevalence(
            test_representation=ts_representation,
            train_representations=self.tr_representation_,
        )

        self.best_distance_ = distance

        return validate_prevalences(self, prevalences, self.classes_)
    
    def get_best_distance(self, X=None):
        if X is not None:
            self._predict(X)
        return self.best_distance_

    @staticmethod
    def _normalize_distribution(values):
        values = np.asarray(values, dtype=float)
        values = np.maximum(values, EPS)
        total = values.sum()
        if total <= EPS:
            return np.ones_like(values) / len(values)
        return values / total

    def get_distance(self, dist_train, dist_test, distance="hellinger"):
        """Compute a distance between two normalized representations."""
        dist_train = np.asarray(dist_train, dtype=float)
        dist_test = np.asarray(dist_test, dtype=float)

        if self.normalize:
            dist_train = self._normalize_distribution(dist_train)
            dist_test = self._normalize_distribution(dist_test)

        if dist_train.shape != dist_test.shape:
            raise ValueError("Distributions must have the same shape.")

        if distance == "hellinger":
            return float(hellinger(dist_train, dist_test))
        if distance == "topsoe":
            return float(topsoe(dist_train, dist_test))
        if distance == "probsymm":
            return float(probsymm(dist_train, dist_test))
        if distance == "sqEuclidean":
            return float(sqEuclidean(dist_train, dist_test))
        if distance == "euclidean":
            return float(np.sqrt(sqEuclidean(dist_train, dist_test)))

        raise ValueError(f"Invalid distance: {distance}")

    
    @staticmethod
    def _mixture(class_representations, prevalences):
        return np.asarray(prevalences) @ np.asarray(class_representations)

    @abstractmethod
    def _solve_prevalence(self, test_representation, train_representations):
        """Solve for class prevalences using train and test representations."""
