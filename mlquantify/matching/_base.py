import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier
from mlquantify.losses import get_loss, normalize_distribution
from mlquantify.utils._constraints import (
    Options,
)


from mlquantify.utils._validation import validate_data, validate_prevalences


class BaseMatchingQuantifier(BaseQuantifier):
    r"""Base class for distribution matching quantifiers.

    Distribution matching quantifiers represent the training class-conditional
    distributions and the test distribution in a common space, then estimate
    class prevalences by matching the test representation with a convex
    combination of the training representations.

    Parameters
    ----------
    representation : representation instance
        A representation object that can be fitted to training data and then
        used to transform both training and test data into a common space.
        Must define a ``class_representations_`` attribute after fitting.
    normalize : bool, default=False
        Whether to normalize the representations to form valid probability
        distributions before computing distances.

    Attributes
    ----------
    representation : representation instance
        The representation object used for fitting and transforming data.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    tr_representation_ : ndarray of shape (n_classes, n_representation_features)
        Class-conditional representations learned from the training data.
    best_distance_ : float
        The distance between the test representation and the best-matching
        convex combination of the training representations.
    distances_ : list of float
        Distances computed during the optimization process (if applicable).

    Examples
    --------
    >>> from mlquantify.matching import BaseMatchingQuantifier
    >>> from mlquantify.representations import KDERepresentation
    >>> import numpy as np
    >>> class MyMatching(BaseMatchingQuantifier):
    ...     def __init__(self):
    ...         super().__init__(representation=KDERepresentation())
    ...     def _solve_prevalence(self, test_representation, train_representations):
    ...         # Dummy implementation for illustration
    ...         prevalences = np.random.dirichlet(np.ones(len(train_representations)))
    ...         distance = np.random.rand()
    ...         return prevalences, distance
    >>> q = MyMatching()   # subclasses implement _solve_prevalence
    """

    _parameter_constraints = {
        "distance": [Options([
            "hellinger",
            "topsoe",
            "probsymm",
            "sqEuclidean",
            "euclidean",
            "least_squares",
            "least-squares",
            "least squares",
            "ls",
            "l2",
            None,
        ])],
        "solver": [Options(["auto", "grid", "ternary", "slsqp"])],
        "normalize": ["boolean", None],
    }

    def __init__(
        self, 
        representation,  
        normalize=False
        ):
        self.representation = representation
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
        return normalize_distribution(values)

    def get_distance(self, dist_train, dist_test, distance="hellinger"):
        """Compute a distance between two normalized representations."""
        loss_function = get_loss(
            loss=distance,
            normalize=self.normalize,
        )
        return loss_function(dist_train, dist_test)

    
    @staticmethod
    def _mixture(class_representations, prevalences):
        return np.asarray(prevalences) @ np.asarray(class_representations)

    @abstractmethod
    def _solve_prevalence(self, test_representation, train_representations):
        """Solve for class prevalences using train and test representations."""
