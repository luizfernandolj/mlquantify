import numpy as np
from abc import abstractmethod
from sklearn.neighbors import KernelDensity

from mlquantify.utils._decorators import _fit_context
from mlquantify.base import BaseQuantifier
from mlquantify.utils import validate_y, validate_predictions, validate_data, check_classes_attribute
from mlquantify.base_aggregative import AggregationMixin, SoftLearnerQMixin, _get_learner_function
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._get_scores import apply_cross_validation
from mlquantify.utils._validation import validate_prevalences

EPS = 1e-12

class BaseKDE(SoftLearnerQMixin, AggregationMixin, BaseQuantifier):
    r"""
    Base class for KDEy quantification methods.
    
    KDEy methods model the class-conditional densities of posterior probabilities
    using Kernel Density Estimation (KDE) in the probability simplex space.
    Given a probabilistic classifier's posterior outputs, each class distribution
    is approximated as a smooth density function via KDE. Class prevalences in
    the test set are estimated as the mixture weights of these densities that best 
    explain the test posterior distribution.
    
    Formally, KDEy approximates the test posterior distribution as:

    \[
    p_{test}(x) \approx \sum_{k=1}^K \alpha_k p_k(x),
    \]

    where \( p_k(x) \) is the KDE of the posterior scores of class \( k \) on training data,
    and \( \alpha_k \) are the unknown class prevalences to be estimated under:

    \[
    \alpha_k \geq 0, \quad \sum_{k=1}^K \alpha_k = 1.
    \]
    
    The quantification task is then to find the vector \( \boldsymbol{\alpha} = (\alpha_1,\dots,\alpha_K) \)
    minimizing an objective function defined on the mixture density and the test posteriors,
    subject to the simplex constraints on \( \boldsymbol{\alpha} \).

    Attributes
    ----------
    learner : estimator
        The underlying probabilistic classifier yielding posterior predictions.
    bandwidth : float
        Bandwidth (smoothing parameter) for the KDE models.
    kernel : str
        Kernel type used in KDE (e.g., 'gaussian').
    _precomputed : bool
        Indicates whether KDE models have been fitted on training data.
    best_distance : float or None
        Stores the best value of the objective (distance or loss) achieved.

    Methods
    -------
    fit(X, y, learner_fitted=False)
        Fits KDE models for each class using posterior predictions of the learner.
    predict(X)
        Aggregates learner’s posterior predictions on X to estimate class prevalences.
    aggregate(predictions, train_predictions, train_y_values)
        Core estimation method that validates inputs, ensures KDE precomputation,
        and calls `_solve_prevalences` implemented by subclasses.
    _fit_kde_models(train_predictions, train_y_values)
        Fits KDE model per class on training data posteriors.
    _solve_prevalences(predictions)
        Abstract method to estimate prevalence vector \( \boldsymbol{\alpha} \) for given posteriors.
        Must be implemented by subclasses.

    Examples
    --------
    To implement a new KDEy quantifier, subclass BaseKDE and implement the method
    `_solve_prevalences`, which receives posterior predictions and returns a tuple

    (estimated prevalences \(\boldsymbol{\alpha}\), objective value).

    >>> class KDEyExample(BaseKDE):
    ...     def _solve_prevalences(self, predictions):
    ...         # Example: simple uniform prevalences, replace with actual optimization
    ...         n_classes = len(self._class_kdes)
    ...         alpha = np.ones(n_classes) / n_classes
    ...         obj_val = 0.0  # Replace with actual objective computation
    ...         return alpha, obj_val

    Mathematical formulation for prevalence estimation typically involves optimizing:

    \[
    \min_{\boldsymbol{\alpha} \in \Delta^{K-1}} \mathcal{L} \bigg( \sum_{k=1}^K \alpha_k p_k(x), \hat{p}(x) \bigg),
    \]

    where \(\hat{p}(x)\) is the test posterior distribution (empirical KDE or direct predictions),
    \(\Delta^{K-1}\) is the probability simplex defined by the constraints on \(\boldsymbol{\alpha}\),
    and \(\mathcal{L}\) is an appropriate divergence or loss function, e.g., negative log-likelihood,
    Hellinger distance, or Cauchy–Schwarz divergence.

    This optimization is typically solved numerically with constrained methods such as
    sequential quadratic programming or projected gradient descent.

    References
    ----------
    [1] Moreo, A., et al. (2023). Kernel Density Quantification methods and applications.
        In *Learning to Quantify*, Springer.
    """

    _parameter_constraints = {
        "bandwidth": [Interval(0, None, inclusive_right=False)],
        "kernel": [Options(["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"])],
    }

    def __init__(self, learner=None, bandwidth: float = 0.1, kernel: str = "gaussian"):
        self.learner = learner
        self.bandwidth = bandwidth
        self.kernel = kernel
        self._precomputed = False
        self.best_distance = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False):
        X, y = validate_data(self, X, y, ensure_2d=True, ensure_min_samples=2)
        validate_y(self, y)
        
        self.classes_ = np.unique(y)
        
        learner_function = _get_learner_function(self)

        if learner_fitted:
            train_predictions = getattr(self.learner, learner_function)(X)
            train_y_values = y
        else:
            train_predictions, train_y_values = apply_cross_validation(
                self.learner, X, y,
                function=learner_function, cv=5,
                stratified=True, shuffle=True
            )

        self.train_predictions = train_predictions
        self.train_y_values = train_y_values
        self._precompute_training(train_predictions, train_y_values)
        return self

    def _fit_kde_models(self, train_predictions, train_y_values):
        P = np.atleast_2d(train_predictions)
        y = np.asarray(train_y_values)
        self._class_kdes = []

        for c in self.classes_:
            Xi = P[y == c]
            if Xi.shape[0] == 0:
                Xi = np.ones((1, P.shape[1])) / P.shape[1]  # fallback
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(Xi)
            self._class_kdes.append(kde)
            
        self._precomputed = True

    def predict(self, X):
        predictions = getattr(self.learner, _get_learner_function(self))(X)
        return self.aggregate(predictions, self.train_predictions, self.train_y_values)
    
    def aggregate(self, predictions, train_predictions, train_y_values):
        predictions = validate_predictions(self, predictions)
        
        if hasattr(self, "classes_") and len(np.unique(train_y_values)) != len(self.classes_):
            self._precomputed = False
        
        self.classes_ = check_classes_attribute(self, np.unique(train_y_values))
        
        if not self._precomputed:
            self._precompute_training(train_predictions, train_y_values)
            self._precomputed = True
            
        prevalence, _ = self._solve_prevalences(predictions)
        prevalence = np.clip(prevalence, EPS, None)
        prevalence = validate_prevalences(self, prevalence, self.classes_)
        return prevalence

    def best_distance(self, predictions, train_predictions, train_y_values):
        """Retorna a melhor distância encontrada durante o ajuste."""
        if self.best_distance is not None:
            return self.best_distance
        
        self.classes_ = check_classes_attribute(self, np.unique(train_y_values))
        
        if not self._precomputed:
            self._precompute_training(train_predictions, train_y_values)
            self._precomputed = True    
    
        _, best_distance = self._solve_prevalences(predictions)
        return best_distance

    @abstractmethod
    def _precompute_training(self, train_predictions, train_y_values):
        raise NotImplementedError

    @abstractmethod
    def _solve_prevalences(self, predictions):
        raise NotImplementedError