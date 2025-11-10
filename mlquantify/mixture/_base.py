import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier

from mlquantify.mixture._utils import sqEuclidean
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_y, validate_data
from mlquantify.multiclass import define_binary
from mlquantify.mixture._utils import (
    hellinger,
    topsoe,
    probsymm,
    sqEuclidean
)

class BaseMixture(BaseQuantifier):
    """
    Base class for mixture-model quantifiers.

    Mixture Models (MM) for quantification estimate class prevalences by modeling 
    the test set score distribution as a mixture of the individual class score 
    distributions learned from training data. The goal is to find the mixture 
    parameters, i.e., class proportions, that best represent the observed test data.

    Mixture-based quantifiers approximate class-conditional distributions typically 
    via histograms or empirical distributions of classifier scores, treating the test 
    distribution as a weighted sum (mixture) of these. Estimation proceeds by finding 
    the mixture weights that minimize a distance or divergence measure between the 
    observed test distribution and the mixture of training class distributions.

    Common distance measures used in evaluating mixtures include:
    - Hellinger distance
    - Topsoe distance (a symmetric Jensen-Shannon type divergence)
    - Probabilistic symmetric divergence
    - Squared Euclidean distance

    These distances compare probability distributions representing class-conditioned 
    scores or histograms, and the choice of distance can affect quantification accuracy 
    and robustness.

    The DyS framework (Maletzke et al. 2019) generalizes mixture models by introducing 
    a variety of distribution dissimilarity measures, enabling flexible and effective 
    quantification methods.
    
    
    Notes
    -----
    Mixture models are defined for only binary quantification problems. For multi-class
    problems, a one-vs-rest strategy is applied, training a binary mixture model for
    each class against the rest.

    Parameters
    ----------
    None directly; subclasses implement fitting and prediction logic.

    Attributes
    ----------
    _precomputed : bool
        Indicates if preprocess computations on data have been performed.
    distances : Any
        Stores intermediate or final distance computations used in model selection.
    classes : ndarray of shape (n_classes,)
        Unique class labels seen during training.

    Methods
    -------
    fit(X, y, *args, **kwargs):
        Fit the mixture quantifier with training data. Validates input and 
        calls internal fitting procedure.
    predict(X, *args, **kwargs):
        Predict class prevalences for input data by leveraging best mixture parameters.
    get_best_distance(*args, **kwargs):
        Return the best distance measure and associated mixture parameters found.
    best_mixture(X):
        Abstract method to determine optimal mixture parameters on input data.
    get_distance(dist_train, dist_test, measure="hellinger"):
        Compute a specified distance between two distributions.

    References
    ----------
    [1] Forman, G. (2005). *Counting Positives Accurately Despite Inaccurate Classification.* ECML, pp. 564-575.
    [2] Forman, G. (2008). *Quantifying Counts and Costs via Classification.* Data Mining and Knowledge Discovery, 17(2), 164-206.
    [3] Maletzke, A., dos Reis, D., Cherman, E., & Batista, G. (2019). *DyS: A Framework for Mixture Models in Quantification.* AAAI Conference on Artificial Intelligence.
    [4] Esuli, A., Moreo, A., & Sebastiani, F. (2023). *Learning to Quantify.* Springer.

    Examples
    --------
    >>> import numpy as np
    >>> class MyMixture(BaseMixture):
    ...     def best_mixture(self, X):
    ...         # Implementation example: estimate mixture weights minimizing Hellinger distance
    ...         pass
    >>> X_train = np.random.rand(100, 10)
    >>> y_train = np.random.randint(0, 2, size=100)
    >>> quantifier = MyMixture()
    >>> quantifier.fit(X_train, y_train)
    >>> prevalences = quantifier.predict(X_train)
    """
    
    def __init__(self):
        self._precomputed = False
        self.distances = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, *args, **kwargs):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, X, y)
        validate_y(self, y)
        self.classes_ = np.unique(y)
        
        self._fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X, *args, **kwargs):
        """Predict class prevalences for the given data."""
        X = validate_data(self, X)
        return self._predict(X, *args, **kwargs)
    
    def get_best_distance(self, *args, **kwargs):
        _, best_distance = self.best_mixture(*args, **kwargs)
        return best_distance

    @abstractmethod
    def best_mixture(self, X):
        """Determine the best mixture parameters for the given data."""
        pass
    
    @classmethod
    def get_distance(cls, dist_train, dist_test, measure="hellinger"):
        """
        Compute distance between two distributions.
        """
        
        if np.sum(dist_train) < 1e-20 or np.sum(dist_test) < 1e-20:
            raise ValueError("One or both vectors are zero (empty)...")
        if len(dist_train) != len(dist_test):
            raise ValueError("Arrays must have the same length.")

        dist_train = np.maximum(dist_train, 1e-20)
        dist_test = np.maximum(dist_test, 1e-20)

        if measure == "topsoe":
            return topsoe(dist_train, dist_test)
        elif measure == "probsymm":
            return probsymm(dist_train, dist_test)
        elif measure == "hellinger":
            return hellinger(dist_train, dist_test)
        elif measure == "euclidean":
            return sqEuclidean(dist_train, dist_test)
        else:
            raise ValueError(f"Invalid measure: {measure}")
    