import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier

from mlquantify.base_aggregative import (
    AggregationMixin,
    _get_learner_function
)
from mlquantify.adjust_counting import CC
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import check_classes_attribute, validate_predictions, validate_y, validate_data, validate_prevalences



class BaseIterativeLikelihood(AggregationMixin, BaseQuantifier):
    r"""Iterative likelihood-based quantification adjustment methods.

    This base class encompasses quantification approaches that estimate class prevalences 
    by maximizing the likelihood of observed data, adjusting prevalence estimates on test 
    sets under the assumption of prior probability shift.

    These methods iteratively refine estimates of class prevalences by maximizing the 
    likelihood of classifier outputs, usually the posterior probabilities provided by 
    a trained model, assuming that the class-conditional distributions remain fixed 
    between training and test domains.

    Mathematical formulation
    ------------------------
    Let:

    - :math:`p_k^t` be the prior probabilities for class \(k\) in the training set, satisfying \( \sum_k p_k^t = 1 \),
    - :math:`s_k(x)` be the posterior probability estimate from the classifier for class \(k\) given instance \(x\),
    - :math:`p_k` be the unknown prior probabilities for class \(k\) in the test set,
    - \( x_1, \dots, x_N \) be unlabeled test set instances.

    The likelihood of the observed data is:

    .. math::

        L = \prod_{i=1}^N \sum_{k=1}^K s_k(x_i) \frac{p_k}{p_k^t}

    Methods in this class seek a solution that maximizes this likelihood via iterative methods.

    Notes
    -----
    - Applicable to binary and multiclass problems as long as the classifier provides calibrated posterior probabilities.
    - Assumes changes only in prior probabilities (prior probability shift).
    - Algorithms converge to local maxima of the likelihood function.
    - Includes methods such as Class Distribution Estimation (CDE), Maximum Likelihood Prevalence Estimation (MLPE), and Expectation-Maximization (EM) based quantification.

    Parameters
    ----------
    learner : estimator, optional
        Probabilistic classifier implementing the methods `fit(X, y)` and `predict_proba(X)`.
    tol : float, default=1e-4
        Convergence tolerance for prevalence update criteria.
    max_iter : int, default=100
        Maximum allowed number of iterations.

    Attributes
    ----------
    learner : estimator
        Underlying classification model.
    tol : float
        Tolerance for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    classes : ndarray of shape (n_classes,)
        Unique classes observed during training.
    priors : ndarray of shape (n_classes,)
        Class distribution in the training set.
    y_train : array-like
        Training labels used to estimate priors.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> class MyQuantifier(BaseIterativeLikelihood):
    ...     def _iterate(self, predictions, priors):
    ...         # Implementation of iterative update logic
    ...         pass
    >>> X = np.random.randn(200, 8)
    >>> y = np.random.randint(0, 3, size=(200,))
    >>> q = MyQuantifier(learner=LogisticRegression(max_iter=200))
    >>> q.fit(X, y)
    >>> q.predict(X)
    {0: 0.32, 1: 0.40, 2: 0.28}

    References
    ----------
    .. [1] Saerens, M., Latinne, P., & Decaestecker, C. (2002). "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure." Neural Computation, 14(1), 2141-2156.

    .. [2] Esuli, A., Moreo, A., & Sebastiani, F. (2023). "Learning to Quantify." The Information Retrieval Series 47, Springer. https://doi.org/10.1007/978-3-031-20467-8
    """

    @abstractmethod
    def __init__(self, 
                 learner=None,
                 tol=1e-4,
                 max_iter=100):
        self.learner = learner
        self.tol = tol
        self.max_iter = max_iter
        
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = False
        return tags
    
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, X, y)
        validate_y(self, y)
        self.classes_ = np.unique(y)
        self.learner.fit(X, y)
        counts = np.array([np.count_nonzero(y == _class) for _class in self.classes_])
        self.priors = counts / len(y)
        self.y_train = y
                
        return self
    
    def predict(self, X):
        """Predict class prevalences for the given data."""
        estimator_function = _get_learner_function(self)
        predictions = getattr(self.learner, estimator_function)(X)
        prevalences = self.aggregate(predictions, self.y_train)
        return prevalences

    def aggregate(self, predictions, y_train):
        predictions = validate_predictions(self, predictions)
        self.classes_ = check_classes_attribute(self, np.unique(y_train))
        
        if not hasattr(self, 'priors') or len(self.priors) != len(self.classes_):
            counts = np.array([np.count_nonzero(y_train == _class) for _class in self.classes_])
            self.priors = counts / len(y_train)
            
        prevalences = self._iterate(predictions, self.priors)
        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences
    
    @abstractmethod
    def _iterate(self, predictions, priors):
        ...
