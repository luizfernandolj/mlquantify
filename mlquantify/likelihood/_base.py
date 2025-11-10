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
    """
    Iterative, likelihood-based quantification via EM adjustment.
    
    This is the base class for quantification methods that estimate class prevalences
    by solving the maximum likelihood problem under prior probability shift, using
    iterative procedures such as the EM (Expectation-Maximization) algorithm 
    [1], [2].
    
    These methods repeatedly adjust the estimated class prevalences for a test set
    by maximizing the likelihood of observed classifier outputs (posterior probabilities),
    under the assumption that the within-class conditional distributions remain fixed
    between training and test domains.
    
    Mathematical formulation
    ------------------------
    Let:
    - \( p_k^t \) denote the prior probability for class \( k \) in the training set (\( \sum_k p_k^t = 1 \)),
    - \( s_k(x) \) be the classifier's posterior probability estimate (for class \( k \), given instance \( x \), fitted on training set),
    - \( p_k \) be the (unknown) prior for the test set,
    - \( x_1, \dots, x_N \) the unlabeled test set instances.

    The procedure iteratively estimates \( p_k \) by maximizing the observed data likelihood

    \[
    L = \prod_{i=1}^N \sum_{k=1}^K s_k(x_i) \frac{p_k}{p_k^t}
    \]
    
    The E-step updates soft memberships:

    \[
    w_{ik}^{(t)} = \frac{s_k(x_i) \cdot (p_k^{(t-1)} / p_k^t)}{\sum_{j=1}^K s_j(x_i) \cdot (p_j^{(t-1)} / p_j^t)}
    \]
    and the M-step re-estimates prevalences:

    \[
    p_k^{(t)} = \frac{1}{N} \sum_{i=1}^N w_{ik}^{(t)}
    \]
    See also [1].

    Notes
    -----
    - Defined for multiclass and binary quantification (single-label), as long as the classifier provides well-calibrated posterior probabilities.
    - Assumes prior probability shift only.
    - Converges to a (local) maximum of the data likelihood.
    - The algorithm is Fisher-consistent under prior probability shift [2].
    - Closely related to the Expectation-Maximization (EM) algorithm for mixture models.

    Parameters
    ----------
    learner : estimator, optional
        Probabilistic classifier instance with `fit(X, y)` and `predict_proba(X)`.
    tol : float, default=1e-4
        Convergence tolerance for prevalence update.
    max_iter : int, default=100
        Maximum number of EM update iterations.

    Attributes
    ----------
    learner : estimator
        Underlying classifier instance.
    tol : float
        Stopping tolerance for EM prevalence estimation.
    max_iter : int
        Maximum updates performed.
    classes : ndarray of shape (n_classes,)
        Unique class labels seen in training.
    priors : ndarray of shape (n_classes,)
        Class distribution of the training set.
    y_train : array-like
        Training labels (used for estimating priors and confusion matrix if needed).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> class MyEM(BaseIterativeLikelihood):
    ...     def _iterate(self, predictions, priors):
    ...         # EM iteration logic here
    ...         pass
    >>> X = np.random.randn(200, 8)
    >>> y = np.random.randint(0, 3, size=(200,))
    >>> q = MyEM(learner=LogisticRegression(max_iter=200))
    >>> q.fit(X, y)
    >>> q.predict(X)
    {0: 0.32, 1: 0.40, 2: 0.28}

    References
    ----------
    [1] Saerens, M., Latinne, P., & Decaestecker, C. (2002). *Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure.* Neural Computation, 14(1), 2141-2156.
    
    [2] Esuli, A., Moreo, A., & Sebastiani, F. (2023). *Learning to Quantify.* The Information Retrieval Series 47, Springer. https://doi.org/10.1007/978-3-031-20467-8
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
