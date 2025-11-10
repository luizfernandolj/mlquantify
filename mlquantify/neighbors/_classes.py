import numpy as np
from mlquantify.utils._constraints import Interval, Options
from mlquantify.neighbors._classification import PWKCLF
from mlquantify.base_aggregative import AggregationMixin, CrispLearnerQMixin
from mlquantify.base import BaseQuantifier
from mlquantify.utils._decorators import _fit_context
from mlquantify.adjust_counting import CC
from mlquantify.utils import validate_y, validate_data
from mlquantify.utils._validation import validate_prevalences


class PWK(BaseQuantifier):
    """
    Probabilistic Weighted k-Nearest Neighbor (PWK) Quantifier.

    This quantifier leverages the PWKCLF classifier to perform quantification by estimating 
    class prevalences through a probabilistically weighted k-nearest neighbor approach.

    The method internally uses a weighted k-NN classifier where neighbors' contributions 
    are adjusted by class-specific weights designed to correct for class imbalance,
    controlled by the hyperparameter alpha.

    Parameters
    ----------
    alpha : float, default=1
        Imbalance correction exponent for class weights. Higher values increase 
        the influence of minority classes.
    n_neighbors : int, default=10
        Number of nearest neighbors considered.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute nearest neighbors.
    metric : str, default='euclidean'
        Distance metric for nearest neighbor search.
    leaf_size : int, default=30
        Leaf size for tree-based neighbors algorithms.
    p : int, default=2
        Power parameter for the Minkowski metric (when metric='minkowski').
    metric_params : dict or None, default=None
        Additional parameters for the metric function.
    n_jobs : int or None, default=None
        Number of parallel jobs for neighbors search.

    Attributes
    ----------
    cc : object
        Internally used Classify & Count quantifier wrapping PWKCLF.
    learner : PWKCLF
        Underlying probabilistic weighted k-NN classifier.

    Methods
    -------
    fit(X, y)
        Fits the quantifier by training the internal PWKCLF and wrapping it with 
        Classify & Count quantification.
    predict(X)
        Predicts class prevalences for input data using the trained model.
    classify(X)
        Returns label predictions by applying the trained PWKCLF classifier.

    Examples
    --------
    >>> q = PWK(alpha=1.5, n_neighbors=5)
    >>> q.fit(X_train, y_train)
    >>> prevalences = q.predict(X_test)
    """
    
    _parameter_constraints = {
        "alpha": [Interval(1, None, inclusive_right=False)],
        "n_neighbors": [Interval(1, None, inclusive_right=False)],
        "algorithm": [Options(["auto", "ball_tree", "kd_tree", "brute"])],
        "metric": [str],
        "leaf_size": [Interval(1, None, inclusive_right=False)],
        "p": [Interval(1, None, inclusive_right=False)],
        "metric_params": [dict, type(None)],
        "n_jobs": [Interval(1, None, inclusive_right=False), type(None)],
    }
    
    def __init__(self,
                 alpha=1,
                 n_neighbors=10,
                 algorithm="auto",
                 metric="euclidean",
                 leaf_size=30,
                 p=2,
                 metric_params=None,
                 n_jobs=None):
        learner = PWKCLF(alpha=alpha,
                         n_neighbors=n_neighbors,
                         algorithm=algorithm,
                         metric=metric,
                         leaf_size=leaf_size,
                         p=p,
                         metric_params=metric_params,
                         n_jobs=n_jobs)
        self.algorithm = algorithm
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.leaf_size = leaf_size
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.learner = learner
        
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the PWK quantifier to the training data.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training features.
        
        y_train : array-like of shape (n_samples,)
            Training labels.
        
        Returns
        -------
        self : object
            The fitted instance.
        """
        X, y = validate_data(self, X, y, ensure_2d=True, ensure_min_samples=2)
        validate_y(self, y)
        self.classes_ = np.unique(y)
        self.cc = CC(self.learner)
        return self.cc.fit(X, y)
    
    def predict(self, X):
        """Predict prevalences for the given data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features for which to predict prevalences.
        
        Returns
        -------
        prevalences : array of shape (n_classes,)
            Predicted class prevalences.
        """
        prevalences = self.cc.predict(X)
        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences
    
    def classify(self, X):
        """Classify samples using the underlying learner.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to classify.
        
        Returns
        -------
        labels : array of shape (n_samples,)
            Predicted class labels.
        """
        return self.learner.predict(X)
        