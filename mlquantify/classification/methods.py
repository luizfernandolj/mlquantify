from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

class PWKCLF(BaseEstimator):
    """
    Learner based on k-Nearest Neighbors (KNN) to use in the PWK method.
    
    This classifier adjusts the influence of neighbors using class weights 
    derived from the `alpha` parameter. The `alpha` parameter controls the 
    influence of class imbalance.

    Parameters
    ----------
    alpha : float, default=1
        Controls the influence of class imbalance. Must be >= 1.

    n_neighbors : int, default=10
        Number of neighbors to use.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm to compute nearest neighbors.

    metric : str, default='euclidean'
        Distance metric to use.

    leaf_size : int, default=30
        Leaf size passed to the tree-based algorithms.

    p : int, default=2
        Power parameter for the Minkowski metric.

    metric_params : dict, optional
        Additional keyword arguments for the metric function.

    n_jobs : int, optional
        Number of parallel jobs to run for neighbors search.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from mlquantify.methods.aggregative import PWK
    >>> from mlquantify.utils.general import get_real_prev
    >>> from mlquantify.classification import PWKCLF
    >>> 
    >>> # Load dataset
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> 
    >>> # Split into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=32)
    >>> 
    >>> # Create and configure the PWKCLF learner
    >>> learner = PWKCLF(alpha=1, n_neighbors=10)
    >>> 
    >>> # Create the PWK quantifier
    >>> model = PWK(learner=learner)
    >>> 
    >>> # Train the model
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Predict prevalences
    >>> y_pred = model.predict(X_test)
    >>> 
    >>> # Display results
    >>> print("Real:", get_real_prev(y_test))
    >>> print("PWK:", y_pred)
    """

    def __init__(self,
                 alpha=1,
                 n_neighbors=10,
                 algorithm="auto",
                 metric="euclidean",
                 leaf_size=30,
                 p=2,
                 metric_params=None,
                 n_jobs=None):
        if alpha < 1:
            raise ValueError("alpha must not be smaller than 1")
        
        self.alpha = alpha
        self.n_neighbors = n_neighbors

        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                     algorithm=algorithm,
                                     leaf_size=leaf_size,
                                     metric=metric,
                                     p=p,
                                     metric_params=metric_params,
                                     n_jobs=n_jobs)

        self.classes_ = None
        self.class_to_index = None
        self.class_weights = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the PWKCLF model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.

        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : object
            The fitted instance.
        """
        n_samples = X.shape[0]
        if n_samples < self.n_neighbors:
            self.nbrs.set_params(n_neighbors=n_samples)
            
        self.y_train = y
        
        if isinstance(y, pd.DataFrame):
            self.y_train = y.reset_index(drop=True)
            
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.classes_ = unique_classes
        self.class_to_index = dict(zip(self.classes_, range(len(self.classes_))))

        min_class_count = np.min(class_counts)
        self.class_weights = (class_counts / min_class_count) ** (-1.0 / self.alpha)
        self.nbrs.fit(X)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        n_samples = X.shape[0]
        nn_indices = self.nbrs.kneighbors(X, return_distance=False)

        CM = np.zeros((n_samples, len(self.classes_)))
        
        for i in range(n_samples):
            for j in nn_indices[i]:
                CM[i, self.class_to_index[self.y_train[j]]] += 1

        CM = np.multiply(CM, self.class_weights)
        predictions = np.apply_along_axis(np.argmax, axis=1, arr=CM)
        
        return self.classes_[predictions]
