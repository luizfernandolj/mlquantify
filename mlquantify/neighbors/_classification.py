
import numpy as np
from sklearn.neighbors import NearestNeighbors



class PWKCLF:
    """
    Probabilistic Weighted k-Nearest Neighbor Classifier (PWKCLF).

    A weighted k-nearest neighbor classifier that assigns class probabilities to 
    instances based on neighbor counts weighted by class-specific inverse frequency 
    factors adjusted by a hyperparameter alpha controlling imbalance compensation. 

    Attributes
    ----------
    alpha : float
        Exponent controlling the degree of imbalance compensation.
    n_neighbors : int
        Number of nearest neighbors considered.
    nbrs : sklearn.neighbors.NearestNeighbors
        The underlying k-NN structure used for neighbor queries.
    classes_ : ndarray
        Unique classes observed during training.
    class_to_index : dict
        Mapping from class label to index used in internal arrays.
    class_weights : ndarray
        Per-class weights computed based on class frequency and alpha.
    y_train : ndarray
        Labels of training samples.

    Methods
    -------
    fit(X, y)
        Fits the k-NN structure and computes class weights.
    predict(X)
        Predicts class labels by weighted voting among neighbors.

    Notes
    -----
    The class weights are defined as:

    \[
    w_c = \left( \frac{N_c}{\min_{c'} N_{c'}} \right)^{-\frac{1}{\alpha}},
    \]

    where \( N_c \) is the count of class \( c \) in the training set.

    This weighting scheme reduces bias towards majority classes by downweighting them
    in the voting process.

    Examples
    --------
    >>> clf = PWKCLF(alpha=2.0, n_neighbors=7)
    >>> clf.fit(X_train, y_train)
    >>> labels = clf.predict(X_test)
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
