
import numpy as np
from sklearn.neighbors import NearestNeighbors



class PWKCLF:
    
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
