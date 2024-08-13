from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

class PWKCLF(BaseEstimator):
    """Learner based on k-Nearest Neighborst (KNN) to use on the method PWK, 
    that also is based on KNN.
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

        self.Y = None
        self.Y_map = None
        self.w = None
        self.y = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        if n_samples < self.n_neighbors:
            self.nbrs.set_params(n_neighbors=n_samples)
            
        self.y = y
        
        if isinstance(y, pd.DataFrame):
            self.y = y.reset_index(drop=True)
            
        Y_cts = np.unique(y, return_counts=True)
        self.Y = Y_cts[0]
        self.Y_map = dict(zip(self.Y, range(len(self.Y))))

        min_class_count = np.min(Y_cts[1])
        self.w = (Y_cts[1] / min_class_count) ** (-1.0 / self.alpha)
        self.nbrs.fit(X)
        return self

    def predict(self, X):    
        n_samples = X.shape[0]
        nn_indices = self.nbrs.kneighbors(X, return_distance=False)

        CM = np.zeros((n_samples, len(self.Y)))
        
        for i in range(n_samples):
            for j in nn_indices[i]:
                CM[i, self.Y_map[self.y[j]]] += 1

        CM = np.multiply(CM, self.w)
        predictions = np.apply_along_axis(np.argmax, axis=1, arr=CM)
        
        return self.Y[predictions]
