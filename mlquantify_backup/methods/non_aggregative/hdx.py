import numpy as np

from ...base import NonAggregativeQuantifier
from ...utils import getHist, hellinger

class HDx(NonAggregativeQuantifier):
    """Hellinger Distance Minimization. The method is similar 
    to the HDy method, but istead of computing the hellinger 
    distance of the scores (generated via classifier), HDx 
    computes the distance of each one of the features of the 
    dataset
    """

    def __init__(self, bins_size:np.ndarray=None):
        if not bins_size:
            bins_size = np.append(np.linspace(2,20,10), 30)
        
        self.bins_size = bins_size
        self.neg_features = None
        self.pos_features = None
        
        
    def _fit_method(self, X, y):
        
        
        self.pos_features = X[y == self.classes[1]]
        self.neg_features = X[y == self.classes[0]]
        
        
        if not isinstance(X, np.ndarray):
            self.pos_features = self.pos_features.to_numpy()
        if not isinstance(y, np.ndarray):
            self.neg_features = self.neg_features.to_numpy()
        
        return self
    
    def _predict_method(self, X) -> dict:
        
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
    
        alpha_values = np.round(np.linspace(0, 1, 101), 2)
        
        best_distances = {}
        
        for x in alpha_values: 
            
            distances = []
            
            for i in range(X.shape[1]):
                for bins in self.bins_size:
                    
                    dist_feature_pos = getHist(self.pos_features[:, i], bins)
                    dist_feature_neg = getHist(self.neg_features[:, i], bins)
                    dist_feature_test = getHist(X[:, i], bins)
    
                    # Combine densities using a mixture of positive and negative densities
                    train_combined_density = (dist_feature_pos * x) + (dist_feature_neg * (1 - x))
                    # Compute the distance using the Hellinger measure
                    distances.append(hellinger(train_combined_density, dist_feature_test))

            best_distances[x] = np.mean(distances)
            
        prevalence = min(best_distances, key=best_distances.get)
        
        return np.asarray([1- prevalence, prevalence])
            
            
                
    
