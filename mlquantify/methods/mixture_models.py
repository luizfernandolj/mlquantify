from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator

from ..base import AggregativeQuantifier

from ..utils.general import get_real_prev
from ..utils.method import *




class MixtureModel(AggregativeQuantifier):
    """Generic Class for the Mixture Models methods, which
    are based oon the assumption that the cumulative 
    distribution of the scores assigned to data points in the test
    is a mixture of the scores in train data
    """
    
    def __init__(self, learner: BaseEstimator):
        self.learner = learner
        self.pos_scores = None
        self.neg_scores = None

    @property
    def multiclass_method(self) -> bool:
        return False

    def _fit_method(self, X, y):
        # Compute scores with cross validation and fit the learner if not already fitted
        y_label, probabilities = get_scores(X, y, self.learner, self.cv_folds, self.learner_fitted)

        # Separate positive and negative scores based on labels
        self.pos_scores = probabilities[y_label == self.classes[1]][:, 1]
        self.neg_scores = probabilities[y_label == self.classes[0]][:, 1]

        return self

    def _predict_method(self, X) -> dict:
        prevalences = {}

        # Get the predicted probabilities for the positive class
        test_scores = self.learner.predict_proba(X)[:, 1]

        # Compute the prevalence using the provided measure
        prevalence = np.clip(self._compute_prevalence(test_scores), 0, 1)

        # Clip the prevalence to be within the [0, 1] range and compute the complement for the other class
        prevalences = np.asarray([1- prevalence, prevalence])

        return prevalences

    @abstractmethod
    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        """ Abstract method for computing the prevalence using the test scores """
        ...

    def get_distance(self, dist_train, dist_test, measure: str) -> float:
        """Compute the distance between training and test distributions using the specified metric"""

        # Check if any vector is too small or if they have different lengths
        if np.sum(dist_train) < 1e-20 or np.sum(dist_test) < 1e-20:
            raise ValueError("One or both vectors are zero (empty)...")
        if len(dist_train) != len(dist_test):
            raise ValueError("Arrays need to be of equal size...")

        # Convert distributions to numpy arrays for efficient computation
        dist_train = np.array(dist_train, dtype=float)
        dist_test = np.array(dist_test, dtype=float)

        # Avoid division by zero by correcting zero values
        dist_train[dist_train < 1e-20] = 1e-20
        dist_test[dist_test < 1e-20] = 1e-20

        # Compute and return the distance based on the selected metric
        if measure == 'topsoe':
            return topsoe(dist_train, dist_test)
        elif measure == 'probsymm':
            return probsymm(dist_train, dist_test)
        elif measure == 'hellinger':
            return hellinger(dist_train, dist_test)
        elif measure == 'euclidean':
            return sqEuclidean(dist_train, dist_test)
        else:
            return 100  # Default value if an unknown measure is provided
        




class DySsyn(MixtureModel):
    """Synthetic Distribution y-Similarity. This method works the
    same as DyS method, but istead of using the train scores, it 
    generates them via MoSS (Model for Score Simulation) which 
    generate a spectrum of score distributions from highly separated
    scores to fully mixed scores.
    """
    
    def __init__(self, learner:BaseEstimator, measure:str="topsoe", merge_factor:np.ndarray=None, bins_size:np.ndarray=None, alpha_train:float=0.5, n:int=None):
        assert measure in ["hellinger", "topsoe", "probsymm"], "measure not valid"
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
        # Set up bins_size
        if not bins_size:
            bins_size = np.append(np.linspace(2,20,10), 30)
        if isinstance(bins_size, list):
            bins_size = np.asarray(bins_size)
            
        if not merge_factor:
            merge_factor = np.linspace(0.1, 0.4, 10)
            
        self.bins_size = bins_size
        self.merge_factor = merge_factor
        self.alpha_train = alpha_train
        self.n = n
        self.measure = measure
        self.m = None
    
    
    
    def _fit_method(self, X, y):
        if not self.learner_fitted:
            self.learner.fit(X, y)
            
        self.alpha_train = list(get_real_prev(y).values())[1]
        
        return self
    
    
    
    def _compute_prevalence(self, test_scores:np.ndarray) -> float:    #creating bins from 10 to 110 with step size 10
        
        distances = self.GetMinDistancesDySsyn(test_scores)
        
        # Use the median of the prevss as the final prevalence estimate
        index = min(distances, key=lambda d: distances[d][0])
        prevalence = distances[index][1]
            
        return prevalence
    
    
    def best_distance(self, X_test):
        
        test_scores = self.learner.predict_proba(X_test)
        
        distances = self.GetMinDistancesDySsyn(test_scores)
        
        index = min(distances, key=lambda d: distances[d][0])
        
        distance = distances[index][0]
        
        return distance
    
    

    def GetMinDistancesDySsyn(self, test_scores) -> list:
        # Compute prevalence by evaluating the distance metric across various bin sizes
        if self.n is None:
            self.n = len(test_scores)
            
        values = {}
        
        # Iterate over each bin size
        for m in self.merge_factor:
            pos_scores, neg_scores = MoSS(self.n, self.alpha_train, m)
            prevs  = []
            for bins in self.bins_size:
                # Compute histogram densities for positive, negative, and test scores
                pos_bin_density = getHist(pos_scores, bins)
                neg_bin_density = getHist(neg_scores, bins)
                test_bin_density = getHist(test_scores, bins)

                # Define the function to minimize
                def f(x):
                    # Combine densities using a mixture of positive and negative densities
                    train_combined_density = (pos_bin_density * x) + (neg_bin_density * (1 - x))
                    # Calculate the distance between combined density and test density
                    return self.get_distance(train_combined_density, test_bin_density, measure=self.measure)
            
                # Use ternary search to find the best x that minimizes the distance
                prevs.append(ternary_search(0, 1, f))
                
            size = len(prevs)
            best_prev = np.median(prevs)

            if size % 2 != 0:  # ODD
                index = np.argmax(prevs == best_prev)
                bin_size = self.bins_size[index]
            else:  # EVEN
                # Sort the values in self.prevs
                ordered_prevs = np.sort(prevs)

                # Find the two middle indices
                middle1 = np.floor(size / 2).astype(int)
                middle2 = np.ceil(size / 2).astype(int)

                # Get the values corresponding to the median positions
                median1 = ordered_prevs[middle1]
                median2 = ordered_prevs[middle2]

                # Find the indices of median1 and median2 in prevs
                index1 = np.argmax(prevs == median1)
                index2 = np.argmax(prevs == median2)

                # Calculate the average of the corresponding bin sizes
                bin_size = np.mean([self.bins_size[index1], self.bins_size[index2]])
                
            
            pos_bin_density = getHist(pos_scores, bin_size)
            neg_bin_density = getHist(neg_scores, bin_size)
            test_bin_density = getHist(test_scores, bin_size)
            
            train_combined_density = (pos_bin_density * best_prev) + (neg_bin_density * (1 - best_prev))
            
            distance = self.get_distance(train_combined_density, test_bin_density, measure=self.measure)
            
            values[m] = (distance, best_prev)
            
        return values
    








class DyS(MixtureModel):
    """Distribution y-Similarity framework. Is a 
    method that generalises the HDy approach by 
    considering the dissimilarity function DS as 
    a parameter of the model
    """
    
    def __init__(self, learner:BaseEstimator, measure:str="topsoe", bins_size:np.ndarray=None):
        assert measure in ["hellinger", "topsoe", "probsymm"], "measure not valid"
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
        # Set up bins_size
        if not bins_size:
            bins_size = np.append(np.linspace(2,20,10), 30)
        if isinstance(bins_size, list):
            bins_size = np.asarray(bins_size)
            
        self.bins_size = bins_size
        self.measure = measure
        self.prevs = None # Array of prevalences that minimizes the distances
        
    
    def _compute_prevalence(self, test_scores:np.ndarray) -> float:    
        
        prevs = self.GetMinDistancesDyS(test_scores)                    
        # Use the median of the prevalences as the final prevalence estimate
        prevalence = np.median(prevs)
            
        return prevalence
    
    
    
    def best_distance(self, X_test) -> float:
        
        test_scores = self.learner.predict_proba(X_test)
        
        prevs = self.GetMinDistancesDyS(test_scores) 
        
        size = len(prevs)
        best_prev = np.median(prevs)

        if size % 2 != 0:  # ODD
            index = np.argmax(prevs == best_prev)
            bin_size = self.bins_size[index]
        else:  # EVEN
            # Sort the values in self.prevs
            ordered_prevs = np.sort(prevs)

            # Find the two middle indices
            middle1 = np.floor(size / 2).astype(int)
            middle2 = np.ceil(size / 2).astype(int)

            # Get the values corresponding to the median positions
            median1 = ordered_prevs[middle1]
            median2 = ordered_prevs[middle2]

            # Find the indices of median1 and median2 in prevs
            index1 = np.argmax(prevs == median1)
            index2 = np.argmax(prevs == median2)

            # Calculate the average of the corresponding bin sizes
            bin_size = np.mean([self.bins_size[index1], self.bins_size[index2]])
            
        
        pos_bin_density = getHist(self.pos_scores, bin_size)
        neg_bin_density = getHist(self.neg_scores, bin_size)
        test_bin_density = getHist(test_scores, bin_size)
        
        train_combined_density = (pos_bin_density * best_prev) + (neg_bin_density * (1 - best_prev))
        
        distance = self.get_distance(train_combined_density, test_bin_density, measure=self.measure)
        
        return distance
        

    def GetMinDistancesDyS(self, test_scores) -> list:
        # Compute prevalence by evaluating the distance metric across various bin sizes
        
        prevs = []
 
        # Iterate over each bin size
        for bins in self.bins_size:
            # Compute histogram densities for positive, negative, and test scores
            pos_bin_density = getHist(self.pos_scores, bins)
            neg_bin_density = getHist(self.neg_scores, bins)
            test_bin_density = getHist(test_scores, bins)

            # Define the function to minimize
            def f(x):
                # Combine densities using a mixture of positive and negative densities
                train_combined_density = (pos_bin_density * x) + (neg_bin_density * (1 - x))
                # Calculate the distance between combined density and test density
                return self.get_distance(train_combined_density, test_bin_density, measure=self.measure)
        
            # Use ternary search to find the best x that minimizes the distance
            prevs.append(ternary_search(0, 1, f))
            
        return prevs
    
    
    





class HDy(MixtureModel):
    """Hellinger Distance Minimization. The method
    is based on computing the hellinger distance of 
    two distributions, test distribution and the mixture
    of the positive and negative distribution of the train.
    """

    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
    
        
    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        
        best_alphas, _ = self.GetMinDistancesHDy(test_scores)
        # Compute the median of the best alpha values as the final prevalence estimate
        prevalence = np.median(best_alphas)
            
        return prevalence
    
    
    
    def best_distance(self, X_test) -> float:
        
        test_scores = self.learner.predict_proba(X_test)
        
        _, distances = self.GetMinDistancesHDy(test_scores)
        
        size = len(distances)
        
        if size % 2 != 0:  # ODD
            index = size // 2
            distance = distances[index]
        else:  # EVEN
            # Find the two middle indices
            middle1 = np.floor(size / 2).astype(int)
            middle2 = np.ceil(size / 2).astype(int)

            # Get the values corresponding to the median positions
            dist1 = distances[middle1]
            dist2 = distances[middle2]
            
            # Calculate the average of the corresponding distances
            distance = np.mean([dist1, dist2])
        
        return distance
        

    def GetMinDistancesHDy(self, test_scores: np.ndarray) -> tuple:
        
        # Define bin sizes and alpha values
        bins_size = np.arange(10, 110, 11)  # Bins from 10 to 110 with a step size of 10
        alpha_values = np.round(np.linspace(0, 1, 101), 2)  # Alpha values from 0 to 1, rounded to 2 decimal places
        
        best_alphas = []
        distances = []
        
        for bins in bins_size:

            pos_bin_density = getHist(self.pos_scores, bins)
            neg_bin_density = getHist(self.neg_scores, bins)
            test_bin_density = getHist(test_scores, bins)
            
            distances = []
            
            # Evaluate distance for each alpha value
            for x in alpha_values:
                # Combine densities using a mixture of positive and negative densities
                train_combined_density = (pos_bin_density * x) + (neg_bin_density * (1 - x))
                # Compute the distance using the Hellinger measure
                distances.append(self.get_distance(train_combined_density, test_bin_density, measure="hellinger"))

            # Find the alpha value that minimizes the distance
            best_alphas.append(alpha_values[np.argmin(distances)])
            distances.append(min(distances)) 
            
        return best_alphas, distances
    
    
    
    
    
    

class SMM(MixtureModel):
    """Sample Mean Matching. The method is 
    a member of the DyS framework that uses 
    simple means to represent the score 
    distribution for positive, negative 
    and unlabelled scores.
    """

    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        mean_pos_score = np.mean(self.pos_scores)
        mean_neg_score = np.mean(self.neg_scores)
        mean_test_score = np.mean(test_scores)
        
        # Calculate prevalence as the proportion of the positive class
        # based on the mean test score relative to the mean positive and negative scores
        prevalence = (mean_test_score - mean_neg_score) / (mean_pos_score - mean_neg_score)
        
        return prevalence
    
    






class SORD(MixtureModel):
    """Sample Ordinal Distance. Is a method 
    that does not rely on distributions, but 
    estimates the prevalence of the positive 
    class in a test dataset by calculating and 
    minimizing a sample ordinal distance measure 
    between the test scores and known positive 
    and negative scores.
    """

    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
        self.best_distance_index = None
        
    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        # Compute alpha values and corresponding distance measures
        alpha_values, distance_measures = self._calculate_distances(test_scores)
        
        # Find the index of the alpha value with the minimum distance measure
        self.best_distance_index = np.argmin(distance_measures)
        prevalence = alpha_values[self.best_distance_index]
        
        return prevalence
    
    
    def _calculate_distances(self, test_scores: np.ndarray):
        # Define a range of alpha values from 0 to 1
        alpha_values = np.linspace(0, 1, 101)
        
        # Get the number of positive, negative, and test scores
        num_pos_scores = len(self.pos_scores)
        num_neg_scores = len(self.neg_scores)
        num_test_scores = len(test_scores)

        distance_measures = []

        # Iterate over each alpha value
        for alpha in alpha_values:
            # Compute weights for positive, negative, and test scores
            pos_weight = alpha / num_pos_scores
            neg_weight = (1 - alpha) / num_neg_scores
            test_weight = -1 / num_test_scores

            # Create arrays with weights
            pos_weights = np.full(num_pos_scores, pos_weight)
            neg_weights = np.full(num_neg_scores, neg_weight)
            test_weights = np.full(num_test_scores, test_weight)

            # Concatenate all scores and their corresponding weights
            all_scores = np.concatenate([self.pos_scores, self.neg_scores, test_scores])
            all_weights = np.concatenate([pos_weights, neg_weights, test_weights])

            # Sort scores and weights based on scores
            sorted_indices = np.argsort(all_scores)
            sorted_scores = all_scores[sorted_indices]
            sorted_weights = all_weights[sorted_indices]

            # Compute the total cost for the current alpha
            cumulative_weight = sorted_weights[0]
            total_cost = 0

            for i in range(1, len(sorted_scores)):
                # Calculate the cost for the segment between sorted scores
                segment_width = sorted_scores[i] - sorted_scores[i - 1]
                total_cost += abs(segment_width * cumulative_weight)
                cumulative_weight += sorted_weights[i]

            distance_measures.append(total_cost)

        return alpha_values, distance_measures