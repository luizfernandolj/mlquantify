import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import AggregationMixin, SoftLearnerQMixin, _get_learner_function
from mlquantify.mixture._base import BaseMixture
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._get_scores import apply_cross_validation
from mlquantify.utils._validation import validate_predictions, validate_prevalences, validate_y
from mlquantify.mixture._utils import (
    getHist,
    ternary_search,
)



# =====================================================
# Base class
# =====================================================

class AggregativeMixture(SoftLearnerQMixin, AggregationMixin, BaseMixture):
    """
    Base class for Mixture-based Quantification Methods.

    These methods assume that the test score distribution is a mixture
    of the positive and negative score distributions from the training data.
    """

    def __init__(self, learner = None):
        super().__init__()
        self.learner = learner
        self.pos_scores = None
        self.neg_scores = None
        self.distances = None
    
    def _fit(self, X, y, learner_fitted=False, *args, **kwargs):
        learner_function = _get_learner_function(self)
        
        if learner_fitted:
            train_predictions = getattr(self.learner, learner_function)(X)
            train_y_values = y
        else:
            train_predictions, train_y_values = apply_cross_validation(
                self.learner,
                X,
                y,
                function= learner_function,
                cv= 5,
                stratified= True,
                random_state= None,
                shuffle= True
            )
        
        self.train_predictions = train_predictions
        self.train_y_values = train_y_values

        self._precompute_training(train_predictions, train_y_values)
        return self

    def _precompute_training(self, train_predictions, train_y_values):
        """
        Fit learner and store score distributions for positive and negative classes.
        """ 
        # Store scores for positive and negative classes
        self.pos_scores = train_predictions[train_y_values == self.classes[1], 1]
        self.neg_scores = train_predictions[train_y_values == self.classes[0], 1]
        self._precomputed = True
        return self
    
    def _predict(self, X):
        """Predict class prevalences for the given data."""
        predictions = getattr(self.learner, _get_learner_function(self))(X)

        prevalences = self.aggregate(predictions, self.train_predictions, self.train_y_values)

        return prevalences
    
    def aggregate(self, predictions, train_predictions, train_y_values):
        predictions = validate_predictions(self, predictions)
        self.classes = self.classes if hasattr(self, 'classes') else np.unique(train_y_values)
        
        if not self._precomputed:
            self._precompute_training(train_predictions, train_y_values)
            self._precomputed = True
        
        pos_test_scores = predictions[:, 1]
        
        best_alpha, _ = self.best_mixture(pos_test_scores, self.pos_scores, self.neg_scores)
        prevalence = np.array([1 - best_alpha, best_alpha])
        prevalence = validate_prevalences(self, prevalence, self.classes)
        return prevalence
    
    @abstractmethod
    def best_mixture(self, predictions, pos_scores, neg_scores):
        ...

# =====================================================
# DyS
# =====================================================

class DyS(AggregativeMixture):
    """
    Distribution y-Similarity (DyS).
    """
    
    _parameter_constraints = {
        "measure": [Options(["hellinger", "topsoe", "probsymm"])],
        "bins_size": ["array-like", None]
    }

    def __init__(self, learner=None, measure="topsoe", bins_size=None):
        super().__init__(learner)
        if bins_size is None:
            bins_size = np.append(np.linspace(2, 20, 10), 30)
            
        self.measure = measure
        self.bins_size = np.asarray(bins_size, dtype=int)

    def best_mixture(self, predictions, pos_scores, neg_scores):
        
        prevs = []
        self.distances = []
        for bins in self.bins_size:
            pos = getHist(pos_scores, bins)
            neg = getHist(neg_scores, bins)
            test = getHist(predictions, bins)

            def f(alpha):
                mix = self._mix(pos, neg, alpha)
                return BaseMixture.get_distance(mix, test, measure=self.measure)

            alpha = ternary_search(0, 1, f)
            prevs.append(alpha)
            self.distances.append(f(alpha))
        alpha = np.median(prevs)
        best_distance = np.median(self.distances)
        return alpha, best_distance
        
    def _mix(self, pos_hist, neg_hist, alpha):
        return alpha * pos_hist + (1 - alpha) * neg_hist


# =====================================================
# HDy
# =====================================================

class HDy(AggregativeMixture):
    """
    Hellinger Distance Minimization (HDy).
    """
    
    def best_mixture(self, predictions, pos_scores, neg_scores):
        bins_size = np.arange(10, 110, 11)
        alpha_values = np.round(np.linspace(0, 1, 101), 2)
        
        alphas, self.distances = [], []
        for bins in bins_size:
            pos = getHist(pos_scores, bins)
            neg = getHist(neg_scores, bins)
            test = getHist(predictions, bins)
            dists = []
            for a in alpha_values:
                mix = self._mix(pos, neg, a)
                dists.append(BaseMixture.get_distance(mix, test, measure="hellinger"))
            a = alpha_values[np.argmin(dists)]
            alphas.append(a)
            self.distances.append(np.min(dists))
        
        best_alpha = np.median(alphas)
        best_distance = np.median(self.distances)    
        
        return best_alpha, best_distance
    
    def _mix(self, pos_hist, neg_hist, alpha):
        return alpha * pos_hist + (1 - alpha) * neg_hist



# =====================================================
# SMM
# =====================================================

class SMM(AggregativeMixture):
    """
    Sample Mean Matching (SMM).
    """
    
    def best_mixture(self, predictions, pos_scores, neg_scores):
        mean_pos = np.mean(pos_scores)
        mean_neg = np.mean(neg_scores)
        mean_test = np.mean(predictions)
        
        alpha = (mean_test - mean_neg) / (mean_pos - mean_neg)
        return alpha, None


# =====================================================
# SORD
# =====================================================

class SORD(AggregativeMixture):
    """
    Sample Ordinal Distance (SORD).
    """


    def best_mixture(self, predictions, pos_scores, neg_scores):
        alphas = np.linspace(0, 1, 101)
        self.distances = []
        
        pos, neg, test = pos_scores, neg_scores, predictions
        n_pos, n_neg, n_test = len(pos), len(neg), len(test)
        for a in alphas:
            pos_w = np.full(n_pos, a / n_pos)
            neg_w = np.full(n_neg, (1 - a) / n_neg)
            test_w = np.full(n_test, -1 / n_test)
            scores = np.concatenate([pos, neg, test])
            weights = np.concatenate([pos_w, neg_w, test_w])
            idx = np.argsort(scores)
            sorted_scores = scores[idx]
            sorted_weights = weights[idx]
            cum_w = sorted_weights[0]
            total = 0
            for i in range(1, len(sorted_scores)):
                seg = sorted_scores[i] - sorted_scores[i - 1]
                total += abs(seg * cum_w)
                cum_w += sorted_weights[i]
            self.distances.append(total)
            
        best_distance_index = np.argmin(self.distances)
        best_alpha = alphas[best_distance_index]
        best_distance = self.distances[best_distance_index]
        return best_alpha, best_distance





# =====================================================
# Non aggregative Mixture-based Quantifiers
# =====================================================


class HDx(BaseMixture):
    """
    Hellinger Distance-based Quantifier (HDx).
    """
    
    _parameter_constraints = {
        "bins_size": ["array-like", None]
    }

    def __init__(self, bins_size=None):
        super().__init__()
        if bins_size is None:
            bins_size = np.append(np.linspace(2, 20, 10), 30)

        self.bins_size = bins_size
        self.neg_features = None
        self.pos_features = None
        
    
    def _fit(self, X, y, *args, **kwargs):
        self.pos_features = X[y == self.classes[1]]
        self.neg_features = X[y == self.classes[0]]

        return self
    
    def _predict(self, X) -> np.ndarray:
        alpha, _ = self.best_mixture(X, self.pos_features, self.neg_features)
        prevalence = np.array([1 - alpha, alpha])
        prevalence = validate_prevalences(self, prevalence, self.classes)
        return prevalence
    
    def best_mixture(self, X, pos, neg):
        alpha_values = np.round(np.linspace(0, 1, 101), 2)
        self.distances = []

        # Iterate over alpha values to compute the prevalence
        for alpha in alpha_values:
            distances = []

            # For each feature, compute the Hellinger distance
            for feature_idx in range(X.shape[1]):
                
                for bins in self.bins_size:
                
                    pos_feature = pos[:, feature_idx]
                    neg_feature = neg[:, feature_idx]
                    test_feature = X[:, feature_idx]

                    pos_hist = getHist(pos_feature, bins)
                    neg_hist = getHist(neg_feature, bins)
                    test_hist = getHist(test_feature, bins)

                    mix_hist = alpha * pos_hist + (1 - alpha) * neg_hist
                    distance = BaseMixture.get_distance(mix_hist, test_hist, measure="hellinger")
                    distances.append(distance)

            avg_distance = np.mean(distances)
            self.distances.append(avg_distance)
        best_alpha = alpha_values[np.argmin(self.distances)]
        best_distance = np.min(self.distances)
        return best_alpha, best_distance