import numpy as np
from abc import abstractmethod

from mlquantify.base_aggregative import AggregationMixin, SoftLearnerQMixin, _get_learner_function
from mlquantify.mixture._base import BaseMixture
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._get_scores import apply_cross_validation



# =====================================================
# Utility functions
# =====================================================

def getHist(scores: np.ndarray, bins: int) -> np.ndarray:
    """
    Compute a normalized histogram (density) for the given scores.

    Parameters
    ----------
    scores : np.ndarray
        Array of score values (e.g., predicted probabilities).
    bins : int
        Number of bins to use in the histogram.

    Returns
    -------
    hist : np.ndarray
        Normalized histogram (sums to 1).
    """
    hist, _ = np.histogram(scores, bins=bins, range=(0, 1), density=False)
    hist = hist.astype(float)
    if np.sum(hist) > 0:
        hist /= np.sum(hist)
    return hist


def ternary_search(left: float, right: float, func, tol: float = 1e-4) -> float:
    """
    Ternary search to find the minimum of a unimodal function in [left, right].

    Parameters
    ----------
    left : float
        Left bound.
    right : float
        Right bound.
    func : callable
        Function to minimize.
    tol : float
        Tolerance for termination.

    Returns
    -------
    float
        Approximate position of the minimum.
    """
    while right - left > tol:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        f1, f2 = func(m1), func(m2)
        if f1 < f2:
            right = m2
        else:
            left = m1
    return (left + right) / 2


def topsoe(p: np.ndarray, q: np.ndarray) -> float:
    """
    Topsoe distance between two probability distributions.

    D_T(p, q) = sum( p*log(2p/(p+q)) + q*log(2q/(p+q)) )
    """
    p = np.maximum(p, 1e-20)
    q = np.maximum(q, 1e-20)
    return np.sum(p * np.log(2 * p / (p + q)) + q * np.log(2 * q / (p + q)))


def probsymm(p: np.ndarray, q: np.ndarray) -> float:
    """
    Probabilistic Symmetric distance.

    D_PS(p, q) = sum( (p - q) * log(p / q) )
    """
    p = np.maximum(p, 1e-20)
    q = np.maximum(q, 1e-20)
    return np.sum((p - q) * np.log(p / q))


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """
    Hellinger distance between two probability distributions.

    H(p, q) = (1/sqrt(2)) * sqrt( sum( (sqrt(p) - sqrt(q))^2 ) )
    """
    p = np.maximum(p, 1e-20)
    q = np.maximum(q, 1e-20)
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def sqEuclidean(p: np.ndarray, q: np.ndarray) -> float:
    """
    Squared Euclidean distance between two vectors.
    """
    return np.sum((p - q) ** 2)



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
        super().__init__(learner)
        self.pos_scores = None
        self.neg_scores = None

    def _precompute_training(self, train_predictions, train_y_values, learner_fitted=False):
        """
        Fit learner and store score distributions for positive and negative classes.
        """ 
        
        learner_function = _get_learner_function(self)
        
        if learner_fitted:
            train_predictions = getattr(self.learner, learner_function)(X)
            y_train_labels = y
        else:
            train_predictions, y_train_labels = apply_cross_validation(
                self.learner,
                X,
                y,
                function= learner_function,
                cv= 5,
                stratified= True,
                random_state= None,
                shuffle= True
            )
        
        # Get predictions for all samples
        self.pos_scores = train_predictions[train_y_values == self.classes[1], 1]
        self.neg_scores = train_predictions[train_y_values == self.classes[0], 1]
        return self
    
    @abstractmethod
    def best_mixture(self, predictions, train_predictions, train_y_values):
        ...


    @classmethod
    def get_distance(cls, dist_train, dist_test, measure="hellinger"):
        """
        Compute distance between two distributions.
        """
        if np.sum(dist_train) < 1e-20 or np.sum(dist_test) < 1e-20:
            raise ValueError("One or both vectors are zero (empty)...")
        if len(dist_train) != len(dist_test):
            raise ValueError("Arrays must have the same length.")

        dist_train = np.maximum(dist_train, 1e-20)
        dist_test = np.maximum(dist_test, 1e-20)

        if measure == "topsoe":
            return topsoe(dist_train, dist_test)
        elif measure == "probsymm":
            return probsymm(dist_train, dist_test)
        elif measure == "hellinger":
            return hellinger(dist_train, dist_test)
        elif measure == "euclidean":
            return sqEuclidean(dist_train, dist_test)
        else:
            raise ValueError(f"Invalid measure: {measure}")
    

# =====================================================
# DyS
# =====================================================

class DyS(BaseMixture):
    """
    Distribution y-Similarity (DyS).
    """
    
    _parameter_constraints = {
        "measure": [Options("hellinger", "topsoe", "probsymm")],
        "bins_size": [Options("array-like", Interval(2, None))]
    }

    def __init__(self, learner=None, measure="topsoe", bins_size=None):
        super().__init__(learner)
        if bins_size is None:
            bins_size = np.append(np.linspace(2, 20, 10), 30)
            
        self.measure = measure
        self.bins_size = np.asarray(bins_size)

    def _best_mixture(self, predictions, train_predictions, train_y_values):
        prevs = []
        distances = []
        for bins in self.bins_size:
            pos = getHist(self.pos_scores, bins)
            neg = getHist(self.neg_scores, bins)
            test = getHist(predictions, bins)

            def f(alpha):
                mix = self._mix(pos, neg, alpha)
                return self.get_distance(mix, test, measure=self.measure)

            best_alpha = ternary_search(0, 1, f)
            prevs.append(best_alpha)
            distances.append(f(best_alpha))
        alpha = np.median(prevs)
        best_distance = min(distances)
        return alpha, best_distance
        
    def _mix(self, pos_hist, neg_hist, alpha):
        return alpha * pos_hist + (1 - alpha) * neg_hist

    def _get_min_distances_dys(self, test_scores):
        prevs = []
        for bins in self.bins_size:
            pos = getHist(self.pos_scores, bins)
            neg = getHist(self.neg_scores, bins)
            test = getHist(test_scores, bins)

            def f(alpha):
                mix = self._mixture(pos, neg, alpha)
                return self.get_distance(mix, test, measure=self.measure)

            prevs.append(ternary_search(0, 1, f))
        return prevs


# =====================================================
# HDy
# =====================================================

class HDy(BaseMixture):
    """
    Hellinger Distance Minimization (HDy).
    """

    def _estimate_prevalences(self, X):
        test_scores = self.predict_learner(X)[:, 1]
        best_alphas, _ = self._get_min_distances_hdy(test_scores)
        alpha = np.median(best_alphas)
        return np.array([1 - alpha, alpha])

    def _get_min_distances_hdy(self, test_scores):
        bins_size = np.arange(10, 110, 11)
        alpha_values = np.round(np.linspace(0, 1, 101), 2)
        best_alphas, distances = [], []
        for bins in bins_size:
            pos = getHist(self.pos_scores, bins)
            neg = getHist(self.neg_scores, bins)
            test = getHist(test_scores, bins)
            dists = []
            for a in alpha_values:
                mix = self._mixture(pos, neg, a)
                dists.append(self.get_distance(mix, test, measure="hellinger"))
            best_a = alpha_values[np.argmin(dists)]
            best_alphas.append(best_a)
            distances.append(np.min(dists))
        return best_alphas, distances


# =====================================================
# SMM
# =====================================================

class SMM(BaseMixture):
    """
    Sample Mean Matching (SMM).
    """

    def _estimate_prevalences(self, X):
        test_scores = self.predict_learner(X)[:, 1]
        mean_pos = np.mean(self.pos_scores)
        mean_neg = np.mean(self.neg_scores)
        mean_test = np.mean(test_scores)
        alpha = (mean_test - mean_neg) / (mean_pos - mean_neg)
        return np.array([1 - alpha, alpha])


# =====================================================
# SORD
# =====================================================

class SORD(BaseMixture):
    """
    Sample Ordinal Distance (SORD).
    """

    def __init__(self, learner=None):
        super().__init__(learner)
        self.best_distance_index = None

    def _estimate_prevalences(self, X):
        test_scores = self.predict_learner(X)[:, 1]
        alphas, distances = self._calculate_distances(test_scores)
        self.best_distance_index = np.argmin(distances)
        alpha = alphas[self.best_distance_index]
        return np.array([1 - alpha, alpha])

    def _calculate_distances(self, test_scores):
        alphas = np.linspace(0, 1, 101)
        pos, neg, test = self.pos_scores, self.neg_scores, test_scores
        n_pos, n_neg, n_test = len(pos), len(neg), len(test)
        dists = []
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
            dists.append(total)
        return alphas, dists
