from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator

from ....base import AggregativeQuantifier
from ....utils import probsymm, sqEuclidean, topsoe, hellinger, get_scores

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
        self.distance = None

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
