from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator

from ....base import AggregativeQuantifier

class MixtureModel(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator, threshold:float=0.5):
        self.learner = learner
        self.threshold = threshold