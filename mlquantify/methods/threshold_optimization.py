from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator

from ..base import AggregativeQuantifier
from ..utils.method import adjust_threshold, get_scores




class ThresholdOptimization(AggregativeQuantifier):
    """Generic Class for methods that are based on adjustments
    of the decision boundary of the underlying classifier in order
    to make the ACC (base method for threshold methods) estimation
    more numerically stable. Most of its strategies involve changing
    the behavior of the denominator of the ACC equation.
    """
    # Class for optimizing classification thresholds

    def __init__(self, learner: BaseEstimator):
        self.learner = learner
        self.threshold = None
        self.cc_output = None
        self.tpr = None
        self.fpr = None
    
    @property
    def is_multiclass(self) -> bool:
        """ All threshold Methods are binary or non multiclass """
        return False
    
    def _fit_method(self, X, y):

        y_labels, probabilities = get_scores(X, y, self.learner, self.cv_folds, self.learner_fitted)
        
        # Adjust thresholds and compute true and false positive rates
        thresholds, tprs, fprs = adjust_threshold(y_labels, probabilities[:, 1], self.classes)
        
        # Find the best threshold based on TPR and FPR
        self.threshold, self.tpr, self.fpr = self.best_tprfpr(thresholds, tprs, fprs)
        
        return self
    
    def _predict_method(self, X) -> dict:
              
        probabilities = self.learner.predict_proba(X)[:, 1]
        
        # Compute the classification count output
        self.cc_output = len(probabilities[probabilities >= self.threshold]) / len(probabilities)
        
        # Calculate prevalence, ensuring it's within [0, 1]
        if self.tpr - self.fpr == 0:
            prevalence = self.cc_output
        else:
            # Equation of all threshold methods to compute prevalence
            prevalence = np.clip((self.cc_output - self.fpr) / (self.tpr - self.fpr), 0, 1)
        
        prevalences = [1- prevalence, prevalence]

        return np.asarray(prevalences)
    
    @abstractmethod
    def best_tprfpr(self, thresholds: np.ndarray, tpr: np.ndarray, fpr: np.ndarray) -> float:
        """Abstract method for determining the best TPR and FPR to use in the equation"""
        ...






class ACC(ThresholdOptimization):
    """ Adjusted Classify and Count or Adjusted Count. Is a 
    base method for the threhold methods.
        As described on the Threshold base class, this method 
    estimate the true positive and false positive rates from
    the training data and utilize them to adjust the output 
    of the CC method.
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        self.threshold = threshold
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        # Get the tpr and fpr where the threshold is equal to the base threshold, default is 0.5
        
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)
    
    
    
    
    
    
    

class MAX(ThresholdOptimization):
    """ Threshold MAX. This method tries to use the
    threshold where it maximizes the difference between
    tpr and fpr to use in the denominator of the equation.
    """
    
    def __init__(self, learner:BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        max_index = np.argmax(np.abs(tprs - fprs))
        
        threshold = thresholds[max_index]
        tpr= tprs[max_index]
        fpr = fprs[max_index]
        return (threshold, tpr, fpr)
    
    






class MS(ThresholdOptimization):
    """ Median Sweep. This method uses an
    ensemble of such threshold-based methods and 
    takes the median prediction.
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        self.threshold = threshold
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        tpr = np.median(tprs)
        fpr = np.median(fprs)
        return (self.threshold, tpr, fpr)
    






class MS2(ThresholdOptimization):
    """ Median Sweep 2. It relies on the same
    strategy of the Median Sweep, but compute 
    the median only for cases in which 
    tpr -fpr > 0.25
    """
    
    def __init__(self, learner:BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        indices = np.where(np.abs(tprs - fprs) > 0.25)[0]
    
        threshold = np.median(thresholds[indices])
        tpr = np.median(tprs[indices])
        fpr = np.median(fprs[indices])
        
        return (threshold, tpr, fpr)
    
    
    
    





class PACC(ThresholdOptimization):
    """ Probabilistic Adjusted Classify and Count. 
    This method adapts the AC approach by using average
    classconditional confidences from a probabilistic 
    classifier instead of true positive and false positive rates.
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        self.threshold = threshold
    
    def _predict_method(self, X):
        prevalences = {}
        
        probabilities = self.learner.predict_proba(X)[:, 1]
        
        mean_scores = np.mean(probabilities)
        
        if self.tpr - self.fpr == 0:
            prevalence = mean_scores
        else:
            prevalence = np.clip(abs(mean_scores - self.fpr) / (self.tpr - self.fpr), 0, 1)
        
        prevalences[self.classes[0]] = 1 - prevalence
        prevalences[self.classes[1]] = prevalence

        return prevalences
    
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)
    
    
    





class T50(ThresholdOptimization):
    """ Threshold 50. This method tries to
    use the threshold where tpr = 0.5.
    """
    
    def __init__(self, learner:BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        min_index = np.argmin(np.abs(tprs - 0.5))
        threshold = thresholds[min_index]
        tpr = tprs[min_index]
        fpr = fprs[min_index]
        return (threshold, tpr, fpr)
    
    
    
    





class X_method(ThresholdOptimization):
    """ Threshold X. This method tries to
    use the threshold where fpr = 1 - tpr
    """
    
    def __init__(self, learner:BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        min_index = np.argmin(abs(1 - (tprs + fprs)))
        
        threshold = thresholds[min_index]
        tpr = tprs[min_index]
        fpr = fprs[min_index]
        
        return (threshold, tpr, fpr)