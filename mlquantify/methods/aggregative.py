from abc import abstractmethod

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ..base import AggregativeQuantifier
from ..utils.method import *
from ..utils.general import get_real_prev

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split





class CC(AggregativeQuantifier):
    """Classify and Count. The simplest quantification method
    involves classifying each instance and then counting the 
    number of instances assigned to each class to estimate 
    the class prevalence.
    """
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
    
    
    def _fit_method(self, X, y):
        if not self.learner_fitted:
            self.learner.fit(X, y)
        return self
    
    
    def _predict_method(self, X) -> dict:
        predicted_labels = self.learner.predict(X)
        
        # Count occurrences of each class in the predictions
        class_counts = np.array([np.count_nonzero(predicted_labels == _class) for _class in self.classes])
        
        # Calculate the prevalence of each class
        prevalences = class_counts / len(predicted_labels)
        
        return prevalences
    
    
    




class EMQ(AggregativeQuantifier):
    """Expectation Maximisation Quantifier. It is a method that
    ajust the priors and posteriors probabilities of a learner
    """
    
    MAX_ITER = 1000
    EPSILON = 1e-6
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.priors = None
    
    def _fit_method(self, X, y):
        
        if not self.learner_fitted:
            self.learner.fit(X, y)
        
        counts = np.array([np.count_nonzero(y == _class) for _class in self.classes])
        self.priors = counts / len(y)
        
        return self
    
    def _predict_method(self, X) -> dict:
        
        posteriors = self.learner.predict_proba(X)
        prevalences, _ = self.EM(self.priors, posteriors)
        
        return prevalences
    
    
    def predict_proba(self, X, epsilon:float=EPSILON, max_iter:int=MAX_ITER) -> np.ndarray:
        posteriors = self.learner.predict_proba(X)
        _, posteriors = self.EM(self.priors, posteriors, epsilon, max_iter)
        return posteriors
    
    
    @classmethod
    def EM(cls, priors, posteriors, epsilon=EPSILON, max_iter=MAX_ITER):
        """Expectaion Maximization function, it iterates several times
        and At each iteration step, both the a posteriori and the a 
        priori probabilities are reestimated sequentially for each new 
        observation and each class. The iterative procedure proceeds 
        until the convergence of the estimated probabilities.

        Args:
            priors (array-like): priors probabilites of the train.
            posteriors (array-like): posteriors probabiblities of the test.
            epsilon (float): value that helps to indify the convergence.
            max_iter (int): max number of iterations.

        Returns:
            the predicted prevalence and the ajusted posteriors.
        """
        
        Px = posteriors
        prev_prevalence = np.copy(priors)
        running_estimate = np.copy(prev_prevalence)  # Initialized with the training prevalence

        iteration, converged = 0, False
        previous_estimate = None

        while not converged and iteration < max_iter:
            # E-step: ps is P(y|xi)
            posteriors_unnormalized = (running_estimate / prev_prevalence) * Px
            posteriors = posteriors_unnormalized / posteriors_unnormalized.sum(axis=1, keepdims=True)

            # M-step:
            running_estimate = posteriors.mean(axis=0)

            if previous_estimate is not None and np.mean(np.abs(running_estimate - previous_estimate)) < epsilon and iteration > 10:
                converged = True

            previous_estimate = running_estimate
            iteration += 1

        if not converged:
            print('[Warning] The method has reached the maximum number of iterations; it might not have converged')

        return running_estimate, posteriors
    
    
    
    
    
    



class FM(AggregativeQuantifier):
    """The Friedman Method. Similar to GPAC, 
    but instead of averaging the confidence scores
    from probabilistic classifiers, it uses the proportion
    of confidence scores that are higher or lower than the
    expected class frequencies found in the training data.
    """
    
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.CM = None
    
    def _fit_method(self, X, y):
        # Get predicted labels and probabilities using cross-validation
        y_labels, probabilities = get_scores(X, y, self.learner, self.cv_folds, self.learner_fitted)
        
        # Fit the learner if it hasn't been fitted already
        if not self.learner_fitted:
            self.learner.fit(X, y)
        
        # Initialize the confusion matrix
        CM = np.zeros((self.n_class, self.n_class))
        
        # Calculate the class priors
        class_counts = np.array([np.count_nonzero(y_labels == _class) for _class in self.classes])
        self.priors = class_counts / len(y_labels)
        
        # Populate the confusion matrix
        for i, _class in enumerate(self.classes):       
            indices = np.where(y_labels == _class)[0]
            CM[:, i] = np.sum(probabilities[indices] > self.priors, axis=0) 
        
        # Normalize the confusion matrix by class counts
        self.CM = CM / class_counts
        
        return self
    
    def _predict_method(self, X) -> dict:
        posteriors = self.learner.predict_proba(X)
        
        # Calculate the estimated prevalences in the test set
        prevs_estim = np.sum(posteriors > self.priors, axis=0) / posteriors.shape[0]
        # Define the objective function for optimization
        def objective(prevs_pred):
            return np.linalg.norm(self.CM @ prevs_pred - prevs_estim)
        
        # Constraints for the optimization problem
        constraints = [{'type': 'eq', 'fun': lambda prevs_pred: np.sum(prevs_pred) - 1.0},
                       {'type': 'ineq', 'fun': lambda prevs_pred: prevs_pred}]
        
        # Initial guess for the optimization
        initial_guess = np.ones(self.CM.shape[1]) / self.CM.shape[1]
        
        # Solve the optimization problem
        result = minimize(objective, initial_guess, constraints=constraints, bounds=[(0, 1)]*self.CM.shape[1])
        
        if result.success:
            prevalences = result.x
        else:
            print("Optimization did not converge")
            prevalences = self.priors
        
        return prevalences
    






class GAC(AggregativeQuantifier):
    """Generalized Adjusted Count. It applies a 
    classifier to build a system of linear equations, 
    and solve it via constrained least-squares regression.
    """
    
    def __init__(self, learner: BaseEstimator, train_size:float=0.6, random_state:int=None):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.cond_prob_matrix = None
        self.train_size = train_size
        self.random_state = random_state
    
    def _fit_method(self, X, y):
        # Ensure X and y are DataFrames
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        if self.learner_fitted:
            y_pred = self.learner.predict(X)
            y_label = y
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=self.train_size, stratify=y, random_state=self.random_state
            )
            
            self.learner.fit(X_train, y_train)
            
            y_label = y_val
            y_pred = self.learner.predict(X_val)
        
        # Compute conditional probability matrix
        self.cond_prob_matrix = GAC.get_cond_prob_matrix(self.classes, y_label, y_pred)
        
        return self
    
    def _predict_method(self, X) -> dict:
        # Predict class labels for the test data
        y_pred = self.learner.predict(X)

        # Distribution of predictions in the test set
        _, counts = np.unique(y_pred, return_counts=True)
        predicted_prevalences = counts / counts.sum()
        
        # Adjust prevalences based on conditional probability matrix
        adjusted_prevalences = self.solve_adjustment(self.cond_prob_matrix, predicted_prevalences)

        return adjusted_prevalences
    
    @classmethod
    def get_cond_prob_matrix(cls, classes:list, y_labels:np.ndarray, predictions:np.ndarray) -> np.ndarray:
        """ Estimate the conditional probability matrix P(yi|yj)"""

        CM = confusion_matrix(y_labels, predictions, labels=classes).T
        CM = CM.astype(float)
        class_counts = CM.sum(axis=0)
        for i, _ in enumerate(classes):
            if class_counts[i] == 0:
                CM[i, i] = 1
            else:
                CM[:, i] /= class_counts[i]
        return CM
    
    @classmethod
    def solve_adjustment(cls, cond_prob_matrix, predicted_prevalences):
        """ Solve the linear system Ax = B with A=cond_prob_matrix and B=predicted_prevalences
        """
        
        #
        A = cond_prob_matrix
        B = predicted_prevalences
        try:
            adjusted_prevalences = np.linalg.solve(A, B)
            adjusted_prevalences = np.clip(adjusted_prevalences, 0, 1)
            adjusted_prevalences /= adjusted_prevalences.sum()
        except (np.linalg.LinAlgError):
            adjusted_prevalences = predicted_prevalences  # No way to adjust them
        return adjusted_prevalences
    
    
    
    
    
    
    
    
    
    



class GPAC(AggregativeQuantifier):
    """Generalized Probabilistic Adjusted Count. Like 
    GAC, it also build a system of linear equations, but 
    utilize the confidence scores from probabilistic 
    classifiers as in the PAC method.
    """
    
    
    def __init__(self, learner: BaseEstimator, train_size:float=0.6, random_state:int=None):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.cond_prob_matrix = None
        self.train_size = train_size
        self.random_state = random_state
    
    def _fit_method(self, X, y):
        # Convert X and y to DataFrames if they are numpy arrays
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        if self.learner_fitted:
            # Use existing model to predict
            y_pred = self.learner.predict(X)
            y_labels = y
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=self.train_size, stratify=y, random_state=self.random_state
            )
            
            self.learner.fit(X_train, y_train)
            
            y_labels = y_val
            y_pred = self.learner.predict(X_val)
        
        # Compute conditional probability matrix using GAC
        self.cond_prob_matrix = GAC.get_cond_prob_matrix(self.classes, y_labels, y_pred)
        
        return self
    
    def _predict_method(self, X) -> dict:
        # Predict class labels for the test data
        predictions = self.learner.predict(X)

        # Calculate the distribution of predictions in the test set
        predicted_prevalences = np.zeros(self.n_class)
        _, counts = np.unique(predictions, return_counts=True)
        predicted_prevalences[:len(counts)] = counts
        predicted_prevalences = predicted_prevalences / predicted_prevalences.sum()
        
        # Adjust prevalences based on the conditional probability matrix from GAC
        adjusted_prevalences = GAC.solve_adjustment(self.cond_prob_matrix, predicted_prevalences)

        # Map class labels to their corresponding prevalences
        return adjusted_prevalences
    
    @classmethod
    def get_cond_prob_matrix(cls, classes:list, y_labels:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        """Estimate the matrix where entry (i,j) is the estimate of P(yi|yj)"""
        
        n_classes = len(classes)
        cond_prob_matrix = np.eye(n_classes)
        
        for i, class_ in enumerate(classes):
            class_indices = y_labels == class_
            if class_indices.any():
                cond_prob_matrix[i] = y_pred[class_indices].mean(axis=0)
        
        return cond_prob_matrix.T










class PCC(AggregativeQuantifier):
    """Probabilistic Classify and Count. This method
    takes the probabilistic predictions and takes the 
    mean of them for each class.
    """
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
    
    def _fit_method(self, X, y):
        if not self.learner_fitted:
            self.learner.fit(X, y)
        return self
    
    def _predict_method(self, X) -> dict:
        # Initialize a dictionary to store the prevalence for each class
        prevalences = []
        
        # Calculate the prevalence for each class
        for class_index in range(self.n_class):
            # Get the predicted probabilities for the current class
            class_probabilities = self.learner.predict_proba(X)[:, class_index]
        
            # Compute the average probability (prevalence) for the current class
            mean_prev = np.mean(class_probabilities)
            prevalences.append(mean_prev)
        
        return np.asarray(prevalences)
    
    
    





class PWK(AggregativeQuantifier):
    """ Nearest-Neighbor based Quantification. This method 
    is based on nearest-neighbor based classification to the
    setting of quantification. In this k-NN approach, it applies
    a weighting scheme which applies less weight on neighbors 
    from the majority class.
    Must be used with PWKCLF to work as expected.
    """
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
    
    def _fit_method(self, X, y):
        if not self.learner_fitted:
            self.learner.fit(X, y)
        return self
    
    def _predict_method(self, X) -> dict:
        # Predict class labels for the given data
        predicted_labels = self.learner.predict(X)
        
        # Compute the distribution of predicted labels
        unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)
        
        # Calculate the prevalence for each class
        class_prevalences = label_counts / label_counts.sum()
        
        # Map each class label to its prevalence
        prevalences  = {label: prevalence for label, prevalence in zip(unique_labels, class_prevalences)}
        
        return prevalences
    
    
    




#===============================================================================================================
#                                              THRESHOLD OPTIMAZTION
#===============================================================================================================




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
    def multiclass_method(self) -> bool:
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
    
    
    
    
    
    
    
    




#===============================================================================================================
#                                                MIXTURE MODELS
#===============================================================================================================




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