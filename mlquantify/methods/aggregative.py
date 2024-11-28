import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ..base import AggregativeQuantifier
from ..utils.method import *

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

    
    

from . import threshold_optimization

ACC = threshold_optimization.ACC
PACC = threshold_optimization.PACC
T50 = threshold_optimization.T50
MAX = threshold_optimization.MAX
X_method  = threshold_optimization.X_method
MS  = threshold_optimization.MS
MS2 = threshold_optimization.MS2



from . import mixture_models

DySsyn = mixture_models.DySsyn
DyS = mixture_models.DyS
HDy = mixture_models.HDy
SMM = mixture_models.SMM
SORD = mixture_models.SORD