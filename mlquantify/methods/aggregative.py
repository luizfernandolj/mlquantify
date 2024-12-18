import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ..base import AggregativeQuantifier
from ..utils.method import *

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import mlquantify as mq





class CC(AggregativeQuantifier):
    """Classify and Count (CC).

    The simplest quantification method involves classifying each instance 
    and then counting the number of instances assigned to each class to 
    estimate the class prevalence.

    This method is based on the concept of classification and counting the 
    number of instances for each class, which is used to estimate the 
    class prevalence.

    Attributes
    ----------
    learner : BaseEstimator
        The machine learning model used to classify the instances. 
        It must be an estimator from scikit-learn (e.g., LogisticRegression, 
        RandomForestClassifier).

    See Also
    --------
    AggregativeQuantifier : Base class for aggregative quantification methods.

    References
    ----------
    FORMAN, George. Quantifying counts and costs via classification. 
    Data Mining and Knowledge Discovery, v. 17, p. 164-206, 2008. 
    Available at: https://link.springer.com/article/10.1007/s10618-008-0097-y

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn-compatible model that serves as the classifier.

    Methods
    -------
    fit(X, y)
        Fits the learner to the data.
    
    predict(X) -> dict
        Predicts the class labels for the given data and calculates 
        the prevalence of each class based on the predictions.

    Examples
    --------
    >>> from mlquantify.utils.general import get_real_prev
    >>> from mlquantify.methods.aggregative import CC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> features, target = load_wine(return_X_y=True)
    >>> 
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=32)
    >>> 
    >>> cc = CC(RandomForestClassifier())
    >>> cc.fit(X_train, y_train)
    >>> y_pred = cc.predict(X_test)
    >>> y_pred
    {0: 0.4305555555555556, 1: 0.2916666666666667, 2: 0.2777777777777778}
    >>> get_real_prev(y_test)
    {0: 0.4166666666666667, 1: 0.3194444444444444, 2: 0.2638888888888889}
    """
    
    def __init__(self, learner: BaseEstimator=None):
        self.learner = learner
    
    def _fit_method(self, X, y):
        """
        Fits the learner to the data. This method is used internally.
        
        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target labels.
        
        Returns
        -------
        self : CC
            The instance of the CC class.
        """
        self.fit_learner(X, y)
        return self
    
    def _predict_method(self, X) -> np.ndarray:
        """
        Predicts the class labels for the given data and calculates 
        the prevalence of each class based on the predictions.
        
        Parameters
        ----------
        X : array-like
            Feature matrix for prediction.
        
        Returns
        -------
        array-like
            An array containing the prevalence of each class.
        """
        predicted_labels = self.predict_learner(X)
        
        # Count occurrences of each class in the predictions
        class_counts = np.array([np.count_nonzero(predicted_labels == _class) for _class in self.classes])
        
        # Calculate the prevalence of each class
        prevalences = class_counts / len(predicted_labels)
        
        return prevalences


    
    
    




class EMQ(AggregativeQuantifier):
    """Expectation Maximisation Quantifier (EMQ).
    
    EMQ is a quantification method that iteratively adjusts the prior 
    and posterior probabilities of a learner using the Expectation-Maximisation (EM) algorithm. 
    It is particularly useful for scenarios where the class distribution in the test set 
    differs from that in the training set.
    
    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn-compatible model used to classify the instances.
    priors : array-like
        Prior probabilities of the classes, estimated from the training data.
    
    References
    ----------
    SAERENS, Marco; LATINNE, Patrice; DECAESTECKER, Christine. Adjusting the outputs of a classifier 
    to new a priori probabilities: a simple procedure. Neural Computation, v. 14, n. 1, p. 21-41, 2002. 
    Available at: https://ieeexplore.ieee.org/abstract/document/6789744

    Examples
    --------
    >>> from mlquantify.methods.aggregative import EMQ
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_wine(return_X_y=True)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=32)
    >>>
    >>> emq = EMQ(RandomForestClassifier())
    >>> emq.fit(X_train, y_train)
    >>> prevalences = emq.predict(X_test)
    >>> print(prevalences)
    {0: 0.4466744706195974, 1: 0.29747794914814046, 2: 0.25584758023226206}
    >>> get_real_prev(y_test)
    {0: 0.4166666666666667, 1: 0.3194444444444444, 2: 0.2638888888888889}
    """
    
    MAX_ITER = 1000
    EPSILON = 1e-6
    
    @property
    def is_probabilistic(self) -> bool:
        return True
    
    def __init__(self, learner: BaseEstimator=None):
        self.learner = learner
        self.priors = None
    
    def _fit_method(self, X, y):
        """
        Fits the learner to the training data and calculates prior probabilities.
        
        Parameters
        ----------
        X : array-like
            Feature matrix for training.
        y : array-like
            Target labels for training.
        
        Returns
        -------
        self : EMQ
            The fitted instance of EMQ.
        """
        self.fit_learner(X, y)
         
        counts = np.array([np.count_nonzero(y == _class) for _class in self.classes])
        self.priors = counts / len(y)
        
        return self
    
    def _predict_method(self, X) -> dict:
        """
        Predicts the prevalence of each class in the test data.
        
        Parameters
        ----------
        X : array-like
            Feature matrix for prediction.
        
        Returns
        -------
        dict
            A dictionary with class labels as keys and their prevalence as values.
        """
        posteriors = self.predict_learner(X)
        prevalences, _ = self.EM(self.priors, posteriors)
        
        return prevalences
    
    def predict_proba(self, X, epsilon: float = EPSILON, max_iter: int = MAX_ITER) -> np.ndarray:
        """
        Predicts the posterior probabilities for the test data after adjustment using EM.
        
        Parameters
        ----------
        X : array-like
            Feature matrix for prediction.
        epsilon : float, optional
            Convergence threshold for the EM algorithm (default: EPSILON).
        max_iter : int, optional
            Maximum number of iterations for the EM algorithm (default: MAX_ITER).
        
        Returns
        -------
        np.ndarray
            Adjusted posterior probabilities.
        """
        posteriors = self.predict_learner(X)
        _, posteriors = self.EM(self.priors, posteriors, epsilon, max_iter)
        return posteriors
    
    @classmethod
    def EM(cls, priors, posteriors, epsilon=EPSILON, max_iter=MAX_ITER):
        """
        Expectation-Maximisation (EM) algorithm for adjusting prior and posterior probabilities.
        
        The algorithm iterates over the data, adjusting the probabilities until convergence 
        or reaching the maximum number of iterations. It estimates the class prevalence 
        and adjusts the posterior probabilities for each class.

        Parameters
        ----------
        priors : array-like
            Initial prior probabilities for each class.
        posteriors : array-like
            Initial posterior probabilities for each test instance and class.
        epsilon : float, optional
            Convergence threshold (default: EPSILON).
        max_iter : int, optional
            Maximum number of iterations (default: MAX_ITER).
        
        Returns
        -------
        tuple
            Adjusted prevalence (array-like) and updated posterior probabilities (array-like).
        """
        Px = posteriors
        prev_prevalence = np.copy(priors)
        running_estimate = np.copy(prev_prevalence)  # Initialized with the training prevalence

        iteration, converged = 0, False
        previous_estimate = None

        while not converged and iteration < max_iter:
            # E-step: Compute unnormalized posteriors
            posteriors_unnormalized = (running_estimate / prev_prevalence) * Px
            posteriors = posteriors_unnormalized / posteriors_unnormalized.sum(axis=1, keepdims=True)

            # M-step: Update the running prevalence estimate
            running_estimate = posteriors.mean(axis=0)

            if previous_estimate is not None and np.mean(np.abs(running_estimate - previous_estimate)) < epsilon and iteration > 10:
                converged = True

            previous_estimate = running_estimate
            iteration += 1

        if not converged:
            print('[Warning] The method has reached the maximum number of iterations; it might not have converged')

        return running_estimate, posteriors

    
    
    
    
    
    



class FM(AggregativeQuantifier):
    """The Friedman Method (FM).
    
    FM is a quantification method similar to GPAC (General Probabilistic Aggregative Classifier), 
    but instead of averaging confidence scores from probabilistic classifiers, 
    it uses the proportion of confidence scores that exceed the expected class frequencies 
    estimated from the training data.

    This method leverages a confusion matrix computed during training to adjust 
    class prevalences in the test set, solving an optimization problem to align 
    predicted and actual distributions.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn-compatible model used for classification.
    CM : np.ndarray
        The confusion matrix, normalized by class counts.
    priors : array-like
        Prior probabilities of the classes, estimated from training data.

    References
    ----------
    Friedman, J. (2001). Quantification via Classification. Presentation. 
    Available at: https://jerryfriedman.su.domains/talks/qc.pdf
    
    Examples
    --------
    >>> from mlquantify.utils.general import get_real_prev
    >>> from mlquantify.methods.aggregative import FM
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> features, target = load_wine(return_X_y=True)
    >>> 
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=32)
    >>> 
    >>> fm = FM(RandomForestClassifier())
    >>> fm.fit(X_train, y_train)
    >>> y_pred = fm.predict(X_test)
    >>> y_pred
    {0: 0.4207283701943278, 1: 0.3049753216939303, 2: 0.27429630811174194}
    >>> get_real_prev(y_test)
    {0: 0.4166666666666667, 1: 0.3194444444444444, 2: 0.2638888888888889}
    """
    
    @property
    def is_probabilistic(self) -> bool:
        return True
    
    
    def __init__(self, learner: BaseEstimator=None):
        self.learner = learner
        self.CM = None
    
    def _fit_method(self, X, y):
        """
        Fits the learner and computes the confusion matrix.

        The confusion matrix is computed based on cross-validated predicted labels 
        and probabilities. It represents the proportions of confidence scores 
        exceeding the priors for each class.

        Parameters
        ----------
        X : array-like
            Feature matrix for training.
        y : array-like
            Target labels for training.
        
        Returns
        -------
        self : FM
            The fitted instance of FM.
        """
        # Get predicted labels and probabilities using cross-validation
        if mq.arguments["y_labels"] is not None and mq.arguments["posteriors_train"] is not None:
            y_labels = mq.arguments["y_labels"]
            probabilities = mq.arguments["posteriors_train"]
        else:
            y_labels, probabilities = get_scores(X, y, self.learner, self.cv_folds, self.learner_fitted)
        
        # Fit the learner if it hasn't been fitted already
        self.fit_learner(X, y)
        
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
        """
        Predicts class prevalences in the test set using the confusion matrix.

        Solves an optimization problem to find class prevalences that best 
        align with the observed proportions in the test set.

        Parameters
        ----------
        X : array-like
            Feature matrix for prediction.
        
        Returns
        -------
        dict
            A dictionary with class labels as keys and their prevalence as values.
        """
        posteriors = self.predict_learner(X)
        
        # Calculate the estimated prevalences in the test set
        prevs_estim = np.sum(posteriors > self.priors, axis=0) / posteriors.shape[0]
        
        # Define the objective function for optimization
        def objective(prevs_pred):
            return np.linalg.norm(self.CM @ prevs_pred - prevs_estim)
        
        # Constraints for the optimization problem
        constraints = [
            {'type': 'eq', 'fun': lambda prevs_pred: np.sum(prevs_pred) - 1.0},
            {'type': 'ineq', 'fun': lambda prevs_pred: prevs_pred}
        ]
        
        # Initial guess for the optimization
        initial_guess = np.ones(self.CM.shape[1]) / self.CM.shape[1]
        
        # Solve the optimization problem
        result = minimize(objective, initial_guess, constraints=constraints, bounds=[(0, 1)] * self.CM.shape[1])
        
        if result.success:
            prevalences = result.x
        else:
            print("Optimization did not converge")
            prevalences = self.priors
        
        return prevalences
    






class GAC(AggregativeQuantifier):
    """
    Generalized Adjusted Count (GAC).

    GAC is a quantification method that applies a classifier to estimate the distribution 
    of class labels in the test set by solving a system of linear equations. This system 
    is constructed using a conditional probability matrix derived from training data and 
    is solved via constrained least-squares regression.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn-compatible model used for classification.
    train_size : float, optional
        Proportion of the dataset to include in the training split, by default 0.6.
    random_state : int, optional
        Random seed for reproducibility of data splits, by default None.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn-compatible model used for classification.
    cond_prob_matrix : np.ndarray
        Conditional probability matrix, representing P(yi|yj).
    train_size : float, optional
        Proportion of the dataset to include in the training split, by default 0.6.
    random_state : int, optional
        Random seed for reproducibility of data splits, by default None.


    References
    ----------
    Firat, Aykut. Unified framework for quantification. arXiv preprint arXiv:1606.00868, 2016. 
    Available at: https://arxiv.org/abs/1606.00868
    
    Examples
    --------
    >>> from mlquantify.utils.general import get_real_prev
    >>> from mlquantify.methods.aggregative import GAC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_wine(return_X_y=True)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=32)
    >>>
    >>> gac = GAC(RandomForestClassifier())
    >>> gac.fit(X_train, y_train)
    >>> y_pred = gac.predict(X_test)
    >>> y_pred
    {0: 0.4305555555555556, 1: 0.2916666666666667, 2: 0.2777777777777778}
    >>> get_real_prev(y_test)
    {0: 0.4166666666666667, 1: 0.3194444444444444, 2: 0.2638888888888889}
    """

    
    def __init__(self, learner: BaseEstimator=None, train_size:float=0.6, random_state:int=None):
        self.learner = learner
        self.cond_prob_matrix = None
        self.train_size = train_size
        self.random_state = random_state
    
    def _fit_method(self, X, y):
        """
        Trains the model and computes the conditional probability matrix.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features of the dataset.
        y : pd.Series or np.ndarray
            Labels of the dataset.

        Returns
        -------
        self : GAC
            Fitted quantifier object.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        if self.learner_fitted or self.learner is None:
            y_pred = mq.arguments["y_pred_train"] if mq.arguments["y_pred_train"] is not None else self.predict_learner(X)
            y_label = y
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=self.train_size, stratify=y, random_state=self.random_state
            )
            self.fit_learner(X_train, y_train)
            y_label = y_val
            y_pred = self.learner.predict(X_val)

        self.cond_prob_matrix = GAC.get_cond_prob_matrix(self.classes, y_label, y_pred)
        return self

    def _predict_method(self, X) -> dict:
        """
        Predicts the class prevalences in the test set and adjusts them.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features of the test dataset.

        Returns
        -------
        dict
            Adjusted class prevalences.
        """
        y_pred = self.predict_learner(X)
        _, counts = np.unique(y_pred, return_counts=True)
        predicted_prevalences = counts / counts.sum()
        adjusted_prevalences = self.solve_adjustment(self.cond_prob_matrix, predicted_prevalences)
        return adjusted_prevalences

    @classmethod
    def get_cond_prob_matrix(cls, classes: list, y_labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Computes the conditional probability matrix P(yi|yj).

        Parameters
        ----------
        classes : list
            List of class labels.
        y_labels : np.ndarray
            True labels from the validation set.
        predictions : np.ndarray
            Predicted labels from the classifier.

        Returns
        -------
        np.ndarray
            Conditional probability matrix.
        """
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
    def solve_adjustment(cls, cond_prob_matrix: np.ndarray, predicted_prevalences: np.ndarray) -> np.ndarray:
        """
        Solves the linear system Ax = B to adjust predicted prevalences.

        Parameters
        ----------
        cond_prob_matrix : np.ndarray
            Conditional probability matrix (A).
        predicted_prevalences : np.ndarray
            Predicted class prevalences (B).

        Returns
        -------
        np.ndarray
            Adjusted class prevalences.
        """
        A = cond_prob_matrix
        B = predicted_prevalences
        try:
            adjusted_prevalences = np.linalg.solve(A, B)
            adjusted_prevalences = np.clip(adjusted_prevalences, 0, 1)
            adjusted_prevalences /= adjusted_prevalences.sum()
        except np.linalg.LinAlgError:
            adjusted_prevalences = predicted_prevalences  # Return unadjusted if adjustment fails
        return adjusted_prevalences
    
    
    
    
    
    
    
    
    
    



class GPAC(AggregativeQuantifier):
    """
    Generalized Probabilistic Adjusted Count (GPAC).

    GPAC is an extension of the Generalized Adjusted Count (GAC) method. It constructs a system of 
    linear equations using the confidence scores from probabilistic classifiers, similar to the PAC method. 
    The system is solved to estimate the prevalence of classes in a test dataset.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn-compatible model used for classification.
    train_size : float, optional
        Proportion of the dataset to include in the training split, by default 0.6.
    random_state : int, optional
        Random seed for reproducibility of data splits, by default None.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn-compatible model used for classification.
    cond_prob_matrix : np.ndarray
        Conditional probability matrix representing P(yi|yj).
    train_size : float, optional
        Proportion of the dataset to include in the training split, by default 0.6.
    random_state : int, optional
        Random seed for reproducibility of data splits, by default None.

    References
    ----------
    Firat, Aykut. Unified framework for quantification. arXiv preprint arXiv:1606.00868, 2016.
    Available at: https://arxiv.org/abs/1606.00868
    
    Examples
    --------
    >>> from mlquantify.utils.general import get_real_prev
    >>> from mlquantify.methods.aggregative import GPAC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_wine(return_X_y=True, random_state=32)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=32)
    >>>
    >>> gpac = GPAC(RandomForestClassifier())
    >>> gpac.fit(X_train, y_train)
    >>> y_pred = gpac.predict(X_test)
    >>> y_pred
    {0: 0.41435185185185186, 1: 0.3078703703703704, 2: 0.2777777777777778}
    >>> get_real_prev(y_test)
    {0: 0.4166666666666667, 1: 0.3194444444444444, 2: 0.2638888888888889}
    """

    def __init__(self, learner: BaseEstimator=None, train_size: float = 0.6, random_state: int = None):
        self.learner = learner
        self.cond_prob_matrix = None
        self.train_size = train_size
        self.random_state = random_state

    def _fit_method(self, X, y):
        """
        Trains the model and computes the conditional probability matrix using validation data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features of the dataset.
        y : pd.Series or np.ndarray
            Labels of the dataset.

        Returns
        -------
        self : GPAC
            Fitted quantifier object.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        if self.learner_fitted or self.learner is None:
            y_pred = mq.arguments["y_pred_train"] if mq.arguments["y_pred_train"] is not None else self.predict_learner(X)
            y_labels = y
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=self.train_size, stratify=y, random_state=self.random_state
            )
            self.fit_learner(X_train, y_train)
            y_labels = y_val
            y_pred = self.predict_learner(X_val)

        # Compute the conditional probability matrix
        self.cond_prob_matrix = GAC.get_cond_prob_matrix(self.classes, y_labels, y_pred)
        return self

    def _predict_method(self, X) -> dict:
        """
        Predicts class prevalences in the test set and adjusts them using the conditional probability matrix.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features of the test dataset.

        Returns
        -------
        dict
            Adjusted class prevalences.
        """
        predictions = self.predict_learner(X)

        # Compute the distribution of predictions
        predicted_prevalences = np.zeros(self.n_class)
        _, counts = np.unique(predictions, return_counts=True)
        predicted_prevalences[:len(counts)] = counts
        predicted_prevalences /= predicted_prevalences.sum()

        # Adjust prevalences using the conditional probability matrix
        adjusted_prevalences = GAC.solve_adjustment(self.cond_prob_matrix, predicted_prevalences)
        return adjusted_prevalences

    @classmethod
    def get_cond_prob_matrix(cls, classes: list, y_labels: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the conditional probability matrix P(yi|yj).

        Parameters
        ----------
        classes : list
            List of class labels.
        y_labels : np.ndarray
            True labels from the validation set.
        y_pred : np.ndarray
            Predicted probabilities or labels from the classifier.

        Returns
        -------
        np.ndarray
            Conditional probability matrix with entry (i, j) representing P(yi|yj).
        """
        n_classes = len(classes)
        cond_prob_matrix = np.eye(n_classes)

        for i, class_ in enumerate(classes):
            class_indices = y_labels == class_
            if class_indices.any():
                cond_prob_matrix[i] = y_pred[class_indices].mean(axis=0)

        return cond_prob_matrix.T










class PCC(AggregativeQuantifier):
    """
    Probabilistic Classify and Count (PCC).

    PCC is a quantification method that uses a probabilistic classifier to estimate 
    class prevalences in a test dataset. It computes the mean of the predicted 
    probabilities for each class to determine their prevalences.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn-compatible probabilistic classifier.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn-compatible probabilistic classifier.
        
    References
    ----------
    BELLA, Antonio et al. Quantification via probability estimators. In: 2010 IEEE International Conference on Data Mining. IEEE, 2010. p. 737-742. Avaliable at: https://ieeexplore.ieee.org/abstract/document/5694031

    Examples
    --------
    >>> from mlquantify.utils.general import get_real_prev
    >>> from mlquantify.methods.aggregative import PCC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_wine(return_X_y=True, random_state=32)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=32)
    >>>
    >>> pcc = PCC(RandomForestClassifier())
    >>> pcc.fit(X_train, y_train)
    >>> y_pred = pcc.predict(X_test)
    >>> y_pred
    {0: 0.4036111111111111, 1: 0.3427777777777778, 2: 0.2536111111111111}
    >>> get_real_prev(y_test)
    {0: 0.4166666666666667, 1: 0.3194444444444444, 2: 0.2638888888888889}
    """
    @property
    def is_probabilistic(self) -> bool:
        return True

    def __init__(self, learner: BaseEstimator=None):
        self.learner = learner

    def _fit_method(self, X, y):
        """
        Fits the learner to the training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features of the training dataset.
        y : pd.Series or np.ndarray
            Labels of the training dataset.

        Returns
        -------
        self : PCC
            Fitted quantifier object.
        """
        self.fit_learner(X, y)
        return self

    def _predict_method(self, X) -> np.ndarray:
        """
        Predicts class prevalences in the test dataset by averaging the predicted probabilities.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features of the test dataset.

        Returns
        -------
        np.ndarray
            Estimated prevalences for each class.
        """
        # Initialize a list to store the prevalence for each class
        prevalences = []

        # Calculate the prevalence for each class
        for class_index in range(self.n_class):
            # Get the predicted probabilities for the current class
            class_probabilities = self.predict_learner(X)[:, class_index]

            # Compute the average probability (prevalence) for the current class
            mean_prev = np.mean(class_probabilities)
            prevalences.append(mean_prev)

        return np.asarray(prevalences)

    
    
    





class PWK(AggregativeQuantifier):
    """
    Nearest-Neighbor Based Quantification (PWK).

    PWK extends nearest-neighbor classification to the quantification setting. 
    This k-NN approach uses a weighting scheme that reduces the influence of 
    neighbors from the majority class to better estimate class prevalences. 
    
    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn-compatible classifier that implements a k-NN approach.

    Notes
    -----
    To get the optimal functionality, you must use the `PWKCLF` classifier, which is a classifier that uses K-NN to classify

    References
    ----------
    BARRANQUERO, Jose et al. On the study of nearest neighbor algorithms for prevalence estimation in binary problems. Pattern Recognition, v. 46, n. 2, p. 472-482, 2013. Available at: https://www.sciencedirect.com/science/article/pii/S0031320312003391?casa_token=qgInkRZdEhgAAAAA:Yu_ttk6Tso0xAZR23I0EGnge_UmA_kWI1eB8kxaRZ5Vg1PFLpMwcbEwNvZ5-4Mep7Jgfj9WsCFMMdQ
    
    Examples
    --------
    >>> from mlquantify.utils.general import get_real_prev
    >>> from mlquantify.methods.aggregative import PWK
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_wine(return_X_y=True, random_state=32)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=32)
    >>>
    >>> pwk = PWK(RandomForestClassifier())
    >>> pwk.fit(X_train, y_train)
    >>> y_pred = pwk.predict(X_test)
    >>> y_pred
    {0: 0.4305555555555556, 1: 0.2916666666666667, 2: 0.2777777777777778}
    >>> get_real_prev(y_test)
    {0: 0.4166666666666667, 1: 0.3194444444444444, 2: 0.2638888888888889}
    """

    def __init__(self, learner: BaseEstimator=None):
        self.learner = learner

    def _fit_method(self, X, y):
        """
        Fits the k-NN learner to the training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features of the training dataset.
        y : pd.Series or np.ndarray
            Labels of the training dataset.

        Returns
        -------
        self : PWK
            Fitted quantifier object.
        """
        self.fit_learner(X, y)
        return self

    def _predict_method(self, X) -> dict:
        """
        Predicts class prevalences in the test dataset by analyzing the distribution of predicted labels.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features of the test dataset.

        Returns
        -------
        dict
            A dictionary mapping each class label to its estimated prevalence.
        """
        # Predict class labels for the given data
        predicted_labels = self.predict_learner(X)

        # Compute the distribution of predicted labels
        unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)

        # Calculate the prevalence for each class
        class_prevalences = label_counts / label_counts.sum()

        # Map each class label to its prevalence
        prevalences = {label: prevalence for label, prevalence in zip(unique_labels, class_prevalences)}

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