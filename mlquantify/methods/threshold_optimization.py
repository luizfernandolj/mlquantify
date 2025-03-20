from abc import abstractmethod
import numpy as np
import warnings
from sklearn.base import BaseEstimator

from ..base import AggregativeQuantifier
from ..utils.method import adjust_threshold, get_scores
import mlquantify as mq




class ThresholdOptimization(AggregativeQuantifier):
    """
    Generic Class for methods that adjust the decision boundary of the underlying classifier
    to make the ACC (base method for threshold methods) estimation more numerically stable. 
    Most strategies involve altering the denominator of the ACC equation.
    
    This class serves as a base for implementing threshold optimization techniques in classification
    tasks. It is designed to adjust thresholds based on true positive and false positive rates, 
    ensuring better quantification performance.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier to be used for threshold optimization.
    threshold : float, optional
        The threshold value to be used for classification decisions. Default is 0.5.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier.
    threshold : float, optional
        The optimized threshold used for classification decisions.
    cc_output : float, optional
        The classification count output, representing the proportion of instances classified 
        as positive based on the threshold.
    tpr : float, optional
        The true positive rate corresponding to the best threshold.
    fpr : float, optional
        The false positive rate corresponding to the best threshold.

    Notes
    -----
    All methods that inherit from this class will be binary quantifiers. In case of multiclass problems, it will be made One vs All.

    Examples
    --------
    >>> from mlquantify.methods.threshold_optimization import ThresholdOptimization
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> class MyThrMethod(ThresholdOptimization):
    ...     def __init__(self, learner, threshold=0.5):
    ...         super().__init__(learner)
    ...         self.threshold = threshold
    ...     def best_tprfpr(self, thresholds, tpr, fpr):
    ...         return thresholds[20], tpr[20], fpr[20]
    >>>  
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> 
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>> 
    >>> mtm = MyThrMethod(learner=SVC(probability=True), threshold=0.5)
    >>> mtm.fit(X_train, y_train)
    >>> y_pred = mtm.predict(X_test)
    """

    def __init__(self, learner: BaseEstimator=None):
        self.learner = learner
        self.threshold = None
        self.cc_output = None
        self.tpr = None
        self.fpr = None
    
    @property
    def is_probabilistic(self) -> bool:
        """
        Returns whether the method is probabilistic.

        This method is used to determine whether the quantification method is probabilistic, 
        meaning it uses class-conditional probabilities to estimate class prevalences.
        
        Returns
        -------
        bool
            True, indicating that this method is probabilistic.
        """
        return True
    
    @property
    def is_multiclass(self) -> bool:
        """
        Returns whether the method is applicable to multiclass quantification.

        Threshold-based methods are typically binary classifiers, so this method 
        returns False.

        Returns
        -------
        bool
            False, indicating that this method does not support multiclass quantification.
        """
        return False
    
    def _fit_method(self, X, y):
        """
        Fits the classifier and adjusts thresholds based on true positive rate (TPR) and false positive rate (FPR).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input features for training.
        y : pd.Series or np.ndarray
            The target labels for training.

        Returns
        -------
        self : ThresholdOptimization
            The fitted quantifier object with the best threshold, TPR, and FPR.
        """
        # Get predicted labels and probabilities
        if mq.arguments["y_labels"] is not None and mq.arguments["posteriors_train"] is not None:
            y_labels = mq.arguments["y_labels"]
            probabilities = mq.arguments["posteriors_train"]
        else:
            y_labels, probabilities = get_scores(X, y, self.learner, self.cv_folds, self.learner_fitted)
        
        # Adjust thresholds and compute true and false positive rates
        thresholds, tprs, fprs = adjust_threshold(y_labels, probabilities[:, 1], self.classes)
        
        # Find the best threshold based on TPR and FPR
        self.threshold, self.tpr, self.fpr = self.best_tprfpr(thresholds, tprs, fprs)
        
        return self
    
    def _predict_method(self, X) -> dict:
        """
        Predicts class prevalences using the adjusted threshold.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input features for prediction.

        Returns
        -------
        np.ndarray
            An array of predicted prevalences for the classes.
        """
        # Get predicted probabilities for the positive class
        probabilities = self.predict_learner(X)[:, 1]
        
        # Compute the classification count output based on the threshold
        self.cc_output = len(probabilities[probabilities >= self.threshold]) / len(probabilities)
        
        # Calculate prevalence, ensuring it is within [0, 1]
        if self.tpr - self.fpr == 0:
            prevalence = self.cc_output
        else:
            # Equation of threshold methods to compute prevalence
            prevalence = np.clip((self.cc_output - self.fpr) / (self.tpr - self.fpr), 0, 1)
        
        prevalences = [1 - prevalence, prevalence]

        return np.asarray(prevalences)
    
    @abstractmethod
    def best_tprfpr(self, thresholds: np.ndarray, tpr: np.ndarray, fpr: np.ndarray) -> float:
        """
        Abstract method for determining the best TPR (True Positive Rate) and FPR (False Positive Rate) 
        to use in the equation for threshold optimization.

        This method needs to be implemented by subclasses to define how the best threshold 
        is chosen based on TPR and FPR.

        Parameters
        ----------
        thresholds : np.ndarray
            An array of threshold values.
        tpr : np.ndarray
            An array of true positive rates corresponding to the thresholds.
        fpr : np.ndarray
            An array of false positive rates corresponding to the thresholds.

        Returns
        -------
        float
            The best threshold value determined based on the true positive and false positive rates.
        """
        ...






class ACC(ThresholdOptimization):
    """
    Adjusted Classify and Count (ACC). This method is a base approach for threshold-based 
    quantification methods.

    As described in the ThresholdOptimization base class, this method estimates the true 
    positive rate (TPR) and false positive rate (FPR) from the training data. It then uses 
    these values to adjust the output of the Classify and Count (CC) method, making the 
    quantification process more accurate and stable.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier to be used for quantification.
    threshold : float, optional
        The decision threshold for classifying instances. Default is 0.5.
    
    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier.
    threshold : float
        The decision threshold used to classify instances as positive or negative. Default is 0.5.

    See Also
    --------
    ThresholdOptimization : Base class for threshold-based quantification methods.
    CC : Classify and Count quantification method.
    
    References
    ----------
    FORMAN, George. Quantifying counts and costs via classification. Data Mining and Knowledge Discovery, v. 17, p. 164-206, 2008. Available at: https://link.springer.com/article/10.1007/s10618-008-0097-y
    
    Examples
    --------
    >>> from mlquantify.methods.aggregative import ACC
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> acc = ACC(learner=SVC(probability=True), threshold=0.5)
    >>> acc.fit(X_train, y_train)
    >>> y_pred = acc.predict(X_test)
    >>> y_pred
    {0: 0.3968506555196656, 1: 0.6031493444803344}
    >>> get_real_prev(y_test)
    {0: 0.3991228070175439, 1: 0.6008771929824561}
    """

    def __init__(self, learner: BaseEstimator=None, threshold: float = 0.5):
        super().__init__(learner)
        self.threshold = threshold

    def best_tprfpr(self, thresholds: np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        """
        Determines the true positive rate (TPR) and false positive rate (FPR) for the specified threshold.

        This method identifies the TPR and FPR corresponding to the threshold provided 
        during initialization. It assumes that the `thresholds`, `tprs`, and `fprs` arrays 
        are aligned, meaning the `i-th` element of each array corresponds to the same threshold.

        Parameters
        ----------
        thresholds : np.ndarray
            An array of threshold values.
        tprs : np.ndarray
            An array of true positive rates corresponding to the thresholds.
        fprs : np.ndarray
            An array of false positive rates corresponding to the thresholds.

        Returns
        -------
        tuple
            A tuple containing the threshold, the true positive rate (TPR), and the false 
            positive rate (FPR) for the specified threshold.

        Raises
        ------
        IndexError
            If the specified threshold is not found in the `thresholds` array.
        """
        # Get the TPR and FPR where the threshold matches the specified value
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)

    
    
    
    
    
    
    

class MAX(ThresholdOptimization):
    """
    Threshold MAX. This quantification method selects the threshold that maximizes 
    the absolute difference between the true positive rate (TPR) and false positive 
    rate (FPR). This threshold is then used in the denominator of the equation for 
    adjusted prevalence estimation.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier to be used for quantification.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier.
        
    See Also
    --------
    ThresholdOptimization : Base class for threshold-based quantification methods.
    ACC : Adjusted Classify and Count quantification method.
    CC : Classify and Count quantification method.
    
    References
    ----------
    FORMAN, George. Counting positives accurately despite inaccurate classification. In: European conference on machine learning. Berlin, Heidelberg: Springer Berlin Heidelberg, 2005. p. 564-575. Available at: https://link.springer.com/chapter/10.1007/11564096_56
    
    Examples
    --------
    >>> from mlquantify.methods.aggregative import MAX
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> maxq = MAX(learner=SVC(probability=True))
    >>> maxq.fit(X_train, y_train)
    >>> y_pred = maxq.predict(X_test)
    >>> y_pred
    {0: 0.3920664352842359, 1: 0.6079335647157641}
    >>> get_real_prev(y_test)
    {0: 0.3991228070175439, 1: 0.6008771929824561}
    """

    def __init__(self, learner: BaseEstimator=None):
        super().__init__(learner)

    def best_tprfpr(self, thresholds: np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        """
        Determines the optimal threshold by maximizing the absolute difference between 
        the true positive rate (TPR) and the false positive rate (FPR).

        This method identifies the index where `|TPR - FPR|` is maximized and retrieves 
        the corresponding threshold, TPR, and FPR.

        Parameters
        ----------
        thresholds : np.ndarray
            An array of threshold values.
        tprs : np.ndarray
            An array of true positive rates corresponding to the thresholds.
        fprs : np.ndarray
            An array of false positive rates corresponding to the thresholds.

        Returns
        -------
        tuple
            A tuple containing:
            - The threshold that maximizes `|TPR - FPR|`.
            - The true positive rate (TPR) at the selected threshold.
            - The false positive rate (FPR) at the selected threshold.

        Raises
        ------
        ValueError
            If `thresholds`, `tprs`, or `fprs` are empty or have mismatched lengths.
        """
        max_index = np.argmax(np.abs(tprs - fprs))
        
        # Retrieve the corresponding threshold, TPR, and FPR
        threshold = thresholds[max_index]
        tpr = tprs[max_index]
        fpr = fprs[max_index]
        return (threshold, tpr, fpr)

    
    






class MS(ThresholdOptimization):
    """
    Median Sweep (MS). This quantification method uses an ensemble 
    of threshold-based methods, taking the median values of the 
    true positive rate (TPR) and false positive rate (FPR) across 
    all thresholds to compute adjusted prevalences.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier to be used for quantification.
    threshold : float, optional
        The default threshold value to use for the quantification method. Default is 0.5.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier.
    threshold : float
        The default threshold to use for the quantification method, typically 0.5.

    See Also
    --------
    ThresholdOptimization : Base class for threshold-based quantification methods.
    ACC : Adjusted Classify and Count quantification method.
    MAX : Threshold MAX quantification method.
    CC : Classify and Count quantification method.
    
    References
    ----------
    FORMAN, George. Quantifying counts and costs via classification. Data Mining and Knowledge Discovery, v. 17, p. 164-206, 2008. Available at: https://link.springer.com/article/10.1007/s10618-008-0097-y
    
    Examples
    --------
    >>> from mlquantify.methods.aggregative import MS
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> ms = MS(learner=SVC(probability=True))
    >>> ms.fit(X_train, y_train)
    >>> y_pred = ms.predict(X_test)
    >>> y_pred
    {0: 0.41287676595138967, 1: 0.5871232340486103}
    >>> get_real_prev(y_test)
    {0: 0.3991228070175439, 1: 0.6008771929824561}
    """

    def __init__(self, learner: BaseEstimator=None, threshold: float = 0.5):
        super().__init__(learner)
        self.threshold = threshold

    def best_tprfpr(self, thresholds: np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        """
        Determines the optimal TPR and FPR by taking the median of 
        all TPR and FPR values across the given thresholds.

        This method computes the median values of TPR and FPR to 
        mitigate the influence of outliers and variability in the 
        performance metrics.

        Parameters
        ----------
        thresholds : np.ndarray
            An array of threshold values.
        tprs : np.ndarray
            An array of true positive rates corresponding to the thresholds.
        fprs : np.ndarray
            An array of false positive rates corresponding to the thresholds.

        Returns
        -------
        tuple
            A tuple containing:
            - The default threshold value (float).
            - The median true positive rate (float).
            - The median false positive rate (float).

        Raises
        ------
        ValueError
            If `thresholds`, `tprs`, or `fprs` are empty or have mismatched lengths.
        """
        # Compute median TPR and FPR
        tpr = np.median(tprs)
        fpr = np.median(fprs)

        return (self.threshold, tpr, fpr)

    






class MS2(ThresholdOptimization):
    """
    Median Sweep 2 (MS2). This method is an extension of the 
    Median Sweep strategy, but it focuses only on cases where 
    the difference between the true positive rate (TPR) and the 
    false positive rate (FPR) exceeds a threshold (0.25). The 
    method computes the median values of TPR, FPR, and thresholds 
    for these selected cases.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier to be used for quantification.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier.

    References
    ----------
    FORMAN, George. Quantifying counts and costs via classification. Data Mining and Knowledge Discovery, v. 17, p. 164-206, 2008. Available at: https://link.springer.com/article/10.1007/s10618-008-0097-y

    See Also
    --------
    ThresholdOptimization : Base class for threshold-based quantification methods.
    ACC : Adjusted Classify and Count quantification method.
    MS : Median Sweep quantification method.
    CC : Classify and Count quantification method.
    
    Examples
    --------
    >>> from mlquantify.methods.aggregative import MS2
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> ms2 = MS2(learner=SVC(probability=True))
    >>> ms2.fit(X_train, y_train)
    >>> y_pred = ms2.predict(X_test)
    >>> y_pred
    {0: 0.41287676595138967, 1: 0.5871232340486103}
    >>> get_real_prev(y_test)
    {0: 0.3991228070175439, 1: 0.6008771929824561}
    """

    def __init__(self, learner: BaseEstimator=None):
        super().__init__(learner)

    def best_tprfpr(self, thresholds: np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        """
        Determines the optimal threshold, TPR, and FPR by focusing only on 
        cases where the absolute difference between TPR and FPR is greater 
        than 0.25. For these cases, the method computes the median values.

        Parameters
        ----------
        thresholds : np.ndarray
            An array of threshold values.
        tprs : np.ndarray
            An array of true positive rates corresponding to the thresholds.
        fprs : np.ndarray
            An array of false positive rates corresponding to the thresholds.

        Returns
        -------
        tuple
            A tuple containing:
            - The median threshold value for cases meeting the condition (float).
            - The median true positive rate for cases meeting the condition (float).
            - The median false positive rate for cases meeting the condition (float).
        
        Raises
        ------
        ValueError
            If no cases satisfy the condition `|TPR - FPR| > 0.25`.
        Warning
            If all TPR or FPR values are zero.
        """
        # Check if all TPR or FPR values are zero
        if np.all(tprs == 0) or np.all(fprs == 0):
            warnings.warn("All TPR or FPR values are zero.")
        
        # Identify indices where the condition is satisfied
        indices = np.where(np.abs(tprs - fprs) > 0.25)[0]
        if len(indices) == 0:
            raise ValueError("No cases meet the condition |TPR - FPR| > 0.25.")

        # Compute medians for the selected cases
        threshold = np.median(thresholds[indices])
        tpr = np.median(tprs[indices])
        fpr = np.median(fprs[indices])

        return (threshold, tpr, fpr)


class PACC(ThresholdOptimization):
    """
    Probabilistic Adjusted Classify and Count (PACC). 
    This method extends the Adjusted Classify and Count (AC) approach 
    by leveraging the average class-conditional confidences obtained 
    from a probabilistic classifier instead of relying solely on true 
    positive and false positive rates.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier to be used for quantification.
    threshold : float, optional
        The decision threshold for classification. Default is 0.5.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier.
    threshold : float
        Decision threshold for classification. Default is 0.5.
    tpr : float
        True positive rate computed during the fitting process.
    fpr : float
        False positive rate computed during the fitting process.
        
    See Also
    --------
    ThresholdOptimization : Base class for threshold-based quantification methods.
    ACC : Adjusted Classify and Count quantification method.
    CC : Classify and Count quantification method.
    
    References
    ----------
    A. Bella, C. Ferri, J. Hernández-Orallo and M. J. Ramírez-Quintana, "Quantification via Probability Estimators," 2010 IEEE International Conference on Data Mining, Sydney, NSW, Australia, 2010, pp. 737-742, doi: 10.1109/ICDM.2010.75. Available at: https://ieeexplore.ieee.org/abstract/document/5694031
    
    Examples
    --------
    >>> from mlquantify.methods.aggregative import PACC
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> pacc = PACC(learner=SVC(probability=True))
    >>> pacc.fit(X_train, y_train)
    >>> y_pred = pacc.predict(X_test)
    >>> y_pred
    {0: 0.4664886119311328, 1: 0.5335113880688672}
    >>> get_real_prev(y_test)
    {0: 0.3991228070175439, 1: 0.6008771929824561}
    """

    def __init__(self, learner: BaseEstimator=None, threshold: float = 0.5):
        super().__init__(learner)
        self.threshold = threshold

    def _predict_method(self, X):
        """
        Predicts the class prevalence using the mean class-conditional 
        probabilities from a probabilistic classifier.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input data for prediction.

        Returns
        -------
        dict
            A dictionary with class labels as keys and their respective 
            prevalence estimates as values.

        Notes
        -----
        The prevalence is adjusted using the formula:
            prevalence = |mean_score - FPR| / (TPR - FPR), 
        where mean_score is the average probability for the positive class.

        Raises
        ------
        ZeroDivisionError
            If `TPR - FPR` equals zero, indicating that the classifier's 
            performance does not vary across the threshold range.
        """
        prevalences = {}

        # Calculate probabilities for the positive class
        probabilities = self.predict_learner(X)[:, 1]

        # Compute the mean score for the positive class
        mean_scores = np.mean(probabilities)

        # Adjust prevalence based on TPR and FPR
        if self.tpr - self.fpr == 0:
            prevalence = mean_scores
        else:
            prevalence = np.clip(abs(mean_scores - self.fpr) / (self.tpr - self.fpr), 0, 1)

        # Map the computed prevalence to the class labels
        prevalences[self.classes[0]] = 1 - prevalence
        prevalences[self.classes[1]] = prevalence

        return prevalences

    def best_tprfpr(self, thresholds: np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        """
        Finds the true positive rate (TPR) and false positive rate (FPR) 
        corresponding to the specified decision threshold.

        Parameters
        ----------
        thresholds : np.ndarray
            An array of threshold values.
        tprs : np.ndarray
            An array of true positive rates corresponding to the thresholds.
        fprs : np.ndarray
            An array of false positive rates corresponding to the thresholds.

        Returns
        -------
        tuple
            A tuple containing the specified threshold, TPR, and FPR.

        Raises
        ------
        IndexError
            If the specified threshold is not found in the `thresholds` array.
        """
        # Locate TPR and FPR for the specified threshold
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)

    
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)
    
    
    





class T50(ThresholdOptimization):
    """
    Threshold 50 (T50). This method adjusts the decision threshold 
    to the point where the true positive rate (TPR) is approximately 
    equal to 0.5. This approach is particularly useful for balancing 
    sensitivity and specificity in binary classification tasks.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier to be used for quantification.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier.
    threshold : float
        Decision threshold determined during training.
    tpr : float
        True positive rate corresponding to the selected threshold.
    fpr : float
        False positive rate corresponding to the selected threshold.

    See Also
    --------
    ThresholdOptimization : Base class for threshold-based quantification methods.
    ACC : Adjusted Classify and Count quantification method.
    CC : Classify and Count quantification method.
    
    References
    ----------
    FORMAN, George. Quantifying counts and costs via classification. Data Mining and Knowledge Discovery, v. 17, p. 164-206, 2008. Available at: https://link.springer.com/article/10.1007/s10618-008-0097-y
    
    Examples
    --------
    >>> from mlquantify.methods.aggregative import T50
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> t50 = T50(learner=SVC(probability=True))
    >>> t50.fit(X_train, y_train)
    >>> y_pred = t50.predict(X_test)
    >>> y_pred
    {0: 0.49563196626070505, 1: 0.504368033739295}
    >>> get_real_prev(y_test)
    {0: 0.3991228070175439, 1: 0.6008771929824561}
    """

    def __init__(self, learner: BaseEstimator=None):
        super().__init__(learner)

    def best_tprfpr(self, thresholds: np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        """
        Determines the threshold, true positive rate (TPR), and false positive 
        rate (FPR) where TPR is closest to 0.5.

        Parameters
        ----------
        thresholds : np.ndarray
            An array of threshold values.
        tprs : np.ndarray
            An array of true positive rates corresponding to the thresholds.
        fprs : np.ndarray
            An array of false positive rates corresponding to the thresholds.

        Returns
        -------
        tuple
            A tuple containing the selected threshold, TPR, and FPR.

        Notes
        -----
        - The method identifies the index where the absolute difference 
          between TPR and 0.5 is minimized.
        - This ensures that the selected threshold represents a balance 
          point in the ROC space.

        Raises
        ------
        ValueError
            If the arrays `thresholds`, `tprs`, or `fprs` are empty or 
            misaligned in length.
        """
        # Find the index where TPR is closest to 0.5
        min_index = np.argmin(np.abs(tprs - 0.5))

        # Retrieve the corresponding threshold, TPR, and FPR
        threshold = thresholds[min_index]
        tpr = tprs[min_index]
        fpr = fprs[min_index]

        return (threshold, tpr, fpr)

    
    
    
    





class X_method(ThresholdOptimization):
    """
    Threshold X. This method identifies the decision threshold where the 
    false positive rate (FPR) is approximately equal to 1 - true positive rate (TPR). 
    This criterion is useful for identifying thresholds that align with a balance 
    point on the ROC curve.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier to be used for quantification.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier.
    threshold : float
        Decision threshold determined during training.
    tpr : float
        True positive rate corresponding to the selected threshold.
    fpr : float
        False positive rate corresponding to the selected threshold.

    See Also
    --------
    ThresholdOptimization : Base class for threshold-based quantification methods.
    ACC : Adjusted Classify and Count quantification method.
    CC : Classify and Count quantification method.
    
    References
    ----------
    FORMAN, George. Quantifying counts and costs via classification. Data Mining and Knowledge Discovery, v. 17, p. 164-206, 2008. Available at: https://link.springer.com/article/10.1007/s10618-008-0097-y
    
    Examples
    --------
    >>> from mlquantify.methods.aggregative import X_method
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> x_method = X_method(learner=SVC(probability=True))
    >>> x_method.fit(X_train, y_train)
    >>> y_pred = x_method.predict(X_test)
    >>> y_pred
    {0: 0.40523495782808205, 1: 0.594765042171918}
    >>> get_real_prev(y_test)
    {0: 0.3991228070175439, 1: 0.6008771929824561}
    """

    def __init__(self, learner: BaseEstimator=None):
        super().__init__(learner)

    def best_tprfpr(self, thresholds: np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        """
        Determines the threshold, true positive rate (TPR), and false positive 
        rate (FPR) where FPR is closest to 1 - TPR.

        Parameters
        ----------
        thresholds : np.ndarray
            An array of threshold values.
        tprs : np.ndarray
            An array of true positive rates corresponding to the thresholds.
        fprs : np.ndarray
            An array of false positive rates corresponding to the thresholds.

        Returns
        -------
        tuple
            A tuple containing the selected threshold, TPR, and FPR.

        Notes
        -----
        - The method identifies the index where the absolute difference 
          between FPR and 1 - TPR is minimized.
        - This ensures that the selected threshold corresponds to a balance 
          point based on the given criterion.

        Raises
        ------
        ValueError
            If the arrays `thresholds`, `tprs`, or `fprs` are empty or 
            misaligned in length.
        """
        # Find the index where FPR is closest to 1 - TPR
        min_index = np.argmin(np.abs(1 - (tprs + fprs)))

        # Retrieve the corresponding threshold, TPR, and FPR
        threshold = thresholds[min_index]
        tpr = tprs[min_index]
        fpr = fprs[min_index]

        return (threshold, tpr, fpr)
