import numpy as np

from mlquantify.base_aggregative import (
    SoftLearnerQMixin,
    CrispLearnerQMixin
)

from mlquantify.adjust_counting._base import BaseCount
from mlquantify.utils._validation import validate_predictions, validate_prevalences, check_classes_attribute
from mlquantify.utils._constraints import Interval
        


class CC(CrispLearnerQMixin, BaseCount):
    """Classify and Count (CC) quantifier.
    
    Implements the Classify and Count method for quantification as described in:
    [1] Forman, G. (2005). *Counting Positives Accurately Despite Inaccurate Classification.*
        ECML, pp. 564-575.
    [2] Forman, G. (2008). *Quantifying Counts and Costs via Classification.*
        Data Mining and Knowledge Discovery, 17(2), 164-206.
        
        
    Parameters
    ----------
    learner : estimator, optional
        A supervised learning estimator with fit and predict methods.
        If None, it is expected that will be used the aggregate method directly.
    threshold : float, default=0.5
        Decision threshold for converting predicted probabilities into class labels.
        Must be in the interval [0.0, 1.0].
        
        
    Attributes
    ----------
    learner : estimator
        Underlying classification model.
    classes : ndarray of shape (n_classes,)
        Unique class labels observed during training.
        
        
    Examples
    --------
    >>> from mlquantify.adjust_counting import CC
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> q = CC(learner=LogisticRegression())
    >>> q.fit(X, y)
    >>> q.predict(X)
    {0: 0.47, 1: 0.53}
    >> q2 = CC()
    >>> predictions = np.random.rand(200)
    >>> q2.aggregate(predictions)
    {0: 0.51, 1: 0.49}
    """
    
    _parameters_constraints = {
        "threshold": [
            Interval(0.0, 1.0),
            Interval(0, 1, discrete=True),
        ],
    }

    def __init__(self, learner=None, threshold=0.5):
        super().__init__(learner=learner)
        self.threshold = threshold

    def aggregate(self, predictions):
        predictions = validate_predictions(self, predictions)
        
        self.classes_ = check_classes_attribute(self, np.unique(predictions))
        class_counts = np.array([np.count_nonzero(predictions == _class) for _class in self.classes_])
        prevalences = class_counts / len(predictions)
        
        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences


class PCC(SoftLearnerQMixin, BaseCount):
    """Probabilistic Classify and Count (PCC) quantifier.
    
    Implements the Probabilistic Classify and Count method for quantification as described in:
    [1] Forman, G. (2005). *Counting Positives Accurately Despite Inaccurate Classification.*
        ECML, pp. 564-575.
    [2] Forman, G. (2008). *Quantifying Counts and Costs via Classification.*
        Data Mining and Knowledge Discovery, 17(2), 164-206.
        
        
    Parameters
    ----------
    learner : estimator, optional
        A supervised learning estimator with fit and predict_proba methods.
        If None, it is expected that will be used the aggregate method directly.
        
        
    Attributes
    ----------
    learner : estimator
        Underlying classification model.
    classes : ndarray of shape (n_classes,)
        Unique class labels observed during training.
        
        
    Examples
    --------
    >>> from mlquantify.adjust_counting import PCC
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> q = PCC(learner=LogisticRegression())
    >>> q.fit(X, y)
    >>> q.predict(X)
    {0: 0.48, 1: 0.52}
    >>> q2 = PCC()
    >>> predictions = np.random.rand(200, 2)
    >>> q2.aggregate(predictions)
    {0: 0.50, 1: 0.50}
    """

    def __init__(self, learner=None):
        super().__init__(learner=learner)

    def aggregate(self, predictions):
        predictions = validate_predictions(self, predictions)
        
        # Handle categorical predictions (1D array with class labels)
        if predictions.ndim == 1 and not np.issubdtype(predictions.dtype, (np.floating, np.integer)):
            self.classes_ = check_classes_attribute(self, np.unique(predictions))
            class_counts = np.array([np.count_nonzero(predictions == _class) for _class in self.classes_])
            prevalences = class_counts / len(predictions)
        else:
            # Handle probability predictions (2D array or 1D probabilities)
            if predictions.ndim == 2:
                self.classes_ = check_classes_attribute(self, np.arange(predictions.shape[1]))
            else:
                self.classes_ = check_classes_attribute(self, np.arange(2))
            prevalences = np.mean(predictions, axis=0) if predictions.ndim == 2 else predictions.mean()
            if predictions.ndim == 1:
                prevalences = np.array([1-prevalences, prevalences])
        
        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences