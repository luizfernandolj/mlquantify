import numpy as np     
from abc import abstractmethod
from scipy.optimize import minimize
from yaml import warnings


from mlquantify.adjust_counting._base import BaseAdjustCount
from mlquantify.adjust_counting._counting import CC, PCC
from mlquantify.base_aggregative import (
    CrispLearnerQMixin,
    SoftLearnerQMixin,
    uses_soft_predictions,
)
from mlquantify.adjust_counting._utils import evaluate_thresholds
from mlquantify.utils._constraints import Interval, Options


class ThresholdAdjustment(SoftLearnerQMixin, BaseAdjustCount): # ACC, X, MAX, T50, MS, MS2

    _parameter_constraints = {
        "threshold": [
            Interval(0.0, 1.0),
            Interval(0, 1, discrete=True),
        ],
    }

    def __init__(self, learner=None, threshold=0.5):
        super().__init__(learner=learner)
        self.threshold = threshold

    def _adjust(self, predictions, train_y_scores, train_y_values):
        
        self.classes = np.unique(train_y_values) if not hasattr(self, 'classes') else self.classes
        
        positive_scores = train_y_scores[:, 1]
        
        # get tpr and fpr values, along with thresholds
        thresholds, tprs, fprs = evaluate_thresholds(train_y_values, positive_scores, self.classes)

        # get best threshold based on some criterion (method's specific)
        threshold, tpr, fpr = self._get_best_threshold(thresholds, tprs, fprs)

        # get predictions for CC
        cc_predictions = CC(threshold).aggregate(predictions)[1]

        # Compute equation of threshold methods to compute prevalence
        if tpr - fpr == 0:
            prevalence = cc_predictions
        else:
            prevalence = np.clip((cc_predictions - fpr) / (tpr - fpr), 0, 1)
        
        prevalence = np.asarray([1-prevalence, prevalence])
        
        # return prevalence
        return prevalence
    
    @abstractmethod
    def _get_best_threshold(self, thresholds, tprs, fprs):
        ...

class MatrixAdjustment(BaseAdjustCount): # FM, GAC, GPAC

    _parameter_constraints = {
        "solver": Options(["optim", "linear"]),
    }

    def __init__(self, learner=None, solver=None):
        super().__init__(learner=learner)
        self.solver = solver
    
    def _adjust(self, predictions, train_y_pred, train_y_values):
        n_class = len(np.unique(train_y_values))
        self.CM = np.zeros((n_class, n_class))
        self.classes = np.unique(train_y_values) if not hasattr(self, 'classes') else self.classes

        if self.solver == 'optim':
            priors = CC().aggregate(train_y_pred)
            self.CM = self._compute_confusion_matrix(train_y_pred, train_y_values, priors)
            
            prevs_estim = self._get_estimations(predictions > priors)
            prevalence = self._solve_optimization(prevs_estim, priors)
        else:
            self.CM = self._compute_confusion_matrix(train_y_pred)
            prevs_estim = self._get_estimations(predictions)
            prevalence = self._solve_linear(prevs_estim)
        

        return prevalence

    def _solve_linear(self, prevs_estim):
        try:
            adjusted = np.linalg.solve(self.CM, prevs_estim)
            adjusted = np.clip(adjusted, 0, 1)
            adjusted /= adjusted.sum()
        except np.linalg.LinAlgError:
            adjusted = prevs_estim
        return adjusted

    def _solve_optimization(self, prevs_estim, priors):
        def objective(prevs_pred):
            return np.linalg.norm(self.CM @ prevs_pred - prevs_estim)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        bounds = [(0, 1)] * self.CM.shape[1]
        initial_guess = np.full(self.CM.shape[1], 1 / self.CM.shape[1])

        result = minimize(objective, initial_guess, constraints=constraints, bounds=bounds)
        if result.success:
            return result.x
        else:
            print("Optimization did not converge")
            return priors


    def _get_estimations(self, predictions):
        if uses_soft_predictions(self):
            prevalences = PCC().aggregate(predictions)
        else:
            prevalences = CC().aggregate(predictions)
        return prevalences

    @abstractmethod
    def _compute_confusion_matrix(self, predictions, *args):
        ...
        
        


class FM(SoftLearnerQMixin, MatrixAdjustment):
    """Forman's Matrix Adjustment method."""
    
    def __init__(self, learner=None):
        super().__init__(learner=learner, solver='optim')
    
    def _compute_confusion_matrix(self, posteriors, y_true, priors):

        for _class in self.classes:
            indices = (y_true == _class)
            self.CM[:, _class] = self._get_estimations(posteriors[indices] > priors)
        
        return self.CM
    
class GAC(CrispLearnerQMixin, MatrixAdjustment):
    """Gonzalez-Castro et al.'s Matrix Adjustment method."""
    
    def __init__(self, learner=None):
        super().__init__(learner=learner, solver='linear')
    
    def _compute_confusion_matrix(self, predictions):
        prev_estim = self._get_estimations(predictions)
        for i, _ in enumerate(self.classes):
            if prev_estim[i] == 0:
                self.CM[i, i] = 1
            else:
                self.CM[:, i] /= prev_estim[i]
        return self.CM
    
    
class GPAC(SoftLearnerQMixin, MatrixAdjustment):
    """Gonzalez-Castro et al.'s Probabilistic Matrix Adjustment method."""
    
    def __init__(self, learner=None):
        super().__init__(learner=learner, solver='linear')
    
    def _compute_confusion_matrix(self, posteriors):
        prev_estim = self._get_estimations(posteriors)
        for i, _ in enumerate(self.classes):
            if prev_estim[i] == 0:
                self.CM[i, i] = 1
            else:
                self.CM[:, i] /= prev_estim[i]
        return self.CM
    
    
    
class ACC(ThresholdAdjustment):
    """Adjusted Count method."""
    
    _parameter_constraints = {
        "threshold": [
            Interval(0.0, 1.0),
            Interval(0, 1, discrete=True),
        ]
    }
    
    def __init__(self, learner=None, threshold=0.5):
        super().__init__(learner=learner)
        self.threshold = threshold
        
    def _get_best_threshold(self, thresholds, tprs, fprs):
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)


class X_method(ThresholdAdjustment):
    """X method for prevalence adjustment."""
    
    def __init__(self, learner=None):
        super().__init__(learner=learner)
    
    def _get_best_threshold(self, thresholds, tprs, fprs):
        # X method: choose threshold that maximizes TPR - FPR
        min_index = np.argmin(np.abs(1 - (tprs + fprs)))
        return thresholds[min_index], tprs[min_index], fprs[min_index]

class MAX(ThresholdAdjustment):
    """Maximum method for prevalence adjustment."""
    
    def __init__(self, learner=None):
        super().__init__(learner=learner)
    
    def _get_best_threshold(self, thresholds, tprs, fprs):
        # MAX method: choose threshold that maximizes TPR
        max_index = np.argmax(np.abs(tprs - fprs))
        return thresholds[max_index], tprs[max_index], fprs[max_index]
    
class T50(ThresholdAdjustment):
    """T50 method for prevalence adjustment."""
    
    def __init__(self, learner=None):
        super().__init__(learner=learner)
    
    def _get_best_threshold(self, thresholds, tprs, fprs):
        # T50 method: choose threshold where TPR is closest to 0.5
        tpr_diffs = np.abs(tprs - 0.5)
        best_index = np.argmin(tpr_diffs)
        return thresholds[best_index], tprs[best_index], fprs[best_index]
    

class MS(ThresholdAdjustment):
    """Minimum Squared method for prevalence adjustment."""
    
    def __init__(self, learner=None):
        super().__init__(learner=learner)
    
    def _get_best_threshold(self, thresholds, tprs, fprs):
        pass
    
    
    def _adjust(self, predictions, train_y_scores, train_y_values):
        
        self.classes = np.unique(train_y_values) if not hasattr(self, 'classes') else self.classes
        
        positive_scores = train_y_scores[:, 1]
        
        # get tpr and fpr values, along with thresholds
        thresholds, tprs, fprs = evaluate_thresholds(train_y_values, positive_scores, self.classes)
        
        prevs = []

        for thr, tpr, fpr in zip(thresholds, tprs, fprs):
            # get predictions for CC
            cc_predictions = CC(thr).aggregate(predictions)
            
            # Compute equation of threshold methods to compute prevalence
            if tpr - fpr == 0:
                prevalence = cc_predictions
            else:
                prevalence = (cc_predictions - fpr) / (tpr - fpr)
            prevs.append(prevalence)

        prevalence = np.median(prevs)
        
        prevalence = np.asarray([1-prevalence, prevalence])
        
        # return prevalence
        return prevalence
    

class MS2(MS):
    """Minimum Squared 2 method for prevalence adjustment."""
    
    def _get_best_threshold(self, thresholds, tprs, fprs):
        # Check if all TPR or FPR values are zero
        if np.all(tprs == 0) or np.all(fprs == 0):
            warnings.warn("All TPR or FPR values are zero.")
        
        # Identify indices where the condition is satisfied
        indices = np.where(np.abs(tprs - fprs) > 0.25)[0]
        if len(indices) == 0:
            warnings.warn("No cases satisfy the condition |TPR - FPR| > 0.25.")
            indices = np.where(np.abs(tprs - fprs) >= 0)[0]
            
        return thresholds[indices], tprs[indices], fprs[indices]
