import numpy as np     
from abc import abstractmethod
from scipy.optimize import minimize


from mlquantify.adjust_counting._base import BaseAdjustCount
from mlquantify.adjust_counting._counting import CC, PCC
from mlquantify.base_aggregative import (
    CrispLearnerQMixin,
    SoftLearnerQMixin,
    uses_soft_predictions,
)
from mlquantify.adjust_counting._utils import evaluate_thresholds
from mlquantify.utils._constraints import Interval, Options


class ThresholdAdjustment(BaseAdjustCount):

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

        # get tpr and fpr values, along with thresholds
        thresholds, tprs, fprs = evaluate_thresholds(train_y_values, train_y_scores)

        # get best threshold based on some criterion (method's specific)
        threshold, tpr, fpr = self._get_best_threshold(thresholds, tprs, fprs)

        # get predictions for CC
        cc_predictions = CC().aggregate(predictions[predictions >= threshold])

        # Compute equation of threshold methods to compute prevalence
        if tpr - fpr == 0:
            prevalence = cc_predictions
        else:
            prevalence = (cc_predictions - fpr) / (tpr - fpr)
        
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