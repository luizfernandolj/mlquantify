import numpy as np
from mlquantify.base_aggregative import SoftLearnerQMixin
from mlquantify.likelihood._base import BaseIterativeLikelihood
from mlquantify.metrics._slq import MAE
from mlquantify.utils._constraints import (
    Interval,
    CallableConstraint,
    Options
)

class EMQ(SoftLearnerQMixin, BaseIterativeLikelihood):

    _parameter_constraints = {
        "tol": [Interval(0, None, inclusive_left=False)],
        "max_iter": [Interval(1, None, inclusive_left=True)],
        "calib_function": [
            Options(["bcts", "ts", "vs", "nbvs", None]),
        ],
        "criteria": [CallableConstraint()],
    }

    def __init__(self, 
                 learner=None, 
                 tol=1e-4, 
                 max_iter=100, 
                 calib_function=None,
                 criteria=MAE):
        super().__init__(learner=learner, tol=tol, max_iter=max_iter)
        self.calib_function = calib_function
        self.criteria = criteria
        
    def _iterate(self, predictions, priors):
        calibrated_predictions = self._apply_calibration(predictions)
        prevalences, _ = self.EM(
            posteriors=calibrated_predictions,
            priors=priors,
            tolerance=self.tol,
            max_iter=self.max_iter,
            criteria=self.criteria
        )
        return prevalences


    @classmethod
    def EM(cls, posteriors, priors, tolerance=1e-6, max_iter=100, criteria=MAE):

        Px = np.array(posteriors, dtype=np.float64)
        Ptr = np.array(priors, dtype=np.float64)

        if np.prod(Ptr) == 0:
            Ptr += tolerance
            Ptr /= Ptr.sum()

        qs = np.copy(Ptr)
        s, converged = 0, False
        qs_prev_ = None

        while not converged and s < max_iter:
            # E-step:
            ps_unnormalized = (qs / Ptr) * Px
            ps = ps_unnormalized / ps_unnormalized.sum(axis=1, keepdims=True)

            # M-step:
            qs = ps.mean(axis=0)

            if qs_prev_ is not None and criteria(qs_prev_, qs) < tolerance and s > 10:
                converged = True

            qs_prev_ = qs
            s += 1

        if not converged:
            print('[warning] the method has reached the maximum number of iterations; it might have not converged')

        return qs, ps


    def _apply_calibration(self, predictions):
        if self.calib_function is None:
            return predictions

        if isinstance(self.calib_function, str):
            method = self.calib_function.lower()
            if method == "ts":
                return self._temperature_scaling(predictions)
            elif method == "bcts":
                return self._bias_corrected_temperature_scaling(predictions)
            elif method == "vs":
                return self._vector_scaling(predictions)
            elif method == "nbvs":
                return self._no_bias_vector_scaling(predictions)

        elif callable(self.calib_function):
            return self.calib_function(predictions)

        raise ValueError(
            f"Invalid calib_function '{self.calib_function}'. Expected one of {{'bcts', 'ts', 'vs', 'nbvs', None, callable}}."
        )

    def _temperature_scaling(self, preds):
        T = 1.0
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        scaled = logits / T
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

    def _bias_corrected_temperature_scaling(self, preds):
        T = 1.0
        bias = np.zeros(preds.shape[1])
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        logits = logits / T + bias
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _vector_scaling(self, preds):
        W = np.ones(preds.shape[1])
        b = np.zeros(preds.shape[1])
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        scaled = logits * W + b
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

    def _no_bias_vector_scaling(self, preds):
        W = np.ones(preds.shape[1])
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        scaled = logits * W
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)



class MLPE(SoftLearnerQMixin, BaseIterativeLikelihood):
    """Maximum Likelihood Prevalence Estimation quantifier."""

    def __init__(self, learner=None):
        super().__init__(learner=learner, max_iter=1)
        
    def _iterate(self, predictions, priors):
        return priors
    

class CDE(SoftLearnerQMixin, BaseIterativeLikelihood):
    """CDE-Iterate (Class Distribution Estimation Iterate) — binary version.

    Based on Xue & Weiss (2009) as described in Esuli et al. (2023), Chapter 4.2.10.
    This implementation expects `predictions` to be posterior probabilities with
    shape (n_samples, 2) and returns estimated prevalences as a length-2 array.
    """

    _parameter_constraints = {
        "tol": [Interval(0, None, inclusive_left=False)],
        "max_iter": [Interval(1, None, inclusive_left=True)],
        "init_cfp": [Interval(0, None, inclusive_left=False)]
    }

    def __init__(self, learner=None, tol=1e-4, max_iter=100, init_cfp=1.0):
        """
        :param learner: (optional) wrapped learner object (not strictly used here;
                        CDE-Iterate original retrains a cost-sensitive classifier,
                        but here we implement a transductive thresholding variant).
        :param tol: convergence tolerance on prevalence change
        :param max_iter: maximal number of iterations
        :param init_cfp: initial false-positive cost (cFP); cFN is kept = 1
        """
        super().__init__(learner=learner, tol=tol, max_iter=max_iter)
        self.init_cfp = float(init_cfp)

    def _iterate(self, predictions, priors):
        """
        :param predictions: np.ndarray shape (n_samples, 2) of posterior probabilities
                            for classes [neg, pos] or [class0, class1]. We will assume
                            column 1 is the positive class.
        :param priors: array-like length 2 with pL(y) (training priors) OR initial priors;
                       CDE uses pL in cost update. We expect priors to be training priors.
        :return: prevalences estimate for U: array length 2
        """
        P = np.asarray(predictions, dtype=np.float64)
        Ptr = np.asarray(priors, dtype=np.float64)

        # basic checks
        if P.ndim != 2 or P.shape[1] != 2:
            raise ValueError("CDE implementation here supports binary case only: predictions shape (n,2).")

        # ensure no zeros
        eps = 1e-12
        P = np.clip(P, eps, 1.0)

        # training priors pL(+), pL(-)
        # assume Ptr order matches columns of P; if Ptr sums to 1 but order unknown, user must match.
        pL_pos = Ptr[1]
        pL_neg = Ptr[0]
        if pL_pos <= 0 or pL_neg <= 0:
            # keep them positive to avoid divisions by zero
            pL_pos = max(pL_pos, eps)
            pL_neg = max(pL_neg, eps)

        # initialize costs
        cFN = 1.0
        cFP = float(self.init_cfp)

        prev_prev_pos = None
        s = 0

        # iterate: compute threshold from costs, classify, estimate prevalences via CC,
        # update cFP via eq. (4.27), repeat
        while s < self.max_iter:
            # decision threshold tau for positive class:
            # Derivation:
            # predict positive if cost_FP * p(-|x) < cost_FN * p(+|x)
            # => predict positive if p(+|x) / p(-|x) > cost_FP / cost_FN
            # since p(+|x) / p(-|x) = p(+|x) / (1 - p(+|x)):
            # p(+|x) > cost_FP / (cost_FP + cost_FN)
            tau = cFP / (cFP + cFN)

            # hard predictions for positive class using threshold on posterior for positive (col 1)
            pos_probs = P[:, 1]
            hard_pos = (pos_probs > tau).astype(float)

            # classify-and-count prevalence estimate on U
            prev_pos = hard_pos.mean()
            prev_neg = 1.0 - prev_pos

            # update cFP according to Eq. 4.27:
            # cFP_new = (pL_pos / pL_neg) * (pU_hat(neg) / pU_hat(pos)) * cFN
            # guard against zero prev_pos / prev_neg
            prev_pos_safe = max(prev_pos, eps)
            prev_neg_safe = max(prev_neg, eps)

            cFP_new = (pL_pos / pL_neg) * (prev_neg_safe / prev_pos_safe) * cFN

            # check convergence on prevalences (absolute change)
            if prev_prev_pos is not None and abs(prev_pos - prev_prev_pos) < self.tol:
                break

            # prepare next iter
            cFP = cFP_new
            prev_prev_pos = prev_pos
            s += 1

        # if didn't converge within max_iter we keep last estimate (book warns about lack of fisher consistency)
        if s >= self.max_iter:
            # optional: warning
            # print('[warning] CDE-Iterate reached max_iter without converging')
            pass

        prevalences = np.array([prev_neg, prev_pos], dtype=np.float64)
        # ensure sums to 1 (numerical safety)
        prevalences = prevalences / prevalences.sum()

        return prevalences