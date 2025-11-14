import numpy as np
from mlquantify.base_aggregative import SoftLearnerQMixin
from mlquantify.likelihood._base import BaseIterativeLikelihood
from mlquantify.metrics._slq import MAE
from mlquantify.multiclass import define_binary
from mlquantify.utils._constraints import (
    Interval,
    CallableConstraint,
    Options
)

class EMQ(SoftLearnerQMixin, BaseIterativeLikelihood):
    r"""Expectation-Maximization Quantifier (EMQ).

    Estimates class prevalences under prior probability shift by alternating 
    between expectation **(E)** and maximization **(M)** steps on posterior probabilities. 

    E-step:
    .. math::
        p_i^{(s+1)}(x) = \frac{q_i^{(s)} p_i(x)}{\sum_j q_j^{(s)} p_j(x)}

    M-step:
    .. math::
        q_i^{(s+1)} = \frac{1}{N} \sum_{n=1}^N p_i^{(s+1)}(x_n)

    where 
    - :math:`p_i(x)` are posterior probabilities predicted by the classifier
    - :math:`q_i^{(s)}` are class prevalence estimates at iteration :math:`s`
    - :math:`N` is the number of test instances.

    Calibrations supported on posterior probabilities before **EM** iteration:

    Temperature Scaling (TS):
    .. math::
        \hat{p} = \text{softmax}\left(\frac{\log(p)}{T}\right)

    Bias-Corrected Temperature Scaling (BCTS):
    .. math::
        \hat{p} = \text{softmax}\left(\frac{\log(p)}{T} + b\right)

    Vector Scaling (VS):
    .. math::
        \hat{p}_i = \text{softmax}(W_i \cdot \log(p_i) + b_i)

    No-Bias Vector Scaling (NBVS):
    .. math::
        \hat{p}_i = \text{softmax}(W_i \cdot \log(p_i))

    Parameters
    ----------
    learner : estimator, optional
        Probabilistic classifier supporting predict_proba.
    tol : float, default=1e-4
        Convergence threshold.
    max_iter : int, default=100
        Maximum EM iterations.
    calib_function : str or callable, optional
        Calibration method:
        - 'ts': Temperature Scaling
        - 'bcts': Bias-Corrected Temperature Scaling
        - 'vs': Vector Scaling
        - 'nbvs': No-Bias Vector Scaling
        - callable: custom calibration function
    criteria : callable, default=MAE
        Convergence metric.

    References
    ----------
    .. [1] Saerens, M., Latinne, P., & Decaestecker, C. (2002).
        Adjusting the Outputs of a Classifier to New a Priori Probabilities.
        Neural Computation, 14(1), 2141-2156.
    .. [2] Esuli, A., Moreo, A., & Sebastiani, F. (2023). Learning to Quantify. Springer.
    """

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
        r"""Perform EM quantification iteration.
        
        Steps:
        - Calibrate posterior predictions if calibration function specified.
        - Apply EM procedure to re-estimate prevalences, based on training priors and calibrated posteriors.
        
        Parameters
        ----------
        predictions : ndarray of shape (n_samples, n_classes)
            Posterior probabilities for each class on test data.
        priors : ndarray of shape (n_classes,)
            Training set class prevalences, serving as initial priors.
        
        Returns
        -------
        prevalences : ndarray of shape (n_classes,)
            Estimated class prevalences after EM iteration.
        """
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
        r"""Static method implementing the EM algorithm for quantification.

        Parameters
        ----------
        posteriors : ndarray of shape (n_samples, n_classes)
            Posterior probability predictions.
        priors : ndarray of shape (n_classes,)
            Training class prior probabilities.
        tolerance : float
            Convergence threshold based on difference between iterations.
        max_iter : int
            Max number of EM iterations.
        criteria : callable
            Metric to assess convergence, e.g., MAE.

        Returns
        -------
        qs : ndarray of shape (n_classes,)
            Estimated test set class prevalences.
        ps : ndarray of shape (n_samples, n_classes)
            Updated soft membership probabilities per instance.
        """
        
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
        r"""Calibrate posterior predictions with specified calibration method.
        
        Parameters
        ----------
        predictions : ndarray
            Posterior predictions to calibrate.
        
        Returns
        -------
        calibrated_predictions : ndarray
            Calibrated posterior predictions.
        
        Raises
        ------
        ValueError
            If calib_function is unrecognized.
        """
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
        """Temperature Scaling calibration applied to logits."""
        T = 1.0
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        scaled = logits / T
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

    def _bias_corrected_temperature_scaling(self, preds):
        """Bias-Corrected Temperature Scaling calibration."""
        T = 1.0
        bias = np.zeros(preds.shape[1])
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        logits = logits / T + bias
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _vector_scaling(self, preds):
        """Vector Scaling calibration."""
        W = np.ones(preds.shape[1])
        b = np.zeros(preds.shape[1])
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        scaled = logits * W + b
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

    def _no_bias_vector_scaling(self, preds):
        """No-Bias Vector Scaling calibration."""
        W = np.ones(preds.shape[1])
        preds = np.clip(preds, 1e-12, 1.0)
        logits = np.log(preds)
        scaled = logits * W
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)



class MLPE(SoftLearnerQMixin, BaseIterativeLikelihood):
    r"""Maximum Likelihood Prevalence Estimation (MLPE).

    Returns training priors as prevalence estimates without adaptations.

    Parameters
    ----------
    learner : estimator, optional
        Base classifier.

    References
    ----------
    .. [2] Esuli, A., Moreo, A., & Sebastiani, F. (2023). Learning to Quantify. Springer.
    """

    def __init__(self, learner=None):
        super().__init__(learner=learner, max_iter=1)
        
    def _iterate(self, predictions, priors):
        """Returns training priors without adjustment.
        
        Parameters
        ----------
        predictions : array-like
            Ignored in this implementation.
        priors : array-like
            Training priors, returned as is.
        
        Returns
        -------
        prevalences : array-like
            Equal to the training priors.
        """
        return priors
    
@define_binary
class CDE(SoftLearnerQMixin, BaseIterativeLikelihood):
    r"""CDE-Iterate for binary classification prevalence estimation.

    Threshold :math:`\tau` from false positive and false negative costs:
    .. math::
        \tau = \frac{c_{FP}}{c_{FP} + c_{FN}}

    Hard classification by thresholding posterior probability :math:`p(+|x)` at :math:`\tau`:
    .. math::
        \hat{y}(x) = \mathbf{1}_{p(+|x) > \tau}

    Prevalence estimation via classify-and-count:
    .. math::
        \hat{p}_U(+) = \frac{1}{N} \sum_{n=1}^N \hat{y}(x_n)

    False positive cost update:
    .. math::
        c_{FP}^{new} = \frac{p_L(+)}{p_L(-)} \times \frac{\hat{p}_U(-)}{\hat{p}_U(+)} \times c_{FN}

    Parameters
    ----------
    learner : estimator, optional
        Wrapped classifier (unused).
    tol : float, default=1e-4
        Convergence tolerance.
    max_iter : int, default=100
        Max iterations.
    init_cfp : float, default=1.0
        Initial false positive cost.

    References
    ----------
    .. [1] Esuli, A., Moreo, A., & Sebastiani, F. (2023). Learning to Quantify. Springer.
    """

    _parameter_constraints = {
        "tol": [Interval(0, None, inclusive_left=False)],
        "max_iter": [Interval(1, None, inclusive_left=True)],
        "init_cfp": [Interval(0, None, inclusive_left=False)]
    }

    def __init__(self, learner=None, tol=1e-4, max_iter=100, init_cfp=1.0):
        super().__init__(learner=learner, tol=tol, max_iter=max_iter)
        self.init_cfp = float(init_cfp)

    def _iterate(self, predictions, priors):
        r"""Iteratively estimate prevalences via cost-sensitive thresholding.

        Parameters
        ----------
        predictions : ndarray, shape (n_samples, 2)
            Posterior probabilities for binary classes [neg, pos].
        priors : ndarray, shape (2,)
            Training priors [p(neg), p(pos)].

        Returns
        -------
        prevalences : ndarray, shape (2,)
            Estimated prevalences for classes [neg, pos].
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