import numpy as np
from sklearn.neighbors import KernelDensity
from mlquantify.utils._constraints import Interval
from mlquantify.neighbors._base import BaseKDE
from mlquantify.neighbors._utils import (
    gaussian_kernel,
    negative_log_likelihood,
    EPS,
)
from mlquantify.utils import check_random_state
from scipy.optimize import minimize


# ============================================================
# Auxiliary functions
# ============================================================

def _optimize_on_simplex(objective, n_classes, x0=None):
    """
    Optimize an objective function over the probability simplex.
    
    This function performs constrained optimization to find the mixture weights
    \( \alpha \) on the simplex \( \Delta^{n-1} = \{ \alpha \in \mathbb{R}^n : \alpha_i \geq 0, \sum_i \alpha_i = 1 \} \)
    that minimize the given objective function.

    Parameters
    ----------
    objective : callable
        Function from \( \mathbb{R}^n \to \mathbb{R} \) to minimize.
    n_classes : int
        Dimensionality of the simplex (number of classes).
    x0 : array-like, optional
        Initial guess for the optimization, defaults to uniform vector.

    Returns
    -------
    alpha_opt : ndarray of shape (n_classes,)
        Optimized weights summing to 1.
    min_loss : float
        Objective function value at optimum.

    Notes
    -----
    The optimization uses scipy's `minimize` with bounds and equality constraint.
    """
    if x0 is None:
        x0 = np.ones(n_classes) / n_classes

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0, 1)] * n_classes

    res = minimize(objective, x0, bounds=bounds, constraints=constraints)
    alpha_opt = res.x / np.sum(res.x)
    return alpha_opt, res.fun


# ============================================================
# KDEy-ML — Maximum Likelihood
# ============================================================

class KDEyML(BaseKDE):
    """KDEy Maximum Likelihood quantifier.
    
    Models class-conditional densities of posterior probabilities via Kernel Density
    Estimation (KDE) and estimates class prevalences by maximizing the likelihood of 
    test data under a mixture model of these KDEs.
    
    The mixture weights correspond to class prevalences, optimized under the simplex 
    constraint. The optimization minimizes the negative log-likelihood of the mixture
    density evaluated at test posteriors.

    This approach generalizes EM-based quantification methods by using KDE instead 
    of discrete histograms, allowing smooth multivariate density estimation over 
    the probability simplex.

    References
    ----------
    The method is based on ideas presented by Moreo et al. (2023), extending KDE-based 
    approaches for distribution matching and maximum likelihood estimation.
    """

    def _precompute_training(self, train_predictions, train_y_values):
        """
        Fit KDE models on class-specific training posterior predictions.
        """
        super()._fit_kde_models(train_predictions, train_y_values)

    def _solve_prevalences(self, predictions):
        """
        Estimate class prevalences by maximizing log-likelihood under KDE mixture.
        
        Parameters
        ----------
        predictions : ndarray, shape (n_samples, n_features)
            Posterior probabilities of test set instances.

        Returns
        -------
        alpha_opt : ndarray, shape (n_classes,)
            Estimated class prevalences.
        min_loss : float
            Minimum negative log-likelihood achieved.

        Notes
        -----
        The optimization is solved over the probability simplex.
        """
        n_classes = len(self._class_kdes)
        class_likelihoods = np.array([
            np.exp(kde.score_samples(predictions)) + EPS for kde in self._class_kdes
        ])  # (n_classes, n_samples)

        def objective(alpha):
            mixture = np.dot(alpha, class_likelihoods)
            return negative_log_likelihood(mixture)

        alpha_opt, min_loss = _optimize_on_simplex(objective, n_classes)
        
        self.best_distance = min_loss

        return alpha_opt, min_loss


# ============================================================
# KDEy-HD — Hellinger Distance Minimization
# ============================================================

class KDEyHD(BaseKDE):
    r"""KDEy Hellinger Distance Minimization quantifier.
    
    Estimates class prevalences by minimizing the Hellinger distance \( HD \) between
    the KDE mixture of class-conditional densities and the KDE of test data, estimated
    via Monte Carlo sampling and importance weighting.
    
    This stochastic approximation enables practical optimization of complex divergence
    measures otherwise lacking closed-form expressions for Gaussian Mixture Models.

    Parameters
    ----------
    montecarlo_trials : int
        Number of Monte Carlo samples used in approximation.
    random_state : int or None
        Seed or random state for reproducibility.

    References
    ----------
    Builds on f-divergence Monte Carlo approximations for KDE mixtures as detailed 
    by Moreo et al. (2023) and importance sampling techniques.
    """

    _parameter_constraints = {
        "montecarlo_trials": [Interval(1, None)],
    }

    def __init__(self, learner=None, bandwidth=0.1, kernel="gaussian", montecarlo_trials=1000, random_state=None):
        super().__init__(learner, bandwidth, kernel)
        self.montecarlo_trials = montecarlo_trials
        self.random_state = random_state

    def _precompute_training(self, train_predictions, train_y_values):
        """
        Precompute reference samples from class KDEs and their densities.
        """
        super()._fit_kde_models(train_predictions, train_y_values)
        n_class = len(self._class_kdes)
        trials = int(self.montecarlo_trials)
        rng = check_random_state(self.random_state)
        # Convert to integer seed for sklearn compatibility
        seed = rng.integers(0, 2**31 - 1) if hasattr(rng, 'integers') else self.random_state

        samples = np.vstack([
            kde.sample(max(1, trials // n_class), random_state=seed)
            for kde in self._class_kdes
        ])

        ref_classwise = np.array([np.exp(k.score_samples(samples)) + EPS for k in self._class_kdes])
        ref_density = np.mean(ref_classwise, axis=0) + EPS

        self._ref_samples = samples
        self._ref_classwise = ref_classwise
        self._ref_density = ref_density

    def _solve_prevalences(self, predictions):
        """
        Minimize Hellinger distance between test KDE and mixture KDE via importance sampling.
        """
        test_kde = KernelDensity(bandwidth=self.bandwidth).fit(predictions)
        qs = np.exp(test_kde.score_samples(self._ref_samples)) + EPS
        iw = qs / self._ref_density
        fracs = self._ref_classwise / qs

        def objective(alpha):
            alpha = np.clip(alpha, EPS, None)
            alpha /= np.sum(alpha)
            ps_div_qs = np.dot(alpha, fracs)
            vals = (np.sqrt(ps_div_qs) - 1.0) ** 2 * iw
            return np.mean(vals)

        alpha_opt, min_loss = _optimize_on_simplex(objective, len(self._class_kdes))

        self.best_distance = min_loss

        return alpha_opt, min_loss


# ============================================================
# KDEy-CS — Cauchy–Schwarz Divergence
# ============================================================

class KDEyCS(BaseKDE):
    """
    KDEy Cauchy-Schwarz Divergence quantifier.
    
    Uses a closed-form solution for minimizing the Cauchy-Schwarz (CS) divergence between
    Gaussian Mixture Models representing class-conditional densities fitted via KDE.
    
    This mathematically efficient approach leverages precomputed kernel Gram matrices
    of train-train, train-test, and test-test instances for fast divergence evaluation,
    enabling scalable multiclass quantification.

    References
    ----------
    Based on closed-form CS divergence derivations by Kampa et al. (2011) and KDEy 
    density representations, as discussed by Moreo et al. (2023).
    """

    def _precompute_training(self, train_predictions, train_y_values):
        """
        Precompute kernel sums and Gram matrices needed for CS divergence evaluation.
        """
        P = np.atleast_2d(train_predictions)
        y = np.asarray(train_y_values)
        centers = [P[y == c] for c in self.classes_]
        counts = np.array([len(x) if len(x) > 0 else 1 for x in centers])
        h_eff = np.sqrt(2) * self.bandwidth

        B_bar = np.zeros((len(self.classes_), len(self.classes_)))
        for i, Xi in enumerate(centers):
            for j, Xj in enumerate(centers[i:], start=i):
                val = np.sum(gaussian_kernel(Xi, Xj, h_eff))
                B_bar[i, j] = B_bar[j, i] = val
        self._centers = centers
        self._counts = counts
        self._B_bar = B_bar
        self._h_eff = h_eff

    def _solve_prevalences(self, predictions):
        """
        Minimize Cauchy-Schwarz divergence over class mixture weights on the probability simplex.
        """
        Pte = np.atleast_2d(predictions)
        n = len(self.classes_)
        a_bar = np.array([np.sum(gaussian_kernel(Xi, Pte, self._h_eff)) for Xi in self._centers])
        counts = self._counts + EPS
        B_bar = self._B_bar + EPS
        t = 1.0 / max(1, Pte.shape[0])

        def objective(alpha):
            alpha = np.clip(alpha, EPS, None)
            alpha /= np.sum(alpha)
            rbar = alpha / counts
            partA = -np.log(np.dot(rbar, a_bar) * t + EPS)
            partB = 0.5 * np.log(rbar @ (B_bar @ rbar) + EPS)
            return partA + partB

        alpha_opt, min_loss = _optimize_on_simplex(objective, n)

        self.best_distance = min_loss

        return alpha_opt, min_loss
