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
# Função auxiliar: otimização no simplex com retorno do valor mínimo
# ============================================================

def _optimize_on_simplex(objective, n_classes, x0=None):
    """Otimiza uma função no simplex e retorna (alpha*, loss_min)."""
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
    """KDEy baseado em máxima verossimilhança."""

    def _precompute_training(self, train_predictions, train_y_values):
        super()._fit_kde_models(train_predictions, train_y_values)

    def _solve_prevalences(self, predictions):
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
    """KDEy minimizando a distância de Hellinger via Monte Carlo."""

    _parameter_constraints = {
        "montecarlo_trials": [Interval(1, None)],
    }

    def __init__(self, learner=None, bandwidth=0.1, kernel="gaussian", montecarlo_trials=1000, random_state=None):
        super().__init__(learner, bandwidth, kernel)
        self.montecarlo_trials = montecarlo_trials
        self.random_state = random_state

    def _precompute_training(self, train_predictions, train_y_values):
        super()._fit_kde_models(train_predictions, train_y_values)
        n_class = len(self._class_kdes)
        trials = int(self.montecarlo_trials)
        rng = check_random_state(self.random_state)

        samples = np.vstack([
            kde.sample(max(1, trials // n_class), random_state=rng)
            for kde in self._class_kdes
        ])

        ref_classwise = np.array([np.exp(k.score_samples(samples)) + EPS for k in self._class_kdes])
        ref_density = np.mean(ref_classwise, axis=0) + EPS

        self._ref_samples = samples
        self._ref_classwise = ref_classwise
        self._ref_density = ref_density

    def _solve_prevalences(self, predictions):
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
    """KDEy usando divergência de Cauchy–Schwarz (forma fechada)."""

    def _precompute_training(self, train_predictions, train_y_values):
        P = np.atleast_2d(train_predictions)
        y = np.asarray(train_y_values)
        centers = [P[y == c] for c in self.classes]
        counts = np.array([len(x) if len(x) > 0 else 1 for x in centers])
        h_eff = np.sqrt(2) * self.bandwidth

        B_bar = np.zeros((len(self.classes), len(self.classes)))
        for i, Xi in enumerate(centers):
            for j, Xj in enumerate(centers[i:], start=i):
                val = np.sum(gaussian_kernel(Xi, Xj, h_eff))
                B_bar[i, j] = B_bar[j, i] = val

        self._centers = centers
        self._counts = counts
        self._B_bar = B_bar
        self._h_eff = h_eff

    def _solve_prevalences(self, predictions):
        Pte = np.atleast_2d(predictions)
        n = len(self.classes)
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
