import numpy as np
from math import sqrt, pi
from abc import abstractmethod
from typing import Optional

from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

from mlquantify.mixture._base import BaseMixture  # seu BaseMixture
from mlquantify.base_aggregative import AggregationMixin, SoftLearnerQMixin

EPS = 1e-12


# -------------------------
# Utility functions
# -------------------------
def gaussian_kernel(X, Y, bandwidth):
    """
    Gaussian kernel matrix K(x,y) with bandwidth `bandwidth` (std dev).
    Returns matrix of shape (len(X), len(Y)) or (len(X), len(X)) when Y is None.
    Normalized so that it integrates properly w.r.t. dimensionality.
    """
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(Y)
    sqd = pairwise_distances(X, Y, metric="euclidean") ** 2
    D = X.shape[1]
    # kernel value: (2π)^{-D/2} * bandwidth^{-D} * exp(-sqd / (2*bandwidth^2))
    norm = (bandwidth ** D) * ((2 * pi) ** (D / 2))
    return np.exp(-sqd / (2 * (bandwidth ** 2))) / (norm + EPS)


def negative_log_likelihood_from_mixture(mixture_likelihoods):
    """
    Numerically stable -sum(log(likelihoods))
    """
    mixture_likelihoods = np.clip(mixture_likelihoods, EPS, None)
    return -np.sum(np.log(mixture_likelihoods))


def _simplex_constraints(n):
    """Return constraints and bounds for optimization on the simplex."""
    cons = [
        {"type": "eq", "fun": lambda a: np.sum(a) - 1.0},
        # inequality a_i >= 0 is handled via bounds instead of inequality constraints
    ]
    bounds = [(0.0, 1.0) for _ in range(n)]
    return cons, bounds


def _optimize_simplex(objective, n, x0=None, method="SLSQP", options=None):
    """Minimize `objective` on the probability simplex of dimension n."""
    if x0 is None:
        x0 = np.ones(n) / n
    cons, bounds = _simplex_constraints(n)
    res = minimize(objective, x0, method=method, constraints=cons, bounds=bounds, options=options or {"maxiter": 500})
    if not res.success:
        # fallback: project to simplex uniform
        x = np.clip(res.x if hasattr(res, 'x') else x0, 0.0, None)
        s = np.sum(x)
        if s <= 0:
            x = np.ones(n) / n
        else:
            x = x / s
        return x
    x = np.clip(res.x, 0.0, None)
    s = np.sum(x)
    if s <= 0:
        x = np.ones(n) / n
    else:
        x = x / s
    return x


# -------------------------
# Base KDEy class
# -------------------------
class BaseKDEy(SoftLearnerQMixin, AggregationMixin, BaseMixture):
    """
    Base KDEy using KernelDensity on posterior-space representations.
    Subclasses must implement _mixture which returns a prevalence vector alpha.
    """

    def __init__(self, learner=None, bandwidth: float = 0.1, kernel: str = 'gaussian'):
        # Note: BaseMixture.fit will prepare train_predictions and train_y_values
        self._precomputed = False
        self.learner = learner
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def aggregate(self, predictions, train_predictions, train_y_values):
        self.classes = self.classes if hasattr(self, 'classes') else np.unique(train_y_values)
        
        if not self._precomputed:
            self.models = self._precompute_training(train_predictions, train_y_values)
            self._precomputed = True
        

        return 
    
    def __get_representations(self, predictions, train_predictions, train_y_values):
        
        if not self._precomputed:
            self.models = self._precompute_training(train_predictions, train_y_values)
            self._precomputed = True
            
        return train_repr, test_repr
        
    @abstractmethod
    def _precompute_training(self, train_predictions, train_y_values):
        ...
    
    @abstractmethod
    def _mixture(self, predictions, train_representations):
        ...


# -------------------------
# KDEy - Maximum Likelihood (KDEyML)
# -------------------------
class KDEyML(BaseKDEy):
    """
    KDEy using maximum likelihood (minimizing -E_q[log p_alpha(x)] equivalently).
    """

    def _mixture(self, predictions, train_predictions, train_y_values):

        # Precompute class-wise densities p_i(x) for every x in test set
        # p_classwise shape: (n_classes, M)
        
        kde_models = self._train_representations
        n_classes = len(kde_models)
        
        M = predictions.shape[0]
        classwise = np.zeros((n_classes, M), dtype=float)
        for i, kde in enumerate(kde_models):
            # sklearn KernelDensity.score_samples returns log density
            classwise[i] = np.exp(kde.score_samples(predictions)) + EPS

        def neg_loglik(alpha):
            # ensure alpha is on simplex (but optimizer handles bounds/cons)
            alpha = np.clip(alpha, EPS, None)
            alpha = alpha / np.sum(alpha)
            # mixture density: (alpha @ classwise) per sample
            mixture = np.dot(alpha, classwise)  # shape (M,)
            return negative_log_likelihood_from_mixture(mixture)

        alpha = _optimize_simplex(neg_loglik, n_classes)
        return alpha


# -------------------------
# KDEy - Hellinger Distance (Monte Carlo) (KDEyHD)
# -------------------------
class KDEyHD(BaseKDEy):
    """
    KDEy minimizing squared Hellinger distance via Monte Carlo importance sampling.
    """
    
    def __init__(self, learner=None, bandwidth: float = 0.1, kernel: str = 'gaussian', montecarlo_trials: int = 1000, random_state = None):
        super().__init__(learner, bandwidth, kernel)
        self.montecarlo_trials = montecarlo_trials
        self.random_state = random_state

    def _precompute_training(self, train_predictions, train_y_values):
        # build KDEs per class and prepare reference samples & precomputed densities
        super()._precompute_training(train_predictions, train_y_values)

        n_class = len(self._classes)
        trials = int(self.montecarlo_trials)
        rng = np.random.RandomState(self.random_state)

        # sample equally from each class KDE to build reference_samples
        samples_per_class = max(1, trials // n_class)
        samples = []
        for kde in self._kde_models:
            s = kde.sample(samples_per_class, random_state=rng)
            samples.append(s)
        ref_samples = np.vstack(samples)
        # classwise densities at reference samples
        ref_classwise = np.asarray([np.exp(k.score_samples(ref_samples)) + EPS for k in self._kde_models])
        # reference density r = mean over classes (uniform mixture)
        ref_density = np.mean(ref_classwise, axis=0) + EPS

        self._ref_samples = ref_samples
        self._ref_classwise = ref_classwise
        self._ref_density = ref_density
        self._precomputed = True

    def _mixture(self, predictions, train_predictions, train_y_values):
        if not self._precomputed:
            self._precompute_training(train_predictions, train_y_values)

        # fit a KDE on test posteriors (to evaluate q at ref samples)
        Pte = np.atleast_2d(predictions)
        test_kde = KernelDensity(bandwidth=self.bandwidth).fit(Pte)
        qs = np.exp(test_kde.score_samples(self._ref_samples)) + EPS  # q(x_i)
        rs = self._ref_density + EPS  # r(x_i) precomputed
        iw = qs / rs  # importance weights
        p_class = self._ref_classwise + EPS  # shape (n_classes, t)
        fracs = p_class / qs  # p_class / q(x) -> shape (n, t)

        def f_squared_hellinger(u):
            return (np.sqrt(u) - 1.0) ** 2

        def divergence(alpha):
            alpha = np.clip(alpha, EPS, None)
            alpha = alpha / np.sum(alpha)
            # ps / qs = prev @ fracs  (vectorized)
            ps_div_qs = np.dot(alpha, fracs)  # shape (t,)
            vals = f_squared_hellinger(ps_div_qs) * iw
            return np.mean(vals)

        alpha = _optimize_simplex(divergence, len(self._kde_models))
        return alpha


# -------------------------
# KDEy - Cauchy-Schwarz closed form (KDEyCS)
# -------------------------
class KDEyCS(BaseKDEy):
    """
    KDEy using closed-form Cauchy–Schwarz divergence manipulations.
    Uses summed Gram matrices (train-train and train-test) to compute objective efficiently.
    """

    def _precompute_training(self, train_predictions, train_y_values):
        # we'll compute the sums used in closed-form: a_bar (train-test sums), B_bar (train-train sums), counts
        P = np.atleast_2d(train_predictions)
        y = np.asarray(train_y_values)
        classes = np.unique(y)
        self._classes = classes
        n = len(classes)

        # centers: list of arrays for each class
        centers = [P[y == c] for c in classes]
        counts = np.array([c.shape[0] if c.shape[0] > 0 else 1 for c in centers], dtype=float)

        # h_eff for Gaussian kernel used in paper is sqrt(2) * h (because kernel sums involve 2h^2)
        h_eff = sqrt(2) * self.bandwidth

        # compute B_bar[i, j] = sum_{u in Li} sum_{v in Lj} K(u, v; h_eff)
        B_bar = np.zeros((n, n), dtype=float)
        for i in range(n):
            Xi = centers[i]
            for j in range(i, n):
                Xj = centers[j]
                K = gaussian_kernel(Xi, Xj, h_eff)
                s = np.sum(K)
                B_bar[i, j] = s
                B_bar[j, i] = s

        # store centers, counts, and B_bar
        self._centers = centers
        self._counts = counts
        self._B_bar = B_bar
        self._h_eff = h_eff
        self._precomputed = True

    def _mixture(self, predictions, train_predictions, train_y_values):
        if not self._precomputed:
            self._precompute_training(train_predictions, train_y_values)

        Pte = np.atleast_2d(predictions)
        n = len(self._classes)
        centers = self._centers
        counts = self._counts + EPS
        B_bar = self._B_bar + EPS
        t = 1.0 / max(1, Pte.shape[0])

        # compute a_bar: for each class i, sum_j sum_k K(x_train^ij, x_test^k; 2h)
        a_bar = np.zeros(n, dtype=float)
        for i in range(n):
            K = gaussian_kernel(centers[i], Pte, self._h_eff)
            a_bar[i] = np.sum(K)

        def objective(alpha):
            # ensure valid alpha on simplex
            alpha = np.clip(alpha, EPS, None)
            alpha = alpha / np.sum(alpha)
            rbar = alpha / counts
            partA = -np.log(np.dot(rbar, a_bar) * t + EPS)
            partB = 0.5 * np.log(rbar @ (B_bar @ rbar) + EPS)
            return partA + partB

        alpha = _optimize_simplex(objective, n)
        return alpha
