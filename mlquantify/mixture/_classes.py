import numpy as np
from abc import abstractmethod
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.optimize import minimize

from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import AggregationMixin, SoftLearnerQMixin, _get_learner_function
from mlquantify.mixture._base import BaseMixture
from mlquantify.multiclass import define_binary
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._get_scores import apply_cross_validation
from mlquantify.utils._validation import check_classes_attribute, validate_predictions, validate_prevalences, validate_y
from mlquantify.mixture._utils import (
    getHist,
    ternary_search,
)



# =====================================================
# Base class
# =====================================================
@define_binary
class AggregativeMixture(SoftLearnerQMixin, AggregationMixin, BaseMixture):
    r"""Base class for Mixture-based Quantification Methods.

    These methods assume that the test score distribution is a mixture
    of the positive and negative score distributions from the training data.
    """
    
    _parameter_constraints = {
        "strategy": [Options(["ovr", "ovo"])]
    }

    def __init__(self, learner = None, strategy="ovr", n_jobs=None):
        super().__init__()
        self.learner = learner
        self.pos_scores = None
        self.neg_scores = None
        self.distances = None
        self.strategy = strategy
        self.n_jobs = n_jobs
    
    def _fit(self, X, y, learner_fitted=False, cv=5, stratified=True, shuffle=False):
        learner_function = _get_learner_function(self)
        
        if learner_fitted:
            train_predictions = getattr(self.learner, learner_function)(X)
            y_train = y
        else:
            train_predictions, y_train = apply_cross_validation(
                self.learner,
                X,
                y,
                function= learner_function,
                cv= cv,
                stratified= stratified,
                random_state= None,
                shuffle= shuffle
            )
            self.learner.fit(X, y)
            
        self.train_predictions = train_predictions
        self.y_train = y_train

        self._precompute_training(train_predictions, y_train)
        return self

    def _precompute_training(self, train_predictions, y_train):
        """
        Fit learner and store score distributions for positive and negative classes.
        """ 
        # Store scores for positive and negative classes
        self.pos_scores = train_predictions[y_train == self.classes_[1], 1]
        self.neg_scores = train_predictions[y_train == self.classes_[0], 1]
        self._precomputed = True
        return self
    
    def _predict(self, X):
        """Predict class prevalences for the given data."""
        predictions = getattr(self.learner, _get_learner_function(self))(X)
        prevalences = self.aggregate(predictions, self.train_predictions, self.y_train)

        return prevalences
    
    def aggregate(self, predictions, train_predictions, y_train):
        predictions = validate_predictions(self, predictions)
        self.classes_ = check_classes_attribute(self, np.unique(y_train))

        if not self._precomputed:
            self._precompute_training(train_predictions, y_train)
            self._precomputed = True
        
        pos_test_scores = predictions[:, 1]
        
        best_alpha, _ = self.best_mixture(pos_test_scores, self.pos_scores, self.neg_scores)
        prevalence = np.array([1 - best_alpha, best_alpha])
        prevalence = validate_prevalences(self, prevalence, self.classes_)
        return prevalence
    
    @abstractmethod
    def best_mixture(self, predictions, pos_scores, neg_scores):
        ...

# =====================================================
# DyS
# =====================================================

class DyS(AggregativeMixture):
    r"""Distribution y-Similarity (DyS) quantification method.

    Uses mixture modeling with a dissimilarity measure between distributions
    computed on histograms of classifier scores. This method optimizes mixture 
    weights by minimizing a chosen distance measure: Hellinger, Topsoe, or ProbSymm.

    Parameters
    ----------
    learner : estimator, optional
        Base probabilistic classifier.
    measure : {'hellinger', 'topsoe', 'probsymm'}, default='topsoe'
        Distance function to minimize.
    bins_size : array-like or None
        Histogram bin sizes to try for score representation. Defaults to a set of 
        bin sizes between 2 and 30.

    References
    ----------
    .. [1] Maletzke et al. (2019). DyS: A Framework for Mixture Models in Quantification. AAAI 2019.
    .. [2] Esuli et al. (2023). Learning to Quantify. Springer.

    Examples
    --------
    >>> from mlquantify.mixture import DyS
    >>> from sklearn.linear_model import LogisticRegression
    >>> q = DyS(learner=LogisticRegression(), measure="hellinger")
    >>> q.fit(X_train, y_train)
    >>> prevalences = q.predict(X_test)
    """
    
    _parameter_constraints = {
        "measure": [Options(["hellinger", "topsoe", "probsymm"])],
        "bins_size": ["array-like", None]
    }

    def __init__(self, learner=None, measure="topsoe", bins_size=None):
        super().__init__(learner)
        if bins_size is None:
            bins_size = np.append(np.linspace(2, 20, 10), 30)
            
        self.measure = measure
        self.bins_size = np.asarray(bins_size, dtype=int)

    def best_mixture(self, predictions, pos_scores, neg_scores):
        r"""Determine the best mixture parameters for the given data.
        
        Applies ternary search to find the mixture weight minimizing the distance
        between the test score histogram and the mixture of positive and negative
        
        The mixture weight :math:`\alpha` is estimated as:

        .. math::

            \alpha = \arg \min_{\alpha \in [0, 1]} D \left( H_{test}, \alpha H_{pos} + (1 - \alpha) H_{neg} \right)
            
        where :math:`D` is the selected distance measure and :math:`H` denotes histograms.
        
        
        Parameters
        ----------
        predictions : ndarray
            Classifier scores for the test data.
        pos_scores : ndarray
            Classifier scores for the positive class from training data.
        neg_scores : ndarray
            Classifier scores for the negative class from training data.
            
            
        Returns
        -------
        alpha : float
            Estimated mixture weight.
        best_distance : float
            Distance corresponding to the best mixture weight.
        """
        
        prevs = []
        self.distances = []
        for bins in self.bins_size:
            pos = getHist(pos_scores, bins)
            neg = getHist(neg_scores, bins)
            test = getHist(predictions, bins)

            def f(alpha):
                mix = self._mix(pos, neg, alpha)
                return BaseMixture.get_distance(mix, test, measure=self.measure)

            alpha = ternary_search(0, 1, f)
            prevs.append(alpha)
            self.distances.append(f(alpha))
        alpha = np.median(prevs)
        best_distance = np.median(self.distances)
        return alpha, best_distance
        
    def _mix(self, pos_hist, neg_hist, alpha):
        return alpha * pos_hist + (1 - alpha) * neg_hist


# =====================================================
# HDy
# =====================================================

class HDy(AggregativeMixture):
    r"""Hellinger Distance Minimization (HDy) quantification method.

    Estimates class prevalences by finding mixture weights that minimize
    the Hellinger distance between the histogram of test scores and the mixture
    of positive and negative class score histograms, evaluated over multiple bin sizes.

    Parameters
    ----------
    learner : estimator, optional
        Base probabilistic classifier.

    References
    ----------
    .. [2] Esuli et al. (2023). Learning to Quantify. Springer.

    """
    
    def best_mixture(self, predictions, pos_scores, neg_scores):
        r"""Determine the best mixture parameters for the given data.

        Compute the mixture weight :math:`\alpha` that minimizes the Hellinger distance between the test score histogram and the mixture of positive and negative class score histograms.

        The mixture weight :math:`\alpha` is estimated as:

        .. math::

            \alpha = \arg \min_{\alpha \in [0, 1]} Hellinger \left( H_{test}, \alpha H_{pos} + (1 - \alpha) H_{neg} \right)
            
        where :math:`H` denotes histograms.
        
        
        Parameters
        ----------
        predictions : ndarray
            Classifier scores for the test data.
        pos_scores : ndarray
            Classifier scores for the positive class from training data.
        neg_scores : ndarray
            Classifier scores for the negative class from training data.
            
            
        Returns
        -------
        alpha : float
            Estimated mixture weight.
        best_distance : float
            Distance corresponding to the best mixture weight.
        """
        
        bins_size =  np.linspace(10, 110, 11)
        alpha_values = np.round(np.linspace(0, 1, 101), 2)
        
        alphas, self.distances = [], []
        for bins in bins_size:
            pos = getHist(pos_scores, bins)
            neg = getHist(neg_scores, bins)
            test = getHist(predictions, bins)
            dists = []
            for a in alpha_values:
                mix = self._mix(pos, neg, a)
                dists.append(BaseMixture.get_distance(mix, test, measure="hellinger"))
            a = alpha_values[np.argmin(dists)]
            alphas.append(a)
            self.distances.append(np.min(dists))
        
        best_alpha = np.median(alphas)
        best_distance = np.median(self.distances)    
        
        return best_alpha, best_distance
    
    def _mix(self, pos_hist, neg_hist, alpha):
        return alpha * pos_hist + (1 - alpha) * neg_hist



# =====================================================
# SMM
# =====================================================

class SMM(AggregativeMixture):
    r"""Sample Mean Matching (SMM) quantification method.

    Estimates class prevalence by matching the mean score of the test samples 
    to a convex combination of positive and negative training scores. The mixture 
    weight :math:`\alpha` is computed as:

    .. math::

        \alpha = \frac{\bar{s}_{test} - \bar{s}_{neg}}{\bar{s}_{pos} - \bar{s}_{neg}}

    where :math:`\bar{s}` denotes the sample mean.

    Parameters
    ----------
    learner : estimator, optional
        Base probabilistic classifier.

    References
    ----------
    .. [2] Esuli et al. (2023). Learning to Quantify. Springer.
    """
    
    def best_mixture(self, predictions, pos_scores, neg_scores):
        mean_pos = np.mean(pos_scores)
        mean_neg = np.mean(neg_scores)
        mean_test = np.mean(predictions)
        if mean_pos - mean_neg == 0:
            alpha = mean_test
        else:
            alpha = np.clip((mean_test - mean_neg) / (mean_pos - mean_neg), 0, 1)
        return alpha, None


# =====================================================
# SORD
# =====================================================

class SORD(AggregativeMixture):
    """Sample Ordinal Distance (SORD) quantification method.

    Estimates prevalence by minimizing the weighted sum of absolute score differences
    between test data and training classes. The method creates weighted score 
    vectors for positive, negative, and test samples, sorts them, and computes
    a cumulative absolute difference as the distance measure.

    Parameters
    ----------
    learner : estimator, optional
        Base probabilistic classifier.

    References
    ----------
    .. [2] Esuli et al. (2023). Learning to Quantify. Springer.
    """

    def best_mixture(self, predictions, pos_scores, neg_scores):
        alphas = np.linspace(0, 1, 101)
        self.distances = []
        
        pos, neg, test = pos_scores, neg_scores, predictions
        n_pos, n_neg, n_test = len(pos), len(neg), len(test)
        for a in alphas:
            pos_w = np.full(n_pos, a / n_pos)
            neg_w = np.full(n_neg, (1 - a) / n_neg)
            test_w = np.full(n_test, -1 / n_test)
            scores = np.concatenate([pos, neg, test])
            weights = np.concatenate([pos_w, neg_w, test_w])
            idx = np.argsort(scores)
            sorted_scores = scores[idx]
            sorted_weights = weights[idx]
            cum_w = sorted_weights[0]
            total = 0
            for i in range(1, len(sorted_scores)):
                seg = sorted_scores[i] - sorted_scores[i - 1]
                total += abs(seg * cum_w)
                cum_w += sorted_weights[i]
            self.distances.append(total)
            
        best_distance_index = np.argmin(self.distances)
        best_alpha = alphas[best_distance_index]
        best_distance = self.distances[best_distance_index]
        return best_alpha, best_distance





# =====================================================
# Non aggregative Mixture-based Quantifiers
# =====================================================
@define_binary
class HDx(BaseMixture):
    """
    Hellinger Distance-based Quantifier (HDx).

    A non-aggregative mixture quantifier that estimates class prevalences by 
    minimizing the average Hellinger distance between class-wise feature histograms 
    of training data and test data. It iterates over mixture weights and histogram bin sizes,
    evaluating distance per feature and aggregates the results.

    Parameters
    ----------
    bins_size : array-like, optional
        Histogram bin sizes to consider for discretizing features.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass quantification strategy.

    Attributes
    ----------
    pos_features : ndarray
        Training samples of the positive class.
    neg_features : ndarray
        Training samples of the negative class.

    References
    ----------
    .. [2] Esuli et al. (2023). Learning to Quantify. Springer.
    """
    
    _parameter_constraints = {
        "bins_size": ["array-like", None],
        "strategy": [Options(["ovr", "ovo"])]
    }

    def __init__(self, bins_size=None, strategy="ovr", n_jobs=None):
        super().__init__()
        if bins_size is None:
            bins_size = np.linspace(10, 110, 11)

        self.bins_size = bins_size
        self.neg_features = None
        self.pos_features = None
        self.strategy = strategy
        self.n_jobs = n_jobs
    
    def _fit(self, X, y, *args, **kwargs):
        self.pos_features = X[y == self.classes_[1]]
        self.neg_features = X[y == self.classes_[0]]
        return self
    
    def _predict(self, X) -> np.ndarray:
        alpha, _ = self.best_mixture(X, self.pos_features, self.neg_features)
        prevalence = np.array([1 - alpha, alpha])
        prevalence = validate_prevalences(self, prevalence, self.classes_)
        return prevalence
    
    def best_mixture(self, X, pos, neg):
        alpha_values = np.round(np.linspace(0, 1, 101), 2)
        self.distances = []

        # Iterate over alpha values to compute the prevalence
        for alpha in alpha_values:
            distances = []

            # For each feature, compute the Hellinger distance
            for feature_idx in range(X.shape[1]):
                
                 for bins in self.bins_size:
                
                    pos_feature = pos[:, feature_idx]
                    neg_feature = neg[:, feature_idx]
                    test_feature = X[:, feature_idx]

                    pos_hist = getHist(pos_feature, bins)
                    neg_hist = getHist(neg_feature, bins)
                    test_hist = getHist(test_feature, bins)

                    mix_hist = alpha * pos_hist + (1 - alpha) * neg_hist
                    distance = BaseMixture.get_distance(mix_hist, test_hist, measure="hellinger")
                    distances.append(distance)

            avg_distance = np.mean(distances)
            self.distances.append(avg_distance)
        best_alpha = alpha_values[np.argmin(self.distances)]
        best_distance = np.min(self.distances)
        return best_alpha, best_distance



class MMD_RKHS(BaseMixture):
    r"""
    Maximum Mean Discrepancy in RKHS (MMD-RKHS) quantification method.

    This method estimates class prevalences in an unlabeled test set by
    matching the kernel mean embedding of the test distribution to a
    convex combination of the class-conditional training embeddings.

    Let :math:`\mathcal{X} \subseteq \mathbb{R}^d` be the input space and
    :math:`\mathcal{Y} = \{0, \dots, C-1\}` the label set. Let
    :math:`K` be a positive definite kernel with RKHS :math:`\mathcal{H}`
    and feature map :math:`\phi`, so that
    :math:`K(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}`.

    For each class :math:`y`, the class-conditional kernel mean embedding is

    .. math::

        \mu_y \;=\; \mathbb{E}_{x \sim P_{D}(x \mid y)}[\phi(x)] \in \mathcal{H},

    and the test mean embedding is

    .. math::

        \mu_U \;=\; \mathbb{E}_{x \sim P_{U}(x)}[\phi(x)] \in \mathcal{H}.

    Under prior probability shift, the test distribution satisfies

    .. math::

        P_U(x) = \sum_{y=0}^{C-1} \theta_y \, P_D(x \mid y),

    which implies

    .. math::

        \mu_U = \sum_{y=0}^{C-1} \theta_y \, \mu_y,

    where :math:`\theta \in \Delta^{C-1}` is the class prevalence vector.
    The MMD-RKHS estimator solves

    .. math::

        \hat{\theta}
        \;=\;
        \arg\min_{\theta \in \Delta^{C-1}}
        \big\lVert \textstyle\sum_{y=0}^{C-1} \theta_y \mu_y - \mu_U
        \big\rVert_{\mathcal{H}}^2.

    In practice, embeddings are approximated by empirical means. Using the
    kernel trick, the objective can be written as a quadratic program

    .. math::

        \hat{\theta}
        \;=\;
        \arg\min_{\theta \in \Delta^{C-1}}
        \big( \theta^\top G \, \theta - 2 \, h^\top \theta \big),

    with

    .. math::

        G_{yy'} = \langle \hat{\mu}_y, \hat{\mu}_{y'} \rangle_{\mathcal{H}},
        \qquad
        h_y = \langle \hat{\mu}_y, \hat{\mu}_U \rangle_{\mathcal{H}}.

    The solution :math:`\hat{\theta}` is the estimated prevalence vector.

    Parameters
    ----------
    kernel : {'rbf', 'linear', 'poly', 'sigmoid', 'cosine'}, default='rbf'
        Kernel used to build the RKHS where MMD is computed.
    gamma : float or None, default=None
        Kernel coefficient for 'rbf' and 'sigmoid'.
    degree : int, default=3
        Degree of the polynomial kernel.
    coef0 : float, default=0.0
        Independent term in 'poly' and 'sigmoid' kernels.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass quantification strategy flag (for consistency with
        other mixture-based quantifiers).

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during fitting.
    X_train_ : ndarray of shape (n_train, n_features)
        Training feature matrix.
    y_train_ : ndarray of shape (n_train,)
        Training labels.
    class_means_ : ndarray of shape (n_classes, n_train)
        Empirical class-wise kernel mean embeddings in the span of training
        samples.
    K_train_ : ndarray of shape (n_train, n_train)
        Gram matrix of training samples under the chosen kernel.

    References
    ----------
    .. [1] Iyer, A., Nath, S., & Sarawagi, S. (2014).
        Maximum Mean Discrepancy for Class Ratio Estimation:
        Convergence Bounds and Kernel Selection. ICML.

    .. [2] Esuli, A., Moreo, A., & Sebastiani, F. (2023).
        Learning to Quantify. Springer.
    """

    _parameter_constraints = {
        "kernel": [Options(["rbf", "linear", "poly", "sigmoid", "cosine"])],
        "gamma": [Interval(0, None, inclusive_left=False), Options([None])],
        "degree": [Interval(1, None, inclusive_left=True)],
        "coef0": [Interval(0, None, inclusive_left=True)],
        "strategy": [Options(["ovr", "ovo"])],
    }

    def __init__(self,
                 kernel="rbf",
                 gamma=None,
                 degree=3,
                 coef0=0.0):
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.X_train_ = None
        self.y_train_ = None
        self.class_means_ = None  # class-wise kernel means
        self.K_train_ = None      # train Gram matrix


    @_fit_context(prefer_skip_nested_validation=True)
    def _fit(self, X, y, *args, **kwargs):
        """
        Store X, y, validate labels and precompute class-wise kernel means.
        """
        self.X_train_ = X
        self.y_train_ = y

        class_means, K_train = self._compute_class_means(X, y)
        self.class_means_ = class_means
        self.K_train_ = K_train

        return self

    def _predict(self, X) -> np.ndarray:
        """
        Estimate the prevalence vector on X using MMD.
        """
        self.classes_ = check_classes_attribute(self, np.unique(self.y_train_))

        theta, _ = self.best_mixture(X, self.X_train_, self.y_train_)
        prevalence = validate_prevalences(self, theta, self.classes_)
        return prevalence

    def best_mixture(self, X_test, X_train, y_train):
        """
        Implements the MMD-based class ratio estimation:

        .. math::

            \min_{\theta \in \Delta^{C-1}} \| \sum_{y=0}^{C-1} \theta_y \mu_y - \mu_U \|^2

        and returns (theta, objective_value).
        """
        # Use precomputed means if available
        if self.class_means_ is None or self.X_train_ is None:
            class_means, _ = self._compute_class_means(X_train, y_train)
        else:
            class_means = self.class_means_

        mu_u = self._compute_unlabeled_mean(X_test)
        G, h = self._build_QP_matrices(class_means, mu_u)

        theta = self._solve_simplex_qp(G, h)
        # Objective value: ||A theta - a||^2 = theta^T G theta - 2 h^T theta + const
        obj = float(theta @ G @ theta - 2.0 * (h @ theta))

        return theta, obj


    def _kernel_kwargs(self):
        params = {}
        if self.kernel == "rbf" and self.gamma is not None:
            params["gamma"] = self.gamma
        if self.kernel == "poly":
            params["degree"] = self.degree
            params["coef0"] = self.coef0
        if self.kernel == "sigmoid":
            if self.gamma is not None:
                params["gamma"] = self.gamma
            params["coef0"] = self.coef0
        return params

    def _compute_class_means(self, X, y):
        """
        Compute kernel mean embeddings per class in the RKHS.

        X: (n_train, d)
        y: (n_train,)
        Returns:
            class_means: (n_classes, n_train)
            K:           (n_train, n_train)
        """
        classes = self.classes_
        K = pairwise_kernels(X, X, metric=self.kernel, **self._kernel_kwargs())
        means = []
        for c in classes:
            mask = (y == c)
            Kc = K[mask]              # rows of class c
            mu_c = Kc.mean(axis=0)    # mean over rows
            means.append(mu_c)
        means = np.vstack(means)
        return means, K

    def _compute_unlabeled_mean(self, X_test):
        """
        Compute the kernel mean embedding of the test set.

        mu_U = E_{x in U} K(x, ·)
        """
        K_ut = pairwise_kernels(
            X_test,
            self.X_train_,
            metric=self.kernel,
            **self._kernel_kwargs()
        )
        mu_u = K_ut.mean(axis=0)  # shape (n_train,)
        return mu_u

    def _build_QP_matrices(self, class_means, mu_u):
        """
        Build G and h for the objective

            min_theta  theta^T G theta - 2 h^T theta

        with theta in the simplex (dimension = n_classes).

        class_means: (n_classes, n_train)
        mu_u:        (n_train,)
        """
        # Gram of means in RKHS: G_ij = <mu_i, mu_j>
        G = class_means @ class_means.T       # (C, C)
        # Inner products with mu_U: h_i = <mu_i, mu_U>
        h = class_means @ mu_u                # (C,)
        return G, h

    def _solve_simplex_qp(self, G, h):
        """
        Solve:

            min_theta  theta^T G theta - 2 h^T theta
            s.t.       theta >= 0, sum(theta) = 1

        using SciPy's SLSQP solver.
        """
        C = G.shape[0]

        def obj(theta):
            return float(theta @ G @ theta - 2.0 * (h @ theta))

        def grad(theta):
            # gradient: 2 G theta - 2 h
            return 2.0 * (G @ theta - h)

        # equality constraint: sum(theta) = 1
        cons = {
            "type": "eq",
            "fun": lambda t: np.sum(t) - 1.0,
            "jac": lambda t: np.ones_like(t),
        }

        # bounds: theta_i >= 0
        bounds = [(0.0, 1.0) for _ in range(C)]

        # initial point: uniform distribution on the simplex
        x0 = np.ones(C) / C

        res = minimize(
            obj,
            x0,
            method="SLSQP",
            jac=grad,
            bounds=bounds,
            constraints=[cons],
            options={"maxiter": 100, "ftol": 1e-9},
        )

        theta = res.x
        theta = np.maximum(theta, 0)
        s = theta.sum()
        if s <= 0:
            theta = np.ones_like(theta) / len(theta)
        else:
            theta /= s
        return theta
