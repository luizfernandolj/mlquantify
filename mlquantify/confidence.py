import numpy as np
from scipy.stats import chi2


import numpy as np
from scipy.stats import chi2


import numpy as np
from scipy.stats import chi2


class BaseConfidenceRegion:
    """Base class for confidence regions of prevalence estimates.

    This class defines the interface and core structure for constructing 
    confidence regions around class prevalence estimates obtained from 
    quantification models.

    Confidence regions capture the uncertainty associated with prevalence 
    estimates, typically derived from bootstrap resampling as proposed by [1].
    Subclasses define specific types of regions (e.g., intervals, ellipses).

    Parameters
    ----------
    prev_estims : array-like of shape (m, n)
        Collection of `m` bootstrap prevalence estimates for `n` classes.
    confidence_level : float, default=0.95
        Desired confidence level (1 - α) of the region.

    Attributes
    ----------
    prev_estims : ndarray of shape (m, n)
        Bootstrap prevalence estimates.
    confidence_level : float
        Confidence level associated with the region.

    Notes
    -----
    The general goal is to construct a confidence region :math:`CR_α` such that:

    .. math::
        P(π^* \\in CR_α) = 1 - α

    where :math:`π^*` is the true (unknown) class prevalence vector.

    Examples
    --------
    >>> import numpy as np
    >>> class DummyRegion(BaseConfidenceRegion):
    ...     def _compute_region(self):
    ...         self.mean_ = np.mean(self.prev_estims, axis=0)
    ...     def get_region(self):
    ...         return self.mean_
    ...     def get_point_estimate(self):
    ...         return self.mean_
    ...     def contains(self, point):
    ...         return np.allclose(point, self.mean_, atol=0.1)
    >>> X = np.random.dirichlet(np.ones(3), size=100)
    >>> region = DummyRegion(X, confidence_level=0.9)
    >>> region.get_point_estimate().round(3)
    array([0.33, 0.33, 0.34])

    References
    ----------
    [1] Moreo, A., & Salvati, N. (2025). 
        *An Efficient Method for Deriving Confidence Intervals in Aggregative Quantification*.
        Istituto di Scienza e Tecnologie dell’Informazione, CNR, Pisa.
    """

    def __init__(self, prev_estims, confidence_level=0.95):
        self.prev_estims = np.asarray(prev_estims)
        self.confidence_level = confidence_level
        self._compute_region()

    def _compute_region(self):
        raise NotImplementedError("Subclasses must implement _compute_region().")

    def get_region(self):
        """Return the parameters defining the confidence region."""
        raise NotImplementedError

    def get_point_estimate(self):
        """Return the point estimate of prevalence (e.g., mean of bootstrap samples)."""
        raise NotImplementedError

    def contains(self, point):
        """Check whether a prevalence vector lies within the region."""
        raise NotImplementedError


# ==========================================================
# Confidence Intervals (via percentiles)
# ==========================================================

class ConfidenceInterval(BaseConfidenceRegion):
    """Bootstrap confidence intervals for each class prevalence.

    Constructs independent percentile-based confidence intervals 
    for each class dimension from bootstrap samples.

    The confidence region is defined as:

    .. math::
        CI_α(π) = 
        \\begin{cases}
        1 & \\text{if } L_i \\le π_i \\le U_i, \\forall i=1,...,n \\\\
        0 & \\text{otherwise}
        \\end{cases}

    where :math:`L_i` and :math:`U_i` are the empirical 
    α/2 and 1−α/2 quantiles for class i.

    Parameters
    ----------
    prev_estims : array-like of shape (m, n)
        Bootstrap prevalence estimates.
    confidence_level : float, default=0.95
        Desired confidence level.

    Attributes
    ----------
    I_low : ndarray of shape (n,)
        Lower confidence bounds.
    I_high : ndarray of shape (n,)
        Upper confidence bounds.

    Examples
    --------
    >>> X = np.random.dirichlet(np.ones(3), size=200)
    >>> ci = ConfidenceInterval(X, confidence_level=0.9)
    >>> ci.get_region()
    (array([0.05, 0.06, 0.05]), array([0.48, 0.50, 0.48]))
    >>> ci.contains([0.3, 0.4, 0.3])
    array([[ True]])

    References
    ----------
    [1] Moreo, A., & Salvati, N. (2025).
        *An Efficient Method for Deriving Confidence Intervals in Aggregative Quantification*.
        Section 3.3, Equation (1).
    """

    def _compute_region(self):
        alpha = 1 - self.confidence_level
        low_perc = (alpha / 2.) * 100
        high_perc = (1 - alpha / 2.) * 100
        self.I_low, self.I_high = np.percentile(self.prev_estims, q=[low_perc, high_perc], axis=0)

    def get_region(self):
        return self.I_low, self.I_high
    
    def get_point_estimate(self):
        return np.mean(self.prev_estims, axis=0)

    def contains(self, point):
        point = np.asarray(point)
        within = np.logical_and(self.I_low <= point, point <= self.I_high)
        return np.all(within, axis=-1, keepdims=True)


# ==========================================================
# Confidence Ellipse in Simplex
# ==========================================================

class ConfidenceEllipseSimplex(BaseConfidenceRegion):
    """Confidence ellipse for prevalence estimates in the simplex.

    Defines a multivariate confidence region based on a chi-squared threshold:

    .. math::
        CE_α(π) =
        \\begin{cases}
        1 & \\text{if } (π - μ)^T Σ^{-1} (π - μ) \\le χ^2_{n-1}(1-α) \\\\
        0 & \\text{otherwise}
        \\end{cases}

    Parameters
    ----------
    prev_estims : array-like of shape (m, n)
        Bootstrap prevalence estimates.
    confidence_level : float, default=0.95
        Confidence level.

    Attributes
    ----------
    mean_ : ndarray of shape (n,)
        Mean prevalence estimate.
    precision_matrix : ndarray of shape (n, n)
        Inverse covariance matrix of estimates.
    chi2_val : float
        Chi-squared cutoff threshold defining the ellipse.

    Examples
    --------
    >>> X = np.random.dirichlet(np.ones(3), size=200)
    >>> ce = ConfidenceEllipseSimplex(X, confidence_level=0.95)
    >>> ce.get_point_estimate().round(3)
    array([0.33, 0.34, 0.33])
    >>> ce.contains(np.array([0.4, 0.3, 0.3]))
    True

    References
    ----------
    [1] Moreo, A., & Salvati, N. (2025).
        *An Efficient Method for Deriving Confidence Intervals in Aggregative Quantification*.
        Section 3.3, Equation (2).
    """

    def _compute_region(self):
        cov_ = np.cov(self.prev_estims, rowvar=False, ddof=1)
        try:
            self.precision_matrix = np.linalg.inv(cov_)
        except np.linalg.LinAlgError:
            self.precision_matrix = None

        dim = self.prev_estims.shape[-1]
        ddof = dim - 1
        self.chi2_val = chi2.ppf(self.confidence_level, ddof)
        self.mean_ = np.mean(self.prev_estims, axis=0)

    def get_region(self):
        return self.mean_, self.precision_matrix, self.chi2_val
    
    def get_point_estimate(self):
        return self.mean_

    def contains(self, point):
        if self.precision_matrix is None:
            return False
        diff = point - self.mean_
        dist2 = diff.T @ self.precision_matrix @ diff
        return bool(np.mean(dist2 <= self.chi2_val))


# ==========================================================
# Confidence Ellipse in CLR (Centered Log-Ratio) Space
# ==========================================================

class ConfidenceEllipseCLR(ConfidenceEllipseSimplex):
    r"""Confidence ellipse for prevalence estimates in CLR-transformed space.

    Applies the Centered Log-Ratio (CLR) transformation:

    .. math::
        T(π) = [\log(π_1/g(π)), ..., \log(π_n/g(π))], \\
        g(π) = (\prod_i π_i)^{1/n}

    A confidence ellipse is then built in the transformed space:

    .. math::
        CT_α(π) =
        \\begin{cases}
        1 & \\text{if } (T(π) - μ_{CLR})^T Σ^{-1} (T(π) - μ_{CLR}) \\le χ^2_{n-1}(1-α) \\\\
        0 & \\text{otherwise}
        \\end{cases}

    Parameters
    ----------
    prev_estims : array-like of shape (m, n)
        Bootstrap prevalence estimates.
    confidence_level : float, default=0.95
        Confidence level.

    Attributes
    ----------
    mean_ : ndarray of shape (n,)
        Mean vector in CLR space.
    precision_matrix : ndarray of shape (n, n)
        Inverse covariance matrix in CLR space.
    chi2_val : float
        Chi-squared threshold.

    Examples
    --------
    >>> X = np.random.dirichlet(np.ones(3), size=200)
    >>> clr = ConfidenceEllipseCLR(X, confidence_level=0.9)
    >>> clr.get_point_estimate().round(3)
    array([ 0.,  0., -0.])
    >>> clr.contains(np.array([0.4, 0.4, 0.2]))
    True

    References
    ----------
    [1] Moreo, A., & Salvati, N. (2025).
        *An Efficient Method for Deriving Confidence Intervals in Aggregative Quantification*.
        Section 3.3, Equation (3).
    """

    def _compute_region(self, eps=1e-6):
        x = self.prev_estims
        G = np.exp(np.mean(np.log(x + eps), axis=1, keepdims=True))
        x_clr = np.log((x + eps) / (G + eps))
        self.x_clr = x_clr
        cov_ = np.cov(x_clr, rowvar=False, ddof=1)
        try:
            self.precision_matrix = np.linalg.inv(cov_)
        except np.linalg.LinAlgError:
            self.precision_matrix = None

        dim = x_clr.shape[-1]
        ddof = dim - 1
        self.chi2_val = chi2.ppf(self.confidence_level, ddof)
        self.mean_ = np.mean(x_clr, axis=0)
        
    def get_point_estimate(self):
        Gp = np.exp(np.mean(np.log(self.prev_estims + 1e-6), axis=1, keepdims=True))
        x_clr = np.log((self.prev_estims + 1e-6) / (Gp + 1e-6))
        return np.mean(x_clr, axis=0)

    def contains(self, point, eps=1e-6):
        if self.precision_matrix is None:
            return False
        Gp = np.exp(np.mean(np.log(point + eps)))
        point_clr = np.log((point + eps) / (Gp + eps))
        diff = point_clr - self.mean_
        dist2 = diff.T @ self.precision_matrix @ diff
        return dist2 <= self.chi2_val



# ==========================================================
#   Factory Method for Confidence Regions
# ==========================================================

def construct_confidence_region(prev_estims, confidence_level=0.95, method="intervals"):
    method = method.lower()
    if method == "intervals":
        return ConfidenceInterval(prev_estims, confidence_level)
    elif method == "ellipse":
        return ConfidenceEllipseSimplex(prev_estims, confidence_level)
    elif method in ("elipse-clr", "ellipse-clr", "clr"):
        return ConfidenceEllipseCLR(prev_estims, confidence_level)
    else:
        raise NotImplementedError(f"Método '{method}' desconhecido.")
