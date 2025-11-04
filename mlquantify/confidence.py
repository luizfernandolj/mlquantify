import numpy as np
from scipy.stats import chi2


class BaseConfidenceRegion:
    """Classe base para regiões de confiança."""
    def __init__(self, prev_estims, confidence_level=0.95):
        self.prev_estims = np.asarray(prev_estims)
        self.confidence_level = confidence_level
        self._compute_region()

    def _compute_region(self):
        raise NotImplementedError("Subclasses devem implementar este método.")

    def get_region(self):
        """Retorna os parâmetros da região de confiança."""
        raise NotImplementedError

    def get_point_estimate(self):
        """Retorna a estimativa pontual da prevalência."""
        raise NotImplementedError

    def contains(self, point):
        """Verifica se um ponto pertence à região."""
        raise NotImplementedError


# ==========================================================
# Intervalos (por percentis)
# ==========================================================

class ConfidenceInterval(BaseConfidenceRegion):
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
        within_intervals = np.logical_and(self.I_low <= point, point <= self.I_high)
        within_all_intervals = np.all(within_intervals, axis=-1, keepdims=True)
        return within_all_intervals


# ==========================================================
# Elipse no simplex
# ==========================================================

class ConfidenceEllipseSimplex(BaseConfidenceRegion):
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
        
        if dist2.ndim == 2:
            dist2 = np.diag(dist2)
        
        within_ellipse = (dist2 <= self.chi2_val)
        
        if isinstance(within_ellipse, np.ndarray):
            within_ellipse = np.mean(within_ellipse)
            
        within_ellipse = bool(within_ellipse * 1)
            
        return within_ellipse


# ==========================================================
# Elipse em CLR (Centered Log-Ratio)
# ==========================================================

class ConfidenceEllipseCLR(ConfidenceEllipseSimplex):
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
        
    def get_region(self):
        return self.mean_, self.precision_matrix, self.chi2_val
    
    def get_point_estimate(self):
        Gp = np.exp(np.mean(np.log(self.prev_estims + 1e-6), axis=1, keepdims=True))
        x_clr = np.log((self.prev_estims + 1e-6) / (Gp + 1e-6))
        mean_clr = np.mean(x_clr, axis=0)
        return mean_clr

    def contains(self, point, eps=1e-6):
        if self.precision_matrix is None:
            return False
        Gp = np.exp(np.mean(np.log(point + eps)))
        point_clr = np.log((point + eps) / (Gp + eps))
        diff = point_clr - self.mean_
        dist2 = diff.T @ self.precision_matrix @ diff
        return dist2 <= self.chi2_val


# ==========================================================
# Fábrica prática para criar regiões
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
