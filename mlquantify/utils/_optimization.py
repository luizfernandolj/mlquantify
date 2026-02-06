import numpy as np
from scipy.optimize import minimize


def _optimize_on_simplex(objective, n_classes, constraints=None):
    r"""Optimize an objective function over the probability simplex.
    
    Parameters
    ----------
    objective : callable
        Função a minimizar (R^n -> R).
    n_classes : int
        Dimensionalidade do simplex.
    extra_constraints : list of dict, optional
        Constraints adicionais (ex: [{'type': 'ineq', 'fun': lambda x: x}]).
    x0 : array-like, optional
        Ponto inicial (uniforme se None).
    priors : array-like, optional
        Fallback se falhar.
    
    Returns
    -------
    alpha_opt : ndarray (n_classes,)
        Pesos otimizados.
    min_loss : float
        Valor mínimo.
    """
    x0 = np.ones(n_classes) / n_classes
    
    if constraints is None:
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    else:
        constraints = constraints
    
    bounds = [(0, 1)] * n_classes
    
    res = minimize(objective, x0=x0, constraints=constraints, bounds=bounds, method='SLSQP')
    
    if res.success:
        alpha_opt = res.x
        return alpha_opt, res.fun
    else:
        return x0, objective(x0)