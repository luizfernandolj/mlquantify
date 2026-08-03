import numpy as np
from scipy.optimize import minimize
from sklearn.utils import check_X_y, check_array


# --------------------------------------------------------------------------
# Multivariate losses computed from the binary contingency table.
#
# Every function receives broadcastable grids ``a`` (true positives, column)
# and ``b`` (false positives, row) plus the class totals ``n_pos`` / ``n_neg``
# and returns the loss grid on the 0-100 scale used by svmperf, so that
# Algorithm 2 (the most-violated-constraint search) can evaluate all
# O(n^2) contingency tables in one vectorized pass.
# --------------------------------------------------------------------------

def _loss_error(a, b, n_pos, n_neg):
    """Error rate, ``100 * (FP + FN) / S`` (recovers the standard SVM)."""
    c = n_pos - a
    return 100.0 * (b + c) / (n_pos + n_neg)


def _loss_f1(a, b, n_pos, n_neg):
    """F1 loss, ``100 * (1 - F1)``."""
    c = n_pos - a
    denominator = 2 * a + b + c
    f1 = np.where(denominator > 0, 2 * a / np.maximum(denominator, 1), 0.0)
    return 100.0 * (1.0 - f1)


def _make_loss_q(beta):
    def _loss_q(a, b, n_pos, n_neg):
        """Q-measure loss, ``100 * (1 - Q_beta)`` (Barranquero et al., 2015)."""
        c = n_pos - a
        recall = a / n_pos
        nas = 1.0 - np.abs(c - b) / max(n_pos, n_neg)
        denominator = beta ** 2 * recall + nas
        q = np.where(
            denominator > 0,
            (1 + beta ** 2) * recall * nas / np.where(denominator > 0, denominator, 1.0),
            0.0,
        )
        return 100.0 * (1.0 - q)
    return _loss_q


def _kld_grid(a, b, n_pos, n_neg):
    total = n_pos + n_neg
    # Half-count (Forman) backoff keeps the log finite when a class is
    # predicted for no document at all.
    eps = 0.5 / total
    p_pos, p_neg = n_pos / total, n_neg / total
    pred_pos = (a + b + eps) / (total + 2 * eps)
    pred_neg = 1.0 - pred_pos
    return (
        p_pos * np.log(p_pos / pred_pos)
        + p_neg * np.log(p_neg / pred_neg)
    )


def _loss_kld(a, b, n_pos, n_neg):
    """Kullback-Leibler divergence loss (Esuli & Sebastiani's SVM(KLD))."""
    return 100.0 * _kld_grid(a, b, n_pos, n_neg)


def _loss_nkld(a, b, n_pos, n_neg):
    """Normalised KLD loss (Esuli & Sebastiani's SVM(NKLD))."""
    euler = np.exp(_kld_grid(a, b, n_pos, n_neg))
    return 100.0 * (2.0 * euler / (1.0 + euler) - 1.0)


def _loss_ae(a, b, n_pos, n_neg):
    """Absolute (prevalence) error loss, ``100 * |FP - FN| / S``."""
    c = n_pos - a
    return 100.0 * np.abs(b - c) / (n_pos + n_neg)


def _loss_rae(a, b, n_pos, n_neg):
    """Relative absolute error loss, with half-count smoothing."""
    total = n_pos + n_neg
    eps = 0.5 / total
    c = n_pos - a
    p_pos, p_neg = n_pos / total, n_neg / total
    error = np.abs(b - c) / total
    p_pos_s = (p_pos + eps) / (1 + 2 * eps)
    p_neg_s = (p_neg + eps) / (1 + 2 * eps)
    return 100.0 * (error / p_pos_s + error / p_neg_s) / 2.0


_LOSSES = {
    "error": _loss_error,
    "f1": _loss_f1,
    "kld": _loss_kld,
    "nkld": _loss_nkld,
    "ae": _loss_ae,
    "rae": _loss_rae,
}


class MultivariateLossSVM:
    r"""Linear SVM trained to minimize a multivariate (sample-based) loss.

    A pure-Python reimplementation of Joachims' :math:`SVM^{\Delta}_{multi}`
    (the algorithm behind ``svmperf``): instead of the example-wise hinge
    loss of a standard SVM, it optimizes a convex upper bound on any loss
    :math:`\Delta` computed from the **contingency table of the whole
    training sample**,

    .. math::

        \min_{w,\,\xi \ge 0}\ \tfrac{1}{2}\|w\|^2 + C\,\xi
        \quad \text{s.t.} \quad
        w^\top[\Psi(\bar{x},\bar{y}) - \Psi(\bar{x},\bar{y}')] \ge
        \Delta(\bar{y}',\bar{y}) - \xi \;\; \forall \bar{y}'

    with :math:`\Psi(\bar{x},\bar{y}') = \sum_i y_i' x_i`. The exponentially
    many constraints are handled by the cutting-plane Algorithm 1 of
    Joachims (2005): at each iteration the most violated constraint is found
    in a single vectorized pass over all :math:`O(n^2)` contingency tables
    (Algorithm 2) and a small quadratic program over the working set is
    re-solved.

    This is the learner behind the ELM quantifiers: with a
    quantification-oriented loss such as the Q-measure, the learned
    hyperplane balances false positives against false negatives on the
    training sample, making its Classify-and-Count estimates reliable
    (Barranquero et al., 2015).

    Parameters
    ----------
    loss : {'q', 'kld', 'nkld', 'ae', 'rae', 'error', 'f1'} or callable, default='q'
        Multivariate loss to minimize, on svmperf's 0-100 scale:

        - ``'q'`` : ``100*(1 - Q_beta)``, the Q-measure combining recall and
          the Normalized Absolute Score (Barranquero et al., 2015).
        - ``'kld'`` / ``'nkld'`` : (normalised) Kullback-Leibler divergence
          between true and predicted prevalences (Esuli & Sebastiani, 2015).
        - ``'ae'`` / ``'rae'`` : (relative) absolute prevalence error.
        - ``'error'`` : error rate — recovers the standard linear SVM
          (Joachims, 2005, Theorem 3).
        - ``'f1'`` : ``100*(1 - F1)``.

        A callable receives broadcastable grids ``(a, b, n_pos, n_neg)``
        (true positives, false positives, class totals) and must return the
        loss grid.
    C : float, default=1.0
        Trade-off between the regularizer and the (0-100 scaled) slack.
    beta : float, default=1.0
        Q-measure trade-off between classification (recall) and
        quantification (NAS) performance; used only by ``loss='q'``.
    tol : float, default=0.1
        Stopping tolerance :math:`\epsilon` of the cutting-plane algorithm
        (on the 0-100 loss scale).
    max_iter : int, default=300
        Maximum number of cutting-plane iterations (constraints added).

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The two class labels; the second is treated as the positive class.
    coef_ : ndarray of shape (n_features,)
        Weight vector of the learned hyperplane (no intercept, as in the
        multivariate formulation).
    n_iter_ : int
        Number of cutting-plane iterations performed.

    Notes
    -----
    The formulation has no bias term (the joint feature map does not include
    one); append a constant feature if an intercept is needed. The
    most-violated-constraint search materialises a
    ``(n_pos+1, n_neg+1)`` grid per iteration, so memory grows quadratically
    with the training-set size.

    Examples
    --------
    >>> import numpy as np
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> clf = MultivariateLossSVM(loss='q').fit(X, y)
    >>> labels = clf.predict(X)

    References
    ----------
    Joachims, T. (2005). A Support Vector Method for Multivariate
    Performance Measures. *ICML*, pp. 377-384.

    Joachims, T. (2006). Training Linear SVMs in Linear Time. *KDD*.

    Barranquero, J., Díez, J., & del Coz, J. J. (2015).
    Quantification-oriented learning based on reliable classifiers.
    *Pattern Recognition*, 48(2), 591-604.
    """

    def __init__(self, loss="q", C=1.0, beta=1.0, tol=0.1, max_iter=300):
        self.loss = loss
        self.C = C
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter

    def get_params(self, deep=True):
        return {
            "loss": self.loss,
            "C": self.C,
            "beta": self.beta,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _resolve_loss(self):
        if callable(self.loss):
            return self.loss
        if self.loss == "q":
            return _make_loss_q(self.beta)
        if self.loss in _LOSSES:
            return _LOSSES[self.loss]
        raise ValueError(
            f"loss must be one of {['q', *_LOSSES]} or a callable, got {self.loss!r}."
        )

    def _most_violated(self, w, X_pos, X_neg, loss_grid, a_col, b_row):
        """Vectorized Algorithm 2: argmax over all contingency tables of
        ``loss(a, b) + w^T Psi(x, y')``, returning the constraint's
        ``(delta_psi, loss_value)``."""
        scores_pos = X_pos @ w
        scores_neg = X_neg @ w

        order_pos = np.argsort(-scores_pos)
        order_neg = np.argsort(-scores_neg)
        sorted_pos_scores = scores_pos[order_pos]
        sorted_neg_scores = scores_neg[order_neg]

        # Labeling the top-a positives / top-b negatives as +1 maximises the
        # score term for a fixed contingency table.
        cum_pos = np.concatenate([[0.0], np.cumsum(sorted_pos_scores)])
        cum_neg = np.concatenate([[0.0], np.cumsum(sorted_neg_scores)])
        score_term = (
            (2.0 * cum_pos - cum_pos[-1])[:, None]
            + (2.0 * cum_neg - cum_neg[-1])[None, :]
        )

        objective = loss_grid + score_term
        best = np.unravel_index(np.argmax(objective), objective.shape)
        a_star, b_star = int(best[0]), int(best[1])

        # delta_psi = Psi(y) - Psi(y'): the (n_pos - a) lowest-scored
        # positives are flipped to -1 (+2 x_i each) and the top-b negatives
        # to +1 (-2 x_i each).
        flipped_pos = X_pos[order_pos[a_star:]].sum(axis=0) if a_star < len(scores_pos) else 0.0
        flipped_neg = X_neg[order_neg[:b_star]].sum(axis=0) if b_star > 0 else 0.0
        delta_psi = 2.0 * flipped_pos - 2.0 * flipped_neg
        return np.asarray(delta_psi, dtype=float), float(loss_grid[a_star, b_star])

    def _solve_working_set(self, gram, losses, alpha0):
        """Dual of the working-set QP: maximize ``alpha^T losses - 1/2
        alpha^T G alpha`` subject to ``alpha >= 0`` and ``sum(alpha) <= C``."""
        n = len(losses)

        def objective(alpha):
            return 0.5 * alpha @ gram @ alpha - alpha @ losses

        def gradient(alpha):
            return gram @ alpha - losses

        result = minimize(
            objective,
            x0=alpha0,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, self.C)] * n,
            constraints={"type": "ineq", "fun": lambda a: self.C - a.sum(),
                         "jac": lambda a: -np.ones(n)},
            options={"maxiter": 200, "ftol": 1e-12},
        )
        return np.clip(result.x, 0.0, None)

    def fit(self, X, y):
        """Train the multivariate SVM with the cutting-plane algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        self : MultivariateLossSVM
            The fitted classifier.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"MultivariateLossSVM is a binary classifier; got "
                f"{len(self.classes_)} classes."
            )
        loss_fn = self._resolve_loss()

        positive = y == self.classes_[1]
        X_pos, X_neg = X[positive], X[~positive]
        n_pos, n_neg = len(X_pos), len(X_neg)

        # Loss grid over all contingency tables (constant across iterations).
        a_col = np.arange(n_pos + 1, dtype=float)[:, None]
        b_row = np.arange(n_neg + 1, dtype=float)[None, :]
        loss_grid = np.asarray(loss_fn(a_col, b_row, n_pos, n_neg), dtype=float)
        loss_grid[n_pos, 0] = 0.0  # the true labeling has zero loss

        n_features = X.shape[1]
        w = np.zeros(n_features)
        constraints = []      # delta_psi vectors
        deltas = []           # loss values
        gram = np.zeros((0, 0))
        alpha = np.zeros(0)

        self.n_iter_ = 0
        for _ in range(self.max_iter):
            delta_psi, delta = self._most_violated(
                w, X_pos, X_neg, loss_grid, a_col, b_row
            )
            violation = delta - w @ delta_psi
            if constraints:
                xi = max(
                    0.0,
                    max(d - w @ dp for d, dp in zip(deltas, constraints)),
                )
            else:
                xi = 0.0
            if violation <= xi + self.tol:
                break

            # Grow the Gram matrix and re-solve the dual over the working set.
            cross = np.array([dp @ delta_psi for dp in constraints])
            constraints.append(delta_psi)
            deltas.append(delta)
            gram = np.block([
                [gram, cross[:, None]],
                [cross[None, :], np.array([[delta_psi @ delta_psi]])],
            ]) if len(cross) else np.array([[delta_psi @ delta_psi]])
            alpha = self._solve_working_set(
                gram, np.asarray(deltas), np.append(alpha, 0.0)
            )
            w = np.sum(alpha[:, None] * np.asarray(constraints), axis=0)
            self.n_iter_ += 1

            # Prune inactive constraints to keep the working set small.
            if len(alpha) > 50:
                keep = alpha > 1e-10
                if not keep.all():
                    constraints = [dp for dp, k in zip(constraints, keep) if k]
                    deltas = [d for d, k in zip(deltas, keep) if k]
                    gram = gram[np.ix_(keep, keep)]
                    alpha = alpha[keep]

        self.coef_ = w

        # Platt-style sigmoid on the training margins so predict_proba is
        # available to soft aggregative quantifiers and calibration.
        margins = X @ w
        from sklearn.linear_model import LogisticRegression

        self._sigmoid = LogisticRegression(max_iter=1000)
        self._sigmoid.fit(margins[:, None], positive.astype(int))
        return self

    def decision_function(self, X):
        """Signed distance to the hyperplane (no intercept)."""
        X = check_array(X)
        return X @ self.coef_

    def predict(self, X):
        """Predict binary class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def predict_proba(self, X):
        """Posterior probabilities from a sigmoid fitted on the margins.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Probability of each class.
        """
        margins = self.decision_function(X)
        return self._sigmoid.predict_proba(margins[:, None])
