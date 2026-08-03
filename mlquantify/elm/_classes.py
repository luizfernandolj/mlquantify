from mlquantify.counting import CC
from mlquantify.elm._svmperf import MultivariateLossSVM
from mlquantify.multiclass import binary_quantifier
from mlquantify.utils._constraints import Interval, Options, CallableConstraint


_ELM_PARAMETER_CONSTRAINTS = {
    "loss": [
        Options(["q", "kld", "nkld", "ae", "rae", "error", "f1"]),
        CallableConstraint(),
    ],
    "C": [Interval(0.0, None, inclusive_left=False)],
    "beta": [Interval(0.0, None)],
    "tol": [Interval(0.0, None, inclusive_left=False)],
    "max_iter": [Interval(1, None, inclusive_right=False)],
    "strategy": [Options(["ovr", "ovo"])],
}


@binary_quantifier(strategy_attr="strategy")
class ELM(CC):
    r"""Explicit Loss Minimization (ELM) quantifier.

    Targets prior probability shift. An **aggregative** Classify-and-Count
    quantifier — it shares the standard ``fit`` / ``predict`` / ``aggregate``
    interface of :class:`~mlquantify.counting.CC` — whose classifier is a
    linear SVM *trained to minimize a quantification-oriented multivariate
    loss* (:class:`MultivariateLossSVM`, a pure-Python ``svmperf``): instead
    of correcting a generic classifier after the fact, the hyperplane itself
    is optimized so that plain counting of its predictions estimates the
    prevalences well. Like :class:`~mlquantify.neighbors.PWK`, it takes **no
    external estimator parameter**: the loss-optimized SVM is intrinsic to
    the method.

    Binary-only method. When applied to multiclass problems, a one-vs-rest
    (OvR) strategy is applied automatically.

    Parameters
    ----------
    loss : {'q', 'kld', 'nkld', 'ae', 'rae', 'error', 'f1'} or callable, default='q'
        Multivariate loss the underlying SVM minimizes (see
        :class:`MultivariateLossSVM`). ``'q'`` is the Q-measure of
        Barranquero et al. (2015); ``'kld'``/``'nkld'`` give Esuli &
        Sebastiani's SVM(KLD)/SVM(NKLD); ``'error'`` recovers a standard
        linear SVM.
    C : float, default=1.0
        Regularization/slack trade-off of the SVM.
    beta : float, default=1.0
        Q-measure trade-off between recall and NAS (``loss='q'`` only).
    tol : float, default=0.1
        Cutting-plane stopping tolerance (0-100 loss scale).
    max_iter : int, default=300
        Maximum cutting-plane iterations.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

    Attributes
    ----------
    estimator : MultivariateLossSVM
        The underlying loss-optimized SVM (built from the parameters above;
        not an argument).
    estimator_ : MultivariateLossSVM
        The fitted SVM.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    See Also
    --------
    SVMQ : ELM with the Q-measure loss (the headline method).
    SVMKLD, SVMNKLD, SVMAE, SVMRAE : ELM with the other quantification losses.
    MultivariateLossSVM : The underlying learner.

    Examples
    --------
    >>> from mlquantify.elm import ELM
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = ELM(loss='q').fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Barranquero, J., Díez, J., & del Coz, J. J. (2015).
               Quantification-oriented learning based on reliable
               classifiers. *Pattern Recognition*, 48(2), 591-604.
        .. [2] Joachims, T. (2005). A Support Vector Method for Multivariate
               Performance Measures. *ICML*, pp. 377-384.
        .. [3] Esuli, A., & Sebastiani, F. (2015). Optimizing Text
               Quantifiers for Multivariate Loss Functions.
               *ACM TKDD*, 9(4), 27.
    """

    _parameter_constraints = dict(_ELM_PARAMETER_CONSTRAINTS)

    def __init__(self, loss="q", C=1.0, beta=1.0, tol=0.1, max_iter=300,
                 strategy="ovr"):
        self.loss = loss
        self.C = C
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.strategy = strategy
        # The loss-optimized SVM is intrinsic to the method rather than
        # supplied by the user.
        super().__init__(estimator=self._make_svm())

    def _make_svm(self):
        return MultivariateLossSVM(
            loss=self.loss,
            C=self.C,
            beta=self.beta,
            tol=self.tol,
            max_iter=self.max_iter,
        )

    def fit(self, X, y, estimator_fitted=False, *args, **kwargs):
        r"""Fit the loss-optimized SVM on the provided data."""
        # Rebuild the intrinsic SVM so parameters changed via ``set_params``
        # (e.g. during grid search) take effect.
        if not estimator_fitted:
            self.estimator = self._make_svm()
        return super().fit(X, y, estimator_fitted=estimator_fitted, *args, **kwargs)


class SVMQ(ELM):
    r"""SVM(Q) quantifier (Barranquero et al., 2015).

    :class:`ELM` with the **Q-measure** loss: the harmonic-style combination

    .. math::

        Q_\beta = (1+\beta^2)\,
        \frac{\text{recall} \cdot \text{NAS}}
             {\beta^2\,\text{recall} + \text{NAS}},
        \qquad
        \text{NAS} = 1 - \frac{|FN - FP|}{\max(P, N)}

    which balances *classification reliability* (recall, keeping false
    negatives low) against *quantification performance* (NAS, keeping the
    errors compensated). Optimizing quantification alone admits degenerate
    solutions — any hyperplane with :math:`FP = FN` is a "perfect"
    quantifier on the training sample — and the recall hook selects the
    reliable ones among them.

    Binary-only method. When applied to multiclass problems, a one-vs-rest
    (OvR) strategy is applied automatically.

    Parameters
    ----------
    C : float, default=1.0
        Regularization/slack trade-off of the SVM.
    beta : float, default=1.0
        Trade-off between recall (:math:`\beta \to 0`) and NAS
        (:math:`\beta \to \infty`); the paper analyses 0.5, 1 and 2.
    tol : float, default=0.1
        Cutting-plane stopping tolerance.
    max_iter : int, default=300
        Maximum cutting-plane iterations.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

    Examples
    --------
    >>> from mlquantify.elm import SVMQ
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = SVMQ(beta=1.0).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Barranquero, J., Díez, J., & del Coz, J. J. (2015).
               Quantification-oriented learning based on reliable
               classifiers. *Pattern Recognition*, 48(2), 591-604.
    """

    def __init__(self, C=1.0, beta=1.0, tol=0.1, max_iter=300, strategy="ovr"):
        super().__init__(loss="q", C=C, beta=beta, tol=tol,
                         max_iter=max_iter, strategy=strategy)


class SVMKLD(ELM):
    r"""SVM(KLD) quantifier (Esuli & Sebastiani, 2015).

    :class:`ELM` with the Kullback-Leibler divergence between the true and
    predicted prevalences as the training loss.

    Binary-only method. When applied to multiclass problems, a one-vs-rest
    (OvR) strategy is applied automatically.

    References
    ----------
    .. dropdown:: References

        .. [1] Esuli, A., & Sebastiani, F. (2015). Optimizing Text
               Quantifiers for Multivariate Loss Functions.
               *ACM TKDD*, 9(4), 27.
    """

    def __init__(self, C=1.0, tol=0.1, max_iter=300, strategy="ovr"):
        super().__init__(loss="kld", C=C, tol=tol,
                         max_iter=max_iter, strategy=strategy)


class SVMNKLD(ELM):
    r"""SVM(NKLD) quantifier (Esuli & Sebastiani, 2015).

    :class:`ELM` with the normalised Kullback-Leibler divergence as the
    training loss.

    Binary-only method. When applied to multiclass problems, a one-vs-rest
    (OvR) strategy is applied automatically.

    References
    ----------
    .. dropdown:: References

        .. [1] Esuli, A., & Sebastiani, F. (2015). Optimizing Text
               Quantifiers for Multivariate Loss Functions.
               *ACM TKDD*, 9(4), 27.
    """

    def __init__(self, C=1.0, tol=0.1, max_iter=300, strategy="ovr"):
        super().__init__(loss="nkld", C=C, tol=tol,
                         max_iter=max_iter, strategy=strategy)


class SVMAE(ELM):
    r"""SVM(AE) quantifier.

    :class:`ELM` with the absolute prevalence error ``|FP - FN| / S`` as the
    training loss — a pure quantification objective.

    Binary-only method. When applied to multiclass problems, a one-vs-rest
    (OvR) strategy is applied automatically.

    References
    ----------
    .. dropdown:: References

        .. [1] Moreo, A., & Sebastiani, F. (2021). Tweet sentiment
               quantification: An experimental re-evaluation. *PLOS ONE*,
               16(9), e0263449.
    """

    def __init__(self, C=1.0, tol=0.1, max_iter=300, strategy="ovr"):
        super().__init__(loss="ae", C=C, tol=tol,
                         max_iter=max_iter, strategy=strategy)


class SVMRAE(ELM):
    r"""SVM(RAE) quantifier.

    :class:`ELM` with the (smoothed) relative absolute prevalence error as
    the training loss.

    Binary-only method. When applied to multiclass problems, a one-vs-rest
    (OvR) strategy is applied automatically.

    References
    ----------
    .. dropdown:: References

        .. [1] Moreo, A., & Sebastiani, F. (2021). Tweet sentiment
               quantification: An experimental re-evaluation. *PLOS ONE*,
               16(9), e0263449.
    """

    def __init__(self, C=1.0, tol=0.1, max_iter=300, strategy="ovr"):
        super().__init__(loss="rae", C=C, tol=tol,
                         max_iter=max_iter, strategy=strategy)
