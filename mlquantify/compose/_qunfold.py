import numpy as np

from mlquantify.base import BaseQuantifier


class ComposeQuantifier(BaseQuantifier):
    r"""
    Generic quantification method based on constrained regression using the
    QUnfold framework.

    .. dropdown:: Mathematical formulation

        This method estimates class prevalences by solving the following problem:

        .. math::

            q \approx M \pi

        where:

        - :math:`q` is the representation of the unlabeled test data,
        - :math:`M` is the class-conditional representation matrix estimated from training data,
        - :math:`\pi` is the vector of class prevalences to be estimated.

        The estimation is performed by minimizing a divergence or loss function
        between the observed representation :math:`q` and the expected representation
        :math:`M \pi`:

        .. math::

            \hat{\pi} = \arg\min_{\pi} \; D(q, M\pi)

        subject to:

        .. math::

            \pi_k \ge 0, \quad \sum_k \pi_k = 1

        The behavior of the method is fully determined by:

        - a representation :math:`f(x)` that maps data into a feature space,
        - a distance/loss function :math:`D(\cdot, \cdot)`,
        - an optimization procedure over the probability simplex.

    This implementation wraps the `qunfold` backend while allowing the use of:

    - native `qunfold` representations and losses,
    - custom representations implemented in `mlquantify`,
    - arbitrary distance functions, such as Topsoe or Hellinger.

    Parameters
    ----------
    representation : object
        Representation object defining how to compute :math:`q` and :math:`M`.
        Can be either a `qunfold` representation or a custom implementation.

    loss : object or callable
        Loss or distance function :math:`D(p, q)`. Can be either a `qunfold`
        loss object or a callable accepting two distributions.

    solver : str or callable, optional
        Solver used for the constrained optimization problem. If not provided,
        the default solver from `qunfold` is used.

    seed : int, optional
        Random seed used by the `qunfold` backend for reproducible optimization.

    Notes
    -----
    This formulation unifies several quantification methods:

    - AC / CC: class-based representations
    - Prob / PCC: probability-based representations
    - HDy / DyS: histogram-based representations with divergence measures
    - HDx: feature-based histogram representations

    The method follows the constrained regression framework described in:

    .. math::

        y = X \hat{\pi}_F

    where different choices of representation and loss correspond to different
    quantification algorithms.

    Examples
    --------
    Using a class-based representation to implement an ACC-like quantifier:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from qunfold.sklearn import CVClassifier
    >>> from qunfold.methods.linear.losses import LeastSquaresLoss
    >>> from qunfold.methods.linear.representations import ClassRepresentation
    >>> from mlquantify.compose import ComposeQuantifier
    >>>
    >>> learner = LogisticRegression(max_iter=1000)
    >>> quantifier = ComposeQuantifier(
    ...     representation=ClassRepresentation(
    ...         CVClassifier(learner),
    ...         is_probabilistic=False,
    ...     ),
    ...     loss=LeastSquaresLoss(),
    ... )
    >>> quantifier.fit(X_train, y_train)
    >>> prevalences = quantifier.predict(X_test)

    Using a histogram-based representation with a custom Topsoe distance from
    ``mlquantify`` to implement a DyS-like quantifier:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from qunfold.sklearn import CVClassifier
    >>> from qunfold.methods.linear.representations import (
    ...     ClassRepresentation,
    ...     HistogramRepresentation,
    ... )
    >>> from mlquantify.compose import ComposeQuantifier
    >>> from mlquantify.metrics import topsoe_jax
    >>>
    >>> learner = LogisticRegression(max_iter=1000)
    >>> representation = HistogramRepresentation(
    ...     n_bins=8,
    ...     preprocessor=ClassRepresentation(
    ...         CVClassifier(learner),
    ...         is_probabilistic=True,
    ...     ),
    ...     unit_scale=False,
    ... )
    >>> quantifier = ComposeQuantifier(
    ...     representation=representation,
    ...     loss=topsoe_jax,
    ... )
    >>> quantifier.fit(X_train, y_train)
    >>> prevalences = quantifier.predict(X_test)

    References
    ----------
    .. [1] Firat, A. (2016). Unified Framework for Quantification.
    """

    def __init__(
        self,
        representation,
        loss,
        solver=None,
        seed=None,
    ):
        self.representation = representation
        self.loss = loss
        self.solver = solver
        self.seed = seed

    def fit(self, X, y, **fit_kwargs):
        from qunfold import LinearMethod

        rep = self._adapt_representation(self.representation)
        loss = self._adapt_loss(self.loss)

        kwargs = fit_kwargs.copy()
        kwargs["seed"] = self.seed

        if self.solver is not None:
            kwargs["solver"] = self.solver

        self.method_ = LinearMethod(loss, rep, **kwargs)

        self.method_.fit(X, y)

        return self

    def predict(self, X):
        representation = self.method_.representation

        q = representation.transform(X)
        M = self.method_.M
        return self.method_.solve(q, M, N=self._n_samples(X))

    def _n_samples(self, X):
        if hasattr(X, "shape"):
            return X.shape[0]

        if isinstance(X, dict):
            for key in ("features", "scores", "X", "timestamps", "T", "t"):
                if key in X and X[key] is not None:
                    return len(X[key])

        if isinstance(X, (tuple, list)) and len(X) == 2:
            return len(X[1])

        return len(X)

    # --------------------------
    # ADAPTERS
    # --------------------------

    def _adapt_representation(self, representation):
        """
        If it's already a qunfold representation, return as is.
        Otherwise, wrap mlquantify representation.
        """

        if hasattr(representation, "fit_transform"):
            return representation  # assume qunfold-compatible

        return _RepresentationAdapter(representation)

    def _adapt_loss(self, loss):
        """
        If it's already a qunfold loss, return as is.
        Otherwise wrap callable distance.
        """

        if hasattr(loss, "instantiate") or hasattr(loss, "gradient"):
            return loss  # assume qunfold loss

        return _LossAdapter(loss)


# --------------------------
# REPRESENTATION ADAPTER
# --------------------------

class _RepresentationAdapter:
    """
    Adapts mlquantify representation to qunfold interface.
    """

    def __init__(self, representation):
        self.rep = representation

    def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
        M = self.rep.fit_transform(X, y, classes=np.unique(y))
        return M

    def transform(self, X, sample_weight=None, average=True):
        q = self.rep.transform(X, classes=self.rep.classes_)
        return q

    @property
    def n_output_features(self):
        return self.rep.n_features_


# --------------------------
# LOSS ADAPTER
# --------------------------

class _LossAdapter:
    """
    Adapts a mlquantify distance function to the qunfold loss interface.

    Expected mlquantify function signature:

        loss(p, q)

    QUnfold expects:

        loss.instantiate(q, M, N) -> callable pi -> loss_value
    """

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def instantiate(self, q, M, N=None):
        import jax.numpy as jnp

        q = jnp.asarray(q)
        M = jnp.asarray(M)

        def loss(prevalences):
            expected = jnp.dot(M, prevalences)
            return self.loss_fn(q, expected)

        return loss
