import numpy as np

from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import AggregationMixin
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import (
    check_is_fitted,
    validate_data,
    validate_prevalences,
)


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

    solver_options : dict, optional
        Options passed to the backend optimization solver. If not provided,
        `qunfold` uses its own defaults.

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
    >>> from mlquantify.compose import (
    ...     ClassRepresentation,
    ...     ComposeQuantifier,
    ...     LeastSquaresLoss,
    ... )
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
        solver_options=None,
        seed=None,
    ):
        self.representation = representation
        self.loss = loss
        self.solver = solver
        self.solver_options = solver_options
        self.seed = seed

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, n_classes=None):
        from qunfold import LinearMethod

        X, y = validate_data(self, X, y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        n_classes = n_classes or len(self.classes_)

        rep = self._adapt_representation(self.representation)
        loss = self._adapt_loss(self.loss)

        kwargs = {}
        if self.solver is not None:
            kwargs["solver"] = self.solver
        if self.solver_options is not None:
            kwargs["solver_options"] = self.solver_options
        if self.seed is not None:
            kwargs["seed"] = self.seed

        self.method_ = LinearMethod(loss, rep, **kwargs)
        self.method_.M = rep.fit_transform(
            X,
            y_encoded,
            sample_weight=sample_weight,
            n_classes=n_classes,
        )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X)

        representation = self.method_.representation

        q = representation.transform(X)
        M = self.method_.M
        prevalences = self.method_.solve(q, M, N=self._n_samples(X))
        return validate_prevalences(self, prevalences, self.classes_)

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


class _QUnfoldClassifyAndCount(AggregationMixin, ComposeQuantifier):
    """Base class for QUnfold classify-and-count methods in mlquantify style."""

    is_probabilistic = False

    def __init__(
        self,
        learner=None,
        solver="trust-ncg",
        solver_options=None,
        seed=None,
    ):
        self.learner = learner
        self.solver = solver
        self.solver_options = solver_options
        self.seed = seed

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.has_estimator = True
        tags.estimator_function = "predict_proba"
        tags.estimator_type = "soft" if self.is_probabilistic else "crisp"
        tags.prediction_requirements.requires_train_proba = True
        tags.prediction_requirements.requires_train_labels = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        learner_fitted=False,
        cv=5,
        random_state=None,
        sample_weight=None,
        n_classes=None,
    ):
        self.representation = self._make_representation(
            learner_fitted=learner_fitted,
            cv=cv,
            random_state=random_state,
        )
        self.loss = self._make_loss()

        return ComposeQuantifier.fit(
            self,
            X,
            y,
            sample_weight=sample_weight,
            n_classes=n_classes,
        )

    def _make_loss(self):
        from mlquantify.compose._losses import LeastSquaresLoss

        return LeastSquaresLoss()

    def _make_representation(self, learner_fitted, cv, random_state):
        from mlquantify.compose.representations import ClassRepresentation

        return ClassRepresentation(
            self._make_classifier(cv=cv, random_state=random_state),
            is_probabilistic=self.is_probabilistic,
            fit_classifier=not learner_fitted,
        )

    def _make_classifier(self, cv, random_state):
        if hasattr(self.learner, "oob_score") and self.learner.oob_score:
            return self.learner

        from qunfold.sklearn import CVClassifier

        return CVClassifier(
            self.learner,
            n_estimators=cv,
            random_state=random_state,
        )


class ACC(_QUnfoldClassifyAndCount):
    r"""Adjusted Classify and Count using the QUnfold linear formulation.

    This class mirrors :class:`qunfold.ACC` while exposing the usual
    ``mlquantify`` estimator parameter name, ``learner``.
    """

    is_probabilistic = False


class PACC(_QUnfoldClassifyAndCount):
    r"""Probabilistic Adjusted Classify and Count using QUnfold.

    This class mirrors :class:`qunfold.PACC` while exposing the usual
    ``mlquantify`` estimator parameter name, ``learner``.
    """

    is_probabilistic = True


class AC(ACC):
    r"""Alias-style mlquantify name for :class:`ACC`."""


class PAC(PACC):
    r"""Alias-style mlquantify name for :class:`PACC`."""


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
