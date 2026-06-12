import numpy as np

from mlquantify.base_aggregative import (
    AggregativeMixin,
    SoftPredictionMixin,
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.matching._base import BaseMatchingQuantifier
from mlquantify.losses import get_loss
from mlquantify.representations import HistogramRepresentation
from mlquantify.multiclass import binary_quantifier
from mlquantify.utils._constraints import Options
from mlquantify.utils._validation import validate_data
from mlquantify.solvers import minimize_prevalence, minimize_prevalence_blocks


@binary_quantifier(strategy_attr="strategy")
class MatchingHistogramQuantifier(BaseMatchingQuantifier):
    r"""Abstract base class for histogram-based distribution matching.

    Subclasses learn class-conditional histogram representations from training
    data and estimate the test prevalence by finding the mixture of those
    histograms that best matches the test histogram.

    This is a **binary-only** method. When applied to multiclass problems,
    a one-vs-rest (OvR) strategy is applied automatically.

    Parameters
    ----------
    bins_size : int or array-like
        Number of histogram bins, or array of bin counts to sweep over.
    distance : str, default='hellinger'
        Distance function used to compare histograms.
    solver : str, default='auto'
        Optimization solver; ``'auto'`` selects based on the distance.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.
    histogram_features : int or None, default=None
        Number of score columns used to build histograms.
    bin_strategy : str or None, default=None
        Aggregation strategy across bin sizes (``'median'`` or ``'mean'``).

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.matching._histogram import MatchingHistogramQuantifier
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> class MyHistQ(MatchingHistogramQuantifier):
    ...     def __init__(self):
    ...         super().__init__(bins_size=10)
    ...     def fit(self, X, y):
    ...         self.classes_ = np.unique(y)
    ...         return self._fit(X, y)
    ...     def predict(self, X):
    ...         return self._predict(X)
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> MyHistQ().fit(X, y).predict(X)
    {0: 0.5, 1: 0.5}
    """

    def __init__(
        self,
        bins_size,
        distance="hellinger",
        solver="auto",
        strategy="ovr",
        histogram_features=None,
        bin_strategy=None,
        laplace_smoothing=False,
    ):
        if bins_size is None:
            bins_size = np.append(np.linspace(2, 20, 10), 30).astype(int)

        self.bins_size = bins_size
        self.distance = distance
        self.loss_function = get_loss(loss=distance, normalize=True)
        self.solver = solver
        self.strategy = strategy
        self.histogram_features = histogram_features
        self.bin_strategy = bin_strategy
        self.laplace_smoothing = laplace_smoothing
        super().__init__(
            representation=HistogramRepresentation(
                bins=bins_size,
                mode="histogram",
                features=histogram_features,
                partition_blocks=bin_strategy in {"median", "mean"},
                laplace_smoothing=laplace_smoothing,
            ),
            normalize=True,
        )

    def _solve_prevalence(self, test_representation, train_representations):
        solver = self._resolve_solver()

        if self.bin_strategy in {"median", "mean"}:
            return minimize_prevalence_blocks(
                objective_factory=self._block_objective_factory,
                test_representation=test_representation,
                train_representations=train_representations,
                block_slices=self._get_block_slices(),
                n_classes=2,
                solver=solver,
                aggregate=self.bin_strategy,
            )

        return minimize_prevalence(
            objective=self._make_objective(
                test_representation,
                train_representations,
            ),
            n_classes=2,
            solver=solver,
        )

    def _resolve_solver(self):
        if self.solver != "auto":
            return self.solver

        if self.distance in {"hellinger", "topsoe", "probsymm"}:
            return "ternary"

        return "grid"

    def _make_objective(self, test_representation, train_representations):
        train_representations = np.asarray(train_representations)

        def objective(alpha):
            prevalence = np.asarray([1.0 - alpha, alpha])
            mix_representation = self._mixture(
                train_representations,
                prevalence,
            )
            return self.loss_function(
                mix_representation,
                test_representation,
            )

        return objective

    def _block_objective_factory(self, test_block, train_block):
        return self._make_objective(
            test_representation=test_block,
            train_representations=train_block,
        )

    def _get_block_slices(self):
        if not hasattr(self.representation, "block_slices_"):
            raise AttributeError(
                "HistogramRepresentation must define block_slices_ "
                "to use bin_strategy='median' or 'mean'."
            )

        return self.representation.block_slices_


@binary_quantifier(strategy_attr="strategy")
class DyS(SoftPredictionMixin, AggregativeMixin, MatchingHistogramQuantifier):
    r"""Distribution y-Similarity (DyS) quantifier.

    Targets prior probability shift. A general mixture-model (distribution
    matching) quantifier: it builds class-conditional score histograms from
    cross-validated classifier scores and searches for the positive-class
    mixture proportion whose mixed histogram minimises a chosen dissimilarity
    to the test score histogram. Generalises :class:`HDy` to any distance
    (Topsoe by default). Binary-only; requires soft scores; multiclass via
    one-vs-rest (OvR).

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    bins_size : int or array-like or None, default=None
        Histogram bin count(s) to sweep over; controls histogram resolution.
        Defaults to a logarithmic grid. Estimates over the bin counts are
        aggregated by their median.
    distance : str, default='topsoe'
        Dissimilarity minimised between the mixed and test histograms.

        - ``'topsoe'`` : symmetric information-theoretic distance; the best
          general performer (recommended).
        - ``'hellinger'`` : bounded sqrt-probability distance (the HDy choice).
        - ``'probsymm'`` : probabilistic symmetric chi-square distance.
        - ``'sqEuclidean'`` : squared Euclidean distance between bin vectors.
    solver : {'auto', 'ternary', 'grid', 'bounded'}, default='auto'
        Search strategy over the mixture proportion alpha.

        - ``'auto'`` : pick a sensible default (bounded scalar search).
        - ``'ternary'`` : interval-trisection search; fast on the near-unimodal objective.
        - ``'grid'`` : exhaustive search over an evenly-spaced grid of alpha.
        - ``'bounded'`` : scipy bounded scalar minimiser.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

        - ``'ovr'`` : one-vs-rest, one binary quantifier per class.
        - ``'ovo'`` : one-vs-one, one binary quantifier per class pair.
    cv : int or None, default=None
        Cross-validation folds for computing training scores.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.
    bin_strategy : {'median', 'mean', None}, default=None
        How per-bin-block prevalence estimates are aggregated when the
        histogram is partitioned into blocks; ``None`` disables block
        partitioning.
    laplace_smoothing : bool, default=False
        If ``True``, add a small count to every bin before normalising,
        stabilising ratio/log-based distances when some bins are empty.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    DyS generalises :class:`HDy`: with ``distance='hellinger'`` the two
    coincide. Best bin counts are typically small (around 20 or fewer); the
    median over swept bin counts makes the estimate robust to bin
    mis-specification.

    See Also
    --------
    HDy : Hellinger-distance special case.
    SORD : Bin-free ordinal-distance member of the same framework.
    SMM : Mean-matching member of the same framework.

    Examples
    --------
    >>> from mlquantify.matching import DyS
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = DyS(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}
    >>> # call aggregate with pre-computed scores
    >>> import numpy as np
    >>> train_scores = np.random.rand(200)
    >>> test_scores = np.random.rand(100)
    >>> y_train = np.random.randint(0, 2, 200)
    >>> q.aggregate(test_scores, train_scores, y_train)
    {0: 0.48, 1: 0.52}

    References
    ----------
    .. dropdown:: References

        .. [1] Maletzke, A., dos Reis, D., Cherman, E., & Batista, G. (2019).
               DyS: A Framework for Mixture Models in Quantification.
               *AAAI*, pp. 4552–4560.
    """

    def __init__(
        self,
        estimator=None,
        bins_size=None,
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
        distance="topsoe",
        solver="auto",
        bin_strategy=None,
        laplace_smoothing=False,
    ):
        super().__init__(
            bins_size=bins_size,
            distance=distance,
            solver=solver,
            strategy=strategy,
            histogram_features=1,
            bin_strategy=bin_strategy,
            laplace_smoothing=laplace_smoothing,
        )
        self.estimator = estimator
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        estimator_fitted=False,
        sample_weight=None,
        cv_prediction="refit",
    ):

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        X, y = self._fit_estimator_predictions(
            X,
            y,
            estimator_fitted=estimator_fitted,
            cv_prediction=cv_prediction,
        )

        return self._fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        X = validate_data(self, X, ensure_2d=True)
        test_scores = self._predict_estimator(X)
        return self._predict(test_scores)

    def aggregate(self, test_scores, train_scores, y_train):
        if not getattr(self, "_precomputed", False):
            self._fit(train_scores, y_train)
        return self._predict(test_scores)


@binary_quantifier(strategy_attr="strategy")
class HDy(SoftPredictionMixin, AggregativeMixin, MatchingHistogramQuantifier):
    r"""Hellinger Distance y (HDy) quantifier.

    Targets prior probability shift. The original mixture-model quantifier on
    classifier scores: it sweeps several histogram bin counts, finds the
    mixture of class-conditional score histograms that minimises the Hellinger
    distance to the test histogram at each bin count, and returns the median
    estimate. Binary-only; requires soft scores; multiclass via one-vs-rest.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    bins_size : array-like or None, default=None
        Array of bin counts to sweep; controls histogram resolution. Defaults
        to a range from 10 to 110; the per-bin estimates are aggregated by the
        median.
    distance : str, default='hellinger'
        Histogram dissimilarity minimised (HDy uses ``'hellinger'``; see
        :class:`DyS` for other choices such as ``'topsoe'``).
    solver : {'grid', 'ternary', 'bounded', 'auto'}, default='grid'
        Search strategy over the mixture proportion alpha.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

        - ``'ovr'`` : one-vs-rest, one binary quantifier per class.
        - ``'ovo'`` : one-vs-one, one binary quantifier per class pair.
    cv : int or None, default=None
        Cross-validation folds for computing training scores.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.
    bin_strategy : {'median', 'mean', None}, default='median'
        Aggregation of per-bin-block prevalence estimates across the sweep.
    laplace_smoothing : bool, default=False
        If ``True``, add a small count to every bin before normalising,
        stabilising the Hellinger distance when some bins are empty.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    HDy is the Hellinger instance of :class:`DyS`; switching to the Topsoe
    distance (via :class:`DyS`) often lowers error. The original 10–110 bin
    range is wide — small bin counts frequently suffice.

    See Also
    --------
    DyS : Generalisation to any histogram distance.
    HDx : Classifier-free, feature-space variant.
    GHDy : Multiclass constrained-regression variant.

    Examples
    --------
    >>> from mlquantify.matching import HDy
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = HDy(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}
    >>> # call aggregate with pre-computed scores
    >>> import numpy as np
    >>> train_scores = np.random.rand(200)
    >>> test_scores = np.random.rand(100)
    >>> y_train = np.random.randint(0, 2, 200)
    >>> q.aggregate(test_scores, train_scores, y_train)
    {0: 0.48, 1: 0.52}

    References
    ----------
    .. dropdown:: References

        .. [1] González-Castro, V., Alaiz-Rodriguez, R., & Alegre, E. (2013).
               Class Distribution Estimation Based on the Hellinger Distance.
               *Information Sciences*, 218, 146–164.
    """

    def __init__(
        self,
        estimator=None,
        bins_size=None,
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
        distance="hellinger",
        solver="grid",
        bin_strategy="median",
        laplace_smoothing=False,
    ):
        if bins_size is None:
            bins_size = np.linspace(10, 110, 11, dtype=int)

        super().__init__(
            bins_size=bins_size,
            distance=distance,
            solver=solver,
            strategy=strategy,
            histogram_features=1,
            bin_strategy=bin_strategy,
            laplace_smoothing=laplace_smoothing,
        )
        self.estimator = estimator
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        estimator_fitted=False,
        sample_weight=None,
        cv_prediction="refit",
    ):

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        X, y = self._fit_estimator_predictions(
            X,
            y,
            estimator_fitted=estimator_fitted,
            cv_prediction=cv_prediction,
        )

        return self._fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        X = validate_data(self, X, ensure_2d=True)
        test_scores = self._predict_estimator(X)
        return self._predict(test_scores)

    def aggregate(self, test_scores, train_scores, y_train):
        if not getattr(self, "_precomputed", False):
            self._fit(train_scores, y_train)
        return self._predict(test_scores)


@binary_quantifier(strategy_attr="strategy")
class HDx(MatchingHistogramQuantifier):
    r"""Hellinger Distance x (HDx) quantifier.

    Targets prior probability shift. The classifier-free mixture model: it
    builds a histogram for every input feature and selects the mixture
    proportion whose prevalence-mixed per-feature histograms minimise the
    Hellinger distance to the test histograms. Uses no scorer, only the raw
    features. Binary-only; multiclass via one-vs-rest.

    Parameters
    ----------
    bins_size : array-like or None, default=None
        Array of bin counts to sweep per feature; controls histogram
        resolution. Defaults to a range from 2 to 30. Per-feature and per-bin
        estimates are aggregated by the median.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

        - ``'ovr'`` : one-vs-rest, one binary quantifier per class.
        - ``'ovo'`` : one-vs-one, one binary quantifier per class pair.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    Because it skips the classifier, HDx avoids calibration issues but cannot
    exploit a good scorer, and degrades when there are many weak or correlated
    features.

    See Also
    --------
    HDy : Score-space variant using a classifier.
    GHDx : Multiclass constrained-regression variant.

    Examples
    --------
    >>> from mlquantify.matching import HDx
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = HDx().fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}

    References
    ----------
    .. dropdown:: References

        .. [1] González-Castro, V., Alaiz-Rodriguez, R., & Alegre, E. (2013).
               Class Distribution Estimation Based on the Hellinger Distance.
               *Information Sciences*, 218, 146–164.
    """

    _parameter_constraints = {
        "bins_size": ["array-like", None],
        "strategy": [Options(["ovr", "ovo"])],
    }

    def __init__(
        self,
        bins_size=None,
        strategy="ovr"
    ):
        if bins_size is None:
            bins_size = np.append(np.linspace(2, 20, 10), 30).astype(int)

        super().__init__(
            bins_size=bins_size,
            distance="hellinger",
            solver="ternary",
            strategy=strategy,
        )

    def fit(self, X, y, sample_weight=None):

        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        return self._fit(X, y, sample_weight)

    def predict(self, X):
        X = validate_data(self, X)
        return self._predict(X)
