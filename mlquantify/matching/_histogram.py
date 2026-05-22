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
    """

    def __init__(
        self,
        bins_size,
        distance="hellinger",
        solver="auto",
        strategy="ovr",
        histogram_features=None,
        bin_strategy=None,
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
        super().__init__(
            representation=HistogramRepresentation(
                bins=bins_size,
                mode="histogram",
                features=histogram_features,
                partition_blocks=bin_strategy in {"median", "mean"},
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
    r"""Distribution y-Similarity with histogram score matching."""

    def __init__(
        self,
        estimator=None,
        bins_size=None,
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            bins_size=bins_size,
            distance="hellinger",
            solver="grid",
            strategy=strategy,
            histogram_features=1,
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
    r"""Distribution y-Similarity with histogram score matching."""

    def __init__(
        self,
        estimator=None,
        bins_size=None,
        strategy="ovr",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        if bins_size is None:
            bins_size = np.linspace(10, 110, 11, dtype=int)

        super().__init__(
            bins_size=bins_size,
            distance="hellinger",
            solver="grid",
            strategy=strategy,
            histogram_features=1,
            bin_strategy="median",
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
    r"""Distribution y-Similarity with histogram score matching."""

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
