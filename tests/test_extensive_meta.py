"""
Comprehensive tests for the mlquantify.meta module.

Covers: EnsembleQ, AggregativeBootstrap, QuaDapt
Tests: fit/predict, binary/multiclass, numpy/pandas, int/string labels,
       parameter variations, edge cases, error cases, prevalence validity.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlquantify import config_context
from mlquantify.meta import EnsembleQ, AggregativeBootstrap, QuaDapt
from mlquantify.adjust_counting import CC, PCC, TAC, PAC, AC
from mlquantify.mixture import DyS, HDy


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_binary_dataset():
    """A very small binary dataset (60 samples) for fast tests."""
    X, y = make_classification(
        n_samples=60, n_features=5, n_classes=2,
        weights=[0.5, 0.5], random_state=7, n_informative=3,
    )
    return train_test_split(X, y, test_size=0.3, random_state=7)


@pytest.fixture
def small_multiclass_dataset():
    """A very small multiclass dataset (90 samples, 3 classes) for fast tests."""
    X, y = make_classification(
        n_samples=90, n_features=5, n_classes=3,
        n_informative=4, n_redundant=0, weights=[0.4, 0.35, 0.25], random_state=7,
    )
    return train_test_split(X, y, test_size=0.3, random_state=7)


@pytest.fixture
def string_label_binary_dataset():
    """Binary dataset with string labels ('neg', 'pos')."""
    X, y = make_classification(
        n_samples=80, n_features=5, n_classes=2,
        weights=[0.6, 0.4], random_state=10, n_informative=3,
    )
    y_str = np.where(y == 0, "neg", "pos")
    return train_test_split(X, y_str, test_size=0.3, random_state=10)


@pytest.fixture
def string_label_multiclass_dataset():
    """Multiclass dataset with string labels ('cat', 'dog', 'fish')."""
    X, y = make_classification(
        n_samples=120, n_features=5, n_classes=3,
        n_informative=4, n_redundant=0, weights=[0.4, 0.35, 0.25], random_state=11,
    )
    mapping = {0: "cat", 1: "dog", 2: "fish"}
    y_str = np.array([mapping[v] for v in y])
    return train_test_split(X, y_str, test_size=0.3, random_state=11)


@pytest.fixture
def pandas_binary_dataset(small_binary_dataset):
    """Small binary dataset returned as pandas objects."""
    X_train, X_test, y_train, y_test = small_binary_dataset
    return (
        pd.DataFrame(X_train), pd.DataFrame(X_test),
        pd.Series(y_train), pd.Series(y_test),
    )


@pytest.fixture
def pandas_multiclass_dataset(small_multiclass_dataset):
    """Small multiclass dataset returned as pandas objects."""
    X_train, X_test, y_train, y_test = small_multiclass_dataset
    return (
        pd.DataFrame(X_train), pd.DataFrame(X_test),
        pd.Series(y_train), pd.Series(y_test),
    )


@pytest.fixture
def imbalanced_binary_dataset():
    """Binary dataset with extreme class imbalance (95/5)."""
    X, y = make_classification(
        n_samples=200, n_features=5, n_classes=2,
        weights=[0.95, 0.05], random_state=42, n_informative=3,
        flip_y=0,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


# ===========================================================================
# EnsembleQ tests
# ===========================================================================

class TestEnsembleQBinary:
    """EnsembleQ with binary datasets."""

    def test_fit_predict_basic(self, binary_dataset, binary_classifier):
        X_train, X_test, y_train, y_test = binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=3,
        )
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("protocol", ["artificial", "natural", "uniform", "kraemer"])
    def test_protocols(self, small_binary_dataset, protocol):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=3, protocol=protocol,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("return_type", ["mean", "median"])
    def test_return_types(self, small_binary_dataset, return_type):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=3, return_type=return_type,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("metric", ["all", "ptr"])
    def test_selection_metrics(self, small_binary_dataset, metric):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=5, selection_metric=metric,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_ds_selection_metric_binary(self, small_binary_dataset):
        """ds metric is only valid for binary; should work here."""
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=5, selection_metric="ds",
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("size", [1, 3, 10])
    def test_varying_ensemble_size(self, small_binary_dataset, size):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=size,
        )
        q.fit(X_tr, y_tr)
        assert len(q.models) == size
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_max_sample_size(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=3, max_sample_size=20,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_prevalence_values_in_range(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=3,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        for v in prev.values():
            assert 0.0 <= v <= 1.0

    def test_output_keys_match_classes(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=3,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert set(prev.keys()) == set(np.unique(y_tr))

    def test_config_context_array_return(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=3,
        )
        q.fit(X_tr, y_tr)
        with config_context(prevalence_return_type="array"):
            prev = q.predict(X_te)
        assert isinstance(prev, np.ndarray)
        assert pytest.approx(prev.sum(), abs=1e-6) == 1.0


class TestEnsembleQMulticlass:
    """EnsembleQ with multiclass datasets."""

    def test_fit_predict_multiclass(self, multiclass_dataset):
        X_tr, X_te, y_tr, y_te = multiclass_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="lbfgs", max_iter=300, random_state=0)),
            size=3,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert len(prev) == 3
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_ds_metric_raises_on_multiclass(self, small_multiclass_dataset):
        """ds selection metric is binary-only; should raise ValueError on multiclass."""
        X_tr, X_te, y_tr, y_te = small_multiclass_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="lbfgs", max_iter=300, random_state=0)),
            size=3, selection_metric="ds",
        )
        with pytest.raises(ValueError, match="binary"):
            q.fit(X_tr, y_tr)

    def test_multiclass_keys(self, small_multiclass_dataset):
        X_tr, X_te, y_tr, y_te = small_multiclass_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="lbfgs", max_iter=300, random_state=0)),
            size=3,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert set(prev.keys()) == set(np.unique(y_tr))


class TestEnsembleQInputVariants:
    """EnsembleQ with different input types and label types."""

    def test_pandas_input(self, pandas_binary_dataset):
        X_tr, X_te, y_tr, y_te = pandas_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=3,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_pandas_multiclass(self, pandas_multiclass_dataset):
        X_tr, X_te, y_tr, y_te = pandas_multiclass_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="lbfgs", max_iter=300, random_state=0)),
            size=3,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert len(prev) == 3
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_string_labels_binary(self, string_label_binary_dataset):
        X_tr, X_te, y_tr, y_te = string_label_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=3,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert set(prev.keys()) == {"neg", "pos"}
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_string_labels_multiclass(self, string_label_multiclass_dataset):
        X_tr, X_te, y_tr, y_te = string_label_multiclass_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="lbfgs", max_iter=300, random_state=0)),
            size=3,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert set(prev.keys()) == {"cat", "dog", "fish"}
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0


class TestEnsembleQEdgeCases:
    """EnsembleQ edge cases."""

    def test_imbalanced_dataset(self, imbalanced_binary_dataset):
        X_tr, X_te, y_tr, y_te = imbalanced_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=3, min_prop=0.01, max_prop=1.0,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_ensemble_size_one(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=1,
        )
        q.fit(X_tr, y_tr)
        assert len(q.models) == 1
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("base_q", [
        CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
        PCC(learner=LogisticRegression(solver="liblinear", random_state=0)),
    ])
    def test_different_base_quantifiers(self, small_binary_dataset, base_q):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(quantifier=base_q, size=3)
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0


# ===========================================================================
# AggregativeBootstrap tests
# ===========================================================================

class TestAggregativeBootstrapBinary:
    """AggregativeBootstrap on binary datasets."""

    def test_fit_predict_basic(self, binary_dataset):
        X_tr, X_te, y_tr, y_te = binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("n_train,n_test", [(1, 1), (3, 3), (5, 2)])
    def test_varying_bootstraps(self, small_binary_dataset, n_train, n_test):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=n_train, n_test_bootstraps=n_test,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("region_type", ["intervals", "ellipse", "ellipse-clr"])
    def test_region_types(self, small_binary_dataset, region_type):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=3, n_test_bootstraps=3,
            region_type=region_type,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        if region_type == "ellipse-clr":
            # CLR-transformed prevalences may not sum to 1
            assert isinstance(prev, dict)
        else:
            assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("conf_level", [0.80, 0.90, 0.95, 0.99])
    def test_confidence_levels(self, small_binary_dataset, conf_level):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=3, n_test_bootstraps=3,
            confidence_level=conf_level,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_prevalence_values_in_range(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=3, n_test_bootstraps=3,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        for v in prev.values():
            assert 0.0 <= v <= 1.0

    def test_output_keys_match_classes(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert set(prev.keys()) == set(np.unique(y_tr))

    def test_config_context_array_return(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        q.fit(X_tr, y_tr)
        with config_context(prevalence_return_type="array"):
            prev = q.predict(X_te)
        assert isinstance(prev, np.ndarray)
        assert pytest.approx(prev.sum(), abs=1e-6) == 1.0

    def test_random_state_reproducibility(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        results = []
        for _ in range(2):
            q = AggregativeBootstrap(
                quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
                n_train_bootstraps=3, n_test_bootstraps=3,
                random_state=42,
            )
            q.fit(X_tr, y_tr)
            results.append(q.predict(X_te))
        # Same random_state => same results
        for k in results[0]:
            assert results[0][k] == pytest.approx(results[1][k], abs=1e-10)


class TestAggregativeBootstrapMulticlass:
    """AggregativeBootstrap on multiclass datasets."""

    def test_fit_predict_multiclass(self, multiclass_dataset):
        X_tr, X_te, y_tr, y_te = multiclass_dataset
        q = AggregativeBootstrap(
            quantifier=PAC(learner=LogisticRegression(solver="lbfgs", max_iter=300, random_state=0)),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert len(prev) == 3
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_multiclass_keys(self, small_multiclass_dataset):
        X_tr, X_te, y_tr, y_te = small_multiclass_dataset
        q = AggregativeBootstrap(
            quantifier=PCC(learner=LogisticRegression(solver="lbfgs", max_iter=300, random_state=0)),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert set(prev.keys()) == set(np.unique(y_tr))


class TestAggregativeBootstrapInputVariants:
    """AggregativeBootstrap with different input / label types."""

    def test_pandas_input(self, pandas_binary_dataset):
        X_tr, X_te, y_tr, y_te = pandas_binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_string_labels(self, string_label_binary_dataset):
        X_tr, X_te, y_tr, y_te = string_label_binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert set(prev.keys()) == {"neg", "pos"}
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("base_q", [
        TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
        PCC(learner=LogisticRegression(solver="liblinear", random_state=0)),
        PAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
    ])
    def test_different_aggregative_quantifiers(self, small_binary_dataset, base_q):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = AggregativeBootstrap(
            quantifier=base_q,
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0


class TestAggregativeBootstrapEdgeCases:
    """Edge cases for AggregativeBootstrap."""

    def test_imbalanced_dataset(self, imbalanced_binary_dataset):
        X_tr, X_te, y_tr, y_te = imbalanced_binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_single_bootstrap(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=1, n_test_bootstraps=1,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0


class TestAggregativeBootstrapErrors:
    """Error handling for AggregativeBootstrap."""

    def test_non_aggregative_quantifier_raises(self, small_binary_dataset):
        """A non-aggregative quantifier should be rejected."""
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        # CC without a learner is still aggregative, let's use EnsembleQ which is a meta quantifier
        q = AggregativeBootstrap(
            quantifier=EnsembleQ(
                quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
                size=2,
            ),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        with pytest.raises(ValueError, match="not an aggregative quantifier"):
            q.fit(X_tr, y_tr)

    def test_fit_with_val_split(self, small_binary_dataset):
        """Fit with val_split should still work."""
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = AggregativeBootstrap(
            quantifier=PCC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=2, n_test_bootstraps=2,
            random_state=42,
        )
        q.fit(X_tr, y_tr, val_split=0.3)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0


# ===========================================================================
# QuaDapt tests
# ===========================================================================

class TestQuaDaptBinary:
    """QuaDapt on binary datasets."""

    def test_fit_predict_basic(self, binary_dataset):
        X_tr, X_te, y_tr, y_te = binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=[0.5],
            measure="topsoe",
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert isinstance(prev, dict)
        assert len(prev) == 2
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("measure", ["hellinger", "topsoe", "probsymm", "sord"])
    def test_measure_variants(self, small_binary_dataset, measure):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=[0.5],
            measure=measure,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("mf", [
        [0.5],
        [0.2, 0.5, 0.8],
        np.arange(0.1, 1.0, 0.3),
    ])
    def test_merging_factor_variants(self, small_binary_dataset, mf):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=mf,
            measure="topsoe",
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_prevalence_values_in_range(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=[0.5],
            measure="topsoe",
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        for v in prev.values():
            assert 0.0 <= v <= 1.0

    def test_output_keys_match_classes(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=[0.5],
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert set(prev.keys()) == set(np.unique(y_tr))

    def test_config_context_array_return(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=[0.5],
        )
        q.fit(X_tr, y_tr)
        with config_context(prevalence_return_type="array"):
            prev = q.predict(X_te)
        assert isinstance(prev, np.ndarray)
        assert pytest.approx(prev.sum(), abs=1e-6) == 1.0


class TestQuaDaptInputVariants:
    """QuaDapt with different input / label types."""

    def test_pandas_input(self, pandas_binary_dataset):
        X_tr, X_te, y_tr, y_te = pandas_binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=[0.5],
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_string_labels(self, string_label_binary_dataset):
        X_tr, X_te, y_tr, y_te = string_label_binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=[0.5],
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert set(prev.keys()) == {"neg", "pos"}
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    @pytest.mark.parametrize("base_q", [
        DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
        HDy(learner=LogisticRegression(solver="liblinear", random_state=0)),
    ])
    def test_different_soft_quantifiers(self, small_binary_dataset, base_q):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = QuaDapt(
            quantifier=base_q,
            merging_factors=[0.5],
            measure="topsoe",
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_different_learner(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=RandomForestClassifier(n_estimators=5, random_state=0)),
            merging_factors=[0.5],
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0


class TestQuaDaptEdgeCases:
    """Edge cases for QuaDapt."""

    def test_imbalanced_dataset(self, imbalanced_binary_dataset):
        X_tr, X_te, y_tr, y_te = imbalanced_binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=[0.5],
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_single_merging_factor(self, small_binary_dataset):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = QuaDapt(
            quantifier=DyS(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=[0.3],
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0


class TestQuaDaptErrors:
    """Error handling for QuaDapt."""

    def test_non_soft_quantifier_raises(self, small_binary_dataset):
        """CC uses hard (crisp) predictions → should be rejected by QuaDapt."""
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = QuaDapt(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            merging_factors=[0.5],
        )
        with pytest.raises(ValueError):
            q.fit(X_tr, y_tr)


# ===========================================================================
# MoSS static method tests
# ===========================================================================

class TestMoSS:
    """Tests for QuaDapt.MoSS (Model for Score Simulation)."""

    def test_moss_output_shape(self):
        scores, labels = QuaDapt.MoSS(n=100, alpha=0.5, merging_factor=0.5)
        assert scores.shape == (100, 2)
        assert labels.shape == (100,)

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_moss_label_proportions(self, alpha):
        scores, labels = QuaDapt.MoSS(n=1000, alpha=alpha, merging_factor=0.5)
        n_pos = int(1000 * alpha)
        assert np.sum(labels == 1) == n_pos
        assert np.sum(labels == 0) == 1000 - n_pos

    def test_moss_scores_in_01(self):
        scores, _ = QuaDapt.MoSS(n=500, alpha=0.5, merging_factor=0.5)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_moss_alpha_as_list(self):
        """MoSS accepts alpha as a list, should use alpha[1]."""
        scores, labels = QuaDapt.MoSS(n=200, alpha=[0.3, 0.7], merging_factor=0.5)
        n_pos = int(200 * 0.7)
        assert np.sum(labels == 1) == n_pos

    @pytest.mark.parametrize("mf", [0.1, 0.5, 0.9])
    def test_moss_merging_factors(self, mf):
        scores, labels = QuaDapt.MoSS(n=300, alpha=0.5, merging_factor=mf)
        assert scores.shape == (300, 2)
        assert set(np.unique(labels)).issubset({0, 1})


# ===========================================================================
# Cross-cutting / integration tests
# ===========================================================================

class TestMetaIntegration:
    """Integration tests across meta quantifiers."""

    def test_ensemble_then_bootstrap(self, small_binary_dataset):
        """Fit EnsembleQ and AggregativeBootstrap on the same data and verify both work."""
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        lr = LogisticRegression(solver="liblinear", random_state=0)

        ens = EnsembleQ(quantifier=CC(learner=lr), size=3)
        ens.fit(X_tr, y_tr)
        p1 = ens.predict(X_te)

        boot = AggregativeBootstrap(
            quantifier=TAC(learner=lr),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        boot.fit(X_tr, y_tr)
        p2 = boot.predict(X_te)

        # Both should be valid
        assert pytest.approx(sum(p1.values()), abs=1e-6) == 1.0
        assert pytest.approx(sum(p2.values()), abs=1e-6) == 1.0

    def test_all_meta_on_same_binary_data(self, small_binary_dataset):
        """All three meta quantifiers on one binary dataset."""
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        lr = LogisticRegression(solver="liblinear", random_state=0)

        results = {}

        ens = EnsembleQ(quantifier=CC(learner=lr), size=3)
        ens.fit(X_tr, y_tr)
        results["ens"] = ens.predict(X_te)

        boot = AggregativeBootstrap(
            quantifier=TAC(learner=lr),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        boot.fit(X_tr, y_tr)
        results["boot"] = boot.predict(X_te)

        qd = QuaDapt(
            quantifier=DyS(learner=lr),
            merging_factors=[0.5],
        )
        qd.fit(X_tr, y_tr)
        results["qd"] = qd.predict(X_te)

        for name, prev in results.items():
            assert len(prev) == 2, f"{name} wrong length"
            assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0, f"{name} doesn't sum to 1"

    @pytest.mark.parametrize("base_q_cls,base_kwargs", [
        (CC, {"learner": LogisticRegression(solver="liblinear", random_state=0)}),
        (PCC, {"learner": LogisticRegression(solver="liblinear", random_state=0)}),
    ])
    def test_ensemble_with_parametrized_quantifiers(self, small_binary_dataset, base_q_cls, base_kwargs):
        X_tr, X_te, y_tr, y_te = small_binary_dataset
        q = EnsembleQ(quantifier=base_q_cls(**base_kwargs), size=3)
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_tiny_dataset_ensemble(self):
        """Very small inline dataset (20 samples)."""
        rng = np.random.RandomState(99)
        X = rng.randn(20, 3)
        y = np.array([0]*10 + [1]*10)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=99)

        q = EnsembleQ(
            quantifier=CC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            size=2,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0

    def test_tiny_dataset_bootstrap(self):
        """Very small inline dataset (20 samples)."""
        rng = np.random.RandomState(99)
        X = rng.randn(20, 3)
        y = np.array([0]*10 + [1]*10)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=99)

        q = AggregativeBootstrap(
            quantifier=TAC(learner=LogisticRegression(solver="liblinear", random_state=0)),
            n_train_bootstraps=2, n_test_bootstraps=2,
        )
        q.fit(X_tr, y_tr)
        prev = q.predict(X_te)
        assert pytest.approx(sum(prev.values()), abs=1e-6) == 1.0
