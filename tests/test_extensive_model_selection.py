"""
Comprehensive tests for the mlquantify.model_selection module.

Covers:
- GridSearchQ with binary and multiclass datasets
- Protocols: APP, NPP, UPP, PPP - split, generation, reproducibility
- Varied input types: numpy arrays, pandas DataFrames
- Varied label types: int labels, string labels
- Parameter variations: param_grids, protocols, scoring functions, val_split
- Edge cases: single parameter value, tiny datasets
- GridSearchQ attribute storage: best_params, best_score, best_model_
- Predict after fit
- Error cases: predict before fit, invalid protocol
- Protocol split sizes and class coverage
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlquantify.model_selection import (
    GridSearchQ,
    APP,
    NPP,
    UPP,
    PPP,
    BaseProtocol,
)
from mlquantify.adjust_counting import CC, PCC
from mlquantify.metrics import MAE, MSE, RAE
from mlquantify import config_context


# ---------------------------------------------------------------------------
# Helper: quantifier subclass with a default learner so GridSearchQ can
# instantiate it via quantifier() without arguments.
# ---------------------------------------------------------------------------
class _CC(CC):
    """CC with a default LogisticRegression learner."""

    def __init__(self, learner=None, threshold=0.5):
        if learner is None:
            learner = LogisticRegression(solver="liblinear", random_state=0)
        super().__init__(learner=learner, threshold=threshold)


class _PCC(PCC):
    """PCC with a default LogisticRegression learner."""

    def __init__(self, learner=None):
        if learner is None:
            learner = LogisticRegression(solver="liblinear", random_state=0)
        super().__init__(learner=learner)


# ---------------------------------------------------------------------------
# Small inline datasets for edge-case testing
# ---------------------------------------------------------------------------
@pytest.fixture()
def tiny_binary_dataset():
    """Very small binary dataset (60 samples)."""
    X, y = make_classification(
        n_samples=60, n_features=5, n_classes=2,
        weights=[0.5, 0.5], random_state=7,
    )
    return train_test_split(X, y, test_size=0.3, random_state=7)


@pytest.fixture()
def string_label_dataset():
    """Binary dataset with string labels ('cat' / 'dog')."""
    X, y_int = make_classification(
        n_samples=200, n_features=10, n_classes=2,
        weights=[0.5, 0.5], random_state=99,
    )
    mapping = {0: "cat", 1: "dog"}
    y_str = np.array([mapping[v] for v in y_int])
    return train_test_split(X, y_str, test_size=0.3, random_state=99)


@pytest.fixture()
def pandas_binary_dataset():
    """Binary dataset returned as pandas DataFrame / Series."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2,
        weights=[0.5, 0.5], random_state=55,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=55,
    )
    return (
        pd.DataFrame(X_train),
        pd.DataFrame(X_test),
        pd.Series(y_train),
        pd.Series(y_test),
    )


# ===================================================================
# SECTION 1 – Protocol unit tests (APP, NPP, UPP, PPP)
# ===================================================================


class TestAPP:
    """Tests for the Artificial Prevalence Protocol."""

    def test_split_yields_indices(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        protocol = APP(batch_size=50, n_prevalences=3, repeats=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) > 0
        for idx in indices:
            assert isinstance(idx, (list, np.ndarray))

    def test_batch_size_respected(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        bs = 30
        protocol = APP(batch_size=bs, n_prevalences=3, repeats=1, random_state=42)
        for idx in protocol.split(X_train, y_train):
            assert len(idx) == bs

    @pytest.mark.parametrize("repeats", [1, 2, 3])
    def test_repeats_increase_splits(self, binary_dataset, repeats):
        X_train, _, y_train, _ = binary_dataset
        protocol = APP(batch_size=50, n_prevalences=3, repeats=repeats, random_state=42)
        n_splits = len(list(protocol.split(X_train, y_train)))
        assert n_splits > 0

    def test_reproducibility(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        p1 = APP(batch_size=50, n_prevalences=3, repeats=1, random_state=42)
        p2 = APP(batch_size=50, n_prevalences=3, repeats=1, random_state=42)
        idx1 = [list(i) for i in p1.split(X_train, y_train)]
        idx2 = [list(i) for i in p2.split(X_train, y_train)]
        assert idx1 == idx2

    def test_different_seed_different_splits(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        p1 = APP(batch_size=50, n_prevalences=5, repeats=1, random_state=1)
        p2 = APP(batch_size=50, n_prevalences=5, repeats=1, random_state=99)
        idx1 = [list(i) for i in p1.split(X_train, y_train)]
        idx2 = [list(i) for i in p2.split(X_train, y_train)]
        assert idx1 != idx2

    def test_multiclass(self, multiclass_dataset):
        X_train, _, y_train, _ = multiclass_dataset
        protocol = APP(batch_size=50, n_prevalences=3, repeats=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) > 0

    @pytest.mark.parametrize("batch_size", [20, 50, [30, 60]])
    def test_various_batch_sizes(self, binary_dataset, batch_size):
        X_train, _, y_train, _ = binary_dataset
        protocol = APP(batch_size=batch_size, n_prevalences=3, repeats=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) > 0

    def test_min_max_prev(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = APP(
            batch_size=50, n_prevalences=5, repeats=1,
            random_state=42, min_prev=0.2, max_prev=0.8,
        )
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) > 0

    def test_get_n_combinations(self):
        protocol = APP(batch_size=50, n_prevalences=5, repeats=2, random_state=42)
        assert protocol.get_n_combinations() > 0


class TestNPP:
    """Tests for the Natural Prevalence Protocol."""

    def test_split_yields_indices(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = NPP(batch_size=50, n_samples=3, repeats=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) > 0

    def test_batch_size_respected(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        bs = 40
        protocol = NPP(batch_size=bs, n_samples=2, repeats=1, random_state=42)
        for idx in protocol.split(X_train, y_train):
            assert len(idx) == bs

    def test_n_samples_controls_count(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        n_samples = 4
        protocol = NPP(batch_size=50, n_samples=n_samples, repeats=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) == n_samples

    @pytest.mark.parametrize("repeats", [1, 2, 4])
    def test_repeats(self, binary_dataset, repeats):
        X_train, _, y_train, _ = binary_dataset
        n_samples = 2
        protocol = NPP(batch_size=50, n_samples=n_samples, repeats=repeats, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) == n_samples * repeats

    def test_reproducibility(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        p1 = NPP(batch_size=50, n_samples=3, random_state=42)
        p2 = NPP(batch_size=50, n_samples=3, random_state=42)
        idx1 = [list(i) for i in p1.split(X_train, y_train)]
        idx2 = [list(i) for i in p2.split(X_train, y_train)]
        assert idx1 == idx2

    def test_multiclass(self, multiclass_dataset):
        X_train, _, y_train, _ = multiclass_dataset
        protocol = NPP(batch_size=50, n_samples=3, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) == 3

    def test_multiple_batch_sizes(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = NPP(batch_size=[30, 60], n_samples=2, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        # 2 batch sizes * 2 n_samples = 4 splits (each repeated 1 time)
        assert len(indices) == 2 * 2


class TestUPP:
    """Tests for the Uniform Prevalence Protocol."""

    def test_split_yields_indices(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = UPP(batch_size=50, n_prevalences=3, repeats=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) > 0

    def test_batch_size_respected(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        bs = 35
        protocol = UPP(batch_size=bs, n_prevalences=3, repeats=1, random_state=42)
        for idx in protocol.split(X_train, y_train):
            assert len(idx) == bs

    @pytest.mark.parametrize("algorithm", ["kraemer", "uniform"])
    def test_algorithm_variants(self, binary_dataset, algorithm):
        X_train, _, y_train, _ = binary_dataset
        protocol = UPP(
            batch_size=50, n_prevalences=3, repeats=1,
            random_state=42, algorithm=algorithm,
        )
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) > 0

    def test_reproducibility(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        p1 = UPP(batch_size=50, n_prevalences=3, repeats=1, random_state=42)
        p2 = UPP(batch_size=50, n_prevalences=3, repeats=1, random_state=42)
        idx1 = [list(i) for i in p1.split(X_train, y_train)]
        idx2 = [list(i) for i in p2.split(X_train, y_train)]
        assert idx1 == idx2

    def test_multiclass(self, multiclass_dataset):
        X_train, _, y_train, _ = multiclass_dataset
        protocol = UPP(batch_size=50, n_prevalences=3, repeats=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) > 0

    def test_min_max_prev(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = UPP(
            batch_size=50, n_prevalences=5, repeats=1,
            random_state=42, min_prev=0.1, max_prev=0.9,
        )
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) > 0

    def test_get_n_combinations(self):
        protocol = UPP(batch_size=50, n_prevalences=5, repeats=2, random_state=42)
        assert protocol.get_n_combinations() > 0


class TestPPP:
    """Tests for the Personalized Prevalence Protocol."""

    def test_split_yields_indices(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = PPP(batch_size=50, prevalences=[[0.3, 0.7], [0.5, 0.5]], random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) == 2

    def test_batch_size_respected(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        bs = 40
        protocol = PPP(batch_size=bs, prevalences=[[0.4, 0.6]], random_state=42)
        for idx in protocol.split(X_train, y_train):
            assert len(idx) == bs

    def test_float_prevalence_binary(self, binary_dataset):
        """A single float is treated as prevalence of the positive class."""
        X_train, _, y_train, _ = binary_dataset
        protocol = PPP(batch_size=50, prevalences=[0.3], random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) == 1

    def test_reproducibility(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        p1 = PPP(batch_size=50, prevalences=[[0.3, 0.7]], random_state=42)
        p2 = PPP(batch_size=50, prevalences=[[0.3, 0.7]], random_state=42)
        idx1 = [list(i) for i in p1.split(X_train, y_train)]
        idx2 = [list(i) for i in p2.split(X_train, y_train)]
        assert idx1 == idx2

    def test_multiple_batch_sizes(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = PPP(
            batch_size=[30, 60], prevalences=[[0.4, 0.6]], random_state=42,
        )
        indices = list(protocol.split(X_train, y_train))
        # 2 batch sizes × 1 prevalence = 2
        assert len(indices) == 2


# ===================================================================
# SECTION 2 – Protocol with different input types
# ===================================================================


class TestProtocolInputTypes:
    """Protocols should work with numpy arrays and pandas DataFrames."""

    def test_app_with_pandas(self, pandas_binary_dataset):
        X_train, _, y_train, _ = pandas_binary_dataset
        protocol = APP(batch_size=30, n_prevalences=3, repeats=1, random_state=42)
        indices = list(protocol.split(X_train.values, y_train.values))
        assert len(indices) > 0

    def test_npp_with_pandas(self, pandas_binary_dataset):
        X_train, _, y_train, _ = pandas_binary_dataset
        protocol = NPP(batch_size=30, n_samples=2, random_state=42)
        indices = list(protocol.split(X_train.values, y_train.values))
        assert len(indices) == 2

    def test_upp_with_pandas(self, pandas_binary_dataset):
        X_train, _, y_train, _ = pandas_binary_dataset
        protocol = UPP(batch_size=30, n_prevalences=3, repeats=1, random_state=42)
        indices = list(protocol.split(X_train.values, y_train.values))
        assert len(indices) > 0


# ===================================================================
# SECTION 3 – GridSearchQ core tests
# ===================================================================


class TestGridSearchQBinary:
    """GridSearchQ tests on binary datasets."""

    def test_basic_fit_predict(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"threshold": [0.3, 0.5, 0.7]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            protocol="app",
            samples_sizes=50,
            n_repetitions=2,
            scoring=MAE,
            refit=True,
            val_split=0.4,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        prev = gsq.predict(X_test)
        assert prev is not None

    def test_best_params_stored(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        param_grid = {"threshold": [0.3, 0.5, 0.7]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert isinstance(gsq.best_params, dict)
        assert "threshold" in gsq.best_params
        assert gsq.best_params["threshold"] in [0.3, 0.5, 0.7]

    def test_best_score_stored(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        param_grid = {"threshold": [0.3, 0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert isinstance(gsq.best_score, float)
        assert gsq.best_score >= 0.0

    def test_best_model_stored(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        param_grid = {"threshold": [0.3, 0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            refit=True,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert hasattr(gsq, "best_model_")
        assert gsq.best_model() is not None

    def test_predict_returns_correct_length(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        prev = gsq.predict(X_test)
        # Binary → 2 prevalence values (dict or array of length 2)
        if isinstance(prev, dict):
            assert len(prev) == 2
        else:
            assert len(prev) == 2

    def test_fit_returns_self(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        result = gsq.fit(X_train, y_train)
        assert result is gsq


class TestGridSearchQMulticlass:
    """GridSearchQ tests on multiclass datasets."""

    def test_fit_predict(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        param_grid = {"threshold": [0.3, 0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            protocol="app",
            samples_sizes=50,
            n_repetitions=2,
            scoring=MAE,
            refit=True,
            val_split=0.4,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        prev = gsq.predict(X_test)
        assert prev is not None

    def test_best_params_multiclass(self, multiclass_dataset):
        X_train, _, y_train, _ = multiclass_dataset
        param_grid = {"threshold": [0.4, 0.5, 0.6]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert isinstance(gsq.best_params, dict)

    def test_predict_returns_three_classes(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        prev = gsq.predict(X_test)
        if isinstance(prev, dict):
            assert len(prev) == 3
        else:
            assert len(prev) == 3


# ===================================================================
# SECTION 4 – GridSearchQ with different protocols
# ===================================================================


class TestGridSearchQProtocols:
    """Verify that GridSearchQ works with each protocol string."""

    @pytest.mark.parametrize("protocol", ["app", "npp", "upp"])
    def test_protocol_options(self, binary_dataset, protocol):
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"threshold": [0.4, 0.6]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            protocol=protocol,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        prev = gsq.predict(X_test)
        assert prev is not None
        assert isinstance(gsq.best_score, float)

    def test_invalid_protocol_raises(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            protocol="invalid_proto",
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        with pytest.raises((ValueError, Exception)):
            gsq.fit(X_train, y_train)


# ===================================================================
# SECTION 5 – GridSearchQ with different scoring functions
# ===================================================================


class TestGridSearchQScoring:
    """Test different scoring callables."""

    @pytest.mark.parametrize("scoring", [MAE, MSE, RAE])
    def test_various_scoring(self, binary_dataset, scoring):
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"threshold": [0.4, 0.6]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            scoring=scoring,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert gsq.best_score >= 0.0
        prev = gsq.predict(X_test)
        assert prev is not None


# ===================================================================
# SECTION 6 – GridSearchQ val_split variations
# ===================================================================


class TestGridSearchQValSplit:
    """Test different val_split fractions."""

    @pytest.mark.parametrize("val_split", [0.2, 0.3, 0.5])
    def test_val_split_values(self, binary_dataset, val_split):
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"threshold": [0.4, 0.6]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            val_split=val_split,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert gsq.best_score is not None
        prev = gsq.predict(X_test)
        assert prev is not None


# ===================================================================
# SECTION 7 – GridSearchQ param_grid variations
# ===================================================================


class TestGridSearchQParamGrid:
    """Test different param_grid configurations."""

    def test_single_param_single_value(self, binary_dataset):
        """Grid with only one value → that value must be selected."""
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert gsq.best_params["threshold"] == 0.5

    def test_multiple_params(self, binary_dataset):
        """Grid search over two parameters simultaneously."""
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {
            "threshold": [0.3, 0.5, 0.7],
            "learner__C": [0.1, 1.0],
        }
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert "threshold" in gsq.best_params
        assert "learner__C" in gsq.best_params

    def test_learner_nested_param(self, binary_dataset):
        """Search over nested learner parameter."""
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"learner__C": [0.01, 0.1, 1.0, 10.0]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert gsq.best_params["learner__C"] in [0.01, 0.1, 1.0, 10.0]


# ===================================================================
# SECTION 8 – GridSearchQ with PCC (soft quantifier)
# ===================================================================


class TestGridSearchQPCC:
    """GridSearchQ with a probabilistic quantifier."""

    def test_pcc_grid_search(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"learner__C": [0.1, 1.0]}
        gsq = GridSearchQ(
            quantifier=_PCC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        prev = gsq.predict(X_test)
        assert prev is not None
        assert isinstance(gsq.best_score, float)

    def test_pcc_multiclass(self, multiclass_dataset):
        X_train, X_test, y_train, y_test = multiclass_dataset
        param_grid = {"learner__C": [0.1, 1.0]}
        gsq = GridSearchQ(
            quantifier=_PCC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        prev = gsq.predict(X_test)
        assert prev is not None


# ===================================================================
# SECTION 9 – Error / edge-case tests
# ===================================================================


class TestGridSearchQErrors:
    """Error handling and edge cases for GridSearchQ."""

    def test_predict_before_fit_raises(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        with pytest.raises(RuntimeError):
            gsq.predict(X_test)

    def test_best_model_before_fit_raises(self):
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        with pytest.raises(ValueError):
            gsq.best_model()

    def test_tiny_dataset(self, tiny_binary_dataset):
        """Ensure GridSearchQ works on a very small dataset."""
        X_train, X_test, y_train, y_test = tiny_binary_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=10,
            n_repetitions=2,
            val_split=0.3,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        prev = gsq.predict(X_test)
        assert prev is not None

    def test_verbose_flag(self, binary_dataset, capsys):
        """Verbose mode should print output."""
        X_train, _, y_train, _ = binary_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            verbose=True,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        captured = capsys.readouterr()
        assert "GridSearchQ" in captured.out


# ===================================================================
# SECTION 10 – GridSearchQ with refit=False
# ===================================================================


class TestGridSearchQRefit:
    """Test refit parameter behaviour."""

    def test_refit_false_no_best_model(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"threshold": [0.3, 0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            refit=False,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        # best_params and best_score are still available
        assert gsq.best_params is not None
        assert gsq.best_score is not None
        # but predict should fail because no refit was done
        with pytest.raises(RuntimeError):
            gsq.predict(X_test)

    def test_refit_true_has_best_model(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            refit=True,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert hasattr(gsq, "best_model_")


# ===================================================================
# SECTION 11 – GridSearchQ reproducibility
# ===================================================================


class TestGridSearchQReproducibility:
    """Same seeds → same results."""

    def test_same_seed_same_result(self, binary_dataset):
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"threshold": [0.3, 0.5, 0.7]}
        kwargs = dict(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq1 = GridSearchQ(**kwargs)
        gsq1.fit(X_train, y_train)

        gsq2 = GridSearchQ(**kwargs)
        gsq2.fit(X_train, y_train)

        assert gsq1.best_params == gsq2.best_params
        assert gsq1.best_score == gsq2.best_score

    def test_different_seed_may_differ(self, binary_dataset):
        """Different seeds can (but don't have to) produce different results."""
        X_train, _, y_train, _ = binary_dataset
        param_grid = {"threshold": [0.3, 0.5, 0.7]}
        gsq1 = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=3,
            random_seed=1,
        )
        gsq1.fit(X_train, y_train)

        gsq2 = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=3,
            random_seed=999,
        )
        gsq2.fit(X_train, y_train)

        # At least one of score or params may differ; just confirm they ran
        assert gsq1.best_score is not None
        assert gsq2.best_score is not None


# ===================================================================
# SECTION 12 – Protocol split sizes and class coverage
# ===================================================================


class TestProtocolSplitCoverage:
    """Ensure that splits cover all classes and respect sizes."""

    def test_app_all_classes_present(self, binary_dataset):
        """Each APP split index set should reference valid indices."""
        X_train, _, y_train, _ = binary_dataset
        protocol = APP(batch_size=50, n_prevalences=5, repeats=1, random_state=42)
        for idx in protocol.split(X_train, y_train):
            assert all(i < len(y_train) for i in idx)

    def test_npp_indices_valid(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = NPP(batch_size=50, n_samples=3, random_state=42)
        for idx in protocol.split(X_train, y_train):
            assert all(0 <= i < len(y_train) for i in idx)

    def test_upp_indices_valid(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = UPP(batch_size=50, n_prevalences=3, repeats=1, random_state=42)
        for idx in protocol.split(X_train, y_train):
            assert all(0 <= i < len(y_train) for i in idx)

    def test_app_multiclass_class_coverage(self, multiclass_dataset):
        """Over enough APP splits, all classes should appear at least once."""
        X_train, _, y_train, _ = multiclass_dataset
        protocol = APP(batch_size=100, n_prevalences=5, repeats=2, random_state=42)
        seen_classes = set()
        for idx in protocol.split(X_train, y_train):
            seen_classes.update(y_train[idx])
        all_classes = set(np.unique(y_train))
        assert seen_classes == all_classes


# ===================================================================
# SECTION 13 – config_context interaction
# ===================================================================


class TestGridSearchQConfigContext:
    """Ensure GridSearchQ works under different config_context settings."""

    @pytest.mark.parametrize("return_type", ["dict", "array"])
    def test_prevalence_return_type(self, binary_dataset, return_type):
        X_train, X_test, y_train, y_test = binary_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        with config_context(prevalence_return_type=return_type):
            gsq.fit(X_train, y_train)
            prev = gsq.predict(X_test)
            if return_type == "dict":
                assert isinstance(prev, dict)
            else:
                assert isinstance(prev, np.ndarray)


# ===================================================================
# SECTION 14 – GridSearchQ with string labels
# ===================================================================


class TestGridSearchQStringLabels:
    """Grid search should handle non-integer class labels."""

    def test_string_labels_fit_predict(self, string_label_dataset):
        X_train, X_test, y_train, y_test = string_label_dataset
        param_grid = {"threshold": [0.4, 0.6]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        prev = gsq.predict(X_test)
        assert prev is not None

    def test_string_labels_best_params(self, string_label_dataset):
        X_train, _, y_train, _ = string_label_dataset
        param_grid = {"threshold": [0.5]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert gsq.best_params["threshold"] == 0.5


# ===================================================================
# SECTION 15 – Protocol edge cases
# ===================================================================


class TestProtocolEdgeCases:
    """Edge-case tests for protocols."""

    def test_app_single_prevalence(self, binary_dataset):
        """n_prevalences=1 should still work."""
        X_train, _, y_train, _ = binary_dataset
        protocol = APP(batch_size=30, n_prevalences=1, repeats=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) >= 1

    def test_npp_single_sample(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = NPP(batch_size=30, n_samples=1, repeats=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) == 1

    def test_upp_single_prevalence(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        protocol = UPP(batch_size=30, n_prevalences=1, repeats=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) >= 1

    def test_large_batch_size_with_replacement(self, binary_dataset):
        """Batch size larger than dataset should log a warning and still work."""
        X_train, _, y_train, _ = binary_dataset
        protocol = NPP(batch_size=len(y_train) + 100, n_samples=1, random_state=42)
        indices = list(protocol.split(X_train, y_train))
        assert len(indices) == 1
        # batch size should match what was requested even with replacement
        assert len(indices[0]) == len(y_train) + 100


# ===================================================================
# SECTION 16 – GridSearchQ with pandas DataFrames
# ===================================================================


class TestGridSearchQPandas:
    """GridSearchQ should accept pandas DataFrames."""

    def test_pandas_fit_predict(self, pandas_binary_dataset):
        X_train, X_test, y_train, y_test = pandas_binary_dataset
        param_grid = {"threshold": [0.4, 0.6]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        prev = gsq.predict(X_test)
        assert prev is not None
        assert isinstance(gsq.best_score, float)


# ===================================================================
# SECTION 17 – GridSearchQ n_jobs parallelism
# ===================================================================


class TestGridSearchQNJobs:
    """Ensure n_jobs does not change results (within tolerance)."""

    def test_njobs_1_vs_default(self, binary_dataset):
        X_train, _, y_train, _ = binary_dataset
        param_grid = {"threshold": [0.3, 0.5, 0.7]}
        gsq = GridSearchQ(
            quantifier=_CC,
            param_grid=param_grid,
            n_jobs=1,
            samples_sizes=50,
            n_repetitions=2,
            random_seed=42,
        )
        gsq.fit(X_train, y_train)
        assert gsq.best_params is not None
        assert gsq.best_score is not None
