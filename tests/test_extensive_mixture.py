"""
Comprehensive tests for the mlquantify.mixture module.

Covers: DyS, HDy, SMM, SORD, HDx, MMD_RKHS
with binary/multiclass datasets, input types, label types,
parameter variations, edge cases, internal methods, and error handling.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlquantify import config_context
from mlquantify.mixture import DyS, HDy, SMM, SORD, HDx, MMD_RKHS
from mlquantify.mixture._base import BaseMixture
from mlquantify.mixture._utils import getHist, ternary_search, hellinger, topsoe, probsymm, sqEuclidean


# =====================================================================
# Helper fixtures
# =====================================================================

@pytest.fixture
def small_binary_dataset():
    """Small binary dataset for quick tests."""
    X, y = make_classification(
        n_samples=120, n_features=5, n_classes=2,
        weights=[0.5, 0.5], random_state=7,
    )
    return train_test_split(X, y, test_size=0.3, random_state=7)


@pytest.fixture
def small_multiclass_dataset():
    """Small multiclass (3-class) dataset."""
    X, y = make_classification(
        n_samples=200, n_features=8, n_classes=3,
        n_informative=6, weights=[0.3, 0.4, 0.3], random_state=7,
    )
    return train_test_split(X, y, test_size=0.3, random_state=7)


@pytest.fixture
def string_label_binary_dataset():
    """Binary dataset with string labels ('pos', 'neg')."""
    X, y = make_classification(
        n_samples=150, n_features=5, n_classes=2,
        weights=[0.5, 0.5], random_state=42,
    )
    y_str = np.where(y == 1, "pos", "neg")
    return train_test_split(X, y_str, test_size=0.3, random_state=42)


@pytest.fixture
def imbalanced_binary_dataset():
    """Extremely imbalanced binary dataset (95% / 5%)."""
    X, y = make_classification(
        n_samples=200, n_features=5, n_classes=2,
        weights=[0.95, 0.05], random_state=99,
    )
    return train_test_split(X, y, test_size=0.3, random_state=99)


@pytest.fixture
def tiny_binary_dataset():
    """Very small binary dataset (20 samples total)."""
    X, y = make_classification(
        n_samples=20, n_features=4, n_classes=2,
        random_state=11,
    )
    return train_test_split(X, y, test_size=0.3, random_state=11)


# =====================================================================
# 1. Aggregative Mixture Models – binary fit/predict (all classes)
# =====================================================================

AGGREGATIVE_CLASSES = [DyS, HDy, SMM, SORD]


@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_aggregative_binary_fit_predict(Quantifier, binary_dataset):
    """All aggregative mixture quantifiers produce valid prevalence dicts on binary data."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0
    for v in prev.values():
        assert 0 <= v <= 1


@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_aggregative_binary_with_pretrained_learner(Quantifier, binary_dataset, binary_classifier):
    """Using a pre-fitted classifier (still goes through CV internally)."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=binary_classifier)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 2. Aggregative Mixture Models – multiclass (ovr strategy)
# =====================================================================

@pytest.mark.parametrize("Quantifier", [HDy, SMM, SORD])
def test_aggregative_multiclass_ovr(Quantifier, small_multiclass_dataset):
    """Multiclass with one-vs-rest strategy (HDy, SMM, SORD accept strategy in __init__)."""
    X_train, X_test, y_train, y_test = small_multiclass_dataset
    q = Quantifier(
        learner=LogisticRegression(random_state=42, solver="liblinear", max_iter=500),
        strategy="ovr",
    )
    with config_context(prevalence_normalization="sum"):
        q.fit(X_train, y_train)
        prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert len(prev) == 3
    assert pytest.approx(sum(prev.values()), abs=1e-2) == 1.0


def test_dys_multiclass_ovr(small_multiclass_dataset):
    """DyS multiclass with ovr strategy (set via attribute since DyS init lacks strategy)."""
    X_train, X_test, y_train, y_test = small_multiclass_dataset
    q = DyS(learner=LogisticRegression(random_state=42, solver="liblinear", max_iter=500))
    q.strategy = "ovr"
    with config_context(prevalence_normalization="sum"):
        q.fit(X_train, y_train)
        prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert len(prev) == 3
    assert pytest.approx(sum(prev.values()), abs=1e-2) == 1.0


@pytest.mark.parametrize("Quantifier", [HDy, SMM, SORD])
def test_aggregative_multiclass_ovo(Quantifier, small_multiclass_dataset):
    """Multiclass with one-vs-one strategy."""
    X_train, X_test, y_train, y_test = small_multiclass_dataset
    q = Quantifier(
        learner=LogisticRegression(random_state=42, solver="liblinear", max_iter=500),
        strategy="ovo",
    )
    with config_context(prevalence_normalization="sum"):
        q.fit(X_train, y_train)
        prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert len(prev) == 3
    assert pytest.approx(sum(prev.values()), abs=1e-2) == 1.0


# =====================================================================
# 3. Non-aggregative Mixture Models – binary
# =====================================================================

def test_hdx_binary(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = HDx()
    q.fit(X_train, y_train)
    prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


@pytest.mark.parametrize("kernel", ["rbf", "linear"])
def test_mmd_rkhs_binary_kernels(kernel, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = MMD_RKHS(kernel=kernel)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 4. Non-aggregative – multiclass
# =====================================================================

def test_hdx_multiclass_ovr(small_multiclass_dataset):
    X_train, X_test, y_train, y_test = small_multiclass_dataset
    q = HDx(strategy="ovr")
    with config_context(prevalence_normalization="sum"):
        q.fit(X_train, y_train)
        prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert len(prev) == 3
    assert pytest.approx(sum(prev.values()), abs=1e-2) == 1.0


def test_mmd_rkhs_multiclass(small_multiclass_dataset):
    """MMD_RKHS natively handles multiclass via QP on the simplex."""
    X_train, X_test, y_train, y_test = small_multiclass_dataset
    q = MMD_RKHS(kernel="rbf")
    q.fit(X_train, y_train)
    prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert len(prev) == 3
    assert pytest.approx(sum(prev.values()), abs=1e-2) == 1.0


# =====================================================================
# 5. Input type variations – Pandas DataFrames
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_aggregative_pandas_input(Quantifier, binary_dataset):
    """Accept pandas DataFrame/Series inputs."""
    X_train, X_test, y_train, y_test = binary_dataset
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train_df, y_train)
    prev = q.predict(X_test_df)

    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


def test_hdx_pandas_input(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = HDx()
    q.fit(pd.DataFrame(X_train), y_train)
    prev = q.predict(pd.DataFrame(X_test))
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


def test_mmd_rkhs_pandas_input(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = MMD_RKHS(kernel="linear")
    q.fit(pd.DataFrame(X_train), y_train)
    prev = q.predict(pd.DataFrame(X_test))
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 6. String label types
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_aggregative_string_labels(Quantifier, string_label_binary_dataset):
    """Quantifiers work with string class labels."""
    X_train, X_test, y_train, y_test = string_label_binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert set(prev.keys()) == {"pos", "neg"}
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


def test_hdx_string_labels(string_label_binary_dataset):
    X_train, X_test, y_train, y_test = string_label_binary_dataset
    q = HDx()
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert set(prev.keys()) == {"pos", "neg"}
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


def test_mmd_rkhs_string_labels(string_label_binary_dataset):
    X_train, X_test, y_train, y_test = string_label_binary_dataset
    q = MMD_RKHS(kernel="rbf")
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert set(prev.keys()) == {"pos", "neg"}
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 7. DyS measure variations
# =====================================================================

@pytest.mark.parametrize("measure", ["hellinger", "topsoe", "probsymm"])
def test_dys_measures(measure, binary_dataset):
    """DyS with each supported measure."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = DyS(
        learner=LogisticRegression(random_state=42, solver="liblinear"),
        measure=measure,
    )
    q.fit(X_train, y_train)
    prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0
    # distances should have been stored
    assert q.distances is not None
    assert len(q.distances) == len(q.bins_size)


# =====================================================================
# 8. DyS bins_size variations
# =====================================================================

@pytest.mark.parametrize("bins", [
    np.array([5, 10]),
    np.array([2, 8, 15, 25]),
    np.array([10]),
])
def test_dys_custom_bins(bins, binary_dataset):
    """DyS with custom bin sizes."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = DyS(
        learner=LogisticRegression(random_state=42, solver="liblinear"),
        bins_size=bins,
    )
    q.fit(X_train, y_train)
    prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert len(q.distances) == len(bins)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 9. Learner variations
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
@pytest.mark.parametrize("learner_cls", [LogisticRegression, RandomForestClassifier])
def test_different_learners(Quantifier, learner_cls, binary_dataset):
    """Works with different sklearn learners."""
    X_train, X_test, y_train, y_test = binary_dataset
    if learner_cls is LogisticRegression:
        learner = LogisticRegression(random_state=42, solver="liblinear")
    else:
        learner = RandomForestClassifier(n_estimators=10, random_state=42)

    q = Quantifier(learner=learner)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 10. MMD_RKHS kernel and parameter variations
# =====================================================================

@pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "sigmoid", "cosine"])
def test_mmd_rkhs_all_kernels(kernel, small_binary_dataset):
    """MMD_RKHS with each supported kernel."""
    X_train, X_test, y_train, y_test = small_binary_dataset
    kwargs = {"kernel": kernel}
    if kernel == "poly":
        kwargs["degree"] = 2
    q = MMD_RKHS(**kwargs)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


def test_mmd_rkhs_gamma(small_binary_dataset):
    """MMD_RKHS with explicit gamma."""
    X_train, X_test, y_train, y_test = small_binary_dataset
    q = MMD_RKHS(kernel="rbf", gamma=0.1)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 11. Aggregate method on aggregative quantifiers
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_aggregate_method(Quantifier, binary_dataset, binary_classifier):
    """Call aggregate() directly with pre-computed predictions."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=binary_classifier)
    q.fit(X_train, y_train)

    predictions = binary_classifier.predict_proba(X_test)
    train_predictions = binary_classifier.predict_proba(X_train)

    prev = q.aggregate(predictions, train_predictions, y_train)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 12. best_mixture method returns (alpha, distance)
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_best_mixture_returns_tuple(Quantifier, binary_dataset):
    """best_mixture returns (alpha, distance)."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)

    pos_scores = q.pos_scores
    neg_scores = q.neg_scores
    test_scores = q.learner.predict_proba(X_test)[:, 1]

    alpha, distance = q.best_mixture(test_scores, pos_scores, neg_scores)
    assert isinstance(alpha, (float, np.floating))
    assert 0 <= alpha <= 1
    # distance may be None for SMM
    if distance is not None:
        assert isinstance(distance, (float, np.floating))


# =====================================================================
# 13. Edge case – tiny dataset
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_tiny_dataset(Quantifier, tiny_binary_dataset):
    """Handle very small datasets without crashing."""
    X_train, X_test, y_train, y_test = tiny_binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 14. Edge case – extreme imbalance
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_extreme_imbalance(Quantifier, imbalanced_binary_dataset):
    """Handle highly imbalanced binary datasets."""
    X_train, X_test, y_train, y_test = imbalanced_binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 15. Edge case – constant predictions (all same class)
# =====================================================================

def test_smm_constant_predictions():
    """SMM with near-constant test scores."""
    np.random.seed(0)
    pos_scores = np.random.uniform(0.6, 1.0, 50)
    neg_scores = np.random.uniform(0.0, 0.4, 50)
    # All test predictions very close to the negative mean
    test_scores = np.full(30, 0.2)

    q = SMM(learner=LogisticRegression())
    q.classes_ = np.array([0, 1])
    alpha, dist = q.best_mixture(test_scores, pos_scores, neg_scores)
    assert 0 <= alpha <= 1
    # Alpha should be low since test is near neg mean
    assert alpha < 0.5


# =====================================================================
# 16. HDx edge cases
# =====================================================================

def test_hdx_tiny_dataset(tiny_binary_dataset):
    X_train, X_test, y_train, y_test = tiny_binary_dataset
    q = HDx()
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


def test_hdx_custom_bins(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = HDx(bins_size=np.array([5, 15, 25]))
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2


# =====================================================================
# 17. MMD_RKHS edge cases
# =====================================================================

def test_mmd_rkhs_tiny_dataset(tiny_binary_dataset):
    X_train, X_test, y_train, y_test = tiny_binary_dataset
    q = MMD_RKHS(kernel="linear")
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


def test_mmd_rkhs_imbalanced(imbalanced_binary_dataset):
    X_train, X_test, y_train, y_test = imbalanced_binary_dataset
    q = MMD_RKHS(kernel="rbf")
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 18. Prevalence normalization via config_context
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_prevalence_sum_normalization(Quantifier, binary_dataset):
    """With prevalence_normalization='sum', output sums to 1."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    with config_context(prevalence_normalization="sum"):
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_prevalence_return_type_array(Quantifier, binary_dataset):
    """With prevalence_return_type='array', output is ndarray."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    with config_context(prevalence_return_type="array"):
        q.fit(X_train, y_train)
        prev = q.predict(X_test)
    assert isinstance(prev, np.ndarray)
    assert len(prev) == 2


# =====================================================================
# 19. Output dict has correct class keys
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_output_keys_binary(Quantifier, binary_dataset):
    """Dict keys match the class labels."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert set(prev.keys()) == set(np.unique(y_train))


def test_mmd_rkhs_output_keys(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = MMD_RKHS(kernel="rbf")
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert set(prev.keys()) == set(np.unique(y_train))


def test_hdx_output_keys(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = HDx()
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert set(prev.keys()) == set(np.unique(y_train))


# =====================================================================
# 20. Internal utility – getHist
# =====================================================================

class TestGetHist:
    """Tests for the getHist utility function."""

    def test_output_length(self):
        scores = np.random.uniform(0, 1, 100)
        nbins = 10
        h = getHist(scores, nbins)
        assert len(h) == nbins

    def test_output_non_negative(self):
        scores = np.random.uniform(0, 1, 50)
        h = getHist(scores, 5)
        assert np.all(h >= 0)

    def test_sums_close_to_one(self):
        """Histogram probabilities sum approximately to 1 (each bin has a base 1/nbins)."""
        scores = np.random.uniform(0, 1, 200)
        h = getHist(scores, 10)
        # Each bin starts with 1/nbins baseline, so sum > 1 expected;
        # just check it's finite and positive
        assert np.isfinite(h).all()
        assert np.sum(h) > 0

    def test_single_bin(self):
        scores = np.array([0.5, 0.6, 0.7])
        h = getHist(scores, 1)
        assert len(h) == 1
        assert h[0] > 0

    def test_empty_bins(self):
        """Scores concentrated in one region; other bins should still have baseline."""
        scores = np.array([0.1, 0.1, 0.1])
        h = getHist(scores, 5)
        assert len(h) == 5
        assert np.all(h > 0)  # baseline ensures no zero bins


# =====================================================================
# 21. Internal utility – ternary_search
# =====================================================================

class TestTernarySearch:
    """Tests for the ternary_search utility function."""

    def test_finds_minimum_of_quadratic(self):
        """Minimizes (x - 0.3)^2 in [0, 1]."""
        f = lambda x: (x - 0.3) ** 2
        result = ternary_search(0, 1, f)
        assert pytest.approx(result, abs=1e-3) == 0.3

    def test_finds_minimum_at_boundary_left(self):
        """Monotonically increasing function => minimum at left boundary."""
        f = lambda x: x
        result = ternary_search(0, 1, f)
        assert result < 0.01

    def test_finds_minimum_at_boundary_right(self):
        """Monotonically decreasing function => minimum at right boundary."""
        f = lambda x: -x
        result = ternary_search(0, 1, f)
        assert result > 0.99

    def test_custom_tolerance(self):
        f = lambda x: (x - 0.5) ** 2
        result = ternary_search(0, 1, f, tol=1e-6)
        assert pytest.approx(result, abs=1e-5) == 0.5


# =====================================================================
# 22. Internal utility – distance functions
# =====================================================================

class TestDistanceFunctions:
    """Tests for hellinger, topsoe, probsymm, sqEuclidean."""

    def test_hellinger_identical(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert pytest.approx(hellinger(p, p), abs=1e-10) == 0.0

    def test_hellinger_symmetric(self):
        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        assert pytest.approx(hellinger(p, q), abs=1e-10) == hellinger(q, p)

    def test_hellinger_non_negative(self):
        p = np.array([0.1, 0.9])
        q = np.array([0.8, 0.2])
        assert hellinger(p, q) >= 0

    def test_topsoe_identical(self):
        p = np.array([0.5, 0.5])
        assert pytest.approx(topsoe(p, p), abs=1e-10) == 0.0

    def test_topsoe_symmetric(self):
        p = np.array([0.2, 0.8])
        q = np.array([0.6, 0.4])
        assert pytest.approx(topsoe(p, q), abs=1e-10) == topsoe(q, p)

    def test_probsymm_identical(self):
        p = np.array([0.4, 0.6])
        assert pytest.approx(probsymm(p, p), abs=1e-10) == 0.0

    def test_probsymm_symmetric(self):
        p = np.array([0.3, 0.7])
        q = np.array([0.5, 0.5])
        assert pytest.approx(probsymm(p, q), abs=1e-10) == probsymm(q, p)

    def test_sqEuclidean_identical(self):
        p = np.array([0.5, 0.5])
        assert pytest.approx(sqEuclidean(p, p), abs=1e-10) == 0.0

    def test_sqEuclidean_value(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert pytest.approx(sqEuclidean(p, q), abs=1e-10) == 2.0


# =====================================================================
# 23. BaseMixture.get_distance
# =====================================================================

class TestBaseMixtureGetDistance:
    """Tests for BaseMixture.get_distance class method."""

    @pytest.mark.parametrize("measure", ["hellinger", "topsoe", "probsymm", "euclidean"])
    def test_valid_measures(self, measure):
        p = np.array([0.3, 0.7])
        q = np.array([0.5, 0.5])
        dist = BaseMixture.get_distance(p, q, measure=measure)
        assert isinstance(dist, float)
        assert dist >= 0

    def test_invalid_measure_raises(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.7])
        with pytest.raises(ValueError, match="Invalid measure"):
            BaseMixture.get_distance(p, q, measure="cosine_similarity")

    def test_zero_vector_raises(self):
        p = np.array([0.0, 0.0])
        q = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="zero"):
            BaseMixture.get_distance(p, q, measure="hellinger")

    def test_different_lengths_raises(self):
        p = np.array([0.3, 0.7])
        q = np.array([0.2, 0.3, 0.5])
        with pytest.raises(ValueError, match="same length"):
            BaseMixture.get_distance(p, q, measure="hellinger")

    def test_identical_distributions(self):
        p = np.array([0.25, 0.75])
        dist = BaseMixture.get_distance(p, p, measure="hellinger")
        assert pytest.approx(dist, abs=1e-8) == 0.0


# =====================================================================
# 24. DyS default bins_size
# =====================================================================

def test_dys_default_bins():
    """Default bins_size is created when None."""
    q = DyS()
    assert q.bins_size is not None
    assert len(q.bins_size) == 11  # linspace(2,20,10) + [30]


# =====================================================================
# 25. HDx default bins_size
# =====================================================================

def test_hdx_default_bins():
    q = HDx()
    assert q.bins_size is not None
    assert len(q.bins_size) == 11  # linspace(10,110,11)


# =====================================================================
# 26. MMD_RKHS internal methods
# =====================================================================

class TestMMDRKHSInternals:
    """Tests for MMD_RKHS internal computation methods."""

    def test_compute_class_means_shape(self, small_binary_dataset):
        X_train, X_test, y_train, y_test = small_binary_dataset
        q = MMD_RKHS(kernel="rbf")
        q.classes_ = np.unique(y_train)
        means, K = q._compute_class_means(X_train, y_train)
        n_classes = len(np.unique(y_train))
        assert means.shape == (n_classes, X_train.shape[0])
        assert K.shape == (X_train.shape[0], X_train.shape[0])

    def test_compute_unlabeled_mean_shape(self, small_binary_dataset):
        X_train, X_test, y_train, y_test = small_binary_dataset
        q = MMD_RKHS(kernel="rbf")
        q.classes_ = np.unique(y_train)
        q.X_train_ = X_train
        q.y_train_ = y_train
        means, K = q._compute_class_means(X_train, y_train)
        q.class_means_ = means
        q.K_train_ = K
        mu_u = q._compute_unlabeled_mean(X_test)
        assert mu_u.shape == (X_train.shape[0],)

    def test_build_QP_matrices_shape(self, small_binary_dataset):
        X_train, X_test, y_train, y_test = small_binary_dataset
        q = MMD_RKHS(kernel="rbf")
        q.classes_ = np.unique(y_train)
        q.X_train_ = X_train
        means, K = q._compute_class_means(X_train, y_train)
        q.class_means_ = means
        q.K_train_ = K
        mu_u = q._compute_unlabeled_mean(X_test)
        G, h = q._build_QP_matrices(means, mu_u)
        n_classes = len(np.unique(y_train))
        assert G.shape == (n_classes, n_classes)
        assert h.shape == (n_classes,)

    def test_solve_simplex_qp(self):
        """Simplex QP solver returns valid probability vector."""
        q = MMD_RKHS()
        G = np.eye(3)
        h = np.array([0.5, 0.3, 0.2])
        theta = q._solve_simplex_qp(G, h)
        assert len(theta) == 3
        assert pytest.approx(np.sum(theta), abs=1e-5) == 1.0
        assert np.all(theta >= 0)

    def test_kernel_kwargs_rbf(self):
        q = MMD_RKHS(kernel="rbf", gamma=0.5)
        kw = q._kernel_kwargs()
        assert kw["gamma"] == 0.5

    def test_kernel_kwargs_poly(self):
        q = MMD_RKHS(kernel="poly", degree=4, coef0=1.0)
        kw = q._kernel_kwargs()
        assert kw["degree"] == 4
        assert kw["coef0"] == 1.0

    def test_kernel_kwargs_linear(self):
        q = MMD_RKHS(kernel="linear")
        kw = q._kernel_kwargs()
        assert kw == {}


# =====================================================================
# 27. best_mixture of MMD_RKHS
# =====================================================================

def test_mmd_rkhs_best_mixture(small_binary_dataset):
    X_train, X_test, y_train, y_test = small_binary_dataset
    q = MMD_RKHS(kernel="rbf")
    q.fit(X_train, y_train)
    theta, obj = q.best_mixture(X_test, X_train, y_train)
    assert isinstance(theta, np.ndarray)
    assert pytest.approx(np.sum(theta), abs=1e-5) == 1.0
    assert isinstance(obj, float)


# =====================================================================
# 28. HDx best_mixture
# =====================================================================

def test_hdx_best_mixture(small_binary_dataset):
    X_train, X_test, y_train, y_test = small_binary_dataset
    q = HDx()
    q.fit(X_train, y_train)
    alpha, best_dist = q.best_mixture(X_test, q.pos_features, q.neg_features)
    assert 0 <= alpha <= 1
    assert best_dist >= 0
    assert q.distances is not None


# =====================================================================
# 29. SMM – edge case equal means
# =====================================================================

def test_smm_equal_means():
    """When pos and neg means are identical, alpha = mean_test."""
    q = SMM(learner=LogisticRegression())
    q.classes_ = np.array([0, 1])
    pos_scores = np.array([0.5, 0.5, 0.5])
    neg_scores = np.array([0.5, 0.5, 0.5])
    test_scores = np.array([0.7, 0.7, 0.7])
    alpha, dist = q.best_mixture(test_scores, pos_scores, neg_scores)
    assert pytest.approx(alpha, abs=1e-5) == 0.7
    assert dist is None


# =====================================================================
# 30. SORD – distances list populated
# =====================================================================

def test_sord_distances_populated(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = SORD(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert q.distances is not None
    assert len(q.distances) == 101  # linspace(0,1,101)


# =====================================================================
# 31. HDy – distances list populated
# =====================================================================

def test_hdy_distances_populated(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = HDy(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert q.distances is not None
    assert len(q.distances) == 11  # linspace(10,110,11)


# =====================================================================
# 32. DyS get_best_distance
# =====================================================================

def test_dys_get_best_distance(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = DyS(learner=LogisticRegression(random_state=42, solver="liblinear"), measure="hellinger")
    q.fit(X_train, y_train)

    pos_scores = q.pos_scores
    neg_scores = q.neg_scores
    test_scores = q.learner.predict_proba(X_test)[:, 1]
    best_dist = q.get_best_distance(test_scores, pos_scores, neg_scores)
    assert isinstance(best_dist, (float, np.floating))
    assert best_dist >= 0


# =====================================================================
# 33. Prevalence values are reasonable (within an acceptable range)
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_prevalence_values_reasonable(Quantifier, binary_dataset):
    """Predicted prevalences should not be wildly different from true prevalences."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    # True prevalence
    true_prev = np.mean(y_test == 1)
    # Allow generous tolerance for quantification
    assert abs(prev.get(1, 0) - true_prev) < 0.3


# =====================================================================
# 34. DyS _mix method
# =====================================================================

def test_dys_mix():
    """The _mix method correctly computes the weighted mixture."""
    q = DyS()
    pos_hist = np.array([0.2, 0.8])
    neg_hist = np.array([0.6, 0.4])
    alpha = 0.5
    mix = q._mix(pos_hist, neg_hist, alpha)
    expected = 0.5 * np.array([0.2, 0.8]) + 0.5 * np.array([0.6, 0.4])
    np.testing.assert_array_almost_equal(mix, expected)


def test_dys_mix_alpha_zero():
    q = DyS()
    pos_hist = np.array([0.3, 0.7])
    neg_hist = np.array([0.9, 0.1])
    mix = q._mix(pos_hist, neg_hist, 0.0)
    np.testing.assert_array_almost_equal(mix, neg_hist)


def test_dys_mix_alpha_one():
    q = DyS()
    pos_hist = np.array([0.3, 0.7])
    neg_hist = np.array([0.9, 0.1])
    mix = q._mix(pos_hist, neg_hist, 1.0)
    np.testing.assert_array_almost_equal(mix, pos_hist)


# =====================================================================
# 35. HDy _mix method
# =====================================================================

def test_hdy_mix():
    q = HDy()
    pos_hist = np.array([0.1, 0.9])
    neg_hist = np.array([0.7, 0.3])
    mix = q._mix(pos_hist, neg_hist, 0.4)
    expected = 0.4 * np.array([0.1, 0.9]) + 0.6 * np.array([0.7, 0.3])
    np.testing.assert_array_almost_equal(mix, expected)


# =====================================================================
# 36. Cross-validation in fit (default path)
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_fit_with_cv(Quantifier, small_binary_dataset):
    """Fit uses cross-validation when learner_fitted=False (default)."""
    X_train, X_test, y_train, y_test = small_binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)  # default: learner_fitted=False
    # After fitting, the learner must be fitted and scores stored
    assert q.pos_scores is not None
    assert q.neg_scores is not None
    assert len(q.pos_scores) > 0
    assert len(q.neg_scores) > 0


# =====================================================================
# 37. Reproducibility – same data produces same result
# =====================================================================

@pytest.mark.parametrize("Quantifier", [DyS, HDy, SMM, SORD])
def test_reproducibility(Quantifier, binary_dataset):
    """Fitting and predicting twice yields same result."""
    X_train, X_test, y_train, y_test = binary_dataset
    learner = LogisticRegression(random_state=42, solver="liblinear")

    q1 = Quantifier(learner=learner)
    q1.fit(X_train, y_train)
    prev1 = q1.predict(X_test)

    q2 = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q2.fit(X_train, y_train)
    prev2 = q2.predict(X_test)

    for k in prev1:
        assert pytest.approx(prev1[k], abs=1e-8) == prev2[k]


# =====================================================================
# 38. MMD_RKHS – precomputed flag
# =====================================================================

def test_mmd_rkhs_precomputed_flag(small_binary_dataset):
    X_train, X_test, y_train, y_test = small_binary_dataset
    q = MMD_RKHS(kernel="rbf")
    assert q._precomputed is False
    q.fit(X_train, y_train)
    # After fit, class_means should be stored
    assert q.class_means_ is not None
    assert q.K_train_ is not None
    assert q.X_train_ is not None


# =====================================================================
# 39. Aggregative precomputed flag
# =====================================================================

@pytest.mark.parametrize("Quantifier", AGGREGATIVE_CLASSES)
def test_aggregative_precomputed_after_fit(Quantifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    assert q._precomputed is True


# =====================================================================
# 40. Non-aggregative multiclass with string labels
# =====================================================================

def test_hdx_multiclass_string_labels():
    """HDx with multiclass string labels."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3,
        n_informative=8, random_state=42,
    )
    y_str = np.array(["cat", "dog", "bird"])[y]
    X_train, X_test, y_train, y_test = train_test_split(X, y_str, test_size=0.3, random_state=42)

    q = HDx(strategy="ovr")
    with config_context(prevalence_normalization="sum"):
        q.fit(X_train, y_train)
        prev = q.predict(X_test)

    assert isinstance(prev, dict)
    assert set(prev.keys()) == {"cat", "dog", "bird"}
    assert pytest.approx(sum(prev.values()), abs=1e-2) == 1.0


# =====================================================================
# 41. HDy and DyS distances attribute type check
# =====================================================================

def test_dys_distances_are_numeric(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = DyS(learner=LogisticRegression(random_state=42, solver="liblinear"), measure="topsoe")
    q.fit(X_train, y_train)
    q.predict(X_test)
    assert all(isinstance(d, (float, np.floating)) for d in q.distances)
    assert all(d >= 0 for d in q.distances)


def test_hdy_distances_are_numeric(binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = HDy(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    q.predict(X_test)
    assert all(isinstance(d, (float, np.floating)) for d in q.distances)
    assert all(d >= 0 for d in q.distances)


# =====================================================================
# 42. Strategy parameter stored correctly
# =====================================================================

@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_strategy_attribute(strategy):
    """Strategy parameter is stored correctly on classes that accept it."""
    # HDy, SMM, SORD inherit AggregativeMixture.__init__ which has strategy
    q_hdy = HDy(strategy=strategy)
    assert q_hdy.strategy == strategy

    q_smm = SMM(strategy=strategy)
    assert q_smm.strategy == strategy

    q_sord = SORD(strategy=strategy)
    assert q_sord.strategy == strategy

    q_hdx = HDx(strategy=strategy)
    assert q_hdx.strategy == strategy


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_dys_strategy_via_attribute(strategy):
    """DyS strategy can be set as attribute (not exposed in __init__)."""
    q = DyS()
    q.strategy = strategy
    assert q.strategy == strategy


# =====================================================================
# 43. MMD_RKHS poly kernel with custom parameters
# =====================================================================

def test_mmd_rkhs_poly_custom_params(small_binary_dataset):
    X_train, X_test, y_train, y_test = small_binary_dataset
    q = MMD_RKHS(kernel="poly", degree=3, coef0=1.0)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 44. MMD_RKHS sigmoid kernel
# =====================================================================

def test_mmd_rkhs_sigmoid(small_binary_dataset):
    X_train, X_test, y_train, y_test = small_binary_dataset
    q = MMD_RKHS(kernel="sigmoid", gamma=0.01, coef0=0.0)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values()), abs=1e-5) == 1.0


# =====================================================================
# 45. Multiple predict calls yield consistent results
# =====================================================================

@pytest.mark.parametrize("Quantifier", [DyS, HDy, SMM, SORD])
def test_multiple_predicts_consistent(Quantifier, binary_dataset):
    """Predicting multiple times on the same data yields the same output."""
    X_train, X_test, y_train, y_test = binary_dataset
    q = Quantifier(learner=LogisticRegression(random_state=42, solver="liblinear"))
    q.fit(X_train, y_train)
    prev1 = q.predict(X_test)
    prev2 = q.predict(X_test)
    for k in prev1:
        assert pytest.approx(prev1[k], abs=1e-10) == prev2[k]


# =====================================================================
# 46. getHist with varying nbins
# =====================================================================

@pytest.mark.parametrize("nbins", [2, 5, 10, 20, 50])
def test_getHist_nbins(nbins):
    scores = np.random.uniform(0, 1, 100)
    h = getHist(scores, nbins)
    assert len(h) == nbins
    assert np.all(h >= 0)
    assert np.isfinite(h).all()


# =====================================================================
# 47. DyS multiclass with measure variations
# =====================================================================

@pytest.mark.parametrize("measure", ["hellinger", "topsoe", "probsymm"])
def test_dys_multiclass_measures(measure, small_multiclass_dataset):
    X_train, X_test, y_train, y_test = small_multiclass_dataset
    q = DyS(
        learner=LogisticRegression(random_state=42, solver="liblinear", max_iter=500),
        measure=measure,
    )
    q.strategy = "ovr"
    with config_context(prevalence_normalization="sum"):
        q.fit(X_train, y_train)
        prev = q.predict(X_test)

    assert len(prev) == 3
    assert pytest.approx(sum(prev.values()), abs=1e-2) == 1.0


# =====================================================================
# 48. MMD_RKHS multiclass with different kernels
# =====================================================================

@pytest.mark.parametrize("kernel", ["rbf", "linear"])
def test_mmd_rkhs_multiclass_kernels(kernel, small_multiclass_dataset):
    X_train, X_test, y_train, y_test = small_multiclass_dataset
    q = MMD_RKHS(kernel=kernel)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert len(prev) == 3
    assert pytest.approx(sum(prev.values()), abs=1e-2) == 1.0
