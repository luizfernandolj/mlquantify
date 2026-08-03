
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

from mlquantify._config import config_context
from mlquantify.elm import (
    ELM,
    SVMQ,
    SVMKLD,
    SVMNKLD,
    SVMAE,
    SVMRAE,
    MultivariateLossSVM,
)

ALL_QUANTIFIERS = [SVMQ, SVMKLD, SVMNKLD, SVMAE, SVMRAE]


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ learner

def test_error_loss_recovers_standard_svm(binary_dataset):
    """With the error-rate loss, SVM_multi is equivalent to a standard
    linear SVM (Joachims, 2005, Theorem 3)."""
    X, y = binary_dataset
    Xb = np.hstack([X, np.ones((len(X), 1))])  # constant feature = intercept
    elm = MultivariateLossSVM(loss="error").fit(Xb, y)
    svc = LinearSVC().fit(X, y)
    acc_elm = (elm.predict(Xb) == y).mean()
    acc_svc = (svc.predict(X) == y).mean()
    assert abs(acc_elm - acc_svc) < 0.05


@pytest.mark.parametrize("loss", ["q", "kld", "nkld", "ae", "rae", "error", "f1"])
def test_learner_all_losses(binary_dataset, loss):
    X, y = binary_dataset
    clf = MultivariateLossSVM(loss=loss, max_iter=100).fit(X, y)
    assert clf.n_iter_ <= 100
    preds = clf.predict(X)
    assert set(preds) <= set(clf.classes_)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_learner_callable_loss(binary_dataset):
    X, y = binary_dataset
    def my_loss(a, b, n_pos, n_neg):
        c = n_pos - a
        return 100.0 * np.abs(b - c) / (n_pos + n_neg)
    clf = MultivariateLossSVM(loss=my_loss, max_iter=60).fit(X, y)
    assert set(clf.predict(X)) <= set(clf.classes_)


def test_learner_deterministic(binary_dataset):
    X, y = binary_dataset
    w1 = MultivariateLossSVM(loss="q", max_iter=60).fit(X, y).coef_
    w2 = MultivariateLossSVM(loss="q", max_iter=60).fit(X, y).coef_
    np.testing.assert_allclose(w1, w2)


def test_learner_rejects_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    with pytest.raises(ValueError):
        MultivariateLossSVM().fit(X, y)


def test_quantification_loss_balances_errors():
    """The AE loss drives training FP and FN toward each other, more so
    than the plain error loss on an imbalanced problem."""
    X, y = make_classification(n_samples=600, n_features=10, n_informative=6,
                               weights=[0.75, 0.25], class_sep=0.8,
                               random_state=0)
    def imbalance(clf):
        pred = clf.predict(X)
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        return abs(fp - fn)
    balanced = imbalance(MultivariateLossSVM(loss="ae", max_iter=150).fit(X, y))
    plain = imbalance(MultivariateLossSVM(loss="error", max_iter=150).fit(X, y))
    assert balanced <= plain


# --------------------------------------------------------------- quantifiers

@pytest.mark.parametrize("quantifier_class", ALL_QUANTIFIERS)
def test_quantifier_fit_predict_binary(quantifier_class, binary_dataset):
    X, y = binary_dataset
    q = quantifier_class(max_iter=100)
    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=2)


def test_quantifier_multiclass_ovr(multiclass_dataset):
    X, y = multiclass_dataset
    q = SVMQ(max_iter=60)
    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=3)


def test_svmq_recovers_shifted_prevalence():
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6,
                               class_sep=1.5, random_state=0)
    q = SVMQ(max_iter=150).fit(X, y)
    rng = np.random.RandomState(0)
    idx1, idx0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for prev in (0.2, 0.8):
        n1 = int(prev * 400)
        take = np.concatenate([
            rng.choice(idx0, 400 - n1, replace=True),
            rng.choice(idx1, n1, replace=True),
        ])
        with config_context(prevalence_return_type="array"):
            estimate = q.predict(X[take])
        true = np.array([1 - prev, prev])
        assert np.abs(estimate - true).mean() < 0.2


def test_elm_set_params_propagates(binary_dataset):
    X, y = binary_dataset
    q = ELM().set_params(loss="ae", C=2.0, max_iter=60).fit(X, y)
    assert q.estimator_.loss == "ae"
    assert q.estimator_.C == 2.0


def test_elm_get_params_roundtrip():
    q = ELM(loss="kld", C=0.5, beta=2.0, max_iter=50)
    params = q.get_params()
    q2 = ELM().set_params(**params)
    assert q2.get_params() == params


def test_elm_invalid_loss(binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError):
        ELM(loss="gini").fit(X, y)
