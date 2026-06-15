"""Correctness + parity tests for the performance optimisations.

Each optimisation keeps a brute-force reference (the previous behaviour) inline
and asserts the fast path produces identical results.
"""
import time

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


# ---------------------------------------------------------------------------
# 1) evaluate_thresholds: vectorised vs original O(T*n) loop
# ---------------------------------------------------------------------------
def _evaluate_thresholds_reference(y, probabilities, score_edges="fixed"):
    classes = np.unique(y)
    scores = np.linspace(0, 1, 101) if score_edges == "fixed" else np.unique(probabilities)
    tprs, fprs = [], []
    for t in scores:
        y_pred = np.where(probabilities >= t, classes[1], classes[0])
        TP = np.logical_and(y == y_pred, y == classes[1]).sum()
        FP = np.logical_and(y != y_pred, y == classes[0]).sum()
        FN = np.logical_and(y != y_pred, y == classes[1]).sum()
        TN = np.logical_and(y == y_pred, y == classes[0]).sum()
        tprs.append(TP / (TP + FN) if (TP + FN) else 0)
        fprs.append(FP / (FP + TN) if (FP + TN) else 0)
    return scores, np.asarray(tprs), np.asarray(fprs)


@pytest.mark.parametrize("score_edges", ["fixed", "auto"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_evaluate_thresholds_parity(score_edges, seed):
    from mlquantify.counting import evaluate_thresholds

    rng = np.random.default_rng(seed)
    n = 500
    y = (rng.random(n) > 0.45).astype(int)
    proba = rng.random(n)

    t_fast, tpr_fast, fpr_fast = evaluate_thresholds(y, proba, score_edges)
    t_ref, tpr_ref, fpr_ref = _evaluate_thresholds_reference(y, proba, score_edges)

    np.testing.assert_allclose(t_fast, t_ref)
    np.testing.assert_allclose(tpr_fast, tpr_ref)
    np.testing.assert_allclose(fpr_fast, fpr_ref)


def test_evaluate_thresholds_is_subquadratic():
    """The vectorised version must not blow up quadratically on unique scores."""
    from mlquantify.counting import evaluate_thresholds

    rng = np.random.default_rng(0)

    def run(n):
        y = (rng.random(n) > 0.5).astype(int)
        proba = rng.random(n)
        t0 = time.perf_counter()
        evaluate_thresholds(y, proba, score_edges="auto")
        return time.perf_counter() - t0

    run(1000)  # warmup
    t_small = max(run(1000), 1e-4)
    t_large = run(8000)
    # 8x the data must cost far less than the ~64x a quadratic loop would.
    assert t_large / t_small < 20


# ---------------------------------------------------------------------------
# 2) Median Sweep: vectorised CC-per-threshold vs the original loop
# ---------------------------------------------------------------------------
def _median_sweep_reference(get_best, predictions, train_scores, y_train):
    from mlquantify.counting import evaluate_thresholds, CC
    from mlquantify import config_context

    thr, tprs, fprs = evaluate_thresholds(y_train, train_scores[:, 1])
    thr, tprs, fprs = get_best(thr, tprs, fprs)
    prevs = []
    for t, tpr, fpr in zip(thr, tprs, fprs):
        with config_context(prevalence_return_type="array"):
            cc = CC(threshold=t).aggregate(predictions, y_train)[1]
        prevs.append(cc if (tpr - fpr) == 0 else np.clip((cc - fpr) / (tpr - fpr), 0, 1))
    p = float(np.median(prevs))
    return np.array([1 - p, p])


@pytest.mark.parametrize("cls_name", ["MS", "MS2"])
@pytest.mark.parametrize("seed", [0, 1])
def test_median_sweep_parity(cls_name, seed):
    from mlquantify.counting import MS, MS2

    cls = {"MS": MS, "MS2": MS2}[cls_name]
    rng = np.random.default_rng(seed)
    n, m = 600, 300
    y = (rng.random(n) > 0.5).astype(int)
    train_scores = np.column_stack([1 - rng.random(n), rng.random(n)])
    predictions = np.column_stack([1 - rng.random(m), rng.random(m)])

    q = cls()
    fast = q._adjust(predictions, train_scores, y)
    ref = _median_sweep_reference(q.get_best_threshold, predictions, train_scores, y)
    np.testing.assert_allclose(fast, ref, atol=1e-9)


@pytest.mark.parametrize("cls_name", ["MS", "MS2"])
def test_median_sweep_end_to_end(cls_name):
    """The full fit/predict path still yields a valid prevalence."""
    from mlquantify.counting import MS, MS2

    cls = {"MS": MS, "MS2": MS2}[cls_name]
    X, y = make_classification(n_samples=400, random_state=0)
    pred = cls(LogisticRegression(max_iter=200)).fit(X, y).predict(X)
    vals = np.array(list(pred.values()) if isinstance(pred, dict) else np.ravel(pred))
    assert np.all(vals >= 0) and abs(vals.sum() - 1) < 1e-6


# ---------------------------------------------------------------------------
# 3) Cython matching kernel: compiled == pure-Python, and == generic solver
# ---------------------------------------------------------------------------
def test_matching_kernel_matches_fallback():
    """Compiled kernel (if built) must match the pure-Python reference."""
    import mlquantify.matching._histogram as H
    if not H._HAS_FAST_KERNEL:
        pytest.skip("compiled _histogram_sweep not built")

    from mlquantify.matching._histogram_sweep import match_sweep as fast
    from mlquantify.matching._histogram_sweep_py import match_sweep as py

    rng = np.random.default_rng(0)
    worst = 0.0
    for n_bins in (8, 16, 64):
        for _ in range(100):
            neg, pos, test = (rng.random(n_bins) for _ in range(3))
            for metric in (0, 1, 2):
                for solver in (0, 1):
                    worst = max(worst, abs(
                        fast(neg, pos, test, metric, solver)
                        - py(neg, pos, test, metric, solver)))
    assert worst < 1e-6


@pytest.mark.parametrize("name", ["DyS", "HDy", "HDx", "DyS-hellinger"])
def test_histogram_kernel_predict_parity(name):
    """DyS/HDy/HDx predictions are unchanged by the sweep kernel."""
    import mlquantify.matching._histogram as H
    from mlquantify.matching import DyS, HDy, HDx

    X, y = make_classification(n_samples=800, weights=[0.6, 0.4], random_state=1)
    Xtr, Xte, ytr = X[:500], X[500:], y[:500]
    build = {
        "DyS": lambda: DyS(LogisticRegression(max_iter=300)),
        "HDy": lambda: HDy(LogisticRegression(max_iter=300)),
        "HDx": lambda: HDx(),
        "DyS-hellinger": lambda: DyS(LogisticRegression(max_iter=300), distance="hellinger"),
    }[name]
    q = build().fit(Xtr, ytr)

    def prev():
        p = q.predict(Xte)
        return np.array(list(p.values()) if isinstance(p, dict) else np.ravel(p))

    old = H.USE_SWEEP_KERNEL
    try:
        H.USE_SWEEP_KERNEL = True
        with_kernel = prev()
        H.USE_SWEEP_KERNEL = False
        without_kernel = prev()
    finally:
        H.USE_SWEEP_KERNEL = old

    np.testing.assert_allclose(with_kernel, without_kernel, atol=1e-3)


# ---------------------------------------------------------------------------
# 4) GridSearchQ: precomputed batches must give IDENTICAL results to the
#    original per-combination loop (protocol regenerated each time).
# ---------------------------------------------------------------------------
def _gridsearch_reference(make_q, grid, X, y, *, protocol, samples_sizes,
                          n_repetitions, scoring, val_split=0.4, random_seed=42):
    import itertools
    from copy import deepcopy
    from sklearn.model_selection import train_test_split
    from mlquantify.utils.prevalence import get_prev_from_labels
    from mlquantify.model_selection import APP, NPP, UPP

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=val_split, random_state=random_seed)
    base = make_q()
    combos = list(itertools.product(*grid.values()))

    def proto():
        if protocol == "app":
            return APP(batch_size=samples_sizes, n_prevalences=n_repetitions,
                       repeats=n_repetitions, random_state=random_seed, min_prev=0.0, max_prev=1.0)
        if protocol == "npp":
            return NPP(batch_size=samples_sizes, n_samples=n_repetitions,
                       repeats=n_repetitions, random_state=random_seed)
        return UPP(batch_size=samples_sizes, n_prevalences=n_repetitions,
                   repeats=n_repetitions, random_state=random_seed, min_prev=0.0, max_prev=1.0)

    best_score, best_combo = None, None
    for combo in combos:
        model = deepcopy(base)
        model.set_params(**dict(zip(grid.keys(), combo)))
        p = proto()
        model.fit(Xtr, ytr)
        errs = [scoring(get_prev_from_labels(yva[idx]), model.predict(Xva[idx]))
                for idx in p.split(Xva, yva)]
        s = float(np.mean(errs))
        if best_score is None or s < best_score:
            best_score, best_combo = s, combo
    return dict(zip(grid.keys(), best_combo)), best_score


@pytest.mark.parametrize("protocol", ["app", "npp", "upp"])
def test_gridsearch_matches_reference_counting(protocol):
    from mlquantify.model_selection import GridSearchQ
    from mlquantify.counting import CC
    from mlquantify.metrics import MAE

    X, y = make_classification(n_samples=500, weights=[0.6, 0.4], random_state=0)
    make_q = lambda: CC(LogisticRegression(max_iter=300))
    grid = {"threshold": [0.3, 0.5, 0.7]}

    gs = GridSearchQ(quantifier=make_q(), param_grid=grid, protocol=protocol,
                     samples_sizes=60, n_repetitions=3, scoring=MAE, refit=False).fit(X, y)
    ref_params, ref_score = _gridsearch_reference(
        make_q, grid, X, y, protocol=protocol, samples_sizes=60, n_repetitions=3, scoring=MAE)

    assert gs.best_params_ == ref_params
    assert gs.best_score_ == pytest.approx(ref_score, rel=1e-12)


def test_gridsearch_matches_reference_multiparam():
    from mlquantify.model_selection import GridSearchQ
    from mlquantify.matching import DyS
    from mlquantify.metrics import MAE

    X, y = make_classification(n_samples=500, weights=[0.55, 0.45], random_state=2)
    make_q = lambda: DyS(LogisticRegression(max_iter=300))
    grid = {"distance": ["topsoe", "hellinger"], "bins_size": [[8], [16]]}

    gs = GridSearchQ(quantifier=make_q(), param_grid=grid, protocol="app",
                     samples_sizes=60, n_repetitions=3, scoring=MAE, refit=False).fit(X, y)
    ref_params, ref_score = _gridsearch_reference(
        make_q, grid, X, y, protocol="app", samples_sizes=60, n_repetitions=3, scoring=MAE)

    assert gs.best_params_ == ref_params
    assert gs.best_score_ == pytest.approx(ref_score, rel=1e-12)


def test_gridsearch_deterministic_and_refit():
    from mlquantify.model_selection import GridSearchQ
    from mlquantify.counting import CC
    from mlquantify.metrics import MAE

    X, y = make_classification(n_samples=400, random_state=1)
    make_q = lambda: CC(LogisticRegression(max_iter=300))
    grid = {"threshold": [0.4, 0.5, 0.6]}

    def run():
        return GridSearchQ(quantifier=make_q(), param_grid=grid, protocol="app",
                           samples_sizes=50, n_repetitions=2, scoring=MAE).fit(X, y)

    a, b = run(), run()
    assert a.best_params_ == b.best_params_
    assert a.best_score_ == pytest.approx(b.best_score_, rel=1e-12)
    # refit=True (default) exposes a usable best model
    assert hasattr(a, "best_model_")
    pred = a.predict(X)
    vals = np.array(list(pred.values()) if isinstance(pred, dict) else np.ravel(pred))
    assert abs(vals.sum() - 1) < 1e-6
