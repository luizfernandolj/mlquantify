"""Quick performance benchmark for the matching kernel and threshold sweeps.

Run::

    python benchmarks/bench_matching.py

Prints a Python-vs-Cython table so you can see the speedup directly. For tracked
regression testing across commits use ``asv`` (see docs/dev/cython_plan.md).
"""
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import mlquantify.matching._histogram as H
from mlquantify.matching import DyS, HDy
from mlquantify.counting import evaluate_thresholds


def _ms(fn, n):
    fn()
    s = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - s) / n * 1e3


def bench_kernel():
    from mlquantify.matching._matching_py import match_sweep as py
    try:
        from mlquantify.matching._matching_fast import match_sweep as cy
        built = True
    except ImportError:
        cy, built = py, False

    rng = np.random.default_rng(0)
    neg, pos, test = (rng.random(20) for _ in range(3))
    print(f"\n[1] single sweep, 20 bins, topsoe/ternary   (compiled: {built})")
    tp = _ms(lambda: py(neg, pos, test, 1), 5000) * 1000
    tc = _ms(lambda: cy(neg, pos, test, 1), 20000) * 1000
    print(f"    python : {tp:8.1f} us")
    print(f"    cython : {tc:8.1f} us    speedup {tp / tc:5.1f}x")


def bench_predict():
    X, y = make_classification(n_samples=4000, weights=[0.6, 0.4], random_state=0)
    Xtr, Xte, ytr = X[:2000], X[2000:], y[:2000]
    print("\n[2] end-to-end predict, 1000 test samples")
    print(f"    {'method':8}{'kernel ms':>12}{'solver ms':>12}{'speedup':>10}")
    for name, Q in (("DyS", DyS), ("HDy", HDy)):
        q = Q(LogisticRegression(max_iter=300)).fit(Xtr, ytr)
        old = H.USE_SWEEP_KERNEL
        H.USE_SWEEP_KERNEL = True
        tk = _ms(lambda: q.predict(Xte[:1000]), 30)
        H.USE_SWEEP_KERNEL = False
        ts = _ms(lambda: q.predict(Xte[:1000]), 30)
        H.USE_SWEEP_KERNEL = old
        print(f"    {name:8}{tk:12.2f}{ts:12.2f}{ts / tk:9.1f}x")


def bench_thresholds():
    rng = np.random.default_rng(0)
    print("\n[3] evaluate_thresholds, score_edges='auto' (vectorised; was O(n^2))")
    for n in (2000, 8000):
        y = (rng.random(n) > 0.5).astype(int)
        p = rng.random(n)
        t = _ms(lambda: evaluate_thresholds(y, p, score_edges="auto"), 20)
        print(f"    n={n:>5}: {t:7.2f} ms")


if __name__ == "__main__":
    bench_kernel()
    bench_predict()
    bench_thresholds()
