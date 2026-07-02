.. _sphx_grid_search:

====================================
Tuning a quantifier with GridSearchQ
====================================

Hyper-parameters should be selected to minimise *quantification* error, not
classification error — and they should be evaluated across a range of
prevalences, not on a single validation split.
:class:`~mlquantify.model_selection.GridSearchQ` does both: it scores every
candidate with an evaluation protocol and a quantification metric, then refits
the best one.

This example tunes the ``bandwidth`` of a :class:`~mlquantify.matching.KDEyML`
quantifier. We let ``GridSearchQ`` pick the winner, then draw the validation
error across the whole grid to show *why* that value won.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    from mlquantify import set_config
    from mlquantify.datasets import make_quantification
    from mlquantify.matching import KDEyML
    from mlquantify.metrics import MAE
    from mlquantify.model_selection import GridSearchQ

    set_config(prevalence_return_type="array")

    # A balanced training sample plus validation bags on a grid of
    # prevalence values across the whole [0, 1] range.
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        batch_size=100, return_train=True, train_size=1500,
        train_prevalence=[0.5, 0.5],
        prevalence="grid", n_prevalences=11, repeats=3,
        n_features=20, random_state=0,
    )

    bandwidths = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]

    search = GridSearchQ(
        quantifier=KDEyML(LogisticRegression(max_iter=1000)),
        param_grid={"bandwidth": bandwidths},
        protocol="app", samples_sizes=100, n_repetitions=5,
        scoring=MAE, random_seed=0,
    ).fit(Xtr, ytr)

    # Re-trace the validation-error curve the search optimised over,
    # scoring each candidate on the prevalence-swept validation bags.
    scores = []
    for bw in bandwidths:
        q = KDEyML(LogisticRegression(max_iter=1000), bandwidth=bw).fit(Xtr, ytr)
        pred = np.vstack([q.predict(Xb) for Xb in Xs])
        scores.append(MAE(prevs, pred))

    best_bw = search.best_params_["bandwidth"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bandwidths, scores, "o-", color="#264653")
    ax.axvline(best_bw, color="#e63946", ls="--",
               label=f"GridSearchQ pick: bandwidth={best_bw}")
    ax.set_xlabel("KDEyML bandwidth")
    ax.set_ylabel("Validation MAE (APP)")
    ax.set_title("Quantification-aware hyper-parameter selection")
    ax.legend()
    fig.tight_layout()

The curve is U-shaped: a too-small bandwidth spikes each class density on its
training points, a too-large one blurs the classes together, and
``GridSearchQ`` lands on the bandwidth at the bottom. After fitting,
``search.predict(X)`` uses the refit best model directly, and
``search.best_params_`` / ``search.best_score_`` report the choice.

.. seealso::

   - :class:`~mlquantify.model_selection.GridSearchQ` — protocol, scoring and
     parallelism options.
   - :ref:`sphx_protocols` — the protocols the search can drive.
