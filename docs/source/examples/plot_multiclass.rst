.. _sphx_multiclass:

==========================
Multiclass quantification
==========================

Every quantifier in ``mlquantify`` handles more than two classes. Native
multiclass methods (:class:`~mlquantify.counting.CC`,
:class:`~mlquantify.counting.PCC`, :class:`~mlquantify.likelihood.EMQ`, the KDEy
family, the generalized matching methods) work on the full simplex directly.
**Binary methods** — :class:`~mlquantify.counting.ACC`,
:class:`~mlquantify.matching.HDy`, :class:`~mlquantify.matching.DyS`,
:class:`~mlquantify.matching.SORD` and the threshold-selection counters — are
decomposed into a set of binary sub-problems automatically, using **One-vs-Rest**.
The diagnostics carry over either way.


Native multiclass methods
--------------------------

We start with :class:`~mlquantify.likelihood.EMQ`, evaluate it across many
prevalence vectors with a :class:`~mlquantify.model_selection.UPP`, and show a
per-class diagonal plot.

.. plot::

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from mlquantify import set_config
    from mlquantify.datasets import make_quantification
    from mlquantify.likelihood import EMQ
    from mlquantify.visualization import DiagonalDisplay

    set_config(prevalence_return_type="array")

    # A balanced 3-class training sample plus bags whose prevalence vectors
    # are drawn uniformly over the simplex.
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        n_batches=400, batch_size=120, return_train=True, train_size=2250,
        train_prevalence=[1 / 3, 1 / 3, 1 / 3], prevalence="uniform",
        n_classes=3, n_features=20, n_informative=6,
        n_clusters_per_class=1, random_state=0,
    )

    q = EMQ(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    pred = np.vstack([q.predict(Xb) for Xb in Xs])

    # DiagonalDisplay colour-codes the three classes on one axes for multiclass.
    disp = DiagonalDisplay.from_predictions(
        prevs, pred, alpha=0.4, s=16,
    )
    disp.ax_.set_title("EMQ — 3-class diagonal (one colour per class)")
    disp.figure_.set_size_inches(6, 6)
    disp.figure_.tight_layout()

Each class gets its own colour; all three clouds hug the diagonal, confirming
EMQ recovers the full prevalence vector and not just one class.


Binary methods via One-vs-Rest
------------------------------

A binary quantifier cannot, by itself, estimate three prevalences. Under
**One-vs-Rest** the problem is split into "class :math:`k` vs. the rest" for
every class; each sub-quantifier estimates the prevalence of its own class, and
the results are normalised to sum to one. ``mlquantify`` does this transparently:
**you just fit and predict the binary method exactly as in the binary case** —
One-vs-Rest is applied automatically, since it is the default decomposition. No
manual class loop, no extra configuration.

The grid below runs four binary methods on the same three-class problem.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    from mlquantify import set_config
    from mlquantify.datasets import make_quantification
    from mlquantify.counting import ACC
    from mlquantify.matching import HDy, DyS, SORD
    from mlquantify.visualization import DiagonalDisplay

    set_config(prevalence_return_type="array")

    Xtr, ytr, Xs, ys, prevs = make_quantification(
        n_batches=200, batch_size=120, return_train=True, train_size=2250,
        train_prevalence=[1 / 3, 1 / 3, 1 / 3], prevalence="uniform",
        n_classes=3, n_features=20, n_informative=6,
        n_clusters_per_class=1, random_state=0,
    )

    # All four are binary quantifiers — they only know "positive vs. rest".
    methods = {
        "ACC": ACC(LogisticRegression(max_iter=1000)),
        "HDy": HDy(LogisticRegression(max_iter=1000)),
        "DyS": DyS(LogisticRegression(max_iter=1000)),
        "SORD": SORD(LogisticRegression(max_iter=1000)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for (name, q), ax in zip(methods.items(), axes.ravel()):
        # No manual decomposition: the binary method handles 3 classes via OvR.
        q.fit(Xtr, ytr)
        pred = np.vstack([q.predict(Xb) for Xb in Xs])
        DiagonalDisplay.from_predictions(
            prevs, pred, ax=ax, alpha=0.4, s=14,
        )
        ax.set_title(f"{name}  (One-vs-Rest)")
    fig.suptitle("Binary quantifiers on a 3-class problem via OvR", y=0.99)
    fig.tight_layout()

Despite being binary at heart, all four methods track the diagonal across the
whole simplex: One-vs-Rest extends them to three classes with no change to your
code. Each sub-quantifier estimates the prevalence of its own class, and
``mlquantify`` normalises the per-class estimates so the prediction stays a
valid prevalence vector.

.. seealso::

   - :ref:`sphx_confidence_regions` — a ternary confidence ellipse for these
     simplex-valued predictions.
   - :ref:`sphx_method_comparison` — the same methods on a binary problem.
