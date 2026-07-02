.. _sphx_quantification_trees:

===============================================
Quantification trees vs. a standard CART
===============================================

Quantification trees (Milli et al., 2013) are decision trees whose split
criterion balances **false positives against false negatives per class**
instead of minimising impurity: when :math:`FP_c = FN_c` for every class,
plain counting of the leaf predictions estimates the prevalences exactly. This
example runs a standard CART wrapped in Classify & Count, a single
:class:`~mlquantify.tree.QuantificationTree`, and the
:class:`~mlquantify.tree.QuantificationForest` (which averages per-tree
Adjusted Count estimates — the configuration with the best results in the
paper) through an Artificial Prevalence Protocol, and reads their bias off
diagonal plots.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier

    from mlquantify import set_config
    from mlquantify.datasets import make_quantification
    from mlquantify.counting import CC
    from mlquantify.tree import QuantificationTree, QuantificationForest
    from mlquantify.visualization import DiagonalDisplay

    set_config(prevalence_return_type="array")

    # One balanced training sample plus test bags on a grid of prevalence
    # values from 0 to 1 (the Artificial Prevalence Protocol).
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        batch_size=200, return_train=True, train_size=2000,
        train_prevalence=[0.5, 0.5],
        prevalence="grid", n_prevalences=11, repeats=3,
        n_features=10, n_informative=6, class_sep=1.0, random_state=0,
    )

    methods = {
        "CC + CART": CC(DecisionTreeClassifier(random_state=0)),
        "QuantificationTree": QuantificationTree(random_state=0),
        "QuantificationForest": QuantificationForest(
            n_estimators=200, sample_fraction=0.7, n_jobs=-1, random_state=0,
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for (name, q), ax, color in zip(
        methods.items(), axes, ["#264653", "#2a9d8f", "#e76f51"],
    ):
        q.fit(Xtr, ytr)
        pred = np.vstack([q.predict(Xb) for Xb in Xs])
        DiagonalDisplay.from_predictions(
            prevs, pred, ax=ax, color=color, alpha=0.5, s=20,
        )
        mae = float(np.mean(np.abs(pred - prevs)))
        ax.set_title(f"{name}  (MAE = {mae:.3f})")
    fig.suptitle("Trees grown for classification vs. grown for quantification", y=0.99)
    fig.tight_layout()

Read the three panels together. CC's expected estimate is
:math:`\hat{p} = fpr + (tpr - fpr)\,p`, so the *slope* of each cloud is the
classifier's :math:`tpr - fpr` and the tilt away from the diagonal grows with
the shift. The single quantification tree achieves its guarantee exactly —
its training errors are perfectly balanced (:math:`FP = FN`), so it is
unbiased *at the training prevalence* (its cloud crosses the diagonal at 0.5)
— but the gain-based stopping rule halts it after very few splits, and the
resulting weak classifier has a smaller :math:`tpr - fpr`: a flatter response
that errs more at the extremes than the deeply grown CART. This mirrors the
original paper, where the plain-counting single trees were not consistently
better than C4.5 either. The method's real deliverable is the third panel:
the forest averages **Adjusted Count** estimates (which divide out the
:math:`tpr - fpr` slope, per tree) over many trees built on random record and
feature subsets, and hugs the diagonal across the whole range. For a single
tree, the same correction is available by composition:
``ACC(estimator=QuantificationTreeClassifier())``.

.. seealso::

   - :ref:`quantification_trees` — the EB/CQB split criteria and every
     hyper-parameter.
   - :ref:`sphx_cc_under_shift` — why plain counting fails under prior shift.
