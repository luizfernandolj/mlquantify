.. _sphx_readme_methods:

============================================
Quantifying without a classifier: ReadMe
============================================

The ReadMe methods estimate class prevalences **directly from features** — no
classifier is trained at any point. They solve the accounting identity
:math:`P(S) = P(S \mid D)\,P(D)` by least squares on the probability simplex:
:class:`~mlquantify.readme.ReadMe` (Hopkins & King, 2010) tabulates the joint
distribution of small random subsets of binary features, while
:class:`~mlquantify.readme.ReadMe2` (Jerzak, King & Strezhnev, 2022) learns
continuous feature projections by SGD and matches the labeled set to the
unlabeled documents before estimating.

.. note::

   ``ReadMe2`` requires PyTorch (``pip install mlquantify[neural]``). The
   settings below are deliberately small so the example runs quickly.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    from mlquantify import set_config
    from mlquantify.datasets import make_quantification
    from mlquantify.readme import ReadMe, ReadMe2
    from mlquantify.visualization import DiagonalDisplay

    set_config(prevalence_return_type="array")

    # A balanced labeled sample plus unlabeled bags spanning a grid of
    # prevalence values from 0 to 1.
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        batch_size=200, return_train=True, train_size=2000,
        train_prevalence=[0.5, 0.5],
        prevalence="grid", n_prevalences=11, repeats=2,
        n_features=20, n_informative=10, class_sep=1.5, random_state=0,
    )

    methods = {
        "ReadMe": ReadMe(n_subsets=30, subset_size=12, n_jobs=-1, random_state=0),
        "ReadMe2": ReadMe2(n_boot=3, sgd_iters=150, n_boot_match=10, random_state=0),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for (name, q), ax, color in zip(
        methods.items(), axes, ["#2a9d8f", "#e76f51"],
    ):
        q.fit(Xtr, ytr)
        pred = np.vstack([q.predict(Xb) for Xb in Xs])
        DiagonalDisplay.from_predictions(
            prevs, pred, ax=ax, color=color, alpha=0.5, s=20,
        )
        mae = float(np.mean(np.abs(pred - prevs)))
        ax.set_title(f"{name}  (MAE = {mae:.3f})")
    fig.suptitle("Classifier-free quantification across the prevalence range", y=0.99)
    fig.tight_layout()

Both methods track the diagonal without any per-document prediction step.
ReadMe's estimates shrink toward the training proportions at the extremes —
the binarized profiles discriminate the categories only weakly here, the bias
analysed by Jerzak et al. (2022) — while ReadMe2's learned projections and
matching visibly tighten the cloud. On genuinely binary indicator features
(word stems), plain ReadMe is at its best.

.. seealso::

   - :ref:`readme_methods` — the accounting identity, its assumptions, and
     every hyper-parameter.
   - :ref:`sphx_method_comparison` — the same read-out for the
     classifier-based families.
