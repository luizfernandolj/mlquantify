.. _sphx_error_by_shift:

=====================================
Robustness to prior-probability shift
=====================================

The diagonal plot in :ref:`sphx_method_comparison` shows *where* a quantifier
errs; this example collapses that into a single, comparable curve: quantification
**error as a function of the amount of prior-probability shift** between the test
sample and the training set. A flat, low curve is the goal — it means the method
is insensitive to how far the test prevalence has drifted.

We use :class:`~mlquantify.visualization.ErrorByShiftDisplay`, which bins the
protocol samples by their shift and draws the mean error with a ``±std`` band.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    from mlquantify import set_config
    from mlquantify.datasets import make_quantification
    from mlquantify.counting import CC, ACC
    from mlquantify.likelihood import EMQ
    from mlquantify.visualization import ErrorByShiftDisplay

    set_config(prevalence_return_type="array")

    # A balanced training sample plus bags whose prevalences are drawn
    # uniformly over the simplex, so the shift varies from none to extreme.
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        n_batches=300, batch_size=100, return_train=True, train_size=2000,
        train_prevalence=[0.5, 0.5],
        prevalence="uniform", n_features=20, random_state=0,
    )
    train_prevalence = np.bincount(ytr) / len(ytr)

    methods = {
        "CC": (CC(LogisticRegression(max_iter=1000)), "#e76f51"),
        "ACC": (ACC(LogisticRegression(max_iter=1000)), "#2a9d8f"),
        "EMQ": (EMQ(LogisticRegression(max_iter=1000)), "#264653"),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (q, color) in methods.items():
        q.fit(Xtr, ytr)
        pred = np.vstack([q.predict(Xb) for Xb in Xs])
        ErrorByShiftDisplay.from_predictions(
            prevs, pred,
            train_prevalence=train_prevalence, error_metric="ae",
            n_bins=10, name=name, ax=ax, color=color,
        )
    ax.set_title("Absolute error vs. prior-probability shift")
    fig.tight_layout()

CC's error grows steadily as the shift increases — exactly the bias from
:ref:`sphx_cc_under_shift`, now quantified — while ACC and EMQ stay low and flat
across the whole range. This is the plot to reach for when you need to *defend*
a method choice: it summarises hundreds of test samples into one honest picture
of robustness.

.. seealso::

   - :ref:`sphx_method_comparison` — the per-sample scatter behind these curves.
   - :class:`~mlquantify.visualization.ErrorByShiftDisplay` — options and metrics.
