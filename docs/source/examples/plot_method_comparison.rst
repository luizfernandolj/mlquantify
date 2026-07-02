.. _sphx_method_comparison:

=========================================
Comparing quantifiers with diagonal plots
=========================================

A *diagonal plot* is the standard way to compare quantifiers: run an
:class:`~mlquantify.model_selection.APP` (Artificial Prevalence Protocol) to
generate many test samples spanning the whole prevalence range, then scatter the
predicted prevalence against the true one. Points hugging the :math:`y = x` line
mean low bias; a tight cloud means low variance.

Here we compare one method from each major family — counting
(:class:`~mlquantify.counting.ACC`), the EM likelihood method
(:class:`~mlquantify.likelihood.EMQ`), distribution matching
(:class:`~mlquantify.matching.DyS`), and plain :class:`~mlquantify.counting.CC`
as a baseline — using :class:`~mlquantify.visualization.DiagonalDisplay`.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    from mlquantify import set_config
    from mlquantify.datasets import make_quantification
    from mlquantify.counting import CC, ACC
    from mlquantify.likelihood import EMQ
    from mlquantify.matching import DyS
    from mlquantify.visualization import DiagonalDisplay

    set_config(prevalence_return_type="array")

    # A balanced training sample plus test bags on a grid of prevalence values
    # spanning the whole [0, 1] range (the Artificial Prevalence Protocol).
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        batch_size=100, return_train=True, train_size=2000,
        train_prevalence=[0.5, 0.5],
        prevalence="grid", n_prevalences=21, repeats=5,
        n_features=20, random_state=0,
    )

    methods = {
        "CC": CC(LogisticRegression(max_iter=1000)),
        "ACC": ACC(LogisticRegression(max_iter=1000)),
        "EMQ": EMQ(LogisticRegression(max_iter=1000)),
        "DyS": DyS(LogisticRegression(max_iter=1000)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for (name, q), ax, color in zip(
        methods.items(), axes.ravel(),
        ["#e76f51", "#2a9d8f", "#264653", "#e9c46a"],
    ):
        q.fit(Xtr, ytr)
        pred = np.vstack([q.predict(Xb) for Xb in Xs])
        DiagonalDisplay.from_predictions(
            prevs, pred, ax=ax, color=color, alpha=0.5, s=18,
        )
        ax.set_title(name)
    fig.suptitle("Diagonal plots across quantifier families", y=0.99)
    fig.tight_layout()

Read the panels like this: CC's cloud tilts off the diagonal (bias under shift),
while ACC, EMQ and DyS each pull their estimates back toward the ideal line in
their own way. Swapping in any other method from
:mod:`mlquantify.counting`, :mod:`mlquantify.matching` or
:mod:`mlquantify.likelihood` is a one-line change.

.. seealso::

   - :ref:`sphx_error_by_shift` — collapse each panel into a single error curve.
   - :ref:`visualization` — the full Display gallery.
