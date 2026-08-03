.. _sphx_elm:

===================================================
Training the classifier for quantification: SVM(Q)
===================================================

Explicit Loss Minimization methods do not correct a classifier's counts
after the fact — they **train the hyperplane itself to minimize a
quantification loss**, using Joachims' multivariate SVM (the algorithm
behind ``svmperf``, reimplemented in pure Python in
:class:`~mlquantify.elm.MultivariateLossSVM`).

The effect is easiest to see on an **imbalanced** training set: an
error-optimized SVM skews toward the majority class, and Classify & Count
inherits that bias. :class:`~mlquantify.elm.SVMQ` (Barranquero et al., 2015)
instead maximises the *Q-measure* — recall combined with the balance
:math:`1 - |FN-FP|/\max(P,N)` — so its counts stay honest.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    from mlquantify import set_config
    from mlquantify.datasets import make_quantification
    from mlquantify.elm import ELM, SVMQ, SVMKLD
    from mlquantify.visualization import DiagonalDisplay

    set_config(prevalence_return_type="array")

    # An imbalanced (75/25) training sample plus test bags on a grid of
    # prevalence values across the whole [0, 1] range.
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        batch_size=200, return_train=True, train_size=2000,
        weights=[0.75, 0.25], train_prevalence=[0.75, 0.25],
        prevalence="grid", n_prevalences=11, repeats=3,
        n_features=10, n_informative=6, class_sep=0.8, random_state=0,
    )

    methods = {
        "ELM(error) = plain SVM": ELM(loss="error"),
        "SVMQ": SVMQ(),
        "SVMKLD": SVMKLD(),
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
    fig.suptitle(
        "Counting over an error-trained vs. a quantification-trained SVM "
        "(imbalanced training)", y=0.99,
    )
    fig.tight_layout()

The left panel is the same multivariate SVM trained with the plain error
loss — a standard linear SVM — and its cloud tilts hard: trained on 25%
positives, it keeps under-predicting the positive class as the true
prevalence grows. The Q- and KLD-trained hyperplanes balance false
positives against false negatives *during learning*, and their Classify &
Count estimates track the diagonal markedly better with **no correction
step at all**.

.. seealso::

   - :ref:`explicit_loss_minimization` — the multivariate SVM, the
     Q-measure, and every hyper-parameter.
   - :ref:`sphx_quantification_trees` — the same learn-for-quantification
     idea with decision trees.
   - :ref:`sphx_cc_under_shift` — why plain counting fails under shift.
