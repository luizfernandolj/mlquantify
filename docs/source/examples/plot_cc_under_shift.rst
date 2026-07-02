.. _sphx_cc_under_shift:

====================================
Why counting fails under prior shift
====================================

The central problem of quantification is **prior-probability shift**: the test
class distribution differs from the training one. Plain Classify & Count
(:class:`~mlquantify.counting.CC`) simply counts the labels its classifier
predicts, so it inherits the classifier's misclassification rates and becomes
*biased* as the test prevalence moves away from training.

:class:`~mlquantify.counting.ACC` (Adjusted Classify & Count) corrects exactly
this bias using the classifier's true- and false-positive rates estimated on
training data. The plot below sweeps the test prevalence across the full range
and tracks what each method predicts.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    from mlquantify.datasets import make_quantification
    from mlquantify.counting import CC, ACC

    # One balanced training sample plus test bags whose true prevalence sweeps
    # a grid across the full [0, 1] range.
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        batch_size=500, return_train=True, train_size=2400,
        train_prevalence=[0.5, 0.5],
        prevalence="grid", n_prevalences=19,
        n_features=20, random_state=1,
    )

    cc = CC(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    acc = ACC(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)

    order = np.argsort(prevs[:, 1])
    true_prev = prevs[order, 1]
    cc_pred = [cc.predict(Xs[i])[1] for i in order]
    acc_pred = [acc.predict(Xs[i])[1] for i in order]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(true_prev, cc_pred, "o-", color="#e76f51", label="CC (biased)")
    ax.plot(true_prev, acc_pred, "s-", color="#2a9d8f", label="ACC (adjusted)")
    ax.set_xlabel("True positive-class prevalence")
    ax.set_ylabel("Predicted positive-class prevalence")
    ax.set_title("CC drifts off the diagonal; ACC tracks it")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()

CC systematically pulls its estimate toward the training prevalence (the curve
flattens away from the diagonal), while ACC stays close to the ideal line. This
gap is the reason the rest of the library exists.

.. seealso::

   - :ref:`sphx_method_comparison` — the same idea across more methods.
   - :ref:`sphx_error_by_shift` — error quantified as a function of shift.
