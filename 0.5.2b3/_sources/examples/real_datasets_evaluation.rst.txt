.. _sphx_real_datasets_evaluation:

====================================
Evaluating a quantifier on real data
====================================

Every fetcher has a built-in **quantification protocol**. Pass ``protocol="app"``
(the Artificial Prevalence Protocol) and the returned
:class:`~mlquantify.datasets.Bunch` gains two extra attributes:

- ``.samples`` — a list of index arrays into ``.data``, one per test *bag*;
- ``.prevalences`` — the ``(n_samples, n_classes)`` array of each bag's **true**
  class prevalence.

That is exactly what you need to score a quantifier: predict the prevalence of
every bag and compare it against the known truth.

.. code-block:: python

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    from mlquantify.datasets import fetch_mushroom
    from mlquantify.counting import ACC
    from mlquantify.metrics import MAE

    # 500 bags of 200 instances, drawn across a range of class balances.
    data = fetch_mushroom(protocol="app", n_samples=500, sample_size=200,
                          random_state=0)
    X, y = data.data, data.target

    quantifier = ACC(RandomForestClassifier(random_state=0)).fit(X, y)

    # Score every bag against its known true prevalence (metrics are
    # ``metric(y_true, y_pred)``, like scikit-learn).
    errors = [
        MAE(true_prev, quantifier.predict(X[bag]))
        for bag, true_prev in zip(data.samples, data.prevalences)
    ]
    print(f"mean absolute error over {len(errors)} bags: {np.mean(errors):.4f}")

The Artificial Prevalence Protocol stress-tests the quantifier across the whole
range of class balances, so the mean error summarises how robust it is to prior
shift — not just how it does at the dataset's natural prevalence. Swap
``protocol="app"`` for ``"npp"``, ``"upp"`` or ``"ppp"`` to change how the bags
are sampled.

.. note::

   For brevity the quantifier is fit on the whole dataset, and the bags are
   resampled from that same pool. For a leakage-free benchmark, split the data
   first and draw the evaluation bags from a disjoint test set —
   :func:`~mlquantify.model_selection.apply_protocol` runs exactly that
   fit-then-score loop for you.

.. seealso::

   - :ref:`real_world_datasets` — the fetcher API and protocol options.
   - :ref:`sphx_protocols` — what each sampling protocol looks like.
   - :func:`~mlquantify.model_selection.apply_protocol` — the one-call
     evaluation helper.
