.. _bootstrap_quantification:

.. currentmodule:: mlquantify.meta

===========================
Bootstrap in Quantification
===========================

Bootstrap is used in quantification to estimate uncertainty by constructing confidence regions around class prevalence estimates. Direct application is computationally expensive; thus, bootstrap is applied efficiently only to the adjustment or aggregation phases of aggregative quantifiers.

Bootstrap strategies are classified into three main types:

.. grid:: 1 1 3 3
    :gutter: 2

    .. grid-item-card:: Model-based Bootstrap
        :text-align: center

        Resamples the classifier's cross-validation outputs during training of adjustment functions. Multiple adjustment models are fitted and applied to fixed classifier predictions, effectively avoiding repeated retraining of classifiers.

    .. grid-item-card:: Population-based Bootstrap
        :text-align: center

        Uses a single prediction set on the test data; bootstrap resamples the test predictions to generate multiple test sample bags. A single adjustment function is applied to each bag to produce bootstrap prevalence estimates.

    .. grid-item-card:: Combined Approach
        :text-align: center

        Applies both model-based and population-based resampling, generating a grid of prevalence estimates balancing computational efficiency and robustness under prior probability shift.

The :class:`AggregativeBootstrap` class implements these strategies for aggregative quantifiers by using two parameters: ``n_train_bootstraps`` and ``n_test_bootstraps``. These parameters define the number of bootstrap samples for the training and test phases, respectively.

.. code-block:: python

    from mlquantify.ensemble import AggregativeBootstrap
    from mlquantify.neighbors import EMQ
    from sklearn.ensemble import RandomForestClassifier

    agg_boot = AggregativeBootstrap(
        quantifier=EMQ(RandomForestClassifier()),
        n_train_bootstraps=100,
        n_test_bootstraps=100
    )
    agg_boot.fit(X_train, y_train)
    prevalence, conf_region = agg_boot.predict(X_test)

For information on confidence interval construction from bootstrap samples, see :ref:`confidence_intervals`.

.. dropdown:: References

    .. [1] Moreo, A., & Salvati, N. (2025). An Efficient Method for Deriving Confidence Intervals in Aggregative Quantification.
