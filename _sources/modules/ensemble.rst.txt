.. _ensemble:

.. currentmodule:: mlquantify.meta

===========================
Ensemble for Quantification
===========================

Ensembles for Quantification (:class:`EnsembleQ`) represent a class of algorithms aimed at improving the accuracy and robustness of class prevalence estimation by combining multiple base quantifiers trained on varied data samples with controlled prevalence distributions. Different training subsets simulate varying class distributions to introduce diversity in the ensemble, which helps address predictable changes in class priors (Prior Probability Shift or Label Shift).

The algorithm can be divided into three main phases:

.. grid:: 1 1 3 3
    :gutter: 2

    .. grid-item-card:: Phase 1: Sample Generation
        :text-align: center

        Multiple training subsets with varied prevalence :math:`p_j` sampled from protocol ('artificial', 'natural', 'uniform', 'kraemer').

    .. grid-item-card:: Phase 2: Model Training
        :text-align: center

        Each batch trains a base quantifier independently with parameters estimated via cross-validation.

    .. grid-item-card:: Phase 3: Aggregation
        :text-align: center

        All models predict :math:`\hat{p}_j`, aggregated via mean/median with optional selection ('all', 'ptr', 'ds').

Advantages include risk reduction, correction of instability in base quantifiers, and resilience to widely varying test prevalence.

.. dropdown:: Mathematical Definition

    Given training class-conditional feature distributions :math:`p(x|+)` and :math:`p(x|-)` and an unlabeled test set :math:`U`, each training batch simulates a mixture distribution:

    .. math::

        V_\alpha(x) = \alpha \cdot p(x|+) + (1 - \alpha) \cdot p(x|-)

    A diversity of prevalence values :math:`\alpha` is sampled according to the chosen protocol to generate training batches :math:`D_j`. Each base quantifier is trained on these batches.

    Final ensemble prevalence estimate :math:`\hat{p}_{final}` is computed as:

    .. math::

        \hat{p}_{final} = \text{aggregation} \left( \hat{p}_1, \hat{p}_2, \ldots, \hat{p}_m \right)

    where aggregation is typically mean or median, optionally weighted by selection metrics.

    **Selection policies used during aggregation:**

    - **'all'**: Uses all ensemble members equally without any selection or weighting.
    - **'ptr' (Prevalence Training Ratio)**: Selects models whose training prevalence :math:`p_j` is closest to an initial prevalence estimate of the test set, often computed as the mean of all base predictions.
    - **'ds' (Distribution Similarity)**: Selects models whose training posterior score distributions are most similar to the test set distribution, measured with metrics such as Hellinger Distance. This requires probabilistic quantifiers capable of producing posterior probabilities.

**Example**

.. code-block:: python

    from mlquantify.ensemble import EnsembleQ
    from mlquantify.mixture import DyS
    from sklearn.ensemble import RandomForestClassifier

    ensemble = EnsembleQ(
         quantifier=DyS(RandomForestClassifier()),
         size=30,
         protocol='artificial',
         selection_metric='ptr'
    )
    ensemble.fit(X_train, y_train)
    prevalence_estimates = ensemble.predict(X_test)

.. dropdown:: References

    .. [1] Pérez-Gállego, P., Castaño, A., Ramón Quevedo, J., & José del Coz, J. (2019). Dynamic ensemble selection for quantification tasks. Information Fusion, 45, 1-15. https://doi.org/10.1016/j.inffus.2018.01.001

    .. [2] Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017). Using ensembles for problems with characterizable changes in data distribution: A case study on quantification. Information Fusion, 34, 87-100. https://doi.org/10.1016/j.inffus.2016.07.001
