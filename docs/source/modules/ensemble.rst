.. _ensemble:

.. currentmodule:: mlquantify.meta

===========================
Meta-Quantification Methods
===========================

Meta-quantifiers wrap an existing *base quantifier* and add higher-level
strategies — ensembling, adaptive score correction, or bootstrap confidence
estimation — to improve accuracy or reliability.

.. contents:: Contents
   :local:
   :depth: 2

----

EnsembleQ — Ensemble of Quantifiers
=====================================

:class:`EnsembleQ` (Pérez-Gállego et al., 2017, 2019) creates a diverse
ensemble of base quantifiers, each trained on a subsample with a **different
class prevalence**. Diversity in training prevalences makes the ensemble
robust to test conditions not seen by any single model.

**Three phases:**

1. **Sample generation** — draw :math:`K` training batches with prevalences
   sampled from a chosen protocol (uniform, artificial, natural).
2. **Training** — fit an independent copy of the base quantifier on each
   batch.
3. **Aggregation** — average (or take the median of) all members' predictions,
   optionally keeping only the most relevant members.

**Why it excels:** A single quantifier may be over-tuned to the training
prevalence. The ensemble explores the full prevalence space during training
and aggregates across diverse operating points, reducing both bias and
variance of the final estimate.

Parameters
----------

.. list-table::
   :widths: 22 15 63
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``quantifier``
     - required
     - The base quantifier. Any ``BaseQuantifier`` subclass works. Use a
       reasonably fast method (e.g. :class:`~mlquantify.matching.DyS`) because
       ``size`` copies will be trained.
   * - ``size``
     - ``50``
     - Number of ensemble members. More members → more diversity and smoother
       estimates, but linearly more training time. 20–50 is a good range.
   * - ``min_prop``
     - ``0.1``
     - Minimum class proportion for sampling batches. Set to ``0.0`` to allow
       nearly all-negative or all-positive batches (risky on small datasets).
   * - ``max_prop``
     - ``1.0``
     - Maximum class proportion.
   * - ``selection_metric``
     - ``'all'``
     - Which members to include in the final aggregation:

       - ``'all'`` — use every member equally. Safe default.
       - ``'ptr'`` — keep the top ``p_metric`` fraction whose *training*
         prevalence is closest to an initial estimate of the test prevalence.
         Reduces bias when test prevalences cluster in a specific range.
       - ``'ds'`` — keep members whose *training score distribution* is
         closest to the test score distribution (Hellinger distance). Most
         adaptive but requires a probabilistic base quantifier and an extra
         logistic regression fit. Binary only.
   * - ``p_metric``
     - ``0.25``
     - Fraction of members retained when ``selection_metric`` is ``'ptr'``
       or ``'ds'``. ``0.25`` keeps the top 25%.
   * - ``protocol``
     - ``'uniform'``
     - Sampling protocol for generating training prevalences:

       - ``'uniform'`` — sample uniformly from the simplex. Good general
         choice.
       - ``'artificial'`` — regular grid (like APP). Gives systematic
         coverage.
       - ``'natural'`` — random sub-samples (like NPP). More realistic.
       - ``'kraemer'`` — like uniform but with a fixed step grid.
   * - ``return_type``
     - ``'mean'``
     - Aggregation function across selected members. ``'mean'`` reduces
       variance; ``'median'`` is more robust to outlier members.
   * - ``max_sample_size``
     - ``None``
     - Maximum training-batch size. ``None`` uses the full training set.
       Set to a smaller value to speed up training on large datasets.
   * - ``n_jobs``
     - ``1``
     - Parallel training of ensemble members. ``-1`` uses all CPU cores.
       Highly recommended for ``size`` > 20.
   * - ``verbose``
     - ``False``
     - Print progress during fit and predict.

Examples
--------

Basic ensemble:

.. code-block:: python

   from mlquantify.meta import EnsembleQ
   from mlquantify.matching import DyS
   from sklearn.linear_model import LogisticRegression
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   X, y = make_classification(n_samples=1000, weights=[0.8, 0.2],
                              random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42)

   q = EnsembleQ(
       quantifier=DyS(LogisticRegression()),
       size=30,
       protocol='uniform',
       n_jobs=-1,
   )
   q.fit(X_train, y_train)
   print(q.predict(X_test))

Using PTR selection to adapt to test prevalence:

.. code-block:: python

   q = EnsembleQ(
       quantifier=DyS(LogisticRegression()),
       size=50,
       selection_metric='ptr',  # keep members closest to test prevalence
       p_metric=0.25,           # keep top 25%
       return_type='median',
       n_jobs=-1,
   )
   q.fit(X_train, y_train)
   print(q.predict(X_test))

.. note::

   ``selection_metric='ds'`` requires a probabilistic base quantifier and
   is binary-only. It fits an internal logistic regression to compute
   posterior histograms for the distribution similarity check.

----

QuaDapt — Adaptive Score Simulation
=====================================

:class:`QuaDapt` (Maletzke et al., 2021) improves prevalence estimation by
simulating a synthetic training-score distribution — via the **MoSS** (Model
for Score Simulation) — that best matches the observed test-score distribution.
The best-matching synthetic set is then used as the training reference for the
wrapped quantifier's :meth:`aggregate` call.

**Why it exists:** Histogram and density matching methods rely on training
scores that may come from a very different score distribution than the test
set (due to score variability — the classifier's output range or sharpness
changes at test time). QuaDapt adaptively selects a synthetic distribution
that bridges this gap, achieving state-of-the-art results on tasks with high
score variability.

**Binary-only** (OvR for multiclass).

Parameters
----------

.. list-table::
   :widths: 22 15 63
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``quantifier``
     - required
     - A soft (probabilistic) base aggregative quantifier (e.g.
       :class:`~mlquantify.matching.DyS`, :class:`~mlquantify.matching.HDy`).
       Must support ``aggregate(test_scores, train_scores, labels)``.
   * - ``measure``
     - ``'topsoe'``
     - Distance metric for comparing test and synthetic distributions. Options:
       ``'hellinger'``, ``'topsoe'``, ``'probsymm'``, ``'sord'``. TopSoe is
       recommended for histogram matching.
   * - ``merging_factors``
     - ``np.arange(0.1, 1.0, 0.2)``
     - Candidate merging-factor values for MoSS. The merging factor controls
       how much positive and negative scores overlap in the synthetic set. A
       finer grid (e.g. ``np.arange(0.05, 1.0, 0.05)``) gives better results
       at the cost of more computation.
   * - ``strategy``
     - ``'ovr'``
     - Multiclass decomposition.

Examples
--------

.. code-block:: python

   from mlquantify.meta import QuaDapt
   from mlquantify.matching import DyS
   from sklearn.linear_model import LogisticRegression

   q = QuaDapt(
       quantifier=DyS(LogisticRegression()),
       measure='topsoe',
       merging_factors=[0.1, 0.3, 0.5, 0.7, 0.9],
   )
   q.fit(X_train, y_train)
   print(q.predict(X_test))

----

AggregativeBootstrap — Confidence Intervals via Bootstrap
===========================================================

:class:`AggregativeBootstrap` wraps any aggregative quantifier and applies
**bootstrap resampling** to both training and test predictions, generating a
*distribution* of prevalence estimates. The distribution is summarised as a
point estimate together with a confidence region.

**Why it exists:** A single prevalence estimate gives no indication of
uncertainty. AggregativeBootstrap (Moreo & Salvati, 2025) provides
statistically rigorous confidence intervals for any aggregative quantifier,
enabling uncertainty-aware deployment.

Parameters
----------

.. list-table::
   :widths: 22 15 63
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``quantifier``
     - required
     - The base aggregative quantifier to wrap.
   * - ``n_train_bootstraps``
     - ``1``
     - Number of bootstrap resamples of the training predictions. Increasing
       this to 50–200 gives more accurate confidence region estimation.
   * - ``n_test_bootstraps``
     - ``1``
     - Number of bootstrap resamples of the test predictions. Together with
       ``n_train_bootstraps`` this controls the total number of bootstrap
       rounds: ``n_train × n_test`` calls to the base quantifier's aggregate.
   * - ``region_type``
     - ``'intervals'``
     - Type of confidence region:

       - ``'intervals'`` — per-class credible intervals. Simple and fast.
       - ``'ellipse'`` — joint confidence ellipse on the prevalence simplex.
       - ``'ellipse-clr'`` — CLR-transformed ellipse (compositional data
         approach; recommended for multiclass).
   * - ``confidence_level``
     - ``0.95``
     - Confidence level for the region (e.g. 0.95 for a 95% CI).
   * - ``random_state``
     - ``None``
     - Seed for reproducibility.

Examples
--------

.. code-block:: python

   from mlquantify.meta import AggregativeBootstrap
   from mlquantify.likelihood import EMQ
   from sklearn.linear_model import LogisticRegression

   q = AggregativeBootstrap(
       EMQ(LogisticRegression()),
       n_train_bootstraps=100,
       n_test_bootstraps=100,
       region_type='intervals',
       confidence_level=0.95,
   )
   q.fit(X_train, y_train)
   prevalences = q.predict(X_test)
   print(prevalences)

   # Access the confidence region after prediction
   # (see mlquantify.confidence for the region object API)

.. seealso::

   :ref:`confidence_intervals` for a full guide on confidence regions in
   quantification.

----

Choosing a Meta-Quantifier
============================

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Method
     - When to use
     - Key advantage
   * - EnsembleQ (``'all'``)
     - Moderate shift; need robustness
     - Reduces variance through diversity.
   * - EnsembleQ (``'ptr'``)
     - Unknown test prevalence region
     - Adapts member selection to the test estimate.
   * - EnsembleQ (``'ds'``)
     - Score variability across batches
     - Selects members by distribution similarity.
   * - QuaDapt
     - Score variability; DyS/HDy as base
     - Corrects for score distribution mismatch.
   * - AggregativeBootstrap
     - Need uncertainty quantification
     - Provides confidence intervals for any quantifier.

**Practical recommendation:** Use **EnsembleQ** with ``selection_metric='ptr'``
and ``n_jobs=-1`` when you want the best accuracy with moderate extra cost.
Use **AggregativeBootstrap** when you need to report uncertainty alongside
your prevalence estimate.

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

    from mlquantify.meta import EnsembleQ
    from mlquantify.matching import DyS
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
