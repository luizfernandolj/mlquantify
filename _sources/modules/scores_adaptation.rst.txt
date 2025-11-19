.. _scores_adaptation:

.. currentmodule:: mlquantify.meta

=========================================
QuaDapt: Drift-Resilient Score Adaptation
=========================================

The :class:`QuaDapt` method is a drift-resilient meta-quantification strategy designed to
handle situations where the **classifier's score distributions drift** between
training and test domains.  
Instead of assuming that the original score distributions remain stable, QuaDapt
actively adapts them at prediction time [1]_.

.. figure:: ../images/quadapt.jpeg
   :align: center
   :width: 90%
   :alt: QuaDapt overview

   *QuaDapt: adaptive score simulation and distribution matching. Ortega et al., 2025*

:class:`QuaDapt` is inspired by the principle behind **Distribution Matching (DM)** or
**Mixture Model (MM)** quantifiers (see :ref:`mixture_models`), which estimate test prevalences by finding the
mixture of class-conditional distributions that best fits the test data.  
However, while classical MM methods rely on **empirical training distributions**,  
**QuaDapt replaces them with synthetic score distributions** generated for multiple
hypothetical levels of class separability [1]_.

The method evaluates several merging factors :math:`m` that control synthetic
separability and selects the one producing the closest match to the test-score
distribution.  
The chosen synthetic model is then passed to an aggregative quantifier (e.g., :class:`ACC`,
:class:`T50`, :class:`DyS`, :class:`SORD`.) to compute final prevalence estimates.

This makes QuaDapt a **meta-quantifier** capable of adapting to score drift without
relying on static training distributions.


.. dropdown:: MoSS: Synthetic Score Generator Used by QuaDapt

   QuaDapt relies on **MoSS (Model for Score Simulation)** to generate synthetic
   positive and negative score distributions with controlled overlap [2]_.

   MoSS takes three parameters:

   - :math:`n`: number of synthetic samples  
   - :math:`\alpha`: class proportion  
   - :math:`m`: merging factor controlling the overlap

   Positive and negative scores are generated as:

   .. math::

       s^+ = U^m, 
       \qquad
       s^- = 1 - U^m,
       \qquad
       U \sim \mathrm{Uniform}(0,1)

   Larger values of :math:`m` create more overlapping distributions, modeling weaker
   classifier separability or drift.


**Example Usage**

.. code-block:: python

   from mlquantify.meta import QuaDapt
   from mlquantify.adjust_counting import ACC
   from sklearn.ensemble import RandomForestClassifier

   q = QuaDapt(
       quantifier=ACC(RandomForestClassifier()),
       merging_factors=[0.1, 0.5, 1.0],
       measure="topsoe"
   )

   q.fit(X_train, y_train)
   prevalence = q.predict(X_test)


.. dropdown:: References

    .. [1] Ortega, J. P., Junior, L. F. L., Zalewski, W., & Maletzke, A.  
       *QuaDapt: Drift-Resilient Quantification via Parameters Adaptation.*  
      5th International Workshop on Learning to Quantify.

    .. [2] Maletzke, A., Reis, D. dos, Hassan, W., & Batista, G. (2021).  
       *Accurately Quantifying under Score Variability.* ICDM 2021.
