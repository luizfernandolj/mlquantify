.. _mixture_models_non_agg:

.. currentmodule:: mlquantify.mixture

========================================================
Mixture Models for Non-Aggregative Quantification
========================================================

Currently, the only Mixture Model method specifically designed for non-aggregative quantification is **HDx (Hellinger Distance x-Similarity)**, found at :class:`HDx`.

HDx: Hellinger Distance x-Similarity
====================================

**HDx** is a non-aggregative quantification method based on :class:`HDy` [1]_. While HDy operates on posterior probabilities (y-space), HDx works directly in the feature space (x-space), without aggregating predictions.

The goal of HDx is to estimate the prevalence parameter \(\alpha\) that minimizes the average Hellinger Distance between the empirical feature distribution of the test set and a convex mixture of the class-conditional feature distributions from training data.

.. dropdown:: Mathematical Definition

   .. math::

      V_\alpha(x) = \alpha \cdot p(x|+) + (1 - \alpha) \cdot p(x|-)

   .. math::

      \alpha^* = \underset{0 \leq \alpha \leq 1}{\arg\min}\; \frac{1}{n_f} \sum_{f=1}^{n_f} HD_f(V^\alpha, U)

   .. math::

      \frac{|V_{f,i}|}{|V|} = \frac{|S^+_{f,i}|}{|S^+|} \cdot \alpha + \frac{|S^-_{f,i}|}{|S^-|} \cdot (1 - \alpha)

   where:

    - :math:`V_\alpha(x)`: mixture distribution for feature :math:`x` parameterized by :math:`\alpha`,
    - :math:`p(x|+), p(x|-)`: class-conditional feature distributions from training,
    - :math:`HD_f`: Hellinger distance for each feature :math:`f`,
    - :math:`U`: empirical test distribution,
    - :math:`|S^+_{f,i}|`, :math:`|S^-_{f,i}|`: counts of positive/negative training samples in bin :math:`i` of feature :math:`f`,
    - :math:`n_f`: number of features.

HDx, different from HDy, does not require a learner to estimate posterior probabilities, as it operates directly in the feature space, so it does not have a `aggregate` method.

.. code-block:: python

   from mlquantify.mixture import HDx
   from sklearn.ensemble import RandomForestClassifier

   q = HDx(bins_size=[10, 20, 30])
   q.fit(X_train, y_train)
   q.predict(X_test)

.. dropdown:: References

   .. [1] González-Castro, V., Alaiz-Rodríguez, R., & Alegre, E. (2013). Class distribution estimation based on the Hellinger distance. Information Sciences, 218, 146-164. https://doi.org/10.1016/j.ins.2012.05.028
