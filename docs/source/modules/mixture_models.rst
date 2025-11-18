.. _mixture_models:

.. currentmodule:: mlquantify.mixture

==============
Mixture Models 
==============

Mixture Model (MM) methods, often referred to as **Distribution Matching (DM)** methods, constitute one of the main families of quantification algorithms [1]_.

Mixture Models seeks to model the data distribution observed in the test set as a parametric mixture of the individual class distributions obtained from training :math:`L`.
   
.. note::
   Mixture Models are predominantly designed for binary quantification problems. While extensions to multi-class scenarios exist, such as one-vs-all strategies, they are computationally intensive and less commonly used. If you are dealing with multi-class quantification, consider using methods from the :ref:`density_module` with better scalability.

.. dropdown:: Mathematical details - Mixture Formulation

    The observed distribution in the test set is approximated as:

    .. math::

       D_U \approx \hat{p} \cdot D_+ + (1 - \hat{p}) \cdot D_-

    Unlike methods like :class:`EMQ`, MM methods generally do not refine priors and posteriors mutually. Instead, they use a search process (exhaustive or optimized) to find the parameter :math:`\hat{p}` that minimizes the dissimilarity function.

.. dropdown:: References

    .. [1] Forman, G. (2008). Quantifying counts and costs via classification. Data Mining and Knowledge Discovery, 17(2), 164–206. https://doi.org/10.1007/s10618-008-0097-y


DyS: Distribution y-Similarity Framework
========================================

**DyS** is a generic framework that formalizes the Mixture Models approach. The term **y-Similarity** indicates that it compares the similarity of classification score distributions (y-space) [1]_.

DyS depends on two critical factors: the method used to represent the distributions (such as histograms or means) and the dissimilarity function (DS) used to compare the test distribution with the mixture distribution.

DyS seeks the prevalence parameter :math:`\alpha` that minimizes the dissimilarity (:math:`DS`) between the test score distribution (:math:`f_U`) and the mixture of training score distributions weighted by :math:`\alpha`.

.. dropdown:: Mathematical details - DyS Optimization

    The estimated prevalence is the :math:`\alpha` that satisfies:

    .. math::

       \hat{p}^{DyS}(\oplus) = \alpha^* = \operatorname*{arg\,min}_{0 \le \alpha \le 1} \{ DS(\alpha f_{L^{\oplus}} + (1-\alpha) f_{L^{\ominus}}, f_U) \}

.. dropdown:: References

    .. [2] Maletzke, A., dos Reis, D., Cherman, E., & Batista, G. (2019). DyS: A Framework for Mixture Models in Quantification. www.aaai.org


HDy: Hellinger Distance y-Similarity
====================================

**HDy** is a specific and popular instance of the DyS framework and a variant of Forman's original MM, proposed by [1]_.

**What HDy does:**

1.  **Representation:** HDy uses normalized histograms (PDF estimates) of posterior probabilities (y-scores) to represent the training class distributions and the test set distribution.
2.  **Mixture:** It models the test histogram (:math:`Q`) as a mixture of the positive histogram (:math:`P_+`) and the negative histogram (:math:`P_-`), weighted by the parameter :math:`\hat{p}`.
3.  **Comparison:** HDy uses the **Hellinger Distance (HD)** as the dissimilarity metric to find the value :math:`\hat{p}` that minimizes the distance between the mixture and the test distribution.
4.  

**Example**

.. code-block:: python

   from mlquantify.mixture import HDy
   from sklearn.ensemble import RandomForestClassifier

   q = HDy(learner=RandomForestClassifier(), bins=10)
   q.fit(X_train, y_train)
   q.predict(X_test)

.. dropdown:: Mathematical details - HDy Bin Adjustment

    The bin-level fit for the histogram is given by:

    .. math::

       \frac{|D'_i|}{|D'|} = \frac{|D^+_i|}{|D^+|} \cdot \hat{p} + \frac{|D^-_i|}{|D^-|} \cdot (1 - \hat{p})

    Where :math:`|D'|` and :math:`|D'_i|` are, respectively, the total cardinality and the count in bin :math:`i` for the modified training distribution [1]_[2]_.

.. dropdown:: References

    .. [3] González-Castro, V., Alaiz-Rodríguez, R., & Alegre, E. (2013). Class distribution estimation based on the Hellinger distance. Information Sciences, 218, 146-164. https://doi.org/10.1016/j.ins.2012.05.028


SMM: Sample Mean Matching
=========================

**SMM** is a member of the DyS framework, proposed by [4]_ notable for its simplicity and efficiency, located at the :class:`SMM`. 

**What SMM does:**

1.  **Representation:** Instead of using histograms (like HDy) or CDFs, SMM represents the score distributions of positive (:math:`S_{\oplus}`), negative (:math:`S_{\ominus}`), and test (:math:`S_U`) classes by a single scalar statistic: the **mean (:math:`\mu`)** of the scores.
2.  **Optimization:** SMM assumes that the mean of the test scores is the weighted sum of the training score means.
3.  **Closed Form Solution:** SMM does not require iteration or complex search procedures, as the problem can be solved in closed form.

.. note::
   SMM is mathematically equivalent to the **PACC (Probabilistic Adjusted Classify & Count)** method [4]_.

.. dropdown:: Mathematical details - SMM Closed Form

    SMM seeks the parameter :math:`\alpha` that minimizes the absolute difference between the test mean and the mixture mean:

    .. math::

       \hat{p}^{SMM}(\oplus) = \alpha = \operatorname*{arg\,min}_{0 \le \alpha \le 1} \{ |\alpha \mu[S_{\oplus}] + (1-\alpha)\mu[S_{\ominus}] - \mu[S_U]| \}

    This can be solved directly via the formula:

    .. math::

       \alpha = \frac{\mu[S_U] - \mu[S_{\ominus}]}{\mu[S_{\oplus}] - \mu[S_{\ominus}]}

    Where:
    
    - :math:`\mu[S_U]` is the mean of test scores.
    - :math:`\mu[S_{\oplus}]` is the mean of positive training scores.
    - :math:`\mu[S_{\ominus}]` is the mean of negative training scores.

**Example**

.. code-block:: python

   from mlquantify.mixture import SMM
   q = SMM(learner=LogisticRegression())
   q.fit(X_train, y_train)
   q.predict(X_test)

.. dropdown:: References

    .. [4] Hassan, W., Maletzke, A., & Batista, G. (2020). Accurately quantifying a billion instances per second. Proceedings - 2020 IEEE 7th International Conference on Data Science and Advanced Analytics, DSAA 2020, 1-10. https://doi.org/10.1109/DSAA49011.2020.00012


SORD: Sample Ordinal Distance
=============================

**SORD (Sample Ordinal Distance)** is one of the dissimilarity functions that fall under the DyS framework, located at :class:`SORD`.

SORD is notable for operating directly on **score samples (observations)** rather than discretized distributions (histograms). By eliminating the dependency on the number of bins, it seeks the minimum cost to transform a sample of scores into the weighted mixture sample. SORD provides an alternative that does not lose details after bin discretization [5]_.

.. dropdown:: References

    .. [5] Maletzke, A., dos Reis, D., Hassan, W., & Batista, G. (2021). Accurately Quantifying under Score Variability.