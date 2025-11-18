.. _nearest_neighbors:

.. currentmodule:: mlquantify.neighbors

=================
Nearest Neighbors
=================

PWK: Pair-wise Weighted K-Nearest Neighbors
===========================================

**PWK** and **PWK**\ :math:`\alpha` are neighbor-based algorithms (k-Nearest Neighbors - NN) that apply class weighting techniques to estimate prevalence in binary problems [2]_.

Key Characteristics
-------------------

* **Focus on Simplicity and Stability:** The primary goal of PWK is to offer a competitive and stable quantifier that balances simplicity with effectiveness.
* **Proximity-Based:** By exploring the **topology** of the data, PWK is robust to changes in distribution, retaining the local structure of the data better than global classifiers [2]_.
* **Class Weighting:** These methods complement the k-NN algorithm with weighting policies. The objective is to neutralize the bias in favor of the majority class, which is inherent in classifiers trained on imbalanced datasets [2]_.
* **Correction:** These methods are often evaluated after applying Forman's **ACC correction**. NN methods are advantageous here because the distance matrix can be computed once, optimizing the estimation of error rates (TPR and FPR) via cross-validation [2]_.

PWK (Proportion-Weighted K-Nearest Neighbor)
--------------------------------------------

This is the simplest version of the algorithm.

* **Mechanism:** The weight :math:`w_c` is defined to be **inversely proportional** to the size of class :math:`c` relative to the total training set size [2]_.
* **Simplicity:** The weights are easily interpretable, and the method only requires the calibration of the number of neighbors :math:`k`.

.. dropdown:: Mathematical details - PWK Weights

   The weight for a class :math:`\gamma_j` is calculated as:

   .. math::

       w_{\gamma_j} = \frac{1 - \frac{m_{\gamma_j}}{m}}{m_{\gamma_j}}

   Where :math:`m` is the total training size and :math:`m_{\gamma_j}` is the count of samples in class :math:`\gamma_j`.

PWK\ :math:`\alpha` (Alpha-Adjusted PWK)
----------------------------------------

**PWK**\ :math:`\alpha` is a general structure that encompasses both standard KNN and PWK as special cases [2]_.

* **Mechanism:** It introduces a tunable parameter :math:`\alpha` into the weighting formula.
* **Flexibility:** :math:`\alpha` is a free parameter that allows the model to adapt to specific datasets. As :math:`\alpha` increases, the penalty for larger classes is progressively smoothed [2]_.
* **Relationship with KNN and PWK:**
    * When :math:`\alpha \to \infty`, the class weights converge to 1, and the algorithm behaves like a traditional **KNN**.
    * When :math:`\alpha = 1`, the algorithm is equivalent to standard **PWK**.
* **Performance:** Empirically, there is often no significant statistical difference in Absolute Error between PWK and PWK\ :math:`\alpha`. However, PWK\ :math:`\alpha` tends to be more **conservative and robust** at lower prevalences, while PWK may be more competitive at higher prevalences [2]_.

.. dropdown:: Mathematical details - PWK-Alpha Weights

   The weight for class :math:`\gamma_j` is a ratio between the class cardinality and the minority class cardinality (:math:`M`), raised to the power of :math:`1/\alpha`:

   .. math::

      w_{\gamma_j}(\alpha) = \left( \frac{M}{m_{\gamma_j}} \right)^{-1/\alpha}, \quad \text{with } \alpha \ge 1

**Example**

.. code-block:: python

   from mlquantify.neighbors import PWK
   
   # PWK operates directly on the feature space using k-NN
   q = PWK(n_neighbors=10)
   q.fit(X_train, y_train)
   q.predict(X_test)

.. dropdown:: References

    .. [2] Barranquero, J., González, P., Díez, J., & del Coz, J. J. (2013). On the study of nearest neighbor algorithms for prevalence estimation in binary problems. Pattern Recognition, 46(2), 472-482. https://doi.org/10.1016/j.patcog.2012.07.022