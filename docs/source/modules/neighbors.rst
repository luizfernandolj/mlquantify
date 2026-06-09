.. _nearest_neighbors:

.. currentmodule:: mlquantify.neighbors

=================
Nearest Neighbours
=================

Nearest-neighbour quantifiers estimate class prevalences by leveraging the
local structure of the feature space. They are non-parametric, require no
distributional assumptions, and are naturally robust to non-linear decision
boundaries.

.. contents:: Contents
   :local:
   :depth: 2

----

PWK — Probabilistic Weighted k-Nearest Neighbours
===================================================

:class:`PWK` (Barranquero et al., 2013) wraps a k-NN classifier with a
**class-imbalance-aware weighting scheme** and then applies Classify and
Count (CC) on its predictions. Each neighbour's vote is multiplied by a
class-specific weight that corrects for the size difference between classes,
so that the minority class is not drowned out by the majority.

The weight for class :math:`c` is:

.. math::

   w_c(\alpha) = \left(\frac{M}{m_c}\right)^{1/\alpha}

where :math:`M = \min_c m_c` is the smallest class size and :math:`\alpha`
is the imbalance-correction exponent.

Special cases:

- :math:`\alpha = 1` → standard PWK (weights proportional to inverse class size).
- :math:`\alpha \to \infty` → all weights equal to 1 (standard k-NN).

**Why it exists:** A standard k-NN classifier biases predictions towards
the majority class. PWK's weighting neutralises this bias, producing more
accurate CC estimates under class imbalance. Barranquero et al. (2013) showed
PWK outperforms standard CC + k-NN by a wide margin on imbalanced datasets.

Parameters
----------

.. list-table::
   :widths: 22 15 63
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``alpha``
     - ``1``
     - Imbalance-correction exponent. Higher values reduce the penalty on
       larger classes:

       - ``alpha=1`` — standard PWK weighting (most aggressive correction).
       - ``alpha=2`` — gentler correction.
       - Very large ``alpha`` (e.g. 100) approaches standard k-NN.

       Tune by cross-validation if you are unsure; ``alpha=1`` is a good
       starting point.
   * - ``n_neighbors``
     - ``10``
     - Number of neighbours :math:`k`. Larger :math:`k` gives smoother
       (lower variance) prevalence estimates. Use larger values on bigger
       datasets. Common choices: 5, 10, 20.
   * - ``algorithm``
     - ``'auto'``
     - k-NN search algorithm. ``'auto'`` selects the fastest available
       (ball_tree/kd_tree for low-dim data, brute-force for high-dim).
   * - ``metric``
     - ``'euclidean'``
     - Distance metric for neighbour search. Use ``'euclidean'`` for
       continuous normalised features; ``'cosine'`` for sparse text/embedding
       vectors; ``'manhattan'`` for count data.
   * - ``leaf_size``
     - ``30``
     - Leaf size for tree-based algorithms. Affects speed, not accuracy.
   * - ``p``
     - ``2``
     - Minkowski distance parameter. ``p=2`` → Euclidean, ``p=1`` →
       Manhattan.
   * - ``metric_params``
     - ``None``
     - Extra keyword arguments for custom metric functions.
   * - ``n_jobs``
     - ``None``
     - Parallel jobs for neighbour search. ``-1`` uses all cores.

Examples
--------

Basic usage:

.. code-block:: python

   from mlquantify.neighbors import PWK
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   X, y = make_classification(n_samples=1000, weights=[0.8, 0.2],
                              random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42)

   q = PWK(alpha=1, n_neighbors=10)
   q.fit(X_train, y_train)
   print(q.predict(X_test))
   # {0: 0.80, 1: 0.20}

Tuning alpha and k:

.. code-block:: python

   from mlquantify.model_selection import GridSearchQ, APP
   from mlquantify.metrics import MAE
   from mlquantify.neighbors import PWK

   protocol = APP(batch_size=100, n_prevalences=21, repeats=5)
   gs = GridSearchQ(
       quantifier=PWK(),
       param_grid={
           'alpha': [1, 2, 5],
           'n_neighbors': [5, 10, 20],
       },
       protocol=protocol,
       error=MAE,
   )
   gs.fit(X_train, y_train)
   print(gs.best_params_)

Getting per-instance classifications:

.. code-block:: python

   q = PWK(alpha=1, n_neighbors=10)
   q.fit(X_train, y_train)
   labels = q.classify(X_test)   # hard labels from the weighted k-NN
   print(labels[:10])

When to Use PWK
---------------

- When you want a simple, non-parametric quantifier without tuning a
  full probabilistic classifier.
- When feature space distances are meaningful (normalised continuous features).
- For imbalanced binary problems where a standard classifier would be biased.

**Limitation:** PWK applies CC on top of the k-NN classifier; it inherits
CC's bias under distributional shift. It does not apply any adjustment like
ACC or EMQ. For strong shift correction, use EMQ or DyS instead.

.. seealso::

   :ref:`counters_module` for CC (which PWK builds on top of).
   :ref:`likelihood` for EMQ (stronger shift correction).

Nearest Neighbors
=================

PWK: Pair-wise Weighted K-Nearest Neighbors
===========================================

:class:`PWK` and **PWK**\ :math:`\alpha` are neighbor-based algorithms (k-Nearest Neighbors - NN) that apply class weighting techniques to estimate prevalence in binary problems [2]_. They can be seen as aggregative quantification methods that adjust a k-NN classifier to account for class imbalance.

PWK methods focus on simplicity and stability, offering competitive quantification that balances effectiveness with interpretability. By exploring the topology of the data through proximity-based analysis, these algorithms are robust to distribution changes and retain local structure better than global classifiers [2]_. The methods complement k-NN with class weighting policies designed to neutralize the bias toward majority classes inherent in imbalanced datasets [2]_.

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