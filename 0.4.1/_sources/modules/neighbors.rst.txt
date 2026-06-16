.. _nearest_neighbors:

.. currentmodule:: mlquantify.neighbors

==================
Nearest Neighbours
==================

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

References
==========

.. dropdown:: References

   - Barranquero, J., Díez, J., & del Coz, J. J. (2013). On the study of
     nearest neighbor algorithms for prevalence estimation in binary problems.
     *Pattern Recognition*, 46(2), 472–482.

.. seealso::

   :ref:`counters_module` for CC (which PWK builds on top of).
   :ref:`likelihood` for EMQ (stronger shift correction).
