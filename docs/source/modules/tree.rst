.. _quantification_trees:

.. currentmodule:: mlquantify.tree

====================
Quantification Trees
====================

Quantification trees (Milli et al., 2013) are decision trees whose learning
algorithm is optimized **directly for quantification** rather than for
individual classification, keeping the simplicity and interpretability of the
decision-tree framework. A single tree is exactly unbiased at the training
prevalence (its errors are balanced by construction) but, being a weak
classifier, still tilts under strong shift; the family's strongest member is
the **forest of Adjusted-Count trees**, which stays close to the true
prevalence across the whole range.

.. contents:: Contents
   :local:
   :depth: 2

----

Why optimize the tree for quantification?
=========================================

A Classify-and-Count estimate is exact whenever the classifier's errors
*compensate*: if, for every class :math:`c`, the number of false positives
equals the number of false negatives (:math:`FP_c = FN_c`), the predicted
class counts match the true ones — even if the classifier is individually
wrong on many instances. Standard trees minimise impurity (Gini/entropy),
which reduces the *total* number of errors but says nothing about their
balance. Quantification trees make the FP/FN balance the split criterion
itself.

The split criteria
==================

The tree maintains, over the whole training set, a per-class quantification
error vector :math:`QE` computed from the current tree's leaf predictions
(each leaf predicts its majority class). Two criteria are available:

- **EB — Classification Error Balancing** (``criterion='eb'``):

  .. math::

     QE[c] = |FP_c - FN_c|

  optimises the quantification goal only: perfectly balanced errors give
  :math:`QE = 0` regardless of how many errors there are.

- **CQB — Classification-Quantification Balancing** (``criterion='cqb'``,
  the default):

  .. math::

     QE[c] = |FP_c^2 - FN_c^2| = |FP_c - FN_c| \cdot (FP_c + FN_c)

  trades off the quantification error (:math:`|FP_c - FN_c|`) against the
  classification error (:math:`FP_c + FN_c`).

The goodness of a candidate split is the **gain**

.. math::

   \Delta = \|QE^{parent}\|_2 - \|QE^{child}\|_2

where the child value recomputes the global vector as if the node were
replaced by its two children. The split with the largest gain is chosen;
ties are broken by the reduction in misclassified samples.

.. note::

   The original paper stops growing as soon as no split has a strictly
   positive gain. Because :math:`\|QE\|_2` often reaches zero after a single
   well-balanced split, that rule alone can degenerate into a decision
   stump. mlquantify therefore also accepts zero-gain splits while they
   strictly reduce the misclassification count, so the tree keeps improving
   as a classifier without ever worsening its training quantification error.

The estimators
==============

The family provides three classes:

- :class:`QuantificationTreeClassifier` — the raw tree learner, a plain
  sklearn-style classifier with ``fit`` / ``predict`` / ``predict_proba``.
  Use it to compose with any aggregative quantifier.
- :class:`QuantificationTree` — a Classify-and-Count quantifier over a
  single quantification tree (the paper's :math:`CC(Q)` variant). Like
  :class:`~mlquantify.neighbors.PWK`, the classifier is intrinsic: there is
  no ``estimator`` parameter.
- :class:`QuantificationForest` — the Random Forest quantifier of the paper
  (Algorithm 3): ``n_estimators`` trees, each trained on a random fraction
  of the records with :math:`\lfloor\log_2 d\rfloor + 1` random features per
  split. Each tree produces an **Adjusted Count** estimate

  .. math::

     \hat{p}_c = \frac{\hat{p}_c^{CC} - fpr_c}{tpr_c - fpr_c}

  with the rates estimated by cross-validation on its own training records,
  and the forest reports the average of the per-tree estimates. This is the
  configuration with the best results in the paper. Set ``adjusted=False``
  for plain Classify-and-Count averaging.

Examples
--------

Basic usage:

.. code-block:: python

   from mlquantify.tree import QuantificationTree, QuantificationForest
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   X, y = make_classification(n_samples=2000, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42)

   q = QuantificationForest(n_estimators=100, random_state=42, n_jobs=-1)
   q.fit(X_train, y_train)
   print(q.predict(X_test))
   # {0: 0.50, 1: 0.50}

The paper's single-tree Adjusted Count variant, :math:`AC(Q)`, by
composition:

.. code-block:: python

   from mlquantify.counting import ACC
   from mlquantify.tree import QuantificationTreeClassifier

   q = ACC(estimator=QuantificationTreeClassifier(criterion='cqb'))
   q.fit(X_train, y_train)
   print(q.predict(X_test))

Tuning the criterion and tree size:

.. code-block:: python

   from mlquantify.model_selection import GridSearchQ, APP
   from mlquantify.metrics import MAE
   from mlquantify.tree import QuantificationTree

   protocol = APP(batch_size=100, n_prevalences=21, repeats=5)
   gs = GridSearchQ(
       quantifier=QuantificationTree(),
       param_grid={
           'criterion': ['eb', 'cqb'],
           'max_depth': [None, 5, 10],
       },
       protocol=protocol,
       error=MAE,
   )
   gs.fit(X_train, y_train)
   print(gs.best_params_)

When to Use Quantification Trees
--------------------------------

- When you want an interpretable model whose *learning phase* already
  targets quantification, instead of post-correcting an off-the-shelf
  classifier.
- :class:`QuantificationForest` (with the default ``adjusted=True``) is the
  strongest member of the family and resilient to sharp changes of the class
  distribution in the test set.
- The single :class:`QuantificationTree` applies plain CC and therefore
  inherits CC's bias under strong shift; prefer the forest or the
  ``ACC``/``GACC`` compositions when the test distribution can drift far
  from the training one.

References
==========

.. dropdown:: References

   - Milli, L., Monreale, A., Rossetti, G., Giannotti, F., Pedreschi, D., &
     Sebastiani, F. (2013). Quantification Trees. *IEEE International
     Conference on Data Mining (ICDM)*, pp. 528–536.
   - Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

.. seealso::

   :ref:`counters_module` for CC and ACC (which the trees build on).
   :ref:`nearest_neighbors` for PWK, the other quantifier with an intrinsic
   classifier.
