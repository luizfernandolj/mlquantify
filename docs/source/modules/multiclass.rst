.. _multiclass:

.. currentmodule:: mlquantify.multiclass

==========================
Multiclass Quantification
==========================

Several quantifiers are **binary by design** — they estimate the prevalence of a
*positive* class against everything else (e.g. the threshold-adjustment methods
:class:`~mlquantify.counting.ACC`/:class:`~mlquantify.counting.TAC`, and the
mixture models :class:`~mlquantify.matching.DyS`, :class:`~mlquantify.matching.HDy`,
:class:`~mlquantify.matching.SORD`). To apply them to a problem with more than two
classes, ``mlquantify`` automatically **decomposes** the task into binary
sub-problems, runs the quantifier on each, and **recombines** the results into a
single prevalence vector over all classes.

This page covers the two built-in decomposition strategies (**One-vs-Rest** and
**One-vs-One**), how to choose between them, how to make your own quantifier
binary, and how to register a brand-new strategy.

.. contents:: Contents
   :local:
   :depth: 2

----

Setting the strategy on a binary method
=======================================

Binary methods expose a ``strategy`` parameter. Pass it to the constructor, or
set it afterwards as an attribute — both are equivalent:

.. code-block:: python

   from mlquantify.matching import DyS
   from sklearn.linear_model import LogisticRegression

   # via the constructor
   q = DyS(LogisticRegression(), strategy="ovo")

   # or as an attribute
   q = DyS(LogisticRegression())
   q.strategy = "ovo"

   q.fit(X_train, y_train)        # 4-class data
   q.predict(X_test)
   # {0: 0.26, 1: 0.24, 2: 0.25, 3: 0.25}

If the data has only two classes the decomposition is skipped entirely and the
method runs natively — the ``strategy`` is only consulted for ``> 2`` classes.

.. note::

   Native-multiclass methods (e.g. :class:`~mlquantify.counting.GACC`,
   :class:`~mlquantify.matching.KDEyML`, :class:`~mlquantify.matching.EDy`,
   :class:`~mlquantify.likelihood.EMQ`) do **not** decompose and ignore
   ``strategy``; they model all classes jointly.

----

One-vs-Rest vs One-vs-One
=========================

.. list-table::
   :widths: 14 43 43
   :header-rows: 1

   * - Strategy
     - One-vs-Rest (``'ovr'``, default)
     - One-vs-One (``'ovo'``)
   * - Sub-problems
     - One per class: *class c* vs the rest (``K`` binary quantifiers).
     - One per class pair (``K(K-1)/2`` binary quantifiers).
   * - Recombination
     - Each binary quantifier gives the prevalence of its class; the vector is
       normalised to sum to 1.
     - Each pair gives the prevalence of one class against the other; per-class
       estimates are averaged over all pairs the class appears in.
   * - Cost
     - Linear in ``K``.
     - Quadratic in ``K``.
   * - Use when
     - **Default.** Scales well; good general choice.
     - Few classes, or when one-vs-rest sub-problems are too imbalanced.

Both strategies run the sub-problems in parallel — pass ``n_jobs`` (when the
method supports it) to use multiple cores.

----

Making your own quantifier binary
=================================

Decorate a quantifier with :func:`binary_quantifier` to give it automatic
OvR/OvO handling. The decorator reads the strategy from the attribute named by
``strategy_attr`` (``"strategy"`` by default) and wraps ``fit`` / ``predict`` /
``aggregate`` with the decomposition logic, exposing the original
implementations as ``_original_fit`` / ``_original_predict`` / ``_original_aggregate``:

.. code-block:: python

   import numpy as np
   from mlquantify.base import BaseQuantifier
   from mlquantify.multiclass import binary_quantifier

   @binary_quantifier(strategy_attr="strategy")
   class MyBinaryQuantifier(BaseQuantifier):
       def __init__(self, estimator=None, strategy="ovr"):
           self.estimator = estimator
           self.strategy = strategy

       def fit(self, X, y):                 # always sees a *binary* y
           self.classes_ = np.unique(y)
           # ... fit on the binary problem ...
           return self

       def predict(self, X):                # returns a 2-element prevalence
           # ... estimate [neg, pos] ...
           return np.array([0.5, 0.5])

Your ``fit``/``predict`` only ever deal with the binary case; the decorator takes
care of the multiclass decomposition and recombination.

----

Adding a new strategy
=====================

Decomposition strategies live in a small registry, so a new one (error-correcting
output codes, hierarchical, nested dichotomies, …) is **one class plus one
decorator — no change to the dispatch**. Subclass :class:`MulticlassStrategy` and
register it with :func:`register_strategy`:

.. code-block:: python

   from mlquantify.multiclass import MulticlassStrategy, register_strategy

   @register_strategy("ecoc")
   class ECOCStrategy(MulticlassStrategy):
       def fit(self, q, X, y, n_jobs=None, fit_args=None, fit_kwargs=None):
           # return {key: fitted_binary_quantifier}
           ...

       def predict(self, q, X, n_jobs=None):
           # return per-class prevalences (dict or array)
           ...

       def aggregate(self, q, classes, args_dict, n_jobs=None):
           # return per-class prevalences from pre-computed predictions
           ...

       def fit_predict(self, q, X, y, X_test, classes, n_jobs=None):
           # fit on (X, y) and return per-class prevalences for X_test
           ...

   # now usable on any binary method:
   q = DyS(LogisticRegression(), strategy="ecoc")

The four methods return prevalences *before* the shared normalisation that
:class:`BinaryQuantifier` applies (see :ref:`prevalence_normalization`).
Inspect the available strategies with :func:`available_strategies`:

.. code-block:: python

   from mlquantify.multiclass import available_strategies
   available_strategies()
   # ['ecoc', 'ovo', 'ovr']

.. seealso::

   :ref:`prevalence_normalization` for how the recombined prevalences are
   normalised, and :ref:`building_a_quantifier` for writing quantifiers.
