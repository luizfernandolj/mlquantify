.. _prevalence_normalization:

.. currentmodule:: mlquantify

========================
Prevalence Normalization
========================

Every quantifier returns its estimate through a single normalization step, so
the output is always a valid prevalence vector — non-negative and summing to 1 —
in a consistent format. Two settings control this step: the **return type**
(array or dict) and the **normalization strategy** (how raw estimates are turned
into probabilities and how several estimates are aggregated).

.. contents:: Contents
   :local:
   :depth: 2

----

The ``normalize_prevalence`` helper
===================================

:func:`~mlquantify.utils.normalize_prevalence` is the low-level helper that turns
a raw vector (or dict) of class scores into a normalized prevalence summing to 1,
aligned to a list of classes:

.. code-block:: python

   from mlquantify.utils import normalize_prevalence

   normalize_prevalence([2.0, 3.0, 5.0], classes=[0, 1, 2])
   # {0: 0.2, 1: 0.3, 2: 0.5}

   normalize_prevalence({0: 0.1, 1: 0.1, 2: 0.3}, classes=[0, 1, 2])
   # {0: 0.2, 1: 0.2, 2: 0.6}

Parameters
----------

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Parameter
     - Meaning
   * - ``prevalences``
     - The raw estimate to normalize: a 1-D array, or a ``{class: value}`` dict.
       Values need not sum to 1 (they are rescaled).
   * - ``classes``
     - The class labels, used to order the output and to fill in any class
       missing from a dict input with ``0``.

Quantifiers rarely call this directly — they go through ``validate_prevalences``,
which additionally applies the **configurable** return type and normalization
strategy described below.

----

Configuring normalization
=========================

Two global options drive the final formatting of every prevalence estimate.
Read them with :func:`get_config`, change them with :func:`set_config` (global)
or :func:`config_context` (temporary, scoped):

``prevalence_return_type`` — output format
------------------------------------------

.. list-table::
   :widths: 16 84
   :header-rows: 1

   * - Value
     - Behaviour
   * - ``'array'``
     - Return a :class:`numpy.ndarray` ordered by class. **Global default.**
   * - ``'dict'``
     - Return a ``{class_label: prevalence}`` dictionary.

``prevalence_normalization`` — normalization / aggregation strategy
-------------------------------------------------------------------

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - Value
     - Behaviour
   * - ``'sum'`` / ``'l1'``
     - Rescale so the values sum to 1 (the standard prevalence constraint).
       **Global default.**
   * - ``'softmax'``
     - Apply the softmax function — useful when the raw estimates are logits or
       unbounded scores rather than proportions.
   * - ``'mean'``
     - Average several estimates (rows of a 2-D input) into one prevalence — used
       when many estimates are produced, e.g. by ensembles or bootstrap.
   * - ``'median'``
     - Take the per-class median across several estimates; more robust to outlier
       estimates than ``'mean'``.
   * - ``None``
     - No normalization or aggregation — return the raw values unchanged.

When the input is a 2-D array of several estimates, ``'sum'``/``'l1'`` normalize
each row and then average them, while ``'mean'``/``'median'`` aggregate directly;
for a single 1-D estimate the aggregation options reduce to that estimate.

----

Examples
========

Set a global default for the whole session:

.. code-block:: python

   from mlquantify import set_config, get_config

   set_config(prevalence_return_type="dict", prevalence_normalization="sum")
   get_config()["prevalence_return_type"]
   # 'dict'

Change it only temporarily with the context manager (recommended — it restores
the previous configuration on exit):

.. code-block:: python

   from mlquantify import config_context
   from mlquantify.matching import DyS
   from sklearn.linear_model import LogisticRegression

   q = DyS(LogisticRegression()).fit(X_train, y_train)

   # default: array output, sum-normalized
   q.predict(X_test)                       # array([0.49, 0.51])

   with config_context(prevalence_return_type="dict"):
       q.predict(X_test)                   # {0: 0.49, 1: 0.51}

   with config_context(prevalence_normalization="median"):
       # aggregate many estimates by their per-class median
       ...

.. note::

   Per-class **median** does not in general sum to 1; if you need both robust
   aggregation *and* the simplex constraint, aggregate with ``'median'`` and then
   re-normalize with ``'sum'``.

.. seealso::

   :ref:`multiclass` for how One-vs-Rest / One-vs-One recombine binary estimates
   before this normalization is applied.
