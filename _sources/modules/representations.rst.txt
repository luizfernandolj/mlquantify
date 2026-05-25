.. _representations:

.. currentmodule:: mlquantify.representations

===============
Representations
===============

Representations transform model outputs or feature spaces into vectors that can
be matched against class-conditional summaries. They are used by compose and
matching quantifiers to build the linear system or likelihood objective.

Core representations
====================

- :class:`BaseRepresentation`
- :class:`HistogramRepresentation`
- :class:`KDERepresentation`
- :class:`DistanceRepresentation`
- :class:`KernelMeanRepresentation`
- :class:`PredictionRepresentation`
- :class:`HardPredictionRepresentation`
- :class:`SoftPredictionRepresentation`

Example
=======

.. code-block:: python

   from mlquantify.representations import HistogramRepresentation

   rep = HistogramRepresentation(bins=(10,))
   rep.fit(train_scores, y_train)
   test_representation = rep.transform(test_scores)
