.. _losses:

.. currentmodule:: mlquantify.losses

==============
Loss Functions
==============

Loss functions are the objective minimised by the :ref:`solvers`. They measure
the discrepancy between a test representation and a candidate
prevalence-weighted mixture of the training representations. A quantifier family
is largely defined by the ``(representation, loss, solver)`` triple it composes,
and the same loss is reused across several methods.

Role and mechanism
==================

Given the per-class descriptors and a candidate prevalence :math:`p`, a loss
:math:`\mathcal{L}(p)` scores how well the mixture reproduces the test
descriptor; the solver returns the :math:`p` that minimises it. The factory
:func:`get_loss` builds any of the losses below by name.

.. list-table::
   :widths: 34 38 28
   :header-rows: 1

   * - Loss
     - Objective (in words)
     - Used by
   * - :class:`DistanceLoss`
     - Selectable distribution distance (Hellinger, Topsoe, prob-symm, ...).
     - DyS, HDy, HDx
   * - :class:`LeastSquaresLoss`
     - Squared error :math:`\lVert y - X p \rVert^2` of the linear system.
     - GACC, GPACC, FM, MMD_RKHS
   * - :class:`HellingerSurrogateLoss`
     - :math:`-\sum_i \sqrt{p_i q_i}`, a gradient-friendly squared-Hellinger surrogate.
     - GHDy, GHDx, KDEyHD
   * - :class:`EnergyLoss`
     - Energy-distance quadratic :math:`p^\top(2q - M p)`.
     - EDy, EDx
   * - :class:`NegativeLogLikelihoodLoss`
     - Negative log-likelihood of the mixture density.
     - EMQ, KDEyML, GKDEyML, MLPE
   * - :class:`MixtureNegativeLogLikelihoodLoss`, :class:`RegularizedMixtureNLLLoss`
     - Mixture NLL built from per-class likelihoods, optionally with a
       simplex-smoothness penalty.
     - likelihood-compose quantifiers

:class:`BaseLoss` defines the callable interface; custom losses subclass it.

Choosing a loss
===============

- **Distance / Hellinger-surrogate** losses drive the histogram and KDE matching
  methods; Topsoe is usually the best general distance for :class:`~mlquantify.matching.DyS`.
- **Least squares** implements the unified constrained-regression objective.
- **Energy** is the closed quadratic behind the energy-distance methods.
- **Negative log-likelihood** is the maximum-likelihood objective behind EM and
  KDE-ML quantifiers.

Used by
=======

The loss is the second element of the ``(representation, loss, solver)`` triple;
see :ref:`distribution_matching` and :ref:`likelihood`.

Example
=======

.. code-block:: python

   from mlquantify.losses import get_loss

   loss = get_loss("hellinger")
   value = loss([0.4, 0.6], [0.5, 0.5])

References
==========

.. dropdown:: References

   - González-Castro, V., Alaiz-Rodríguez, R., & Alegre, E. (2013). Class
     Distribution Estimation Based on the Hellinger Distance. *Information
     Sciences*, 218, 146–164.
   - Firat, A. (2016). Unified Framework for Quantification. *arXiv:1606.00868*.
   - Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the Outputs
     of a Classifier to New a Priori Probabilities. *Neural Computation*, 14(1),
     21–41.

.. seealso::

   :ref:`representations` and :ref:`solvers` for the other two elements of the
   matching triple.
