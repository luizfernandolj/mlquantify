.. _losses:

.. currentmodule:: mlquantify.losses

==============
Loss Functions
==============

Loss functions measure the discrepancy between a test representation and a
candidate mixture of training representations. They are used by compose and
matching quantifiers to estimate prevalences.

Available losses
================

- :class:`BaseLoss`
- :class:`DistanceLoss`
- :class:`LeastSquaresLoss`
- :class:`HellingerSurrogateLoss`
- :class:`EnergyLoss`
- :class:`NegativeLogLikelihoodLoss`
- :class:`MixtureNegativeLogLikelihoodLoss`
- :class:`RegularizedMixtureNLLLoss`
- :func:`get_loss`

Example
=======

.. code-block:: python

   from mlquantify.losses import get_loss

   loss = get_loss("hellinger")
   value = loss([0.4, 0.6], [0.5, 0.5])
