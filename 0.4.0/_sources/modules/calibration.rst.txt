.. _calibration:

.. currentmodule:: mlquantify.calibration

===========
Calibration
===========

Calibration utilities provide hooks for adjusting classifier or quantifier
outputs before prevalence estimation. Use these classes to implement custom
calibration strategies when needed.

Available classes
=================

- :class:`Calibrator`
- :class:`ClassifierCalibrator`
- :class:`QuantifierCalibrator`

Example skeleton
================

.. code-block:: python

   from mlquantify.calibration import Calibrator

   class MyCalibrator(Calibrator):
       def fit(self, y_true, y_pred):
           return self

       def predict(self, y_pred):
           return y_pred
