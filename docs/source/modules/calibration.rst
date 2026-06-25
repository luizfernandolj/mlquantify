.. _calibration:

.. currentmodule:: mlquantify.calibration

===========
Calibration
===========

Well-calibrated posterior probabilities improve probabilistic quantifiers such
as :class:`~mlquantify.likelihood.EMQ`. The :mod:`mlquantify.calibration`
subpackage provides post-hoc calibration of classifier posteriors through the
*scaling* family of methods.

Classifier calibration
======================

:class:`ClassifierCalibrator` rescales a classifier's logits to minimise the
negative log-likelihood of a held-out set. Four methods are available via the
``method`` argument:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - ``method``
     - Description
   * - ``'ts'``
     - **Temperature Scaling** -- one shared temperature ``T`` (Guo et al., 2017).
   * - ``'bcts'``
     - **Bias-Corrected Temperature Scaling** -- ``T`` plus per-class biases
       (Alexandari, Kundaje & Shrikumar, 2020). A strong default.
   * - ``'vs'``
     - **Vector Scaling** -- per-class weights and biases (Guo et al., 2017).
   * - ``'nbvs'``
     - **No-Bias Vector Scaling** -- per-class weights only (Alexandari et al.,
       2020).

The calibrator follows the scikit-learn ``fit(y_true, y_pred)`` / ``predict``
convention. ``y_pred`` may be probabilities (``input_type='proba'``, the
default) or raw logits (``input_type='logits'``); :meth:`~ClassifierCalibrator.predict`
always returns calibrated probabilities.

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import train_test_split
   from mlquantify.calibration import ClassifierCalibrator

   X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.3)
   clf = LogisticRegression().fit(X_tr, y_tr)

   # Fit the calibrator on held-out predictions, never on the training set.
   cal = ClassifierCalibrator(method="bcts").fit(y_cal, clf.predict_proba(X_cal))
   calibrated = cal.predict(clf.predict_proba(X_test))

.. note::

   Calibration must be fit on predictions held out from classifier training (a
   validation split or cross-validated predictions); fitting it on the training
   predictions under-estimates the miscalibration.

Use with EMQ
============

:class:`~mlquantify.likelihood.EMQ` can apply calibration internally before the
EM loop -- pass ``calib_function='bcts'`` (or ``'ts'`` / ``'vs'`` / ``'nbvs'``):

.. code-block:: python

   from mlquantify.likelihood import EMQ
   from sklearn.linear_model import LogisticRegression

   emq = EMQ(LogisticRegression(), calib_function="bcts").fit(X_train, y_train)
   prevalence = emq.predict(X_test)

Quantifier calibration
======================

:class:`QuantifierCalibrator` is reserved for post-hoc calibration of quantifier
outputs and is **not implemented yet** (its methods raise
``NotImplementedError``). Use :class:`ClassifierCalibrator` to calibrate the
posteriors a quantifier consumes.

Custom calibrators
==================

Subclass :class:`Calibrator` to implement your own strategy:

.. code-block:: python

   from mlquantify.calibration import Calibrator

   class MyCalibrator(Calibrator):
       def fit(self, y_true, y_pred):
           return self

       def predict(self, y_pred):
           return y_pred

References
==========

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
  *On Calibration of Modern Neural Networks.* ICML.
- Alexandari, A., Kundaje, A., & Shrikumar, A. (2020).
  *Maximum Likelihood with Bias-Corrected Calibration is Hard-to-Beat.* ICML.
