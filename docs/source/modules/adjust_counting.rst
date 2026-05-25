.. _counting:

.. currentmodule:: mlquantify.counting

================
Adjusted Counting
================

Adjusted Counting methods improve upon simple "counting" quantifiers by correcting bias using what is known about the classifier's errors on the training set.  
They aim to produce better estimates of class prevalence (how frequent each class is in a dataset) even when training and test distributions differ.

see :ref:`counters_module` for an overview of the base counters for quantification.

This page focuses on **threshold adjustment methods**, which adjust the decision
threshold of a classifier to optimize prevalence estimation. Examples include
Adjusted Count (TAC) and its threshold selection policies (TX, TMAX, T50, MS,
MS2).



Threshold Adjustment
====================

Threshold-based adjustment methods correct the bias of :class:`CC` by using the classifier's **True Positive Rate (TPR)** and **False Positive Rate (FPR)**.  
They are mainly used for `binary` quantification tasks.

**Threshold Adjusted Count (TAC) Equation**

.. math::

   \hat{p}^U_{TAC}(⊕) = \frac{\hat{p}^U_{CC}(⊕) - FPR_L}{TPR_L - FPR_L}

:caption: *Corrected prevalence estimate using classifier error rates*

The main idea is that by adjusting the observed rate of positive predictions, we can better approximate the real class distribution.

.. figure:: ../images/threshold-selection-policies.png
   :align: center
   :width: 80%
   :alt: Threshold selection policies comparison

   *Comparison of different threshold selection policies showing FPR and 1-TPR curves with optimal thresholds for each method [Adapted from Forman (2008)]*

Different *threshold methods* vary in how they choose the classifier cutoff :math:`\tau` for scores :math:`s(x)` .

+----------------------------+------------------------------------------------------+-----------------------------------------+
| **Method**                 | **Threshold Choice**                                 | **Goal**                                |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`TAC`               | Fixed threshold :math:`\tau = 0.5`                   | Simple baseline adjustment              |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`TX`          | Threshold where :math:`\text{FPR} = 1 - \text{TPR}`  | Avoids unstable prediction tails        |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`TMAX`               | Threshold maximizing :math:`\text{TPR} - \text{FPR}` | Improves numerical stability            |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`T50`               | Threshold where :math:`\text{TPR} = 0.5`             | Uses central part of ROC curve          |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`MS` (Median Sweep) | Median of all thresholds' ACC results                | Reduces effect of threshold outliers    |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`MS2`               | Median Sweep variant with constraint                 | Reduces effect of threshold outliers    |
|                            | :math:`\|\text{TPR} - \text{FPR}\| > 0.25`           |                                         |
+----------------------------+------------------------------------------------------+-----------------------------------------+

All these methods have their `fit`, `predict` and `aggregate` functions, similar to other aggregative quantifiers. However, they also include a specialized method: `get_best_thresholds`, which identifies the optimal threshold, given `y` and predicted `probabilities`. Here is an example of how to use the :class:`T50` method:

.. code-block:: python

   from mlquantify.counting import T50, evaluate_thresholds
   from sklearn.linear_model import LogisticRegression

   clf = LogisticRegression()

   thresholds, tprs, fprs = evaluate_thresholds(
      y=y_test, 
      probabilities=clf.predict_proba(X_test)[:, 1]) # binary proba

   q = T50()
   best_thr, best_tpr, best_fpr = q.get_best_thresholds(X_val, y_val)
   print(f"Best threshold: {best_thr}, TPR: {best_tpr}, FPR: {best_fpr}")

.. note::

   Threshold adjustment methods like :class:`TAC` are primarily designed for binary classification tasks.
