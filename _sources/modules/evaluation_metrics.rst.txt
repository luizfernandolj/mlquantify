.. _evaluation_metrics:

Evaluation Metrics
------------------

.. currentmodule:: mlquantify.metrics

Evaluation metrics for quantification assess the accuracy of estimated class prevalences against true prevalences. These metrics are crucial for understanding how well a quantifier performs, especially under distributional shifts.

The library includes several widely used evaluation metrics:

.. list-table:: Metrics
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - :class:`NMD`
     - Normalized Match Distance
   * - :class:`RNOD`
     - Relative Normalized Overall Deviation
   * - :class:`VSE`
     - Variance Shift Error
   * - :class:`CvM_L1`
     - Cramér-von Mises L1 Distance
   * - :class:`AE`
     - Absolute Error
   * - :class:`SE`
     - Squared Error
   * - :class:`MAE`
     - Mean Absolute Error
   * - :class:`MSE`
     - Mean Squared Error
   * - :class:`KLD`
     - Kullback-Leibler Divergence
   * - :class:`RAE`
     - Relative Absolute Error
   * - :class:`NAE`
     - Normalized Absolute Error
   * - :class:`NRAE`
     - Normalized Relative Absolute Error
   * - :class:`NKLD`
     - Normalized Kullback-Leibler Divergence

=========================================
Single Label Quantification (SLQ) Metrics
=========================================

AE (Absolute Error)
===================

**Parameters:**  

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence (distribution of classes).  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.

AE calculates the simple absolute error across classes:

.. math::

   \text{AE}(p, \hat{p}) = \sum_{c} |p(c) - \hat{p}(c)|

Its primary strength is transparency and ease of interpretation.

SE (Squared Error)
==================

**Parameters:**

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.

SE is the sum of squared differences:

.. math::

   \text{SE}(p, \hat{p}) = \sum_{c} (p(c) - \hat{p}(c))^2

This penalizes larger errors more heavily, making outlier mistakes more obvious.

MAE (Mean Absolute Error)
=========================

**Parameters:**

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.

MAE averages the absolute errors over all classes:

.. math::

   \text{MAE}(p, \hat{p}) = \frac{1}{K} \sum_{c} |p(c) - \hat{p}(c)|

It offers a normalized perspective, useful for comparing performances across datasets.

MSE (Mean Squared Error)
========================

**Parameters:**  

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.

MSE averages the squared errors:

.. math::

   \text{MSE}(p, \hat{p}) = \frac{1}{K} \sum_{c} (p(c) - \hat{p}(c))^2

Ideal for highlighting large deviations in prevalence estimation.

KLD (Kullback-Leibler Divergence)
=================================

**Parameters:** 

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.

KLD measures the information loss between distributions:

.. math::

   \text{KLD}(p, \hat{p}) = \sum_{c} p(c) \log \frac{p(c)}{\hat{p}(c)}

Its key advantage is sensitivity to wrong predictions where the true prevalence is high.

RAE (Relative Absolute Error)
=============================

**Parameters:**  

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.  
- :math:`\epsilon`: float, optional (default=1e-12)  
  Small constant to ensure numerical stability.

RAE scales the absolute error by true prevalence:

.. math::

   \text{RAE}(p, \hat{p}) = \sum_{c} \frac{|p(c) - \hat{p}(c)|}{p(c) + \epsilon}

This is beneficial for identifying relative impact in imbalanced scenarios.

NAE (Normalized Absolute Error)
===============================

**Parameters:**

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.

NAE normalizes the absolute error:

.. math::

   \text{NAE}(p, \hat{p}) = \frac{1}{K} \sum_{c} \frac{|p(c) - \hat{p}(c)|}{\max\{p(c), \hat{p}(c)\}}

Best used for ensuring error scale invariance.

NRAE (Normalized Relative Absolute Error)
=========================================

**Parameters:**

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.  
- :math:`\epsilon`: float, optional (default=1e-12)  
  Small constant for numerical stability.

NRAE further normalizes relative errors:

.. math::

   \text{NRAE}(p, \hat{p}) = \frac{1}{K} \sum_{c} \frac{|p(c) - \hat{p}(c)|}{p(c) + \hat{p}(c) + \epsilon}

This balances error measurement between true and estimated values.

NKLD (Normalized Kullback-Leibler Divergence)
=============================================

**Parameters:** 

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.  
- :math:`\epsilon`: float, optional (default=1e-12)
  Small constant for numerical stability.

NKLD outputs a normalized form of KLD:

.. math::

   \text{NKLD}(p, \hat{p}) = \frac{1}{K} \sum_{c} p(c) \log \frac{p(c)}{\hat{p}(c) + \epsilon}

This makes it robust for comparing across distinct sample sizes.

============================================
Regression-Based Quantification (RQ) Metrics
============================================

VSE (Variance Shift Error)
==========================

**Parameters:** 

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.

The Variance Shift Error quantifies the discrepancy between the variance of true and estimated distributions:

.. math::

   \text{VSE}(p, \hat{p}) = |\text{Var}(p) - \text{Var}(\hat{p})|

This metric emphasizes changes in dispersion, which is useful for detecting model bias towards certain classes.

CvM_L1 (Cramér-von Mises L1 Distance)
=====================================

**Parameters:**  

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.

CvM_L1 compares cumulative distributions using the L1 norm:

.. math::

   \text{CvM\_L1}(p, \hat{p}) = \sum_{c} |F_p(c) - F_{\hat{p}}(c)|

where \(F_p(c)\) is the cumulative distribution. Its advantage lies in capturing distributional differences beyond pointwise errors.

===================================
Ordinal Quantification (OQ) Metrics
===================================

NMD (Normalized Match Distance)
===============================

**Parameters:**  

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.

The NMD metric quantifies the normalized difference between two prevalence distributions:

.. math::

   \text{NMD}(p, \hat{p}) = \frac{1}{2} \sum_{c} |p(c) - \hat{p}(c)|

where \( p(c) \) is the true prevalence and \( \hat{p}(c) \) is the estimated. The advantage of NMD is its straightforward interpretability and normalization, making it ideal for comparing different quantification methods.

RNOD (Relative Normalized Overall Deviation)
============================================

**Parameters:**

- :math:`p`: array-like, shape (n_classes,)  
  True prevalence.  
- :math:`\hat{p}`: array-like, shape (n_classes,)  
  Estimated prevalence.  
- :math:`\epsilon`: float, optional (default=1e-12)  
  Small constant to ensure numerical stability.

RNOD measures the proportional deviation between the true and estimated prevalence, particularly highlighting errors in rare classes:

.. math::

   \text{RNOD}(p, \hat{p}) = \frac{1}{K} \sum_{c} \frac{|p(c) - \hat{p}(c)|}{p(c) + \epsilon}

Its benefit is in handling imbalanced distributions by reducing the influence of dominant classes.
