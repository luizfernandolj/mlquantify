.. _adjust_counting:

.. currentmodule:: mlquantify.adjust_counting

===============
Adjust Counting
===============

Adjusted Counting methods improve upon simple "counting" quantifiers by correcting bias using what is known about the classifier's errors on the training set.  
They aim to produce better estimates of class prevalence (how frequent each class is in a dataset) even when training and test distributions differ.

see :ref:`modules/counters` for an overview of the base counters for quantification.

Currently, there are two types of adjustment methods implemented:

1. **Threshold Adjustment Methods**: These methods adjust the decision threshold of the classifier to optimize prevalence estimation. Examples include Adjusted Classify and Count (ACC) and its probabilistic counterpart PACC.
2. **Matrix Adjustment Methods**: These methods use a confusion matrix derived from the classifier's performance on a validation set to adjust the estimated prevalences. Examples include the EM-based methods and other matrix inversion techniques.



Threshold Adjustment  
====================

Threshold-based adjustment methods correct the bias of :class:`CC` by using the classifier's **True Positive Rate (TPR)** and **False Positive Rate (FPR)**.  
They are mainly used for `binary` quantification tasks.

**Adjusted Classify and Count (ACC) Equation**

.. math::

   \hat{p}^U_{ACC}(⊕) = \frac{\hat{p}^U_{CC}(⊕) - FPR_L}{TPR_L - FPR_L}

:caption: *Corrected prevalence estimate using classifier error rates*

The main idea is that by adjusting the observed rate of positive predictions, we can better approximate the real class distribution.

[Plot Idea: Diagram showing how TPR and FPR move the prevalence estimate]

Different *threshold methods* vary in how they choose the classifier cutoff :math:`\tau` for scores :math:`s(x)` .

+----------------------------+------------------------------------------------------+-----------------------------------------+
| **Method**                 | **Threshold Choice**                                 | **Goal**                                |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`ACC`               | Fixed threshold :math:`\tau = 0.5`                   | Simple baseline adjustment              |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`X_method`          | Threshold where :math:`\text{FPR} = 1 - \text{TPR}`  | Avoids unstable prediction tails        |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`MAX`               | Threshold maximizing :math:`\text{TPR} - \text{FPR}` | Improves numerical stability            |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`T50`               | Threshold where :math:`\text{TPR} = 0.5`             | Uses central part of ROC curve          |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`MS` (Median Sweep) | Median of all thresholds' ACC results                | Reduces effect of threshold outliers    |
+----------------------------+------------------------------------------------------+-----------------------------------------+
| :class:`MS2`               | Median Sweep variant with constraint                 | Reduces effect of threshold outliers    |
|                            | :math:`\|\text{TPR} - \text{FPR}\| > 0.25`           |                                         |
+----------------------------+------------------------------------------------------+-----------------------------------------+

**Example**

.. code-block:: python

   from mlquantify.adjust_counting import ACC
   q = ACC(learner=LogisticRegression())
   q.fit(X, y)
   q.predict(X)
   # -> adjusted prevalence dictionary

[Plot Idea: Show ROC curve with marked chosen thresholds (TPR, FPR, tau)]

.. note::

    Threshold adjustment methods like ACC are primarily designed for binary classification tasks,  
    For multi-class problems, matrix adjustment methods are generally preferred.



Matrix Adjustment  
=================

Matrix-based adjustment methods use a *confusion matrix* or *generalized rate matrix* to adjust predictions for multi-class quantification.  
They treat quantification as solving a small linear system.

**Matrix Equation**

.. math::

   \mathbf{y = X \hat{\pi}_F + \epsilon}, \quad
   \text{subject to } \hat{\pi}_F \ge 0,\ \sum \hat{\pi}_F = 1

:caption: *General linear system linking observed and true prevalences*

Here:

- :math:`\mathbf{y}`: average observed predictions in :math:`U`  
- :math:`\mathbf{X}`: classifier behavior from training (mean conditional rates)  
- :math:`\hat{\pi}_F`: corrected class prevalences in :math:`U` 

[Plot Idea: Matrix illustration showing how confusion corrections map to estimated prevalences]

Generalized Adjusted Classify and Count (GAC) and Generalized Probabilistic Adjusted Classify and Count (GPAC)
--------------------------------------------------------------------------------------------------------------

.. code-block:: python

   from mlquantify.adjust_counting import GAC, GPAC
   from sklearn.linear_model import LogisticRegression
   q = GAC(learner=LogisticRegression())
   q.fit(X_train, y_train)
   q.predict(X_test)
   # -> {0: 0.48, 1: 0.52}

Both :class:`GAC` and :class:`GPAC` are solved using this linear system:

- GAC uses hard classifier decisions (confusion matrix).  
- GPAC uses soft probabilities :math:`P(y=l|x)` .



Friedman's Method (FM)  
----------------------

The :class:`FM` constructs its adjustment matrix :math:`\mathbf{X}` based on a specialized feature transformation function :math:`f_l(x)` that indicates whether the predicted class probability for an item exceeds that class's proportion in the training data :math:`(\pi_l^T)` , a technique chosen because it theoretically minimizes the variance of the resulting prevalence estimates.

.. dropdown:: Mathematical details - Friedman's Method

   To improve stability, **Friedman's Method (FM)** generates the adjustment matrix :math:`\mathbf{X}` using a special transformation function applied to each class :math:`l` and training sample :math:`x` :

   .. math::

      f_l(x) = I \left[ \hat{P}_T(y = l \mid x) > \pi_l^T \right]

   where:

   - :math:`I[\cdot]` is the indicator function, equal to 1 if the condition inside is true, 0 otherwise.  
   - :math:`\hat{P}_T(y = l \mid x)` is the classifier's estimated posterior probability for class :math:`l` on training sample :math:`x`.  
   - :math:`\pi_l^T` is the prevalence of class :math:`l` in the training set.

   The entry :math:`X_{i,l}` of the matrix :math:`\mathbf{X}` is computed as the average of :math:`f_l(x)` over all :math:`x` in class :math:`i` of the training data:

   .. math::

      X_{i,l} = \frac{1}{|L_i|} \sum_{x \in L_i} f_l(x)

   where:

   - :math:`L_i` is the subset of training samples with true class :math:`i`.  
   - :math:`|L_i|` is the number of these samples.

   This matrix is then used in the constrained least squares optimization:

   .. math::

      \min_{\hat{\pi}_F} \frac{1}{2} \hat{\pi}_F^\top D \hat{\pi}_F + d^\top \hat{\pi}_F
      \quad \text{subject to} \quad \hat{\pi}_F \ge 0, \quad \sum \hat{\pi}_F = 1

   to estimate the corrected prevalences :math:`\hat{\pi}_F` on the test set [3]_.

   This thresholding on posterior probabilities ensures that the matrix :math:`\mathbf{X}` highlights regions where the classifier consistently predicts a class more confidently than its baseline prevalence, improving statistical stability and reducing estimation variance [3]_.

   **References**

   .. [3] Friedman, J. H. (2015). CLASS COUNTS IN FUTURE UNLABELED SAMPLES (Detecting and dealing with concept drift).