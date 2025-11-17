.. _threshold_adjustment:

Threshold Adjustment
====================

Threshold Adjustment methods optimize the classifier decision threshold to improve quantification accuracy. The threshold affects the classifier's true positive and false positive rates, which in turn influence correction methods such as Adjust Counting.

.. math::

   \hat{y} = \begin{cases}
   1 & s \geq \tau \\
   0 & \text{otherwise}
   \end{cases}

where \(s\) is the classifier score and \(\tau\) is the decision threshold.

By selecting appropriate \(\tau\), the quantification estimates can be stabilized, avoiding estimation issues with probabilities near zero or one.

Each Threshold Adjustment method typically chooses \(\tau\) based on different criteria, such as minimizing quantification error on validation data or balancing true positive and false positive rates.

