.. _matrix_adjustment:

Matrix Adjustment
-----------------

Matrix Adjustment is an extension of Adjust Counting where the full confusion matrix information is used to adjust the quantification estimates for multi-class problems. This involves inverting the confusion matrix to obtain corrected class prevalence estimates.

.. math::

   \hat{\mathbf{p}} = \mathbf{C}^\top \mathbf{p}

where \(\hat{\mathbf{p}}\) is the vector of observed predicted prevalences, \(\mathbf{C}\) is the confusion matrix, and \(\mathbf{p}\) is the true prevalence vector.

.. math::

   \mathbf{p} = (\mathbf{C}^\top)^{-1} \hat{\mathbf{p}}

