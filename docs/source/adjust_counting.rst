.. _adjust_counting:

Adjust Counting
---------------

Adjusted Counting methods improve upon the basic Classify and Count :ref:`CC` or its probabilistic variant :ref:`PCC` by correcting the estimated prevalences using knowledge from the training set about the behavior of the classifier, particularly accounting for its error characteristics.

Currently, there are two types of adjustment methods implemented:

1. **Threshold Adjustment Methods**: These methods adjust the decision threshold of the classifier to optimize prevalence estimation. Examples include Adjusted Classify and Count (ACC) and its probabilistic counterpart PACC.
2. **Matrix Adjustment Methods**: These methods use a confusion matrix derived from the classifier's performance on a validation set to adjust the estimated prevalences. Examples include the EM-based methods and other matrix inversion techniques.


.. toctree::
   :maxdepth: 2

   modules/classify_and_count.rst
   modules/threshold_adjustment.rst
   modules/matrix_adjustment.rst