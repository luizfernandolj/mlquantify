.. _adjust_counting:

.. currentmodule:: mlquantify.adjust_counting

===============
Adjust Counting
===============

Adjusted Counting methods improve upon the basic Classify and Count :ref:`CC` or its probabilistic variant :ref:`PCC` by correcting the estimated prevalences using knowledge from the training set about the behavior of the classifier, particularly accounting for its error characteristics.

Currently, there are two types of adjustment methods implemented:

1. **Threshold Adjustment Methods**: These methods adjust the decision threshold of the classifier to optimize prevalence estimation. Examples include Adjusted Classify and Count (ACC) and its probabilistic counterpart PACC.
2. **Matrix Adjustment Methods**: These methods use a confusion matrix derived from the classifier's performance on a validation set to adjust the estimated prevalences. Examples include the EM-based methods and other matrix inversion techniques.



.. _classify_and_count:

Classify and Count
==================

Classify and Count (CC) is the most basic aggregative quantification method. It consists of training a standard hard classifier on labeled data and then applying this classifier to the unlabeled set, estimating class prevalences by counting the number of instances assigned to each class.

Despite its simplicity, CC is known to be suboptimal in many scenarios because standard classifiers often exhibit bias that leads to inaccurate prevalence estimates.

This method is implemented in the :ref:`CC` class.

Example of usage
---------------

.. code-block:: python

    from mlquantify.adjust_counting import CC
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    # Using fit and predict methods
    q = CC(learner=LogisticRegression())
    q.fit(X, y)
    q.predict(X)
    {0: 0.47, 1: 0.53}

    # Using the aggregate method directly
    q2 = CC()
    predictions = np.random.rand(200)
    q2.aggregate(predictions)
    {0: 0.51, 1: 0.49}


However, Forman (2008) showed that CC can be biased when train and test class distributions differ, leading to the development of various adjusted counting methods that aim to correct this bias.



.. _probabilistic_classify_and_count:

Probabilistic Classify and Count
================================

There is also a probabilistic variant of Classify and Count called Probabilistic Classify and Count (PCC), which uses the predicted probabilities from a probabilistic classifier instead of hard class labels to estimate prevalences. This method is implemented in the class :ref:`PCC`.

Example of usage
---------------

.. code-block:: python

    from mlquantify.adjust_counting import PCC
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    # Using fit and predict methods
    q = PCC(learner=LogisticRegression())
    q.fit(X, y)
    q.predict(X)
    {0: 0.45, 1: 0.55}

    # Using the aggregate method directly
    q2 = PCC()
    probabilities = np.random.rand(200, 2) # Probabilistic outputs for n classes
    q2.aggregate(probabilities)
    {0: 0.48, 1: 0.52}



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


.. _matrix_adjustment:


Matrix Adjustment
================

Matrix Adjustment is an extension of Adjust Counting where the full confusion matrix information is used to adjust the quantification estimates for multi-class problems. This involves inverting the confusion matrix to obtain corrected class prevalence estimates.

.. math::

   \hat{\mathbf{p}} = \mathbf{C}^\top \mathbf{p}

where \(\hat{\mathbf{p}}\) is the vector of observed predicted prevalences, \(\mathbf{C}\) is the confusion matrix, and \(\mathbf{p}\) is the true prevalence vector.

.. math::

   \mathbf{p} = (\mathbf{C}^\top)^{-1} \hat{\mathbf{p}}

