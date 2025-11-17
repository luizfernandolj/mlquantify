.. _classify_and_count:

==================
Classify and Count
==================

Classify and Count (CC) is the most basic aggregative quantification method. It consists of training a standard hard classifier on labeled data and then applying this classifier to the unlabeled set, estimating class prevalences by counting the number of instances assigned to each class.

Despite its simplicity, CC is known to be suboptimal in many scenarios because standard classifiers often exhibit bias that leads to inaccurate prevalence estimates.

This method is implemented in the :class:`~mlquantify.adjust_counting.CC` class.

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


================================
Probabilistic Classify and Count
================================

There is also a probabilistic variant of Classify and Count called Probabilistic Classify and Count (PCC), which uses the predicted probabilities from a probabilistic classifier instead of hard class labels to estimate prevalences. This method is implemented in the class :class:`~mlquantify.adjust_counting.PCC`.

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
