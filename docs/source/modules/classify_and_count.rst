.. _classify_and_count:

Classify and Count
==================

Classify and Count (CC) is the most basic aggregative quantification method. It consists of training a standard hard classifier on labeled data and then applying this classifier to the unlabeled set, estimating class prevalences by counting the number of instances assigned to each class.

Despite its simplicity, CC is known to be suboptimal in many scenarios because standard classifiers often exhibit bias that leads to inaccurate prevalence estimates.

This method is implemented in the class :class:`CC` in the package.

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


