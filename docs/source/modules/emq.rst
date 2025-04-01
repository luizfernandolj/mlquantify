.. _expectation_maximisation_for_quantification:

Expectation Maximisation for Quantification
-------------------------------------------

The Expectation Maximisation (EM) is an iterative algorithm that is used to find the maximum likelihood estimates of parameters (our case is the class distribution)for models that depend on unobserved latent variables. The algorithm consists by incrementally updating the posterior probabiblites by using the class prevalence values computed in the last step of the iteration, and updates the class prevalence values by using the posterior probabilities computed in the last step of the iteration, in a mutually recursive fashion.

The method is available in :class:`~mlquantify.methods.aggregative.EMQ` class, and can be used as a probabilistic classifier, using the :func:`~mlquantify.methods.aggregative.EMQ.predict_proba` method, or as a quantifier, example below:

.. code-block:: python

    from mlquantify.methods.aggregative import EMQ
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, size=100)
    X_test = np.random.rand(50, 10)

    quantifier = EMQ(LogisticRegression())
    quantifier.fit(X_train, y_train)

    class_distribution = quantifier.predict(X_test)
    scores = quantifier.predict_proba(X_test)

    print("Class distribution:", class_distribution)
    print("Scores:", scores)
