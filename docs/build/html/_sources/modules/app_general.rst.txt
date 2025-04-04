.. _app_general:

=================================================
Artificial Prevalence protocol with general usage
=================================================

.. currentmodule:: mlquantify.evaluation.protocol.APP

In quantification, the APP protocol is a method for evaluating the performance of quantifiers. It is based on the idea of varying class distribution in the test set, to simulate different scenarios and assess the robustness of the quantifier. The APP protocol is widely used in the field of quantification.

One way to use this protocol is the general way, of using all classes of methods in the same evaluation. This is useful when you want to evaluate the performance of a quantifier across different classes of methods, and compare their results.

You can use 4 ways to define the quantifiers to be evaluated in a general format in the `models` argument:

- a list of strings containing the names of the quantifiers to be evaluated, e.g. `['CC', 'DyS', 'EMQ']`
- 'all' to evaluate all the quantifiers available in the library (except for the :class:`~mlquantify.methods.meta.Ensemble` class)
- 'aggregative' to evaluate all the aggregative quantifiers available in the library
- 'non-aggregative' to evaluate all the non-aggregative quantifiers available in the library

.. note::

    When using one of the listed options, you must pass the learner argument to the protocol, unless you are using non-aggregative or the list of methods only containg non-aggregative methods.

basic usage:

.. code-block:: python

    from mlquantify.evaluation.protocol import APP
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X_train = np.random.rand(1000, 20)  # training data
    y_train = np.random.randint(0, 2, size=1000)  # training labels
    X_test = np.random.rand(1000, 20)  # test data
    y_test = np.random.randint(0, 2, size=1000)  # test labels

    # list of quantifiers
    quantifiers = ["CC", "DyS", "EMQ"]  # or 'all', 'aggregative', 'non-aggregative'

    app = APP(models=quantifiers, 
              learner=RandomForestClassifier(), 
              batch_size=100, 
              n_prevs=20,
              measures=["mae"], 
              return_type="table", 
              verbose=True)

    app.fit(X_train, y_train)
    table = app.predict(X_test, y_test)

    print(table)
    

See :ref:`protocol` for more details on all protocol parameters.