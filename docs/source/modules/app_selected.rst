.. _app_selected:

=================================================
Artificial Prevalence protocol with selected usage
=================================================

.. currentmodule:: mlquantify.evaluation.protocol.APP

Other way of using the APP protocol is the selected way, where you can choose a specific set of quantifiers to be evaluated, including the ones implemented by the user, see :ref:`building_a_quantifier` for more details. This is useful when you want to evaluate the performance of your own quantifier or a specific set of quantifiers, and compare their results.

To use this approach, you must pass a list of instantiated quantifiers to the `models` argument. The quantifiers must be any class that inherits from the :class:`~mlquantify.methods.base.Quantifier` class, linha `AggregativeQuantifier` or `NonAggregativeQuantifier` classes, e.g. [CC(), DyS(), EMQ(), myQuantifier()], each one with its parameters setted.

The usage is similar to the general usage, but when using the selected way, you don't need to pass the learner arument:

.. code-block:: python

    from mlquantify.evaluation.protocol import APP
    from mlquantify.methods import CC, DyS, EMQ
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X_train = np.random.rand(1000, 20)  # training data
    y_train = np.random.randint(0, 2, size=1000)  # training labels
    X_test = np.random.rand(1000, 20)  # test data
    y_test = np.random.randint(0, 2, size=1000)  # test labels

    # list of quantifiers
    quantifiers = [CC(RandomForestClassifier()), DyS(RandomForestClassifier()), EMQ(RandomForestClassifier())]  # or 'all', 'aggregative', 'non-aggregative'

    app = APP(models=quantifiers, 
              batch_size=100, 
              n_prevs=20,
              measures=["mae"] 
              return_type="table", 
              verbose=True)

    app.fit(X_train, y_train)
    table = app.predict(X_test, y_test)

    print(table)
    

See :ref:`protocol` for more details on all protocol parameters.

