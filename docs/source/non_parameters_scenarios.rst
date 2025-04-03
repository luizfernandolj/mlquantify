.. _non_parameters_scenarios:

Non Parameters Scenarios
------------------------

When dealing with new reasearch problems, it is common to have new scenarios that are not covered by the current quantifiers. In this case, there are problems when:

- Learner (e.g. classifier) does not have :func:`fit`, :func:`predict` and :func:`predict_proba` methods, or it is an external learner that does not have a `sklearn` interface
- You already have the predictions of the mid task (e.g. classifier) and you don't want to remake the predictions.

The solution for this is to use the :func:`~mlquantify.set_arguments` method to set the arguments of the quantifier.

.. note::
    **Recommended Usage**
    
    When using the set_parameters, you don't need to pass any learner when instantiating the quantifier, but you still need to use the `fit` and `predict` methods of the quantifier.

.. code-block:: python

    from mlquantify import set_arguments
    from mlquantify.methods import EMQ
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(10, 10)
    y_test = np.random.randint(0, 2, 10)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # parameters
    y_pred = model.predict(X_test) # predictions of the test set
    posteriors_train = model.predict_proba(X_train) # predictions of the training set generated via cross validation
    posteriors_test = model.predict_proba(X_test) # predictions of the test set
    y_labels = y_train # Generated via cross validation
    y_pred_train = model.predict(X_train) # predictions of the training set generated via cross validation

    set_arguments(y_pred=y_pred,
                posteriors_train=posteriors_train,
                posteriors_test=posteriors_test,
                y_labels=y_labels,
                y_pred_train=y_pred_train)

    quantifier = EMQ()
    quantifier.fit(X_train, y_train)
    pred = quantifier.predict(X_test)
    print(pred)


Each argument is optinal depending on the quantifier you are using. The arguments are:

- y_pred: Array-like
    - Predictions for the test set, 
    - must be the same length as the test set. 
    - Must be python list, numpy array or pandas series.
- posteriors_train: Array-like
    - Class probabilities for the training set computed via **cross-validation**.
    - Must be of shape (n_samples, n_classes), with each row summing to 1.0.
    - Must be python list, numpy array or pandas Dataframe.
- posteriors_test: Array-like
    - Class probabilities for the test set.
    - Must be of shape (n_samples_test, n_classes), with each row summing to 1.0.
    - Must be python list, numpy array or pandas Dataframe.
- y_labels: Array-like
    - Ground truth labels for the training set used during **cross-validation**.
    - Must be the same length as the posteriors_train parameter.
    - Must be python list, numpy array or pandas series.
- y_pred_train: Array-like
    - Predictions for the training set generated through **cross-validation**.
    - Must be the same length as the posteriors_train parameter.
    - Must be python list, numpy array or pandas series.