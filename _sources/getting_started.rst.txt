Getting Started
===============

**mlquantify** is a comprehensive Python toolkit for **Quantification** (also known as *Class Prevalence Estimation*, *Class Prior Estimation*, or *Shift Estimation*). 

Installation
------------

You can install ``mlquantify`` using pip:

.. code-block:: bash

   pip install mlquantify

Or install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/luizfernandolj/mlquantify.git
   cd mlquantify
   pip install .

Basic Usage
-----------

Most quantifiers in ``mlquantify`` behave like scikit-learn estimators. They implement ``fit(X, y)`` and ``predict(X)`` methods.

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from mlquantify.adjust_counting import CC  # Classify & Count

    # 1. Initialize a base classifier
    estimator = LogisticRegression()

    # 2. Wrap it with an aggregative quantifier (e.g., CC)
    quantifier = CC(estimator)
    
    # 3. Fit on labeled training data
    quantifier.fit(X_train, y_train)

    # 4. Predict class prevalences on new data
    prevalences = quantifier.predict(X_test)
    
    print(prevalences)

The ``fit`` Parameters
----------------------

All aggregative quantifiers in ``mlquantify`` support a consistent set of parameters in their ``fit`` method to control how the underlying classifier is trained or used:

*   ``X``: The training input samples (array-like, sparse matrix).
*   ``y``: The target values (class labels).
*   ``learner_fitted`` (bool): If ``True``, assumes the provided estimator is already trained. If ``False`` (default), trains the estimator on the provided ``X`` and ``y``.
*   ``cv`` (int, cross-validation generator, or iterable): Determines the cross-validation splitting strategy for generating internal predictions (used by methods like ACC, PACC).
*   ``stratified`` (bool): If ``True``, uses stratified folds for cross-validation.
*   ``shuffle`` (bool): Whether to shuffle the data before splitting in cross-validation.

Aggregative Quantifiers & ``aggregate``
---------------------------------------

Aggregative methods (like CC, ACC, PCC) estimate prevalence by aggregating predictions from individual instances.

Unlike standard estimators, they offer an additional **``aggregate``** method. This allows you to perform quantification **without re-predicting** if you already have the classifier's outputs (labels or probabilities) for your test set.

.. code-block:: python

    # Assume we already have predictions for the test set
    predictions = classifier.predict(X_test)
    
    # Use 'aggregate' directly - no need for X_test
    estimated_prevalence = quantifier.aggregate(predictions)

Model evaluation
----------------

Fitting a model to some data does not entail that it will predict well on unseen data. This needs to be directly evaluated. We typically use a ``train_test_split`` to split a dataset into train and test sets, and then use specific metrics to compare the predicted prevalences against the true prevalences.

``mlquantify`` provides many tools for model evaluation in the :mod:`mlquantify.metrics` module.

.. code-block:: python

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from mlquantify.adjust_counting import CC
    from mlquantify.metrics import MAE
    from sklearn.linear_model import LogisticRegression

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, weights=[0.8, 0.2], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Initialize and fit
    quantifier = CC(LogisticRegression())
    quantifier.fit(X_train, y_train)

    # Predict prevalences
    y_pred = quantifier.predict(X_test)

    # Calculate Mean Absolute Error (MAE)
    # y_test contains true labels; we convert them to prevalences for comparison
    error = MAE(y_test, y_pred) 
    print(f"Mean Absolute Error: {error:.4f}")

Quantification Protocols
~~~~~~~~~~~~~~~~~~~~~~~~

In quantification, a single test set is often insufficient because we want to evaluate performance across *different* class distributions (shifts). 

**Protocols** like the **Artificial Prevalence Protocol (APP)** allow you to generate many test samples with varying prevalences from a single dataset.

.. code-block:: python

    from mlquantify.protocols import APP
    from mlquantify.utils import get_prev_from_labels

    # Create an APP generator:
    # - n_prevalences=21: Generate samples with prevalences from 0.0 to 1.0 (step 0.05)
    # - repeats=10: Generate 10 difference samples for each prevalence
    protocol = APP(batch_size=100, n_prevalences=21, repeats=10, random_state=42)

    errors = []
    
    # APP.split() yields indices for each test sample
    for test_index in protocol.split(X_test, y_test):
        X_sample, y_sample = X_test[test_index], y_test[test_index]
        
        # Predict prevalence on this specific sample
        pred_prev = quantifier.predict(X_sample)
        
        # Calculate error for this sample
        errors.append(MAE(y_sample, pred_prev))

    print(f"Mean Absolute Error across {len(errors)} samples: {sum(errors)/len(errors):.4f}")

See :ref:`quantification_protocols` for more details on APP, NPP, and other protocols.

Next steps
----------

We have briefly covered estimator fitting and predicting, aggregative methods, and model evaluation. This guide should give you an overview of some of the main features of the library, but there is much more to ``mlquantify``!

Please refer to our :ref:`user-guide` for details on all the tools that we provide, including **Non-Aggregative Methods**, **Meta Quantification**, and **Confidence Intervals**. You can also find an exhaustive list of the public API in the :ref:`api`.

You can also look at our numerous :ref:`examples` that illustrate the use of ``mlquantify`` in many different contexts.
