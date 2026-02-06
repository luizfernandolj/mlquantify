Getting Started
===============

`mlquantify` is a comprehensive Python toolkit for quantification (also known as class prevalence estimation or class prior estimation). This guide will help you get started with the main features of the library:

- Aggregative methods
- Non-aggregative methods
- Meta methods
- Evaluation metrics
- Confidence Intervals
- Model Selection

It is assumed basic knowledge of machine learning practices (fitting, evaluation, etc.). You can install `mlquantify`

Installation
------------

You can install ``mlquantify`` directly from PyPI:

.. code-block:: bash

   pip install mlquantify

Alternatively, you can install the latest version from source:

.. code-block:: bash

   git clone https://github.com/luizfernandolj/mlquantify.git
   cd mlquantify
   pip install .

Quick Start
-----------

Here is a minimal example of how to use ``mlquantify`` to estimate class prevalences using the **Classify & Count (CC)** method.

1. **Import necessary modules**

   We'll use ``scikit-learn`` for the underlying classifier and data generation.

   .. code-block:: python

      from mlquantify.adjust_counting import CC
      from sklearn.linear_model import LogisticRegression
      from sklearn.datasets import make_classification
      from sklearn.model_selection import train_test_split

2. **Generate Synthetic Data**

   Let's create a binary classification dataset.

   .. code-block:: python

      # Generate a synthetic dataset
      X, y = make_classification(
          n_samples=2000, 
          n_features=20, 
          n_classes=2, 
          weights=[0.8, 0.2],  # Imbalanced dataset
          random_state=42
      )

      # Split into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

3. **Initialize and Train the Quantifier**

   Most aggregative quantifiers in ``mlquantify`` wrap a standard classifier. Here we use Logistic Regression.

   .. code-block:: python

      # Initialize the classifier
      classifier = LogisticRegression()

      # Initialize the Classify & Count (CC) quantifier
      quantifier = CC(classifier)

      # Fit the quantifier on the training data
      quantifier.fit(X_train, y_train)

4. **Estimate Class Prevalences**

   Use the ``predict`` method to estimate the class distribution of the test set.

   .. code-block:: python

      # Estimate class prevalences
      prevalences = quantifier.predict(X_test)

      print(f"Estimated class prevalences: {prevalences}")
      # Example Output: [0.79, 0.21]

Next Steps
----------

*   Explore the :ref:`mlquantify-methods` to see all available algorithms.
*   Check out the :ref:`user-guide` for in-depth tutorials on Aggregative and Non-Aggregative methods.
*   Learn about :ref:`evaluation-metrics` to properly assess your quantifier's performance.
