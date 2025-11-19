.. _building_a_quantifier:

Building a Quantifier
---------------------


General Quantifiers
===================

In MLQuantify, you can build your own quantifier by following the :class:`BaseQuantifier` base class. This base class provides the necessary structure and methods that your custom quantifier should implement. To create a new quantifier, you need to define the following key methods:

- :func:`fit`: This method is responsible for training the quantifier on the provided training data. You should implement the logic to learn from the features and labels of the training set.
- :func:`predict`: This method is used to estimate the class prevalences on the test data. You should implement the logic to make predictions based on the learned model from the `fit` method.


Aggregative Quantifiers
=======================