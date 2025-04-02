.. _non_aggregative_quantifier:

=====================================
Building a Non-Aggregative Quantifier
=====================================

.. currentmodule:: mlquantify.base.NonAggregativeQuantifier

A non-aggregative quantifier does use a mid task, but tries to estimate the class distribution of the test set directly from the training set. For implementing new non-aggregative quantifiers you must subclass from :class:`mlquantify.base.NonAggregativeQuantifier` class, and implement only the :func:`_fit_method` and :func:`_predict_method` methods. 

The :func:`_fit_method` method is used to train the quantifier on a training set, while the :func:`_predict_method` method is used to make predictions on a test set. The :func:`_fit_method` method should return `self` to allow for method chaining. And the predictions should be a disctionary containing the predicted proportions for each class. The keys of the dictionary should be the class labels, sorted in ascending order. The values should be the predicted proportions for each class, which should sum to 1.0.

.. note::
    **Recommended Usage**
    
    When using quantifiers that subclass from :class:`mlquantify.base.NonAggregativeQuantifier` to fit and predict data, you must use the :func:`fit` and :func:`predict`, where these methods will call the :func:`_fit_method` and :func:`_predict_method` methods respectively. 


New non-aggregative quantifiers can be implemented the following way:

.. code-block:: python

    from mlquantify.base import NonAggregativeQuantifier

    class MyNonAggregativeQuantifier(NonAggregativeQuantifier):
    
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

        def _fit_method(self, X, y):
            # Fit the quantifier to the training data
            return self

        def _predict_method(self, X):
            # Make predictions on the test data
            return {0: 0.5, 1: 0.5}  # Example prediction