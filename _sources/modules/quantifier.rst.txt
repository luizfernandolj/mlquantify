.. _quantifier:

=====================
Building a Quantifier
=====================

.. currentmodule:: mlquantify.base.Quantifier

Quantifiers are the main objects of interest in the `mlquantify` library. Since quantification is still emerging as a research area, new quantifiers are being developed all the time. The library is designed to be extensible, so you can easily add your own quantifiers to test out new ideas, and send them to us for inclusion in the library.

Basic quantifiers are easily implemented by subclassing the :class:`~mlquantify.base.Quantifier` class. The `Quantifier` class is an abstract base class that defines the interface for all quantifiers. It provides a number of methods that must be implemented by subclasses, including the :func:`~mlquantify.base.Quantifier.fit` and :func:`~mlquantify.base.Quantifier.predict` methods. The `fit` method is used to train the quantifier on a training set, while the `predict` method is used to make predictions on a test set.


.. warning:: 
   **Recommended usage**

   The `Quantifier` class is **not** intended to be used directly, and it is **recommended** that you subclass from :class:`~mlquantify.base.AggregativeQuantifier` or :class:`~mlquantify.base.NonAggregativeQuantifier` instead, since these classes provide dynamic handling of binary and multiclass problems and has more functionality.

.. note::
    **Method functionality**
    
    The `fit` method should return `self` to allow for method chaining. And the predictions should be a disctionary containing the predicted proportions for each class. The keys of the dictionary should be the class labels, sorted in ascending order. The values should be the predicted proportions for each class, which should sum to 1.0. 

You can implement new quantifiers the following way:

.. code-block:: python

    from mlquantify.base import Quantifier

    class MyQuantifier(Quantifier):
    
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

        def fit(self, X, y):
            # Fit the quantifier to the training data
            return self

        def predict(self, X):
            # Make predictions on the test data
            return {0: 0.5, 1: 0.5}  # Example prediction