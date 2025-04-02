.. _aggregative_quantifier:

==================================
Building an Aggregative Quantifier
==================================

If you are building a new quantifier that aggregates the results of a mid task, such as classification, you can subclass the :class:`~mlquantify.base.AggregativeQuantifier` class. This class provides two methods that must be implemented by subclasses, including the :func:`_fit_method` and :func:`_predict_method` methods. where in the `_fit_method` you can implement the fitting of the quantifier to the training data the way you want, and in the `_predict_method` you can implement the prediction of the quantifier to the test data, returning always a dictionary containing the predicted proportions for each class.

.. note::
    **Dynamic Label Scheme**
    
    In case your quantifier is a binary quantifier, you must include the `is_multiclass` property returning False, and in other case, you don't need to include it.

.. note::
    **Method functionality**
    
    If your quantifier fits a learner (e.g., a classifier) in the `_fit_method`, you can include the `is_probabilistic` property returning False if the learner is not probabilistic, and nothing if it is probabilistic. This will allow you to use the :func:`~mlquantify.base.AggregativeQuantifier.fit_learner` and :func:`~mlquantify.base.AggregativeQuantifier.predict_learner` methods to fit and predict the learner for dinamically handling of multiple scenarios, including the **Non parameters Scenarios**, see :ref:`non_parameters_scenarios` for more information.

An aggregative quantifier can be implemented the following way:

.. code-block:: python
    
    from mlquantify.base import AggregativeQuantifier
    from sklearn.ensemble import RandomForestClassifier

    class MyAggregativeQuantifier(AggregativeQuantifier):
        def is_multiclass(self):
            return False  # Doesn't need to be included if True
        
        def is_probabilistic(self):
            return False  # Doesn't need to be included if True

        def __init__(self, learner=RandomForestClassifier(), param1=None, param2=None):
            self.learner = learner
            self.param1 = param1
            self.param2 = param2

        def _fit_method(self, X, y):
            
            self.fit_learner(X, y)

            return self

        def _predict_method(self, X):
            # Make predictions on the test data
            y_pred = self.predict_learner(X)  # Example usage of learner prediction

            return {0: 0.5, 1: 0.5}  # Example prediction

.. warning::
    **Recommended usage**
    
    When implementing a new aggregative quantifier, it is recommended to use learner as one of the parameters, and it must have `fit`, `predict` and `predict_proba` methods.
    When using quantifiers that subclass from :class:`mlquantify.base.AggregativeQuantifier` to fit and predict data, you must use the :func:`fit` and :func:`predict`, where these methods will call the :func:`_fit_method` and :func:`_predict_method` methods respectively. 