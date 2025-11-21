.. _building_a_quantifier:

Building a Quantifier
=====================

This guide explains how to create custom quantifiers in MLQuantify by extending the base classes and using the provided utilities.


General Quantifiers
-------------------

To build a custom quantifier, inherit from :class:`BaseQuantifier` and implement the following methods:

- :func:`fit`: Train the quantifier on the provided training data (features and labels)
- :func:`predict`: Estimate class prevalences on test data using the learned model

**Example:**

.. code-block:: python

    from mlquantify.base import BaseQuantifier
    import numpy as np
    
    class MyQuantifier(BaseQuantifier):
        def __init__(self, param1=42, param2='default'):
            self.param1 = param1
            self.param2 = param2
        
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            # Add your training logic here
            return self
        
        def predict(self, X):
            _, counts = np.unique(self.classes_, return_counts=True)
            prevalence = counts / counts.sum()
            return prevalence


Aggregative Quantifiers
------------------------

Aggregative quantifiers combine predictions from a base learner. Inherit from both :class:`AggregativeQuantifierMixin` and :class:`BaseQuantifier` to create one.

**Required methods:**

- :func:`fit`: Train the quantifier and its base learner
- :func:`predict`: Generate individual predictions
- :func:`aggregate`: Convert individual predictions into prevalence estimates

.. important::

    - Place :class:`AggregativeQuantifierMixin` **first** in the inheritance list for proper method resolution
    - Include a ``learner`` attribute for automatic parameter handling

**Example:**

.. code-block:: python

    from mlquantify.base import BaseQuantifier
    from mlquantify.mixins import AggregativeQuantifierMixin
    import numpy as np
    
    class MyAggregativeQuantifier(AggregativeQuantifierMixin, BaseQuantifier):
        def __init__(self, learner, param1=42, param2='default'):
            self.learner = learner
            self.param1 = param1
            self.param2 = param2
        
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.learner.fit(X, y)
            return self
        
        def predict(self, X):
            return self.learner.predict(X)
        
        def aggregate(self, individual_predictions):
            _, counts = np.unique(individual_predictions, return_counts=True)
            prevalence = counts / counts.sum()
            return prevalence


Aggregation Types
-----------------

Specify the prediction type by inheriting from one of these mixins:

- :class:`CrispLearnerMixin`: For hard/crisp predictions (class labels)
- :class:`SoftLearnerMixin`: For soft/probabilistic predictions (class probabilities)

.. note::

    These mixins only work with :class:`AggregativeQuantifierMixin` and require a ``learner`` attribute.


Binary Quantifiers
------------------

Use the :func:`@define_binary <mlquantify.multiclass.define_binary>` decorator to adapt multiclass quantifiers for binary problems. The decorator automatically modifies :func:`fit`, :func:`predict`, and :func:`aggregate` methods.

**Multiclass strategies:**

- ``'ovr'`` (One-vs-Rest): Treat each class as positive vs. all others
- ``'ovo'`` (One-vs-One): Compare each pair of classes

**Example:**

.. code-block:: python

    from mlquantify.base import BaseQuantifier
    from mlquantify.multiclass import define_binary
    import numpy as np
    
    @define_binary
    class MyBinaryQuantifier(BaseQuantifier):
        def __init__(self, strategy='ovo', param1=42):
            self.strategy = strategy
            self.param1 = param1
        
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self
        
        def predict(self, X):
            # Binary prediction logic
            return np.array([0.5, 0.5])  # Example prevalence


Validation Utilities
--------------------

The :mod:`mlquantify.utils` module provides validation functions to ensure data consistency:

- :func:`validate_data`: Validate input features and labels
- :func:`validate_prevalences`: Validate prevalence estimates
- :func:`validate_predictions`: Validate learner predictions (for aggregative quantifiers)
- :func:`_fit_context`: Context manager for fitting with automatic parameter validation, with an option to skip multiple validations during fitting if in a loop.

**Example:**

.. code-block:: python

    from mlquantify.utils import (
        validate_data, 
        validate_prevalences, 
        validate_predictions, 
        _fit_context
    )
    from mlquantify.base import BaseQuantifier
    from mlquantify.mixins import SoftLearnerMixin, AggregativeQuantifierMixin
    
    class MyQuantifier(SoftLearnerMixin, AggregativeQuantifierMixin, BaseQuantifier):
        def __init__(self, learner, param1=42):
            self.learner = learner
            self.param1 = param1

        @_fit_context(prefer_skip_validation=True)
        def fit(self, X, y):
            X, y = validate_data(X, y)
            # Training logic
            return self

        def predict(self, X):
            X = validate_data(X)
            # Prediction logic
            return predictions
        
        def aggregate(self, individual_predictions):
            individual_predictions = validate_predictions(
                individual_predictions, self.classes_
            )
            # Aggregation logic
            return prevalence


Parameter Constraints
---------------------

Ensure parameter validity using the :mod:`mlquantify.utils` module:

1. Apply the :func:`@_fit_context <mlquantify.utils._fit_context>` decorator to the ``fit`` method
2. Use constraint validators from :mod:`mlquantify.utils` to check parameter types and values

These validators will raise informative errors when parameters don't meet expectations.

.. seealso::

    - :ref:`api_ref` for detailed API documentation
  
**Example:**

.. code-block:: python

    from mlquantify.base import BaseQuantifier
    from mlquantify.utils import (
        _fit_context,
        Interval,
        Options
    )
    import numpy as np

    class MyQuantifier(BaseQuantifier):
        _param_constraints = {
            'param1': [Interval(0, None, inclusive_left=False)],
            'param2': [Options(['default', 'option1', 'option2'])],
            'param3': "array-like"
        }

        def __init__(self, param1=42, param2='default', param3=None):
            self.param1 = param1
            self.param2 = param2
            self.param3 = param3

        @_fit_context(prefer_skip_validation=True)
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            # Training logic
            return self