.. _model_selection:

==========================================
Tunning the hyper-parameter of a quantifier
==========================================

.. currentmodule:: mlquantify.model_selection

Hyper-parameters are the parameters of the estimator that are set when instantiating it. After that, you can get and set the parameters of the estimator using the `get_params` and `set_params` methods respectively for all the quantifiers, including those you implemented, as long as they inherit from :class:`~mlquantify.base.AggregativeQuantifier`, :class:`~mlquantify.base.NonAggregativeQuantifier` or :class:`~mlquantify.base.Quantifier` classes.

.. code-block:: python

    quantifier.get_params(deep = True) # If true return the parameters for this estimator and contained subobjects that are estimators.
    quantifier.set_params(param = value) # Set the parameters of this estimator.

To tune the hyper-parameters of a quantifier, you can use the :class:`~mlquantify.model_selection.GridSearchQ` class. This class implements a grid search over the hyper-parameters of a quantifier, using the :class:`~mlquantify.evaluation.protocol.APP`, the best combination of hyper-parameters is selected based on the performance metric(s) specified in the `scoring` parameter.

First, you need to create a parameter grid, which is a dictionary containing the hyper-parameters to be tuned and their possible values. The keys of the dictionary should be the names of the hyper-parameters, and the values should be lists of possible values for each hyper-parameter. In case of aggregative quantifiers, the learner inside the quantifier can be tuned as well, using the `learner__` prefix. 

GridSearchQ can be used with any quantifier that inherits from :class:`~mlquantify.base.AggregativeQuantifier`, :class:`~mlquantify.base.NonAggregativeQuantifier` or :class:`~mlquantify.base.Quantifier` classes:

.. code-block:: python

    from mlquantify.model_selection import GridSearchQ
    from mlquantify.methods import DyS
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X_train = np.random.rand(200, 10)
    y_train = np.random.randint(0, 2, 200)

    parameter_grid = {
        'learner__n_estimators': [100, 200], # learner params
        'measure': ["topsoe", "hellinger"] # method params
    }
    quantifier = DyS(RandomForestClassifier())

    gs = GridSearchQ(model=quantifier, param_grid=parameter_grid, scoring='nae', verbose=True)

    gs.fit(X_train, y_train)

.. note::
    **learner__ prefix**
    
    You can use the `learner__` prefix in `set_params` method to set the parameters of the learner inside the quantifier.

To get the main attributes of the grid search, you can use the following methods:
- `best_model()`: Returns the best model found during the grid search.
- `get_params`: Gets the best parameters found during the grid search.

.. Note::
    **Scoring**

    You can pass a list of scoring metrics to the `scoring` parameter, and the best model will be selected based on the average of the scores for each metric. The scoring metrics can be any of the `~mlquantify.evaluation.measures` functions, excluding the absolute error metric. When using scoring, all scoring measures **should** be passed with its acronym, for example "mae" for :func:`~mlquantify.evaluation.measures.mean_absolute_error`. 