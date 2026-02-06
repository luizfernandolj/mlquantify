.. _tuning_hyperparameters:

Tuning Hyperparameters
----------------------

.. currentmodule:: mlquantify.model_selection


Hyperparameter tuning is essential for optimizing any machine learning model, and quantification models are no exception. In quantification, hyperparameter tuning often focuses on parameters that influence the model's ability to estimate class prevalences accurately under distribution shifts, using specific evaluation metrics designed for quantification tasks [1]_.

The :class:`GridSearchQ` class provides a systematic way to perform hyperparameter tuning for quantification models. It extends the traditional grid search approach by incorporating quantification-specific evaluation metrics and protocols.

**Key Features of GridSearchQ:**

- **Quantification Metrics:** Supports metrics like Absolute Error (AE), Relative Absolute Error (RAE), and Kullback-Leibler Divergence (KLD) that are specifically designed to evaluate quantification performance.
- **Protocols Integration:** Seamlessly integrates with quantification protocols such as Artificial-Prevalence Protocol (APP) to generate diverse test samples for robust evaluation.
- **Cross-Validation:** Implements cross-validation strategies tailored for quantification tasks to ensure reliable hyperparameter selection.
- **Parallel Processing:** Supports parallel computation to speed up the hyperparameter search process.

**Example Usage**

.. code-block:: python

    from mlquantify.likelihood import EMQ
    from mlquantify.model_selection import GridSearchQ
    from mlquantify.metrics import MAE
    from sklearn.ensemble import RandomForestClassifier

    param_grid = {'alpha': [0.1, 1.0], 'beta': [10, 20]}
    grid_search = GridSearchQ(quantifier=EMQ(RandomForestClassifier()),
                            param_grid=param_grid,
                            protocol='app',
                            samples_sizes=100,
                            n_repetitions=5,
                            scoring=MAE,
                            refit=True,
                            val_split=0.3,
                            n_jobs=2,
                            random_seed=123,
                            verbose=True)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    best_params = grid_search.best_params()
    best_model = grid_search.best_model()

.. seealso::

   - :class:`GridSearchQ` for detailed implementation.
   - :ref:`quantification_protocols` for available quantification protocols.
   - :ref:`evaluation_metrics` for available quantification metrics.
  
References
==========

.. [1] Esuli, A., Fabris, A., Moreo, A., & Sebastiani, F. (n.d.). Learning to Quantify The Information Retrieval Series.