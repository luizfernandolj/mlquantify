.. _building_a_protocol:

===================
Building a Protocol
===================

.. currentmodule:: mlquantify.evaluation.protocol.Protocol

The construction of all protocols is similar, and you can create your own protocol by inheriting from the :class:`~mlquantify.evaluation.protocol.Protocol` class. The only requirement is to implement:

- `_new_sample` method: This method is responsible for generating the new sample to be used in the protocol. It should receive the X_test and y_test parameters, and return the new sample to be used in the protocol. The new sample should be a tuple of (X, y), where X is the new sample and y is the corresponding labels.
- `predict_protocol` method: This method is responsible for predicting the quantifier. It should receive the X_test and y_test parameters, and return the predictions of the quantifier. **Containing** at least the names of the quantifiers, the real prevalences and the predicted prevalences for each samples. You can use the `_new_sample` function to generate the samples
- `_delayed_fit` method: This method is responsible for fitting the quantifier. It should receive the model (quantifier), X_train and y_train parameters, and return the fitted quantifier.

An example of a custom protocol is:

.. code-block:: python

    import numpy as np
    from mlquantify.evaluation.protocol import Protocol
    from mlquantify.utils import get_real_prev
    from sklearn.ensemble import RandomForestClassifier
    import time as t

    class MyProtocol(Protocol):
        def __init__(self, 
                    models, 
                    n_jobs, 
                    random_state, 
                    verbose, 
                    return_type, 
                    measures, 
                    sample_size, 
                    iterations=10,
                    learner=RandomForestClassifier()):
            
            super().__init__(models, learner, n_jobs, random_state, verbose, return_type, measures, 
                            columns=['QUANTIFIER', 'REAL_PREVS', 'PRED_PREVS', 'TIME'])
            
            # Specific Parameters
            self.sample_size = sample_size
            self.iterations = iterations

        def predict_protocol(self, X_test, y_test):
            predictions = []
            X_sample, y_sample = self._new_sample(X_test, y_test)

            for _ in range(self.iterations):
                for model in self.models:
                    quantifier = model.__class__.__name__
                    real_prev = get_real_prev(y_sample)
                    
                    start_time = t.time()
                    pred_prev = model.predict(X_sample)
                    end_time = t.time()
                    time_elapsed = end_time - start_time
                    
                    predictions.append([quantifier, real_prev, pred_prev, time_elapsed])
            
            return predictions

        def _new_sample(self, X_test, y_test):
            indexes = np.random.choice(len(X_test), size=self.sample_size, replace=False)
            return X_test[indexes], y_test[indexes]


An important note is that this examples uses the :func:`~mlquantify.utils.general.get_real_prev` function from the `mlquantify.utils` module to get the real prevalences of the sample. This function is used to calculate the real prevalences of the sample, and it is important to use it to ensure that the real prevalences are calculated correctly.

Other point is that you need to pass the parameters of :class:`Protocol` class to the constructor of the Protocol class, and the parameters of your custom protocol to the constructor of your custom protocol. The parameters of the Protocol class can be found in the :ref:`protocol` section.

.. warning::

    The custom protocol must pass the columns parameter with **at least** the QUANTIFIER, REAL_PREVS and PRED_PREVS columns, otherwise the protocol will not work. The rest of the columns are optional.