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