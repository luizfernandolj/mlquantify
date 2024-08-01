import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
import quantifyML
from quantifyML import evaluation
from ...base import Quantifier
from ...utils import make_prevs, getHist, normalize_prevalence, parallel, hellinger

class Ensemble(Quantifier):
    
    SELECTION_METRICS = {"ds", "all", "ptr"}
    
    def __init__(self, 
                 method:Quantifier,
                 size:int=50,
                 min_pos_prop:float=0.1,
                 sample_size:int=None,
                 n_jobs:int=1,
                 metric:str = "all",
                 p_metric:float = 0.5,
                 return_type="mean",
                 verbose:bool=False):
        
        assert sample_size is None or sample_size > 0, \
            'wrong value for sample_size; set it to a positive number or None'
        assert metric in Ensemble.SELECTION_METRICS or metric in evaluation.MEASURES, \
            "metric not value, please select one of the options: 'ds', 'all', 'ptr' or one of the measures in evaluation package"
        
        self.method = method
        self.size = size
        self.min_pos_prop = min_pos_prop
        self.sample_size = sample_size
        self.n_jobs = n_jobs
        self.metric = metric
        self.p_metric = p_metric
        self.return_type = return_type
        self.verbose = verbose
        self.ensemble = None

    def sout(self, msg):
        if self.verbose:
            print('[Ensemble] ' + msg)
           
    def fit(self, X, y):
        self.sout("FIT")
        self.classes = np.unique(y)
        
        if self.metric == "ds" and not self.binary_data:
            raise ValueError("ds selection measure is not valid for multiclass problems, plase pass a binary dataset")
    
        sample_size = len(X) if self.sample_size is None else min(self.sample_size, len(X))
        if sample_size > len(X):
            self.sout("ALERT: SAMPLE SIZE GREATER THAN LENGTH OF DATA")
        
        prevs = [make_prevs(ndim=self.n_class, min_val=self.min_pos_prop) for _ in range(self.size)]
        
        posteriors = None
        if self.metric == 'ds':
            # precompute the training posterior probabilities
            posteriors, self.proba_generator = self.ds_get_posteriors(X, y)
        
        args = (
            (X, y, self.method, prev, posteriors, self.verbose, sample_size)
            for prev in prevs
        )
        
        self.ensemble = parallel(
            self._delayed_new_sample,
            tqdm(args, desc='fitting ensamble', total=self.size) if self.verbose else args,
            n_jobs=self.n_jobs)
        
        self.sout('Fit [Done]')
        return self
    
    def predict(self, X):
        if self.verbose:
            self.sout("PREDICTING")
        
        args = ((X, Ei[2]) for Ei in self.ensemble)
        
        prevalences = np.asarray(
            parallel(
                self._delayed_predict_ensemble,
                tqdm(args, desc='predicting ensamble', total=self.size) if self.verbose else args,
                n_jobs=self.n_jobs
            )
        )
        
        prevalences = pd.DataFrame(prevalences).to_numpy()
        
        if self.metric == "ds":
            prevalences = self.ds_metric(prevalences, X)
        elif self.metric == "ptr":
            prevalences = self.ptr_metric(prevalences, evaluation.get_measure("mse"))
        elif self.metric in evaluation.MEASURES:
            prevalences = self.ptr_metric(prevalences, evaluation.get_measure(self.metric))
        
        if self.return_type == "mean":
            prevalences = np.mean(prevalences, axis=0)
        elif self.return_type == "median":
            prevalences = np.median(prevalences, axis=0)
        
        return normalize_prevalence(prevalences, self.classes)
        
    
    def _delayed_new_sample(self, args):
        
        X, y, base_method, prev, posteriors, verbose, sample_size = args
        
        method = deepcopy(base_method)
        
        if verbose:
            ...
            #print(f'\tfit-start for prev {str(prev)}, sample_size={sample_size}')
        
        indexes = self._generate_indexes(y, prev, sample_size)
        X_sample = X[indexes]
        y_sample = y[indexes]
        
  
        method.fit(X_sample, y_sample)
        tr_distribution = getHist(posteriors[indexes], 8) if (posteriors is not None) else None
        if verbose:
            ...
             #print(f'\t\--fit-ended for prev {str(prev)}')
            
        return (X_sample, y_sample, method, prev, tr_distribution)
    
    
    
    def _delayed_predict_ensemble(self, args):
        X, method = args
        return list(method.predict(X).values())
    
    
    
    def _generate_indexes(self, y, prevalence: list, sample_size:int):        
        # Ensure the sum of prevalences is 1
        assert np.isclose(sum(prevalence), 1), "The sum of prevalences must be 1"
        # Ensure the number of prevalences matches the number of classes
        assert len(prevalence) == len(self.classes), "The number of prevalences must match the number of classes"

        sampled_indexes = []
        total_sampled = 0

        for i, class_ in enumerate(self.classes):

            if i == len(self.classes) - 1:
                num_samples = sample_size - total_sampled
            else:
                num_samples = int(sample_size * prevalence[i])
            
            # Get the indexes of the current class
            class_indexes = np.where(y == class_)[0]

            # Sample the indexes for the current class
            sampled_class_indexes = np.random.choice(class_indexes, size=num_samples, replace=True)
            
            sampled_indexes.extend(sampled_class_indexes)
            total_sampled += num_samples

        np.random.shuffle(sampled_indexes)  # Shuffle after collecting all indexes
            
        return sampled_indexes
    
    
    
    def ds_get_posteriors(self, X, y):
        lr_base = LogisticRegression(class_weight='balanced', max_iter=10000)

        optim = GridSearchCV(
            lr_base, param_grid={'C': np.logspace(-4, 4, 9)}, cv=5, n_jobs=self.n_jobs, refit=True
        ).fit(X, y)

        posteriors = cross_val_predict(
            optim.best_estimator_, X, y, cv=5, n_jobs=self.n_jobs, method='predict_proba'
        )
        posteriors_generator = optim.best_estimator_.predict_proba

        return posteriors, posteriors_generator
    
    
    
    def ds_metric(self, predictions, test):
        k = int(len(predictions) * self.p_metric)
        
        test_posteriors = self.proba_generator(test)
        test_distribution = getHist(test_posteriors, 8)
        tr_distributions = [m[4] for m in self.ensemble]
        dist = [hellinger(tr_dist_i, test_distribution) for tr_dist_i in tr_distributions]
        order = np.argsort(dist)
        return _select_k(predictions, order, k)
    
    
    def ptr_metric(self, predictions:np.ndarray, metric:callable):
        """
        Selects the predictions made by models that have been trained on samples with a prevalence that is most similar
        to a first approximation of the test prevalence as made by all models in the ensemble.
        """
        k = int(len(predictions) * self.p_metric)

        test_prev_estim = np.mean(predictions, axis=0)
        tr_prevs = [m[3] for m in self.ensemble]
        ptr_differences = [metric(test_prev_estim, ptr_i) for ptr_i in tr_prevs]
        order = np.argsort(ptr_differences)
        return _select_k(predictions, order, k)
    
    


def _select_k(elements, order, k):
    print(elements)
    return [elements[idx] for idx in order[:k]]