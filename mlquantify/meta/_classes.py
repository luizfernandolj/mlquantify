import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict

from mlquantify.base import BaseQuantifier, MetaquantifierMixin
from mlquantify.metrics._slq import MSE
from mlquantify.mixture._utils import getHist, hellinger
from mlquantify.utils import Options, Interval, CallableConstraint
from mlquantify.utils import _fit_context
from mlquantify.utils._sampling import (
    simplex_grid_sampling, 
    simplex_uniform_sampling, 
    simplex_uniform_kraemer
)
from mlquantify.model_selection import APP, NPP, UPP
from mlquantify.utils._validation import validate_data, validate_prevalences
from mlquantify.utils.prevalence import get_prev_from_labels

class EnsembleQ(MetaquantifierMixin, BaseQuantifier):
        
    _parameter_constraints = {
        "quantifier": [BaseQuantifier],
        "size": [Interval(left=1, right=None, discrete=True)],
        "min_prop": [Interval(left=0.0, right=1.0, inclusive_left=True, inclusive_right=True)],
        "max_prop": [Interval(left=0.0, right=1.0, inclusive_left=True, inclusive_right=True)],
        "selection_metric": [Options(['all', 'ptr', 'ds'])],
        "p_metric": [Interval(left=0.0, right=1.0, inclusive_left=True, inclusive_right=True)],
        "protocol": [Options(['app', 'npp', 'upp', 'upp-k'])],
        "return_type": [Options(['mean', 'median'])],
        "max_sample_size": [Options([Interval(left=1, right=None, discrete=True), None])],
        "max_trials": [Interval(left=1, right=None, discrete=True)],
        "n_jobs": [Interval(left=1, right=None, discrete=True)],
        "verbose": [bool],
    }

    def __init__(self,
                 quantifier,
                 size=50,
                 min_prop=0.1,
                 max_prop=1,
                 selection_metric='all',
                 protocol="upp",
                 p_metric=0.25,
                 return_type="mean",
                 max_sample_size=None,
                 max_trials=100,
                 n_jobs=1,
                 verbose=False):
        
        self.quantifier = quantifier
        self.size = size
        self.min_prop = min_prop
        self.max_prop = max_prop
        self.p_metric = p_metric
        self.protocol = protocol
        self.selection_metric = selection_metric
        self.return_type = return_type
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_sample_size = max_sample_size
        self.max_trials = max_trials

    def sout(self, msg):
        """Prints a message if verbose is True."""
        if self.verbose:
            print('[Ensemble]' + msg)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """ Fits the ensemble model to the given training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : Ensemble
            The fitted ensemble model.
        """
        self.sout('Fit')
        
        self.models = []
        self.train_prevalences = []
        self.train_distributions = []
        self.posteriors_generator = []
        
        self.classes = np.unique(y)
        X, y = validate_data(self, X, y)
        
        if self.selection_metric == 'ds' and not len(self.classes) == 2:
            raise ValueError(f'ds selection_metric is only defined for binary quantification, but this dataset is not binary')
        # randomly chooses the prevalences for each member of the ensemble (preventing classes with less than
        # min_pos positive examples)
        sample_size = len(y) if self.max_sample_size is None else min(self.max_sample_size, len(y))
        
        if self.protocol == 'app':
            protocol = APP(batch_size=sample_size,
                           n_prevalences=self.size,
                           min_prev=self.min_prop,
                           max_prev=self.max_prop)

        elif self.protocol == 'npp':
            protocol = NPP(batch_size=sample_size,
                           n_samples=self.size)

        elif self.protocol == 'upp':
            protocol = UPP(batch_size=sample_size,
                           n_prevalences=self.size,
                           algorithm='uniform',
                           min_prev=self.min_prop,
                           max_prev=self.max_prop)
        elif self.protocol == 'upp-k':
            protocol = UPP(batch_size=sample_size,
                           n_prevalences=self.size,
                           algorithm='kraemer',
                           min_prev=self.min_prop,
                           max_prev=self.max_prop)

        posteriors = None
        if self.selection_metric == 'ds':
            # precompute the training posterior probabilities
            posteriors, self.posteriors_generator = self.ds_get_posteriors(X, y)
            
        for idx in protocol.split(X, y):
            X_batch, y_batch = X[idx], y[idx]
            model = deepcopy(self.quantifier)
            
            model.fit(X_batch, y_batch)
            tr_prev = get_prev_from_labels(y_batch)
            
            if self.selection_metric == 'ds':
                self.train_distributions.append(getHist(posteriors[idx], 8))
            
            self.train_prevalences.append(tr_prev)
            self.models.append(model)

        self.sout('Fit [Done]')
        return self

    def predict(self, X):
        """ Predicts the class prevalences for the given test data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        
        Returns
        -------
        prevalences : array-like of shape (n_samples, n_classes)
            The predicted class prevalences.
        """
        self.sout('Predict')
        
        test_prevalences = []
        
        for model in tqdm(self.models, disable=not self.verbose):
            pred = np.asarray(list(model.predict(X).values()))
            test_prevalences.append(pred)
        
        test_prevalences = np.asarray(test_prevalences)
        self.p_metric = int(len(test_prevalences) * self.p_metric)

        if self.selection_metric == 'ptr':
            test_prevalences = self.ptr_selection_metric(test_prevalences, self.train_prevalences)
        elif self.selection_metric == 'ds':
            test_prevalences = self.ds_selection_metric(X, 
                                                   test_prevalences, 
                                                   self.train_distributions,
                                                   self.posteriors_generator)

        if self.return_type == "median":
            prevalences = np.median(test_prevalences, axis=0)
        else:      
            prevalences = np.mean(test_prevalences, axis=0)
            
        
        self.sout('Predict [Done]')
        prevalences = validate_prevalences(self, prevalences, self.classes)
        return prevalences


    def ptr_selection_metric(self, prevalences, train_prevalences):
        """
        Selects the prevalence estimates from models trained on samples whose prevalence is most similar
        to an initial approximation of the test prevalence as estimated by all models in the ensemble.

        Parameters
        ----------
        prevalences : numpy.ndarray
            An array of prevalence estimates provided by each model in the ensemble.

        Returns
        -------
        numpy.ndarray
            The selected prevalence estimates after applying the PTR selection metric.
        """
        test_prev_estim = prevalences.mean(axis=0)
        ptr_differences = [MSE(test_prev_estim, ptr_i) for ptr_i in train_prevalences]
        order = np.argsort(ptr_differences)
        return _select_k(prevalences, order, k=self.p_metric)

    def ds_get_posteriors(self, X, y):
        """ 
        Generate posterior probabilities using cross-validated logistic regression.
        This method computes posterior probabilities for the training data via cross-validation,
        using a logistic regression classifier with hyperparameters optimized through grid search.
        It also returns a function to generate posterior probabilities for new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix representing the training data.
        y : array-like of shape (n_samples,)
            The target vector representing class labels for the training data.
            
        Returns
        -------
        posteriors : ndarray of shape (n_samples, n_classes)
            Posterior probabilities for the training data obtained through cross-validation.
        posteriors_generator : callable
            A function that computes posterior probabilities for new input data.
            
        Notes
        -----
        - In scenarios where the quantifier is not based on a probabilistic classifier, it's necessary
            to train a separate probabilistic model to obtain posterior probabilities.
        - Using cross-validation ensures that the posterior probabilities for the training data are unbiased,
            as each data point is evaluated by a model not trained on that point.
        - Hyperparameters for the logistic regression classifier are optimized using a grid search with
            cross-validation to improve the model's performance.
        """
        lr_base = LogisticRegression(class_weight='balanced', max_iter=1000)

        optim = GridSearchCV(
            lr_base, param_grid={'C': np.logspace(-4, 4, 9)}, cv=5, n_jobs=self.n_jobs, refit=True
        ).fit(X, y)

        posteriors = cross_val_predict(
            optim.best_estimator_, X, y, cv=5, n_jobs=self.n_jobs, method='predict_proba'
        )
        posteriors_generator = optim.best_estimator_.predict_proba

        return posteriors, posteriors_generator


    def ds_selection_metric(self, X, prevalences, train_distributions, posteriors_generator):
        """
        Selects the prevalence estimates from models trained on samples whose distribution of posterior
        probabilities is most similar to the distribution of posterior probabilities for the test data.
        
        Parameters
        ----------
        prevalences : numpy.ndarray
            An array of prevalence estimates provided by each model in the ensemble.
        test : array-like of shape (n_samples, n_features)
            The feature matrix representing the test data.
        
        Returns
        -------
        numpy.ndarray
            The selected prevalence estimates after applying the DS selection metric.
        """
        test_posteriors = posteriors_generator(X)
        test_distribution = getHist(test_posteriors, 8)
        dist = [hellinger(tr_dist_i, test_distribution) for tr_dist_i in train_distributions]
        order = np.argsort(dist)
        return _select_k(prevalences, order, k=self.p_metric)

def _select_k(elements, order, k):
    """
    Selects the k elements from the list of elements based on the order.
    If the list is empty, it returns the original list.
    
    Parameters
    ----------
    elements : array-like
        The array of elements to be selected from.
    order : array-like
        The order of the elements.
    k : int
        The number of elements to be selected.
    
    Returns
    -------
    array-like
        The selected elements.
    """
    elements_k = [elements[idx] for idx in order[:k]]
    if elements_k:
        return elements_k
    print(f"Unable to take {k} for elements with size {len(elements)}")
    return elements


class QuaDapt(MetaquantifierMixin, BaseQuantifier):
    """Placeholder for QuaDapt class."""
    pass