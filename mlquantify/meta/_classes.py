import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.utils import resample

from mlquantify.base import BaseQuantifier, MetaquantifierMixin
from mlquantify.metrics._slq import MSE
from mlquantify.mixture._classes import SORD, DyS
from mlquantify.mixture._utils import getHist, hellinger
from mlquantify.utils import Options, Interval, CallableConstraint
from mlquantify.utils import _fit_context
from mlquantify.confidence import (
    ConfidenceInterval,
    ConfidenceEllipseSimplex,
    ConfidenceEllipseCLR,
    construct_confidence_region
)
from mlquantify.base_aggregative import (
    _get_learner_function, 
    is_aggregative_quantifier,
    uses_soft_predictions, 
    get_aggregation_requirements)
from mlquantify.utils._sampling import (
    simplex_grid_sampling, 
    simplex_uniform_sampling, 
    simplex_uniform_kraemer,
    bootstrap_sample_indices
)
from mlquantify.model_selection import APP, NPP, UPP
from mlquantify.utils._validation import validate_data, validate_predictions, validate_prevalences
from mlquantify.utils.prevalence import get_prev_from_labels



def get_protocol_sampler(protocol_name, batch_size, n_prevalences, min_prev, max_prev, n_classes):
    """ Returns a prevalence sampler function based on the specified protocol name.
    
    Parameters
    ----------
    protocol_name : str
        The name of the protocol ('app', 'npp', 'upp', 'upp-k').
    batch_size : int
        The size of each batch.
    n_prevalences : int
        The number of prevalences to sample.
    min_prev : float
        The minimum prevalence value.
    max_prev : float
        The maximum prevalence value.
    n_classes : int
        The number of classes.
        
    Returns
    -------
    callable
        A function that generates prevalence samples according to the specified protocol.
    """
    
    if protocol_name == 'artificial':
        protocol = APP(batch_size=batch_size,
                           n_prevalences=n_prevalences,
                           min_prev=min_prev,
                           max_prev=max_prev)

    elif protocol_name == 'natural':
        protocol = NPP(batch_size=batch_size,
                           n_samples=n_prevalences)

    elif protocol_name == 'uniform':
            protocol = UPP(batch_size=batch_size,
                           n_prevalences=n_prevalences,
                           algorithm='uniform',
                           min_prev=min_prev,
                           max_prev=max_prev)
    elif protocol_name == 'kraemer':
        protocol = UPP(batch_size=batch_size,
                           n_prevalences=n_prevalences,
                           algorithm='kraemer',
                           min_prev=min_prev,
                           max_prev=max_prev)
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}")
    return protocol

class EnsembleQ(MetaquantifierMixin, BaseQuantifier):
        
    _parameter_constraints = {
        "quantifier": [BaseQuantifier],
        "size": [Interval(left=1, right=None, discrete=True)],
        "min_prop": [Interval(left=0.0, right=1.0, inclusive_left=True, inclusive_right=True)],
        "max_prop": [Interval(left=0.0, right=1.0, inclusive_left=True, inclusive_right=True)],
        "selection_metric": [Options(['all', 'ptr', 'ds'])],
        "p_metric": [Interval(left=0.0, right=1.0, inclusive_left=True, inclusive_right=True)],
        "protocol": [Options(['artificial', 'natural', 'uniform', 'kraemer'])],
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
                 protocol="uniform",
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
        
        protocol = get_protocol_sampler(
            batch_size=sample_size, 
            n_prevalences=self.size, 
            min_prev=self.min_prop, 
            max_prev=self.max_prop,
            n_classes=len(self.classes)
        )()

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





class AggregativeBootstrap(MetaquantifierMixin, BaseQuantifier):


    _parameter_constraints = {
        "quantifier": [BaseQuantifier],
        "n_train_bootstraps": [Interval(left=1, right=None, discrete=True)],
        "n_test_bootstraps": [Interval(left=1, right=None, discrete=True)],
        "random_state": [Options([None, int])],
        "region_type": [Options(['intervals', 'ellipse', 'ellipse-clr'])],
        "confidence_level": [Interval(left=0.0, right=1.0)],
    }

    def __init__(self, 
                 quantifier, 
                 n_train_bootstraps=1, 
                 n_test_bootstraps=1,
                 random_state=None,
                 region_type='intervals',
                 confidence_level=0.95):
        self.quantifier = quantifier
        self.n_train_bootstraps = n_train_bootstraps
        self.n_test_bootstraps = n_test_bootstraps
        self.random_state = random_state
        self.region_type = region_type
        self.confidence_level = confidence_level
        
    def fit(self, X, y, val_split=None):
        """ Fits the aggregative bootstrap model to the given training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        self : AggregativeBootstrap
            The fitted aggregative bootstrap model.
        """
        X, y = validate_data(self, X, y)
        self.classes = np.unique(y)
        
        if not is_aggregative_quantifier(self.quantifier):
            raise ValueError(f"The quantifier {self.quantifier.__class__.__name__} is not an aggregative quantifier.")
        
        learner_function = _get_learner_function(self.quantifier)
        model = self.quantifier.learner
        
        if val_split is None:
            model.fit(X, y)
            train_y_values = y
            train_predictions = getattr(model, learner_function)(X)
        else:
            X_fit, y_fit, X_val, y_val = train_test_split(X, y, test_size=val_split, random_state=self.random_state)
            model.fit(X_fit, y_fit)
            train_y_values = y_val
            train_predictions = getattr(model, learner_function)(X_val)
        
        self.train_predictions = train_predictions
        self.train_y_values = train_y_values
        
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
        X = validate_data(self, X, None)
        learner_function = _get_learner_function(self.quantifier)
        model = self.quantifier.learner
        
        predictions = getattr(model, learner_function)(X)

        return self.aggregate(predictions, self.train_predictions, self.train_y_values)


    def aggregate(self, predictions, train_predictions, train_y_values):
        """ Aggregates the predictions using bootstrap resampling.
        
        Parameters
        ----------
        predictions : array-like of shape (n_samples, n_classes)
            The input data.
        train_predictions : array-like of shape (n_samples, n_classes)
            The training predictions.
        train_y_values : array-like of shape (n_samples,)
            The training target values.
            
        Returns
        -------
        prevalences : array-like of shape (n_samples, n_classes)
            The predicted class prevalences.
        """
        prevalences = []
        
        self.classes = self.classes if hasattr(self, 'classes') else np.unique(train_y_values)
        
        for train_idx in bootstrap_sample_indices(
            n_samples=len(train_predictions),
            n_bootstraps=self.n_train_bootstraps,
            batch_size=len(train_predictions),
            random_state=self.random_state
        ):
            train_pred_boot = train_predictions[train_idx]
            train_y_boot = train_y_values[train_idx]
            
            for test_idx in bootstrap_sample_indices(
                n_samples=len(predictions),
                n_bootstraps=self.n_test_bootstraps,
                batch_size=len(predictions),
                random_state=self.random_state
            ):
                test_pred_boot = predictions[test_idx]

                requirements = get_aggregation_requirements(self.quantifier)
                
                if requirements.requires_train_proba and requirements.requires_train_labels:
                    prevalences_boot = self.quantifier.aggregate(test_pred_boot, train_pred_boot, train_y_boot)
                elif requirements.requires_train_labels:
                    prevalences_boot = self.quantifier.aggregate(test_pred_boot, train_y_boot)
                else:
                    prevalences_boot = self.quantifier.aggregate(test_pred_boot)

                prevalences_boot = np.asarray(list(prevalences_boot.values()))
                prevalences.append(prevalences_boot)

        prevalences = np.asarray(prevalences)
        confidence_region = construct_confidence_region(
            prev_estims=prevalences,
            method=self.region_type,
            confidence_level=self.confidence_level,
        )

        prevalence = confidence_region.get_point_estimate()
        
        prevalence = validate_prevalences(self, prevalence, self.classes)

        return prevalence




class QuaDapt(MetaquantifierMixin, BaseQuantifier):
    
    _parameter_constraints = {
        "quantifier": [BaseQuantifier],
        "merging_factor": "array-like",
        "measure": [Options(["hellinger", "topsoe", "probsymm", "sord"])],
        "random_state": [Options([None, int])],
    }
    
    def __init__(self, 
                 quantifier,
                 measure="topsoe", 
                 merging_factor=(0.1, 1.0, 0.2)):
        self.quantifier = quantifier
        self.measure = measure
        self.merging_factor = merging_factor
        
    
    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes = np.unique(y)
        
        self.quantifier.learner.fit(X, y)
        self.train_y_values = y
        
        return self
        
    def predict(self, X):

        X = validate_data(self, X, None)
        
        learner_function = _get_learner_function(self.quantifier)
        model = self.quantifier.learner
        
        predictions = getattr(model, learner_function)(X)

        return self.aggregate(predictions, self.train_y_values)
    
    
    def aggregate(self, predictions, train_y_values):

        pos_predictions = predictions[:, 1]
        m = self._get_best_merging_factor(pos_predictions)
        
        self.classes = self.classes if hasattr(self, 'classes') else np.unique(train_y_values)

        moss = QuaDapt.MoSS(1000, 0.5, m)

        moss_scores = moss[:, :2]
        moss_labels = moss[:, 2]

        prevalences = self.quantifier.aggregate(predictions,
                                                moss_scores,
                                                moss_labels)
        
        prevalences = {self.classes[i]: v for i, v in enumerate(prevalences.values())}
        return prevalences

        
    def _get_best_merging_factor(self, predictions):
        
        MF = np.atleast_1d(np.round(self.merging_factor, 2)).astype(float)
        
        distances = []
        
        for mf in MF:
            scores = QuaDapt.MoSS(1000, 0.5, mf)
            pos_scores = scores[scores[:, 2] == 1][:, :2]
            neg_scores = scores[scores[:, 2] == 0][:, :2]
            
            best_distance = self._get_best_distance(predictions, pos_scores, neg_scores)
            
            distances.append(best_distance)
        
        best_m = MF[np.argmin(distances)]
        return best_m
    
    def _get_best_distance(self, predictions, pos_scores, neg_scores):
        
        if self.measure in ["hellinger", "topsoe", "probsymm"]:
            method = DyS(measure=self.measure)
        elif self.measure == "sord":
            method = SORD()
        
        best_distance = method.get_best_distance(predictions, pos_scores, neg_scores)
        return best_distance
        

    @classmethod
    def MoSS(cls, n, alpha, m):
        p_score = np.random.uniform(size=int(n * alpha)) ** m
        n_score = 1 - (np.random.uniform(size=int(round(n * (1 - alpha), 0))) ** m)
        scores = np.column_stack(
            (np.concatenate((p_score, n_score)), 
             np.concatenate((p_score, n_score)), 
             np.concatenate((
                 np.ones(len(p_score)), 
                 np.full(len(n_score), 0))))
        )
        return scores
        