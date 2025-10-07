import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from ..evaluation import measures
from ..base import Quantifier
from ..utils.method import getHist, hellinger
from ..utils.general import make_prevs, normalize_prevalence, parallel, generate_artificial_indexes

class Ensemble(Quantifier):
    """Ensemble of Quantification Models.
    
    This class implements an ensemble of quantification methods, 
    allowing parallel processing for evaluation. The ensemble 
    method is based on the articles by Pérez-Gállego et al. (2017, 2019).
    
    This approach of Ensemble is made of taking multiple
    samples varying class proportions on each, and for the 
    predictions, it takes the k models which as the minimum
    seletion metric
    
    Attributes
    ----------
    base_quantifier : Quantifier
        The base quantifier model to be used in the ensemble.
    size : int
        The number of samples to be generated for the ensemble.
    min_prop : float
        The minimum proportion of each class in the generated samples.
    selection_metric : str
        The metric used for selecting the best models in the ensemble.
        Valid options are 'all', 'ptr', and 'ds'.
        - all -> return all the predictions
        - ptr -> computes the selected error measure
        - ds -> computes the hellinger distance of the train and test
     distributions for each model
    p_metric : float
        The proportion of models to be selected based on the selection metric.
    return_type : str
        The type of aggregation to be used for the final prediction.
        Valid options are 'mean' and 'median'.
    max_sample_size : int or None
        The maximum size of the samples to be generated. If None, the entire dataset is used.
    max_trials : int
        The maximum number of trials to generate valid samples.
    n_jobs : int
        The number of parallel jobs to run.
    verbose : bool
        If True, prints progress messages during fitting and prediction.
        
    See Also
    --------
    joblib.Parallel : Parallel processing utility for Python.
    
    Parameters
    ----------
    quantifier : Quantifier
        The base quantifier model to be used in the ensemble.
    size : int, optional (default=50)
        The number of samples to be generated for the ensemble.
    min_prop : float, optional (default=0.1)
        The minimum proportion of each class in the generated samples.
    selection_metric : str, optional (default='all')
        The metric used for selecting the best models in the ensemble.
        Valid options are 'all', 'ptr', and 'ds'.
    p_metric : float, optional (default=0.25)
        The proportion of models to be selected based on the selection metric.
    return_type : str, optional (default='mean')
        The type of aggregation to be used for the final prediction.
        Valid options are 'mean' and 'median'.
    max_sample_size : int or None, optional (default=None)
        The maximum size of the samples to be generated. If None, the entire dataset is used.
    max_trials : int, optional (default=100)
        The maximum number of trials to generate valid samples.
    n_jobs : int, optional (default=1)
        The number of parallel jobs to run.
    verbose : bool, optional (default=False)
        If True, prints progress messages during fitting and prediction.
        
    References
    ----------
    .. [1] PÉREZ-GÁLLEGO, Pablo; QUEVEDO, José Ramón; DEL COZ, Juan José. Using ensembles for problems with characterizable changes in data distribution: A case study on quantification. Information Fusion, v. 34, p. 87-100, 2017. Avaliable at https://www.sciencedirect.com/science/article/abs/pii/S1566253516300628?casa_token=XblH-3kwhf4AAAAA:oxNRiCdHZQQa1C8BCJM5PBnFrd26p8-9SSBdm8Luf1Dm35w88w0NdpvoCf1RxBBqtshjyAhNpsDd
    .. [2] PÉREZ-GÁLLEGO, Pablo et al. Dynamic ensemble selection for quantification tasks. Information Fusion, v. 45, p. 1-15, 2019. Avaliable at https://www.sciencedirect.com/science/article/abs/pii/S1566253517303652?casa_token=jWmc592j5uMAAAAA:2YNeZGAGD0NJEMkcO-YBr7Ak-Ik7njLEcG8SKdowLdpbJ0mwPjYKKiqvQ-C3qICG8yU0m4xUZ3Yv        

    Examples
    --------
    >>> from mlquantify.methods import FM, Ensemble
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> 
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)
    >>> 
    >>> model = FM(RandomForestClassifier())
    >>> ensemble = Ensemble(quantifier=model,
    ...                     size=50,
    ...                     selection_metric='ptr',
    ...                     return_type='median',
    ...                     n_jobs=-1,
    ...                     verbose=False)
    >>> 
    >>> ensemble.fit(X_train, y_train)
    >>> 
    >>> predictions = ensemble.predict(X_test)
    >>> predictions
    {0: 0.4589857954621449, 1: 0.5410142045378551}
    >>> get_real_prev(y_test)
    {0: 0.45614035087719296, 1: 0.543859649122807}
    """
        
    SELECTION_METRICS = {'all', 'ptr', 'ds'}

    def __init__(self,
                 quantifier:Quantifier,
                 size:int=50,
                 min_prop:float=0.1,
                 selection_metric:str='all',
                 p_metric:float=0.25,
                 return_type:str="mean",
                 max_sample_size:int=None,
                 max_trials:int=100, 
                 n_jobs:int=1,
                 verbose:bool=False):
        
        assert selection_metric in Ensemble.SELECTION_METRICS, \
            f'unknown selection_metric={selection_metric}; valid are {Ensemble.SELECTION_METRICS}'
        assert max_sample_size is None or max_sample_size > 0, \
            'wrong value for max_sample_size; set it to a positive number or None'
            
        self.base_quantifier = quantifier
        self.size = size
        self.min_prop = min_prop
        self.p_metric = p_metric
        self.selection_metric = selection_metric
        self.return_type = return_type
        self.n_jobs = n_jobs
        self.proba_generator = None
        self.verbose = verbose
        self.max_sample_size = max_sample_size
        self.max_trials = max_trials

    def sout(self, msg):
        """Prints a message if verbose is True."""
        if self.verbose:
            print('[Ensemble]' + msg)

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
        
        self.classes = np.unique(y)
        
        if self.selection_metric == 'ds' and not self.binary_data:
            raise ValueError(f'ds selection_metric is only defined for binary quantification, but this dataset is not binary')
        # randomly chooses the prevalences for each member of the ensemble (preventing classes with less than
        # min_pos positive examples)
        sample_size = len(y) if self.max_sample_size is None else min(self.max_sample_size, len(y))
        prevs = [_draw_simplex(ndim=self.n_class, min_val=self.min_prop, max_trials=self.max_trials) for _ in range(self.size)]


        posteriors = None
        if self.selection_metric == 'ds':
            # precompute the training posterior probabilities
            posteriors, self.proba_generator = self.ds_get_posteriors(X, y)


        args = (
            (X, y, self.base_quantifier, prev, posteriors, self.verbose, sample_size)
            for prev in prevs
        )
        
        self.ensemble = parallel(
            _delayed_new_sample,
            tqdm(args, desc='fitting ensemble', total=self.size) if self.verbose else args,
            n_jobs=self.n_jobs)

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
        
        args = ((Qi, X) for Qi in self.ensemble)
        
        prevalences = np.asarray(
            parallel(_delayed_predict, 
                     tqdm(args, desc="Predicting Ensemble", total=len(self.ensemble)) if self.verbose else args, 
                     n_jobs=self.n_jobs)
        )

        prevalences = pd.DataFrame(prevalences).to_numpy()
        
        self.p_metric = int(len(prevalences) * self.p_metric)

        if self.selection_metric == 'ptr':
            prevalences = self.ptr_selection_metric(prevalences)
        elif self.selection_metric == 'ds':
            prevalences = self.ds_selection_metric(prevalences, X)
            
        
        if self.return_type == "median":   
            prevalences = np.median(prevalences, axis=0)
        else:      
            prevalences = np.mean(prevalences, axis=0)
            
        
        self.sout('Predict [Done]')
        return normalize_prevalence(prevalences, self.classes)


    def ptr_selection_metric(self, prevalences):
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
        tr_prevs = [m[1] for m in self.ensemble]
        ptr_differences = [measures.mean_squared_error(test_prev_estim, ptr_i) for ptr_i in tr_prevs]
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


    def ds_selection_metric(self, prevalences, test):
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
        test_posteriors = self.proba_generator(test)
        test_distribution = getHist(test_posteriors, 8)
        tr_distributions = [m[2] for m in self.ensemble]
        dist = [hellinger(tr_dist_i, test_distribution) for tr_dist_i in tr_distributions]
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
    


def _delayed_new_sample(args):
    """
    Fits a new sample for the ensemble, this method is used for parallelization purposes generating a new artificial sample for each quantifier.
    
    Parameters
    ----------
    args : tuple
        A tuple containing the following elements:
        
        X : array-like of shape (n_samples, n_features)
            The feature matrix representing the training data.
        y : array-like of shape (n_samples,)
            The target vector representing class labels for the training data.
        base_quantifier : Quantifier
            The base quantifier model to be used in the ensemble.
        prev : array-like of shape (n_classes,)
            The class prevalences for the new sample.
        posteriors : array-like of shape (n_samples, n_classes)
            The posterior probabilities for the training data obtained through cross-validation.
        verbose : bool
            If True, prints progress messages during fitting and prediction.
        sample_size : int
            The size of the sample to be generated.
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        model : Quantifier
            The fitted quantifier model.
        tr_prevalence : array-like of shape (n_classes,)
            The class prevalences for the new sample.
        tr_distribution : array-like of shape (n_classes,)
            The distribution of posterior probabilities for the new sample.
        X : array-like of shape (n_samples, n_features)
            The feature matrix representing the training data.
        y : array-like of shape (n_samples,)
            The target vector representing class labels for the training data.
    """
    
    
    X, y, base_quantifier, prev, posteriors, verbose, sample_size = args
    if verbose:
        print(f'\tfit-start for prev {str(np.round(prev, 3))}, sample_size={sample_size}')
    model = deepcopy(base_quantifier)

    sample_index = generate_artificial_indexes(y, prev, sample_size, np.unique(y))
    X_sample = np.take(X, sample_index, axis=0)
    y_sample = np.take(y, sample_index, axis=0)
    #print(X_sample)

    model.fit(X_sample, y_sample)

    tr_prevalence = prev
    tr_distribution = getHist(posteriors[sample_index], 8) if (posteriors is not None) else None
    if verbose:
        print(f'\t \\--fit-ended for prev {str(np.round(prev, 3))}')
    return (model, tr_prevalence, tr_distribution, X, y)


def _delayed_predict(args):
    """
    Predicts the class prevalences for the given test data.
    
    Parameters
    ----------
    args : tuple
        A tuple containing the following elements:
        
        quantifier : Quantifier
            The quantifier model to be used for prediction.
        X : array-like of shape (n_samples, n_features)
            The input data.
    
    Returns
    -------
    array-like of shape (n_samples, n_classes)
        The predicted class prevalences.
    """
    quantifier, X = args
    #print(np.asarray(list(quantifier[0].predict(X).values())))
    return list(quantifier[0].predict(X).values())


def _draw_simplex(ndim, min_val, max_trials=100):
    """
    Return a uniform sample from the ndim-dimensional simplex, ensuring all dimensions are >= min_val.

    Note:
        For min_val > 0, the sampling is not truly uniform because the simplex is restricted.

    Parameters:
        ndim (int): Number of dimensions of the simplex.
        min_val (float): Minimum allowed value for each dimension. Must be less than 1 / ndim.
        max_trials (int, optional): Maximum number of attempts to find a valid sample (default is 100).

    Returns:
        numpy.ndarray: A sample from the ndim-dimensional simplex where all dimensions are >= min_val.

    Raises:
        ValueError: If min_val >= 1 / ndim, or if a valid sample cannot be found within max_trials trials.
    """
    if min_val >= 1 / ndim:
        raise ValueError(f'no sample can be draw from the {ndim}-dimensional simplex so that '
                         f'all its values are >={min_val} (try with a larger value for min_pos)')
    trials = 0
    while True:
        u = make_prevs(ndim)
        if all(u >= min_val):
            return u
        trials += 1
        if trials >= max_trials:
            raise ValueError(f'it looks like finding a random simplex with all its dimensions being'
                             f'>= {min_val} is unlikely (it failed after {max_trials} trials)')
            