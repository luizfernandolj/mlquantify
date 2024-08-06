import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from quantifyML.evaluation import measures
from ...base import Quantifier
from ...utils import make_prevs, getHist, normalize_prevalence, parallel, hellinger, generate_indexes

class Ensemble(Quantifier):
    SELECTION_METRICS = {'all', 'ptr', 'ds'}

    """
    Methods from the articles:
    Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017).
    Using ensembles for problems with characterizable changes in data distribution: A case study on quantification.
    Information Fusion, 34, 87-100.
    and
    Pérez-Gállego, P., Castano, A., Quevedo, J. R., & del Coz, J. J. (2019). 
    Dynamic ensemble selection for quantification tasks. 
    Information Fusion, 45, 1-15.
    """

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
        if self.verbose:
            print('[Ensemble]' + msg)

    def fit(self, X, y):
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
            tqdm(args, desc='fitting ensamble', total=self.size) if self.verbose else args,
            n_jobs=self.n_jobs)

        self.sout('Fit [Done]')
        return self

    def predict(self, X):
        self.sout('Predict')
        
        
        prevalences = np.asarray(
            parallel(_delayed_predict, ((Qi, X) for Qi in self.ensemble), n_jobs=self.n_jobs)
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
        Selects the prevalences made by models that have been trained on samples with a prevalence that is most similar
        to a first approximation of the test prevalence as made by all models in the ensemble.
        """
        test_prev_estim = prevalences.mean(axis=0)
        tr_prevs = [m[1] for m in self.ensemble]
        ptr_differences = [measures.mean_squared_error(test_prev_estim, ptr_i) for ptr_i in tr_prevs]
        order = np.argsort(ptr_differences)
        return _select_k(prevalences, order, k=self.p_metric)

    def ds_get_posteriors(self, X, y):
        """
        In the original article, this procedure is not described in a sufficient level of detail. The paper only says
        that the distribution of posterior probabilities from training and test examples is compared by means of the
        Hellinger Distance. However, how these posterior probabilities are generated is not specified. In the article,
        a Logistic Regressor (LR) is used as the classifier device and that could be used for this purpose. However, in
        general, a Quantifier is not necessarily an instance of Aggreggative Probabilistic Quantifiers, and so, that the
        quantifier builds on top of a probabilistic classifier cannot be given for granted. Additionally, it would not
        be correct to generate the posterior probabilities for training documents that have concurred in training the
        classifier that generates them.
        This function thus generates the posterior probabilities for all training documents in a cross-validation way,
        using a LR with hyperparameters that have previously been optimized via grid search in 5FCV.
        :return P,f, where P is a ndarray containing the posterior probabilities of the training data, generated via
        cross-validation and using an optimized LR, and the function to be used in order to generate posterior
        probabilities for test X.
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
        test_posteriors = self.proba_generator(test)
        test_distribution = get_probability_distribution(test_posteriors)
        tr_distributions = [m[2] for m in self.ensemble]
        dist = [hellinger(tr_dist_i, test_distribution) for tr_dist_i in tr_distributions]
        order = np.argsort(dist)
        return _select_k(prevalences, order, k=self.p_metric)


def get_probability_distribution(posterior_probabilities, bins=8):
    assert posterior_probabilities.shape[1] == 2, 'the posterior probabilities do not seem to be for a binary problem'
    posterior_probabilities = posterior_probabilities[:, 1]  # take the positive posteriors only
    distribution, _ = np.histogram(posterior_probabilities, bins=bins, range=(0, 1), density=True)
    return distribution


def _select_k(elements, order, k):
    elements_k = [elements[idx] for idx in order[:k]]
    if elements_k:
        return elements_k
    print(f"Unable to take {k} for elements with size {len(elements)}")
    return elements
    


def _delayed_new_sample(args):
    X, y, base_quantifier, prev, posteriors, verbose, sample_size = args
    if verbose:
        print(f'\tfit-start for prev {str(prev)}, sample_size={sample_size}')
    model = deepcopy(base_quantifier)

    sample_index = generate_indexes(y, prev, sample_size, np.unique(y))
    X_sample = X[sample_index]
    y_sample = y[sample_index]

    model.fit(X_sample, y_sample)

    tr_prevalence = prev
    tr_distribution = get_probability_distribution(posteriors[sample_index]) if (posteriors is not None) else None
    if verbose:
        print(f'\t\--fit-ended for prev {str(prev)}')
    return (model, tr_prevalence, tr_distribution, X, y)


def _delayed_predict(args):
    quantifier, X = args
    #print(np.asarray(list(quantifier[0].predict(X).values())))
    return list(quantifier[0].predict(X).values())


def _draw_simplex(ndim, min_val, max_trials=100):
    """
    returns a uniform sampling from the ndim-dimensional simplex but guarantees that all dimensions
    are >= min_class_prev (for min_val>0, this makes the sampling not truly uniform)
    :param ndim: number of dimensions of the simplex
    :param min_val: minimum class prevalence allowed. If less than 1/ndim a ValueError will be throw since
    there is no possible solution.
    :return: a sample from the ndim-dimensional simplex that is uniform in S(ndim)-R where S(ndim) is the simplex
    and R is the simplex subset containing dimensions lower than min_val
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
            