from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator

from ..base import AggregativeQuantifier

from ..utils.general import get_real_prev
from ..utils.method import *
import mlquantify as mq




class MixtureModel(AggregativeQuantifier):
    """Mixtures of Score Distributions
    
    MixtureModel is a generic class for methods based on mixture models. 
    The main idea is that the cumulative distribution of scores assigned 
    to data points in the test set is a mixture of the score distributions 
    from the training set (positive and negative classes).

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier that supports `predict_proba`.

    Attributes
    ----------
    learner : BaseEstimator
        A scikit-learn compatible classifier that provides predictive probabilities.
    pos_scores : np.ndarray
        Score distribution for the positive class in the training data.
    neg_scores : np.ndarray
        Score distribution for the negative class in the training data.

    Notes
    -----
    All methods that inherits from MixtureModel will be binary quantifiers. In case of multiclass problems will be made One vs All.
    
    Examples
    --------
    >>> from mlquantify.methods.mixture_models import MixtureModel
    >>> from mlquantify.utils.general import get_real_prev
    >>> from mlquantify.utils.method import getHist
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>> 
    >>> class MyMixtureModel(MixtureModel):
    ...     def __init__(self, learner, param):
    ...         super().__init__(learner)
    ...         self.param = param
    ...     def _compute_prevalence(self, test_scores: np.ndarray) -> float:
    ...         hist_pos = getHist(self.pos_scores, self.param)
    ...         hist_neg = getHist(self.neg_scores, self.param)
    ...         hist_test = getHist(test_scores, self.param)
    ...         mixture = hist_test * (hist_pos + hist_neg)
    ...         return np.sum(mixture)
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>> 
    >>> mm = MyMixtureModel(RandomForestClassifier(), 10)
    >>> mm.fit(X_train, y_train)
    >>> prevalence = mm.predict(X_test)
    >>> prevalence
    {0: 0.3622419419517543, 1: 0.6377580580482457}
    >>> get_real_prev(y_test)
    {0: 0.37719298245614036, 1: 0.6228070175438597}
    """

    def __init__(self, learner: BaseEstimator=None):
        self.learner = learner
        self.pos_scores = None
        self.neg_scores = None

    @property
    def is_multiclass(self) -> bool:
        """
        Indicates whether the model supports multiclass classification.

        Returns
        -------
        bool
            Always returns False, as MixtureModel supports only binary classification.
        """
        return False
    
    @property
    def is_probabilistic(self) -> bool:
        return True

    def _fit_method(self, X, y):
        """
        Fits the positive and negative score distributions using cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix.
        y : np.ndarray
            Training labels.

        Returns
        -------
        self : MixtureModel
            The fitted MixtureModel instance.
        """
        if mq.arguments["y_labels"] is not None and mq.arguments["posteriors_train"] is not None:
            y_labels = mq.arguments["y_labels"]
            probabilities = mq.arguments["posteriors_train"]
        else:
            y_labels, probabilities = get_scores(X, y, self.learner, self.cv_folds, self.learner_fitted)

        # Separate positive and negative scores based on labels
        self.pos_scores = probabilities[y_labels == self.classes[1]][:, 1]
        self.neg_scores = probabilities[y_labels == self.classes[0]][:, 1]

        return self

    def _predict_method(self, X) -> dict:
        """
        Predicts class prevalences for the test data.

        Parameters
        ----------
        X : np.ndarray
            Test feature matrix.

        Returns
        -------
        np.ndarray
            An array containing the prevalence for each class.
        """
        # Get the predicted probabilities for the positive class
        test_scores = self.predict_learner(X)[:, 1]

        # Compute the prevalence using the mixture model
        prevalence = np.clip(self._compute_prevalence(test_scores), 0, 1)

        # Return the prevalence as a distribution over the classes
        return np.asarray([1 - prevalence, prevalence])

    @abstractmethod
    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        """
        Abstract method to compute prevalence using the test scores.
        Subclasses must implement this method.

        Parameters
        ----------
        test_scores : np.ndarray
            Probabilities for the positive class in the test set.

        Returns
        -------
        float
            The computed prevalence for the positive class.
        """
        pass

    def get_distance(self, dist_train, dist_test, measure: str) -> float:
        """
        Computes the distance between training and test distributions using a specified metric.

        Parameters
        ----------
        dist_train : np.ndarray
            Distribution of scores for the training data.
        dist_test : np.ndarray
            Distribution of scores for the test data.
        measure : str
            The metric to use for distance calculation. Supported values are 
            'topsoe', 'probsymm', 'hellinger', and 'euclidean'.

        Returns
        -------
        float
            The computed distance between the two distributions.

        Raises
        ------
        ValueError
            If the input distributions have mismatched sizes or are zero vectors.
        """
        # Validate input distributions
        if np.sum(dist_train) < 1e-20 or np.sum(dist_test) < 1e-20:
            raise ValueError("One or both vectors are zero (empty)...")
        if len(dist_train) != len(dist_test):
            raise ValueError("Arrays need to be of equal size...")

        # Avoid division by zero by replacing small values
        dist_train = np.maximum(dist_train, 1e-20)
        dist_test = np.maximum(dist_test, 1e-20)

        # Compute the distance based on the selected metric
        if measure == 'topsoe':
            return topsoe(dist_train, dist_test)
        elif measure == 'probsymm':
            return probsymm(dist_train, dist_test)
        elif measure == 'hellinger':
            return hellinger(dist_train, dist_test)
        elif measure == 'euclidean':
            return sqEuclidean(dist_train, dist_test)
        else:
            return 100  # Default value for unknown metrics

        



class DyS(MixtureModel):
    """
    Distribution y-Similarity (DyS) framework.

    DyS is a method that generalizes the HDy approach by 
    considering the dissimilarity function DS as a parameter 
    of the model.
    
    Parameters
    ----------
    learner : BaseEstimator
        A probabilistic classifier implementing the `predict_proba` method.
    measure : str, optional
        The metric used to compare distributions. Options are:
        - "hellinger"
        - "topsoe"
        - "probsymm"
        Default is "topsoe".
    bins_size : np.ndarray, optional
        Array of bin sizes for histogram computation. 
        Default is np.append(np.linspace(2, 20, 10), 30).
        
    Attributes
    ----------
    bins_size : np.ndarray
        Bin sizes used for histogram calculations.
    measure : str
        Selected distance metric.
    prevs : np.ndarray
        Array of prevalences that minimize the distances.
    
    References
    ----------
    VAN HASSELT, H.; GUEZ, A.; SILVER, D. Proceedings of the AAAI conference on artificial intelligence. 2016. Avaliable at https://ojs.aaai.org/index.php/AAAI/article/view/4376 
    
    Examples
    --------
    >>> from mlquantify.methods.mixture_models import DyS
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> dys = DyS(RandomForestClassifier())
    >>> dys.fit(X_train, y_train)
    >>> prevalence = dys.predict(X_test)
    >>> prevalence
    {0: 0.3736714619191387, 1: 0.6263285380808613}
    >>> get_real_prev(y_test)
    {0: 0.37719298245614036, 1: 0.6228070175438597}
    """

    def __init__(self, learner: BaseEstimator=None, measure: str = "topsoe", bins_size: np.ndarray = None):
        assert measure in ["hellinger", "topsoe", "probsymm"], "Invalid measure."
        super().__init__(learner)

        # Set up bins_size
        if bins_size is None:
            bins_size = np.append(np.linspace(2, 20, 10), 30)
        if isinstance(bins_size, list):
            bins_size = np.asarray(bins_size)

        self.bins_size = bins_size
        self.measure = measure
        self.prevs = None  # Array of prevalences that minimizes the distances

    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        """
        Compute the prevalence estimate based on the test scores.

        Parameters
        ----------
        test_scores : np.ndarray
            Array of predicted probabilities for the test data.

        Returns
        -------
        prevalence : float
            Estimated prevalence.
        """
        prevs = self.GetMinDistancesDyS(test_scores)
        # Use the median of the prevalences as the final estimate
        prevalence = np.median(prevs)

        return prevalence

    def best_distance(self, X_test: np.ndarray) -> float:
        """
        Calculate the minimum distance between test scores and train distributions.

        Parameters
        ----------
        X_test : np.ndarray
            Test data to evaluate.

        Returns
        -------
        distance : float
            The minimum distance value.
        """
        test_scores = self.predict_learner(X_test)
        prevs = self.GetMinDistancesDyS(test_scores)

        size = len(prevs)
        best_prev = np.median(prevs)

        if size % 2 != 0:  # Odd
            index = np.argmax(prevs == best_prev)
            bin_size = self.bins_size[index]
        else:  # Even
            # Sort the prevalences
            ordered_prevs = np.sort(prevs)
            # Get the two middle indices
            middle1 = np.floor(size / 2).astype(int)
            middle2 = np.ceil(size / 2).astype(int)
            # Find the values corresponding to the median positions
            median1 = ordered_prevs[middle1]
            median2 = ordered_prevs[middle2]
            # Find the indices of these medians
            index1 = np.argmax(prevs == median1)
            index2 = np.argmax(prevs == median2)
            # Compute the average bin size
            bin_size = np.mean([self.bins_size[index1], self.bins_size[index2]])

        # Compute histogram densities
        pos_bin_density = getHist(self.pos_scores, bin_size)
        neg_bin_density = getHist(self.neg_scores, bin_size)
        test_bin_density = getHist(test_scores, bin_size)

        # Combine densities
        train_combined_density = (pos_bin_density * best_prev) + (neg_bin_density * (1 - best_prev))

        # Compute the distance
        distance = self.get_distance(train_combined_density, test_bin_density, measure=self.measure)

        return distance

    def GetMinDistancesDyS(self, test_scores: np.ndarray) -> list:
        """
        Compute prevalence by evaluating the distance metric across bin sizes.

        Parameters
        ----------
        test_scores : np.ndarray
            Array of predicted probabilities for the test data.

        Returns
        -------
        prevs : list
            List of prevalence estimates minimizing the distance for each bin size.
        """
        prevs = []

        # Iterate over each bin size
        for bins in self.bins_size:
            # Compute histogram densities
            pos_bin_density = getHist(self.pos_scores, bins)
            neg_bin_density = getHist(self.neg_scores, bins)
            test_bin_density = getHist(test_scores, bins)

            # Define the function to minimize
            def f(x):
                # Combine densities
                train_combined_density = (pos_bin_density * x) + (neg_bin_density * (1 - x))
                # Compute the distance
                return self.get_distance(train_combined_density, test_bin_density, measure=self.measure)

            # Use ternary search to minimize the distance
            prevs.append(ternary_search(0, 1, f))

        return prevs










class DySsyn(MixtureModel):
    """Synthetic Distribution y-Similarity (DySsyn).

    This method works similarly to the DyS method, but instead of using the 
    train scores, it generates them via MoSS (Model for Synthetic Scores). 
    MoSS creates a spectrum of score distributions ranging from highly separated 
    to fully mixed scores.

    Parameters
    ----------
    learner : BaseEstimator
        A probabilistic classifier implementing the `predict_proba` method.
    measure : str, optional
        The metric used to compare distributions. Options are:
        - "hellinger"
        - "topsoe"
        - "probsymm"
        Default is "topsoe".
    merge_factor : np.ndarray, optional
        Array controlling the mixing level of synthetic distributions.
        Default is np.linspace(0.1, 0.4, 10).
    bins_size : np.ndarray, optional
        Array of bin sizes for histogram computation. 
        Default is np.append(np.linspace(2, 20, 10), 30).
    alpha_train : float, optional
        Initial estimate of the training prevalence. Default is 0.5.
    n : int, optional
        Number of synthetic samples generated. Default is None.
    
    Attributes
    ----------
    bins_size : np.ndarray
        Bin sizes used for histogram calculations.
    merge_factor : np.ndarray
        Mixing factors for generating synthetic score distributions.
    alpha_train : float
        True training prevalence.
    n : int
        Number of samples generated during synthetic distribution creation.
    measure : str
        Selected distance metric.
    m : None or float
        Best mixing factor determined during computation.
    
    References
    ----------
    MALETZKE, André et al. Accurately quantifying under score variability. In: 2021 IEEE International Conference on Data Mining (ICDM). IEEE, 2021. p. 1228-1233. Avaliable at https://ieeexplore.ieee.org/abstract/document/9679104
    
    Examples
    --------
    >>> from mlquantify.methods.mixture_models import DySsyn
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> dyssyn = DySsyn(RandomForestClassifier())
    >>> dyssyn.fit(X_train, y_train)
    >>> prevalence = dyssyn.predict(X_test)
    >>> prevalence
    {0: 0.3606413872681201, 1: 0.6393586127318799}
    >>> get_real_prev(y_test)
    {0: 0.37719298245614036, 1: 0.6228070175438597}
    """

    
    def __init__(self, learner:BaseEstimator=None, measure:str="topsoe", merge_factor:np.ndarray=None, bins_size:np.ndarray=None, alpha_train:float=0.5, n:int=None):
        assert measure in ["hellinger", "topsoe", "probsymm"], "measure not valid"
        super().__init__(learner)
        
        # Set up bins_size
        if not bins_size:
            bins_size = np.append(np.linspace(2,20,10), 30)
        if isinstance(bins_size, list):
            bins_size = np.asarray(bins_size)
            
        if not merge_factor:
            merge_factor = np.linspace(0.1, 0.4, 10)
            
        self.bins_size = bins_size
        self.merge_factor = merge_factor
        self.alpha_train = alpha_train
        self.n = n
        self.measure = measure
        self.m = None
    
    
    
    def _fit_method(self, X, y):
        """
        Fits the learner and calculates the training prevalence.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : DySsyn
            The fitted DySsyn instance.
        """
        self.fit_learner(X, y)

        self.alpha_train = list(get_real_prev(y).values())[1]

        return self

    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        """
        Computes the prevalence estimate using test scores.

        Parameters
        ----------
        test_scores : np.ndarray
            Array of predicted probabilities for the test data.

        Returns
        -------
        prevalence : float
            Estimated prevalence based on the minimum distance 
            across synthetic distributions.
        """
        distances = self.GetMinDistancesDySsyn(test_scores)

        # Use the median of the prevalence estimates as the final prevalence
        index = min(distances, key=lambda d: distances[d][0])
        prevalence = distances[index][1]

        return prevalence

    def best_distance(self, X_test):
        """
        Computes the minimum distance between test scores and synthetic distributions of MoSS.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        distance : float
            Minimum distance value for the test data.
        """
        test_scores = self.predict_learner(X_test)

        distances = self.GetMinDistancesDySsyn(test_scores)

        index = min(distances, key=lambda d: distances[d][0])

        distance = distances[index][0]

        return distance

    def GetMinDistancesDySsyn(self, test_scores: np.ndarray) -> list:
        """
        Calculates the minimum distances between test scores and synthetic distributions of MoSS
        across various bin sizes and merge factors.

        Parameters
        ----------
        test_scores : np.ndarray
            Array of predicted probabilities for the test data.

        Returns
        -------
        values : dict
            Dictionary mapping each merge factor (m) to a tuple containing:
            - The minimum distance value.
            - The corresponding prevalence estimate.
        """
        if self.n is None:
            self.n = len(test_scores)

        values = {}

        # Iterate over each merge factor
        for m in self.merge_factor:
            pos_scores, neg_scores = MoSS(self.n, self.alpha_train, m)
            prevs = []
            for bins in self.bins_size:
                # Compute histogram densities for positive, negative, and test scores
                pos_bin_density = getHist(pos_scores, bins)
                neg_bin_density = getHist(neg_scores, bins)
                test_bin_density = getHist(test_scores, bins)

                # Define the function to minimize
                def f(x):
                    # Combine densities using a mixture of positive and negative densities
                    train_combined_density = (pos_bin_density * x) + (neg_bin_density * (1 - x))
                    # Calculate the distance between combined density and test density
                    return self.get_distance(train_combined_density, test_bin_density, measure=self.measure)

                # Use ternary search to find the best x that minimizes the distance
                prevs.append(ternary_search(0, 1, f))

            size = len(prevs)
            best_prev = np.median(prevs)

            if size % 2 != 0:  # ODD
                index = np.argmax(prevs == best_prev)
                bin_size = self.bins_size[index]
            else:  # EVEN
                # Sort the values in self.prevs
                ordered_prevs = np.sort(prevs)

                # Find the two middle indices
                middle1 = np.floor(size / 2).astype(int)
                middle2 = np.ceil(size / 2).astype(int)

                # Get the values corresponding to the median positions
                median1 = ordered_prevs[middle1]
                median2 = ordered_prevs[middle2]

                # Find the indices of median1 and median2 in prevs
                index1 = np.argmax(prevs == median1)
                index2 = np.argmax(prevs == median2)

                # Calculate the average of the corresponding bin sizes
                bin_size = np.mean([self.bins_size[index1], self.bins_size[index2]])

            pos_bin_density = getHist(pos_scores, bin_size)
            neg_bin_density = getHist(neg_scores, bin_size)
            test_bin_density = getHist(test_scores, bin_size)

            train_combined_density = (pos_bin_density * best_prev) + (neg_bin_density * (1 - best_prev))

            distance = self.get_distance(train_combined_density, test_bin_density, measure=self.measure)

            values[m] = (distance, best_prev)

        return values

    
    
    





class HDy(MixtureModel):
    """
    Hellinger Distance Minimization (HDy) framework.

    HDy is based on computing the Hellinger distance between two distributions: 
    the test distribution and the mixture of the positive and negative 
    distributions from the training data.
    
    Parameters
    ----------
    learner : BaseEstimator
        A supervised learning model implementing a `predict_proba` method.
        
    Attributes
    ----------
    pos_scores : np.ndarray
        Score distribution for the positive class in the training data.
    neg_scores : np.ndarray
        Score distribution for the negative class in the training data.
    
    References
    ----------
    GONZÁLEZ-CASTRO, Víctor; ALAIZ-RODRÍGUEZ, Rocío; ALEGRE, Enrique. Class distribution estimation based on the Hellinger distance. Information Sciences, v. 218, p. 146-164, 2013. Avaliable at https://www.sciencedirect.com/science/article/abs/pii/S0020025512004069?casa_token=W6UksOigmp4AAAAA:ap8FK5mtpAzG-s8k2ygfRVgdIBYDGWjEi70ueJ546coP9F-VNaCKE5W_gsAv0bWQiwzt2QoAuLjP
    
    Examples
    --------
    >>> from mlquantify.methods.mixture_models import HDy
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> hdy = HDy(RandomForestClassifier())
    >>> hdy.fit(X_train, y_train)
    >>> prevalence = hdy.predict(X_test)
    >>> prevalence
    {0: 0.33999999999999997, 1: 0.66}
    >>> get_real_prev(y_test)
    {0: 0.37719298245614036, 1: 0.6228070175438597}
    """

    def __init__(self, learner: BaseEstimator=None):
        super().__init__(learner)

    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        """
        Compute the prevalence estimate based on test scores.

        Parameters
        ----------
        test_scores : np.ndarray
            Array of predicted probabilities for the test data.

        Returns
        -------
        prevalence : float
            Estimated prevalence.
        """
        best_alphas, _ = self.GetMinDistancesHDy(test_scores)
        # Use the median of the best alpha values as the final prevalence estimate
        prevalence = np.median(best_alphas)

        return prevalence

    def best_distance(self, X_test: np.ndarray) -> float:
        """
        Calculate the minimum Hellinger distance for the test data.

        Parameters
        ----------
        X_test : np.ndarray
            Test data to evaluate.

        Returns
        -------
        distance : float
            The minimum distance value.
        """
        test_scores = self.predict_learner(X_test)
        _, distances = self.GetMinDistancesHDy(test_scores)

        size = len(distances)

        if size % 2 != 0:  # Odd
            index = size // 2
            distance = distances[index]
        else:  # Even
            # Find the two middle indices
            middle1 = np.floor(size / 2).astype(int)
            middle2 = np.ceil(size / 2).astype(int)
            # Compute the average of the corresponding distances
            distance = np.mean([distances[middle1], distances[middle2]])

        return distance

    def GetMinDistancesHDy(self, test_scores: np.ndarray) -> tuple:
        """
        Compute prevalence by minimizing the Hellinger distance across bins and alphas.

        Parameters
        ----------
        test_scores : np.ndarray
            Array of predicted probabilities for the test data.

        Returns
        -------
        best_alphas : list
            List of alpha values that minimize the Hellinger distance for each bin size.
        distances : list
            List of minimum distances corresponding to the best alphas for each bin size.
        """
        # Define bin sizes and alpha values
        bins_size = np.arange(10, 110, 11)  # Bins from 10 to 110 with a step size of 10
        alpha_values = np.round(np.linspace(0, 1, 101), 2)  # Alpha values from 0 to 1, rounded to 2 decimal places

        best_alphas = []
        distances = []

        for bins in bins_size:
            # Compute histogram densities for positive, negative, and test scores
            pos_bin_density = getHist(self.pos_scores, bins)
            neg_bin_density = getHist(self.neg_scores, bins)
            test_bin_density = getHist(test_scores, bins)

            bin_distances = []

            # Evaluate distance for each alpha value
            for x in alpha_values:
                # Combine densities using a mixture of positive and negative densities
                train_combined_density = (pos_bin_density * x) + (neg_bin_density * (1 - x))
                # Compute the distance using the Hellinger measure
                bin_distances.append(self.get_distance(train_combined_density, test_bin_density, measure="hellinger"))

            # Find the alpha value that minimizes the distance
            best_alpha = alpha_values[np.argmin(bin_distances)]
            min_distance = min(bin_distances)

            best_alphas.append(best_alpha)
            distances.append(min_distance)

        return best_alphas, distances

    
    
    
    
    
    

class SMM(MixtureModel):
    """
    Sample Mean Matching (SMM).

    A member of the DyS framework that estimates the prevalence 
    of the positive class in a test dataset by leveraging simple 
    mean values to represent the score distributions for positive, 
    negative, and unlabeled data.
    
    Parameters
    ----------
    learner : BaseEstimator
        A supervised learning model implementing a `predict_proba` method.
        
    Attributes
    ----------
    pos_scores : np.ndarray
        Score distribution for the positive class in the training data.
    neg_scores : np.ndarray
        Score distribution for the negative class in the training data.
    
    References
    ----------
    HASSAN, Waqar; MALETZKE, André; BATISTA, Gustavo. Accurately quantifying a billion instances per second. In: 2020 IEEE 7th International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2020. p. 1-10. Avaliable at https://ieeexplore.ieee.org/document/9260028
    
    Examples
    --------
    >>> from mlquantify.methods.mixture_models import SMM
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> smm = SMM(RandomForestClassifier())
    >>> smm.fit(X_train, y_train)
    >>> prevalence = smm.predict(X_test)
    >>> prevalence
    {0: 0.38358048188348526, 1: 0.6164195181165147}
    >>> get_real_prev(y_test)
    {0: 0.37719298245614036, 1: 0.6228070175438597}
    """

    def __init__(self, learner: BaseEstimator=None):
        super().__init__(learner)

    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        """
        Compute the prevalence estimate based on mean scores.

        Parameters
        ----------
        test_scores : np.ndarray
            Array of predicted probabilities for the test data.

        Returns
        -------
        prevalence : float
            Estimated prevalence.
        """
        mean_pos_score = np.mean(self.pos_scores)
        mean_neg_score = np.mean(self.neg_scores)
        mean_test_score = np.mean(test_scores)

        # Calculate prevalence as the proportion of the positive class
        prevalence = (mean_test_score - mean_neg_score) / (mean_pos_score - mean_neg_score)

        return prevalence


class SORD(MixtureModel):
    """
    Sample Ordinal Distance (SORD).

    A method that estimates the prevalence of the positive class 
    in a test dataset by calculating and minimizing a sample ordinal 
    distance measure between test scores and known positive and 
    negative scores. This approach does not rely on distributional 
    assumptions.
    
    Parameters
    ----------
    learner : BaseEstimator
        A supervised learning model implementing a `predict_proba` method.
        
    Attributes
    ----------
    pos_scores : np.ndarray
        Score distribution for the positive class in the training data.
    neg_scores : np.ndarray
        Score distribution for the negative class in the training data.
    best_distance_index : int
        Index of the best alpha value.
    
    References
    ----------
    VAN HASSELT, H.; GUEZ, A.; SILVER, D. Proceedings of the AAAI conference on artificial intelligence. 2016. Avaliable at https://ojs.aaai.org/index.php/AAAI/article/view/4376 
    
    Examples
    --------
    >>> from mlquantify.methods.mixture_models import SORD
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    >>>
    >>> sord = SORD(RandomForestClassifier())
    >>> sord.fit(X_train, y_train)
    >>> prevalence = sord.predict(X_test)
    >>> prevalence
    {0: 0.38, 1: 0.62}
    >>> get_real_prev(y_test)
    {0: 0.37719298245614036, 1: 0.6228070175438597}
    """

    def __init__(self, learner: BaseEstimator=None):
        super().__init__(learner)

        self.best_distance_index = None  # Stores the index of the best alpha value

    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        """
        Compute the prevalence estimate by minimizing the ordinal distance.

        Parameters
        ----------
        test_scores : np.ndarray
            Array of predicted probabilities for the test data.

        Returns
        -------
        prevalence : float
            Estimated prevalence.
        """
        # Compute alpha values and corresponding distance measures
        alpha_values, distance_measures = self._calculate_distances(test_scores)

        # Find the index of the alpha value with the minimum distance measure
        self.best_distance_index = np.argmin(distance_measures)
        prevalence = alpha_values[self.best_distance_index]

        return prevalence

    def _calculate_distances(self, test_scores: np.ndarray) -> tuple:
        """
        Calculate distance measures for a range of alpha values.

        Parameters
        ----------
        test_scores : np.ndarray
            Array of predicted probabilities for the test data.

        Returns
        -------
        alpha_values : np.ndarray
            Array of alpha values (from 0 to 1) used for evaluation.
        distance_measures : list
            List of distance measures for each alpha value.
        """
        # Define a range of alpha values from 0 to 1
        alpha_values = np.linspace(0, 1, 101)

        # Get the number of positive, negative, and test scores
        num_pos_scores = len(self.pos_scores)
        num_neg_scores = len(self.neg_scores)
        num_test_scores = len(test_scores)

        distance_measures = []

        # Iterate over each alpha value
        for alpha in alpha_values:
            # Compute weights for positive, negative, and test scores
            pos_weight = alpha / num_pos_scores
            neg_weight = (1 - alpha) / num_neg_scores
            test_weight = -1 / num_test_scores

            # Create arrays with weights
            pos_weights = np.full(num_pos_scores, pos_weight)
            neg_weights = np.full(num_neg_scores, neg_weight)
            test_weights = np.full(num_test_scores, test_weight)

            # Concatenate all scores and their corresponding weights
            all_scores = np.concatenate([self.pos_scores, self.neg_scores, test_scores])
            all_weights = np.concatenate([pos_weights, neg_weights, test_weights])

            # Sort scores and weights based on scores
            sorted_indices = np.argsort(all_scores)
            sorted_scores = all_scores[sorted_indices]
            sorted_weights = all_weights[sorted_indices]

            # Compute the total cost for the current alpha
            cumulative_weight = sorted_weights[0]
            total_cost = 0

            for i in range(1, len(sorted_scores)):
                # Calculate the cost for the segment between sorted scores
                segment_width = sorted_scores[i] - sorted_scores[i - 1]
                total_cost += abs(segment_width * cumulative_weight)
                cumulative_weight += sorted_weights[i]

            distance_measures.append(total_cost)

        return alpha_values, distance_measures
