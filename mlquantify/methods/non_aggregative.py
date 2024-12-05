import numpy as np

from ..base import NonAggregativeQuantifier
from ..utils.method import getHist, hellinger




class HDx(NonAggregativeQuantifier):
    """
    Hellinger Distance Minimization (HDx).

    This method estimates class prevalence by calculating the Hellinger 
    distance for each feature in the dataset, as opposed to HDy, which 
    computes the distance for classifier-generated scores.
    
    Parameters
    ----------
    bins_size : np.ndarray, optional
        An array of bin sizes for histogram calculations. Defaults to an array 
        combining linearly spaced values between 2 and 20 with an additional 
        bin size of 30.
    
    Attributes
    ----------
    bins_size : np.ndarray
        An array of bin sizes for histogram calculations.
    neg_features : np.ndarray
        Features from the negative class.
    pos_features : np.ndarray
        Features from the positive class.
    
    References
    ----------
    .. [1] GONZÁLEZ-CASTRO, Víctor; ALAIZ-RODRÍGUEZ, Rocío; ALEGRE, Enrique. Class distribution estimation based on the Hellinger distance. Information Sciences, v. 218, p. 146-164, 2013. Avaliable at https://www.sciencedirect.com/science/article/abs/pii/S0020025512004069?casa_token=W6UksOigmp4AAAAA:ap8FK5mtpAzG-s8k2ygfRVgdIBYDGWjEi70ueJ546coP9F-VNaCKE5W_gsAv0bWQiwzt2QoAuLjP    
    
    Examples
    --------
    >>> from mlquantify.methods.non_aggregative import HDx
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> 
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    >>> 
    >>> model = HDx()
    >>> model.fit(X_train, y_train)
    >>> 
    >>> predictions = model.predict(X_test)
    >>> predictions
    {0: 0.39, 1: 0.61}
    >>> get_real_prev(y_test)
    {0: 0.3684210526315789, 1: 0.631578947368421}
    """

    def __init__(self, bins_size: np.ndarray = None):
        if bins_size is None:
            bins_size = np.append(np.linspace(2, 20, 10), 30)

        self.bins_size = bins_size
        self.neg_features = None
        self.pos_features = None

    def _fit_method(self, X, y):
        """
        Fit the HDx model by separating the features into positive and negative classes.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target labels.

        Returns
        -------
        self : HDx
            The fitted instance of the class.
        """
        self.pos_features = X[y == self.classes[1]]
        self.neg_features = X[y == self.classes[0]]

        if not isinstance(X, np.ndarray):
            self.pos_features = self.pos_features.to_numpy()
        if not isinstance(y, np.ndarray):
            self.neg_features = self.neg_features.to_numpy()

        return self

    def _predict_method(self, X) -> np.ndarray:
        """
        Predict the prevalence of the positive and negative classes.

        Parameters
        ----------
        X : array-like
            Feature matrix for the test data.

        Returns
        -------
        prevalence : np.ndarray
            A 2-element array representing the prevalence of the negative 
            and positive classes, respectively.
        """
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        alpha_values = np.round(np.linspace(0, 1, 101), 2)
        best_distances = {}

        # Iterate over alpha values to compute the prevalence
        for alpha in alpha_values:
            distances = []

            # For each feature, compute the Hellinger distance
            for i in range(X.shape[1]):
                for bins in self.bins_size:
                    dist_feature_pos = getHist(self.pos_features[:, i], bins)
                    dist_feature_neg = getHist(self.neg_features[:, i], bins)
                    dist_feature_test = getHist(X[:, i], bins)

                    # Combine positive and negative densities using the mixture weight (alpha)
                    train_combined_density = (dist_feature_pos * alpha) + (dist_feature_neg * (1 - alpha))

                    # Compute the Hellinger distance between the combined density and test density
                    distances.append(hellinger(train_combined_density, dist_feature_test))

            # Store the mean distance for the current alpha
            best_distances[alpha] = np.mean(distances)

        # Find the alpha value that minimizes the mean Hellinger distance
        prevalence = min(best_distances, key=best_distances.get)

        return np.asarray([1 - prevalence, prevalence])
