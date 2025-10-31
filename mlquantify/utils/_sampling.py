import numpy as np
import itertools


def get_indexes_with_prevalence(y, prevalence: list, sample_size:int):
    """
    Get indexes for a stratified sample based on the prevalence of each class.
    
    Parameters
    ----------
    y : np.ndarray
        Array of class labels.
    prevalence : list
        List of prevalences for each class.
    sample_size : int
        Number of samples to generate.
    classes : list
        List of unique classes.
        
    Returns
    -------
    list
        List of indexes for the stratified sample.
    """      
    classes = np.unique(y)
        
    # Ensure the sum of prevalences is 1
    assert np.isclose(sum(prevalence), 1), "The sum of prevalences must be 1"
    # Ensure the number of prevalences matches the number of classes
    assert len(prevalence) == len(classes), "The number of prevalences must match the number of classes"

    sampled_indexes = []
    total_sampled = 0

    for i, class_ in enumerate(classes):

        if i == len(classes) - 1:
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



def kraemer_sampling(n_dim: int, n_prev: int, n_iter: int = 1) -> np.ndarray:
    """
    Uniform sampling from the unit simplex using Kraemer's algorithm.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    n_prev : int
        Size of the sample.
    n_iter : int
        Number of iterations.

    Returns
    -------
    np.ndarray
        Array of sampled prevalences.
    """

    def _sampling(n_dim: int, n_prev: int) -> np.ndarray:
        if n_dim == 2:
            u = np.random.rand(n_prev)
            return np.vstack([1 - u, u]).T
        else:
            u = np.random.rand(n_prev, n_dim - 1)
            u.sort(axis=-1)   # sort each row
            _0s = np.zeros((n_prev, 1))
            _1s = np.ones((n_prev, 1))
            a = np.hstack([_0s, u])
            b = np.hstack([u, _1s])
            return b - a

    # repeat n_iter times
    prevs = _sampling(n_dim, n_prev)

    return np.repeat(prevs, n_iter, axis=0) if n_iter > 1 else prevs
 

def artificial_sampling(n_dim: int, n_prev: int, n_iter: int) -> np.ndarray:
    """Generates n artificial prevalences with n dimensions.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    n_prev : int
        Number of prevalences to generate.
    n_iter : int
        Number of iterations.
    
    Returns
    -------
    np.ndarray
        Array of artificial prevalences.
    
    """
    s = np.linspace(0., 1., n_prev, endpoint=True)
    prevs = np.array([p + (1 - sum(p),) for p in itertools.product(*(s,) * (n_dim - 1)) if sum(p) <= 1])
    
    return np.repeat(prevs, n_iter, axis=0) if n_iter > 1 else prevs