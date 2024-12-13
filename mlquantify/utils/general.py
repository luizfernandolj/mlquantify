import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load
from collections import defaultdict
import itertools


def convert_columns_to_arrays(df, columns:list = ['PRED_PREVS', 'REAL_PREVS']):
    """
    Converts specified columns in a DataFrame from strings of arrays to NumPy arrays.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert.
    columns : list
        List of columns to convert.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with the specified columns converted to NumPy arrays.
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x)
    return df





def generate_artificial_indexes(y, prevalence: list, sample_size:int, classes:list):
    """
    Generate indexes for a stratified sample based on the prevalence of each class.
    
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
    # Ensure the sum of prevalences is 1
    assert np.isclose(sum(prevalence), 1), "The sum of prevalences must be 1"
    # Ensure the number of prevalences matches the number of classes

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




def generate_artificial_prevalences(n_dim: int, n_prev: int, n_iter: int) -> np.ndarray:
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








def get_real_prev(y) -> dict:
    """
    Get the real prevalence of each class in the target array.
    
    Parameters
    ----------
    y : np.ndarray or pd.Series
        Array of class labels.
        
    Returns
    -------
    dict
        Dictionary of class labels and their corresponding prevalence.
    """
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    real_prevs = y.value_counts(normalize=True).to_dict()
    real_prevs = dict(sorted(real_prevs.items()))
    return real_prevs









def load_quantifier(path:str):
    """
    Load a quantifier from a file.
    
    Parameters
    ----------
    path : str
        Path to the file containing the quantifier.
    
    Returns
    -------
    Quantifier
        Loaded quantifier.
    """
    return load(path)









def make_prevs(ndim:int) -> list:
    """
    Generate a list of n_dim values uniformly distributed between 0 and 1 that sum exactly to 1.
    
    Parameters
    ----------
    ndim : int
        Number of dimensions.
    
    Returns
    -------
    list
        List of n_dim values uniformly distributed between 0 and 1 that sum exactly to 1.
    """
    # Generate n_dim-1 random u_dist uniformly distributed between 0 and 1
    u_dist = np.random.uniform(0, 1, ndim - 1)
    # Add 0 and 1 to the u_dist
    u_dist = np.append(u_dist, [0, 1])
    # Sort the u_dist
    u_dist.sort()
    # Calculate the differences between consecutive u_dist
    prevs = np.diff(u_dist)

    return prevs













def normalize_prevalence(prevalences: np.ndarray, classes:list):
    """
    Normalize the prevalence of each class to sum to 1.
    
    Parameters
    ----------
    prevalences : np.ndarray
        Array of prevalences.
    classes : list
        List of unique classes.
    
    Returns
    -------
    dict
        Dictionary of class labels and their corresponding prevalence.
    """
    if isinstance(prevalences, dict):
        summ = sum(prevalences.values())
        prevalences = {int(_class):float(value/summ) for _class, value in prevalences.items()}
        return prevalences
    
    summ = np.sum(prevalences, axis=-1, keepdims=True)
    prevalences = np.true_divide(prevalences, sum(prevalences), where=summ>0)
    prevalences = {int(_class):float(prev) for _class, prev in zip(classes, prevalences)}
    prevalences = defaultdict(lambda: 0, prevalences)
    
    # Ensure all classes are present in the result
    for cls in classes:
        prevalences[cls] = prevalences[cls]
    
    return dict(prevalences)







def parallel(func, elements, n_jobs: int = 1, *args):
    """
    Run a function in parallel on a list of elements.
    
    Parameters
    ----------
    func : function
        Function to run in parallel.
    elements : list
        List of elements to run the function on.
    n_jobs : int
        Number of jobs to run in parallel.
    args : tuple
        Additional arguments to pass to the function.
    
    Returns
    -------
    list
        List of results from running the function on each element.
    """
    return Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(func)(e, *args) for e in elements
    )
    
    
    
    






def round_protocol_df(dataframe: pd.DataFrame, frac: int = 3):
    """
    Round the columns of a protocol dataframe to a specified number of decimal places.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Protocol dataframe to round.
    frac : int
        Number of decimal places to round to.
    
    Returns
    -------
    pd.DataFrame
        Protocol dataFrame with the columns rounded to the specified number of decimal places.
    """
    def round_column(col):
        if col.name in ['PRED_PREVS', 'REAL_PREVS']:
            return col.apply(lambda x: np.round(x, frac) if isinstance(x, (np.ndarray, float, int)) else x)
        elif np.issubdtype(col.dtype, np.number):
            return col.round(frac)
        else:
            return col
    
    return dataframe.apply(round_column)




def get_measure(measure:str):
    """
    Get the measure from the evaluation module.
    
    Parameters
    ----------
    measure : str
        Measure to get.
    
    Returns
    -------
    Measure
        Measure function from the evaluation module.
    """
    from ..evaluation import MEASURES
    return MEASURES.get(measure)


def get_method(method: str):
    """
    Get the method from the methods module.
    
    Parameters
    ----------
    method : str
        Method to get.
    
    Returns
    -------
    Method
        Method class from the methods module.
    """
    from ..methods import METHODS
    return METHODS.get(method)