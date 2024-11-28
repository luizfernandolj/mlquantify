import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load
from collections import defaultdict


def convert_columns_to_arrays(df, columns:list = ['PRED_PREVS', 'REAL_PREVS']):
    """Converts the specified columns from string of arrays to numpy arrays

    Args:
        df (array-like): the dataframe from which to change convert the coluns
        columns (list, optional): the coluns with string of arrays, default is the options for
        the protocol dataframes
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x)
    return df





def generate_artificial_indexes(y, prevalence: list, sample_size:int, classes:list):        
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








def get_real_prev(y) -> dict:
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    real_prevs = y.value_counts(normalize=True).to_dict()
    real_prevs = dict(sorted(real_prevs.items()))
    return real_prevs









def load_quantifier(path:str):
    return load(path)









def make_prevs(ndim:int) -> list:
    """
    Generate a list of n_dim values uniformly distributed between 0 and 1 that sum exactly to 1.
    
    Args:
    n_dim (int): Number of values in the list.
    
    Returns:
    list: List of n_dim values that sum to 1.
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
    return Parallel(n_jobs=n_jobs)(
        delayed(func)(e, *args) for e in elements
    )
    
    
    
    






def round_protocol_df(dataframe: pd.DataFrame, frac: int = 3):
    def round_column(col):
        if col.name in ['PRED_PREVS', 'REAL_PREVS']:
            return col.apply(lambda x: np.round(x, frac) if isinstance(x, (np.ndarray, float, int)) else x)
        elif np.issubdtype(col.dtype, np.number):
            return col.round(frac)
        else:
            return col
    
    return dataframe.apply(round_column)




def get_measure(measure:str):
    from ..evaluation import MEASURES
    return MEASURES.get(measure)


def get_method(method: str):
    from ..methods import METHODS
    return METHODS.get(method)