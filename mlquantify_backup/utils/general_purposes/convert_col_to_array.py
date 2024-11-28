import numpy as np

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