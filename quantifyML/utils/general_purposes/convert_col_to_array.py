import numpy as np

def convert_columns_to_arrays(df, columns:list = ['PRED_PREVS', 'REAL_PREVS']):
    for col in columns:
        df[col] = df[col].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x)
    return df