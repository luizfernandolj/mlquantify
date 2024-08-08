import numpy as np
import pandas as pd


def round_protocol_df(dataframe:pd.DataFrame, frac:int=3):
    def round_column(col):
        if col.name in ['PRED_PREVS', 'REAL_PREVS']:
            return col.apply(lambda x: np.round(x, frac))
        elif np.issubdtype(col.dtype, np.number):
            return col.round(frac)
        else:
            return col
    
    return dataframe.apply(round_column)