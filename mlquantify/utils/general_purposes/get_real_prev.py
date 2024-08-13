import numpy as np
import pandas as pd

def get_real_prev(y) -> dict:
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    real_prevs = y.value_counts(normalize=True).to_dict()
    real_prevs = dict(sorted(real_prevs.items()))
    return real_prevs