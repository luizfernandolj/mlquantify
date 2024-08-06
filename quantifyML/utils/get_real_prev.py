import numpy as np
import pandas as pd

def get_real_prev(y) -> dict:
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    real_prevs = np.round(y.value_counts(normalize=True), 3).to_dict()
    real_prevs = dict(sorted(real_prevs.items()))
    return real_prevs