from abc import ABC
from copy import deepcopy
import numpy as np

array = np.random.choice([0, 4], 10)
array = (array == 4).astype(int)
print(array)