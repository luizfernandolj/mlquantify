import numpy as np


class One_vs_All:
    
    def __init__(self, y):
        self.y = y
        
    def generate_trains(self):
        for label in np.unique(self.y):
            y_label = np.asarray([1 if _class == label else 0 for _class in self.y])
            yield label, y_label
