import numpy as np
from .. import get_class
from ...evaluation import get_measure
from ...base import Quantifier

class Ensemble(Quantifier):
    
    def __init__(self, methods:list[Quantifier],
                 size:int=50,
                 min_pos_prop:float=0.1,
                 sample_size:int=None,
                 n_jobs:int=1,
                 return_type="mean",
                 verbose:bool=False):
                
        assert sample_size is None or sample_size > 0, \
            'wrong value for sample_size; set it to a positive number or None'
        
        self.methods = methods
        self.size = size
        self.min_pos_prop = min_pos_prop
        self.sample_size = sample_size
        self.n_jobs = n_jobs
        self.return_type = return_type
        self.verbose = verbose


    def sout(self, msg):
        if self.verbose:
            print('[Ensemble]' + msg)
           
            
    def fit(self, X, y):
        self.sout("FIT")
        self.classes = np.unique(y)
    
        sample_size = len(X) if self.sample_size is None else min(self.sample_size, len(X))
        if self.sample_size > len(X):
            print("ALERT: SAMPLE SIZE GREATER THAN LENGTH OF DATA")
        
        train_prevs = [_make_prevs(ndim=self.n_class, min_val=self.min_pos_prop)]
    
    
    
    
    


        
        
