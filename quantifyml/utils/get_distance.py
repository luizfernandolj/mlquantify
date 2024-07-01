import numpy as np
from .distances import Distances


def get_distance(sc_1, sc_2, measure):
    """This function applies a selected distance metric"""
    
    dist = Distances(sc_1, sc_2)
    
    if measure == 'topsoe':
        return dist.topsoe()
    if measure == 'probsymm':
        return dist.probsymm()
    if measure == 'hellinger':
        return dist.hellinger()
    return 100