import numpy as np
from ..methods import get_class
from . import get_measure

class Ensemble:
    
    def __init__(self, methods:list, n_iterations:int=100, measure:str="ae"):
