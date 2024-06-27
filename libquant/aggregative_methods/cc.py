from ..base import Quantifier

class CC(Quantifier):
    
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2