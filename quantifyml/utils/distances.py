import numpy as np

class Distances():
    
    def __init__(self,P,Q):
        if sum(P)<1e-20 or sum(Q)<1e-20:
            raise "One or both vector are zero (empty)..."
        if len(P)!=len(Q):
            raise "Arrays need to be of equal sizes..."
        #use numpy arrays for efficient coding
        P=np.array(P,dtype=float);Q=np.array(Q,dtype=float)
        #Correct for zero values
        P[np.where(P<1e-20)]=1e-20
        Q[np.where(Q<1e-20)]=1e-20
        self.P=P
        self.Q=Q
        
    def sqEuclidean(self):
        P=self.P; Q=self.Q; d=len(P)
        return sum((P-Q)**2)
    
    def probsymm(self):
        P=self.P; Q=self.Q; d=len(P)
        return 2*sum((P-Q)**2/(P+Q))
    
    def topsoe(self):
        P=self.P; Q=self.Q
        return sum(P*np.log(2*P/(P+Q))+Q*np.log(2*Q/(P+Q)))
    def hellinger(self):
        P=self.P; Q=self.Q
        return 2 * np.sqrt(np.abs(1 - sum(np.sqrt(P * Q))))