import numpy as np

def sqEuclidean(dist1, dist2):
    P=dist1 
    Q=dist2 
    return sum((P-Q)**2)
    
def probsymm(dist1, dist2):
    P=dist1
    Q=dist2
    return 2*sum((P-Q)**2/(P+Q))

def topsoe(dist1, dist2):
    P=dist1
    Q=dist2
    return sum(P*np.log(2*P/(P+Q))+Q*np.log(2*Q/(P+Q)))

def hellinger(dist1, dist2):
    P=dist1
    Q=dist2
    return 2 * np.sqrt(np.abs(1 - sum(np.sqrt(P * Q))))