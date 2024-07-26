import numpy as np

def MoSS(n:int, alpha:float, m:float):
  
  n_pos = int(n*alpha)
  n_neg = int((1-alpha)*n)

  x_pos = np.arange(1, n_pos, 1)
  x_neg = np.arange(1, n_neg, 1)
  
  syn_plus = np.power(x_pos/(n_pos+1), m)
  syn_neg = 1 - np.power(x_neg/(n_neg+1), m)

  #moss = np.union1d(syn_plus, syn_neg)

  return syn_plus, syn_neg