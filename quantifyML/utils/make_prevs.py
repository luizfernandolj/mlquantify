import numpy as np

def make_prevs(ndim:int, min_val:float=0, max_trials=100) -> float:
    """
    Generate a list of n_dim values uniformly distributed between 0 and 1 that sum exactly to 1.
    
    Args:
    n_dim (int): Number of values in the list.
    
    Returns:
    list: List of n_dim values that sum to 1.
    """
    trials = 0
    while True:
        
        # Generate n_dim-1 random u_dist uniformly distributed between 0 and 1
        u_dist = np.random.uniform(0, 1, ndim - 1)
        # Add 0 and 1 to the u_dist
        u_dist = np.append(u_dist, [0, 1])
        # Sort the u_dist
        u_dist.sort()
        # Calculate the differences between consecutive u_dist
        prevs = np.diff(u_dist)

        if all(prevs < min_val):
            return prevs
        trials += 1
        if trials >= max_trials:
            raise ValueError(f'it looks like finding a random simplex with all its dimensions being'
                             f'>= {min_val} is unlikely (it failed after {max_trials} trials)')
        