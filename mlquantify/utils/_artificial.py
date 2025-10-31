import numpy as np


def make_prevs(ndim:int) -> list:
    """
    Generate a list of n_dim values uniformly distributed between 0 and 1 that sum exactly to 1.
    
    Parameters
    ----------
    ndim : int
        Number of dimensions.
    
    Returns
    -------
    list
        List of n_dim values uniformly distributed between 0 and 1 that sum exactly to 1.
    """
    # Generate n_dim-1 random u_dist uniformly distributed between 0 and 1
    u_dist = np.random.uniform(0, 1, ndim - 1)
    # Add 0 and 1 to the u_dist
    u_dist = np.append(u_dist, [0, 1])
    # Sort the u_dist
    u_dist.sort()
    # Calculate the differences between consecutive u_dist
    prevs = np.diff(u_dist)

    return prevs