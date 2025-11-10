import numpy as np
from numpy.random import RandomState, Generator, default_rng


def check_random_state(seed=None):
    """
    Turn seed into a np.random.RandomState or np.random.Generator instance.

    Parameters
    ----------
    seed : None, int, RandomState, Generator
        - If None, return the global RandomState singleton used by np.random.
        - If int, return a new RandomState instance seeded with seed.
        - If RandomState or Generator, return it.
        - Otherwise, raise ValueError.

    Returns
    -------
    rng : np.random.Generator
        A numpy random generator compatible with modern numpy APIs.
    """
    if seed is None or seed is np.random:
        return default_rng()  # new independent generator each call
    if isinstance(seed, (int, np.integer)):
        return default_rng(seed)
    if isinstance(seed, Generator):
        return seed
    if isinstance(seed, RandomState):
        # Wrap legacy RandomState inside a Generator for uniformity
        bitgen = np.random.MT19937()
        bitgen.state = seed.get_state()
        return Generator(bitgen)
    raise ValueError(
        f"{seed!r} cannot be used to seed a numpy random number generator. "
        "Valid options are None, int, RandomState, or Generator."
    )
