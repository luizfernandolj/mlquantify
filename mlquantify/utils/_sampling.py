import numpy as np
from mlquantify.utils import check_random_state
import itertools


def get_indexes_with_prevalence(y, prevalence: list, sample_size:int, random_state: int = None):
    """
    Get indexes for a stratified sample based on the prevalence of each class.
    
    Parameters
    ----------
    y : np.ndarray
        Array of class labels.
    prevalence : list
        List of prevalences for each class.
    sample_size : int
        Number of samples to generate.
    classes : list
        List of unique classes.
        
    Returns
    -------
    list
        List of indexes for the stratified sample.
    """      
    classes = np.unique(y)
    rng = check_random_state(random_state)
        
    # Ensure the sum of prevalences is 1
    assert np.isclose(sum(prevalence), 1), "The sum of prevalences must be 1"
    # Ensure the number of prevalences matches the number of classes
    assert len(prevalence) == len(classes), "The number of prevalences must match the number of classes"

    sampled_indexes = []
    total_sampled = 0
    
    for i, class_ in enumerate(classes):

        if i == len(classes) - 1:
            num_samples = sample_size - total_sampled
        else:
            num_samples = int(sample_size * prevalence[i])
        
        # Get the indexes of the current class
        class_indexes = np.where(y == class_)[0]

        # Sample the indexes for the current class
        sampled_class_indexes = rng.choice(class_indexes, size=num_samples, replace=True)
        
        sampled_indexes.extend(sampled_class_indexes)
        total_sampled += num_samples

    rng.shuffle(sampled_indexes)  # Shuffle after collecting all indexes
        
    return sampled_indexes



def simplex_uniform_kraemer(n_dim: int, 
                            n_prev: int,
                            n_iter: int, 
                            min_val: float = 0.0, 
                            max_val: float = 1.0, 
                            max_tries: int = 1000,
                            random_state: int = None) -> np.ndarray:
    """
    Generates n_prev prevalence vectors of n_dim classes uniformly 
    distributed on the simplex, with optional lower and upper bounds.

    Based on the algorithm of Kramer et al. for uniform sampling on a simplex.

    Parameters
    ----------
    n_dim : int
        Number of dimensions (classes).
    n_prev : int
        Number of prevalence vectors to generate.
    min_val : float, optional
        Minimum allowed prevalence for each class (default=0.0).
    max_val : float, optional
        Maximum allowed prevalence for each class (default=1.0).
    max_tries : int, optional
        Maximum number of sampling iterations to reach the target n_prev.

    Returns
    -------
    np.ndarray
        Array of shape (n_prev, n_dim) with valid prevalence vectors.
    """
    if n_dim < 2:
        raise ValueError("n_dim must be >= 2.")
    if not (0 <= min_val < 1) or not (0 < max_val <= 1):
        raise ValueError("min_val and max_val must be between 0 and 1.")
    if min_val * n_dim > 1 or max_val * n_dim < 1:
        raise ValueError("Invalid bounds: they make it impossible to sum to 1.")

    rng = check_random_state(random_state)

    effective_simplex_size = 1 - n_dim * min_val
    prevs = []

    tries = 0
    batch_size = n_prev

    while len(prevs) < n_prev and tries < max_tries:
        tries += 1
    
        u = rng.uniform(0, 1, (batch_size, n_dim - 1))
        u.sort(axis=1)
        simplex = np.diff(np.concatenate([np.zeros((batch_size, 1)), u, np.ones((batch_size, 1))], axis=1), axis=1)

        scaled = min_val + simplex * effective_simplex_size

        scaled /= scaled.sum(axis=1, keepdims=True)

        mask = np.all((scaled >= min_val) & (scaled <= max_val), axis=1)
        valid = scaled[mask]

        if valid.size > 0:
            prevs.append(valid)

    if not prevs:
        raise RuntimeError("No valid prevalences found with given constraints. Try adjusting min_val/max_val.")
    
    result = np.vstack(prevs)
    result = result[:n_prev]

    if n_iter > 1:
        result = np.repeat(result, n_iter, axis=0)
    
    return result
 
 
 
def simplex_grid_sampling(
    n_dim: int,
    n_prev: int,
    n_iter: int,
    min_val: float,
    max_val: float,
) -> np.ndarray:
    """
    Efficiently generates artificial prevalence vectors that sum to 1
    and respect min_val ≤ p_i ≤ max_val for all i.

    Parameters
    ----------
    n_dim : int
        Number of dimensions (classes).
    n_prev : int
        Number of prevalence points per dimension (grid density).
    n_iter : int
        Number of repetitions.
    min_val : float
        Minimum allowed value for each prevalence component.
    max_val : float
        Maximum allowed value for each prevalence component.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_dim) with all valid prevalence vectors.
    """
    if n_dim < 2:
        raise ValueError("n_dim must be at least 2.")
    if not (0 <= min_val < max_val <= 1):
        raise ValueError("min_val and max_val must satisfy 0 <= min_val < max_val <= 1.")
    if min_val * n_dim > 1 or max_val * n_dim < 1:
        raise ValueError("Impossible combination of min_val, max_val, and n_dim — cannot sum to 1.")

    # Intervalo de possíveis valores para cada dimensão (exceto a última)
    s = np.linspace(min_val, max_val, n_prev)
    grids = np.stack(np.meshgrid(*([s] * (n_dim - 1)), indexing="ij"), axis=-1)
    grid_flat = grids.reshape(-1, n_dim - 1)

    # Calcula o último valor para garantir soma = 1
    last_col = 1.0 - np.sum(grid_flat, axis=1)
    prevs = np.hstack([grid_flat, last_col[:, None]])

    # Filtro de validade: dentro dos limites
    mask = np.all((prevs >= min_val) & (prevs <= max_val), axis=1)
    prevs = prevs[mask]

    # Repetição se necessário
    if n_iter > 1:
        prevs = np.repeat(prevs, n_iter, axis=0)

    return prevs




def simplex_dirichlet_sampling(
    n_dim: int,
    n_prev: int,
    n_iter: int,
    alpha=1.0,
    min_val: float = 0.0,
    max_val: float = 1.0,
    max_tries: int = 1000,
    random_state: int = None,
) -> np.ndarray:
    r"""
    Sample prevalence vectors from a Dirichlet distribution on the simplex,
    constrained by ``min_val`` ≤ :math:`p_i` ≤ ``max_val``.

    The concentration parameter ``alpha`` controls how the probability mass is
    spread over the simplex:

    - ``alpha == 1`` — the flat Dirichlet :math:`\mathrm{Dir}(\mathbf{1})`,
      i.e. a *uniform* distribution over the simplex (every prevalence
      combination is equally likely).
    - ``alpha > 1`` — mass is pulled towards the balanced centre
      :math:`(1/k, \ldots, 1/k)`; extreme prevalences become rare.
    - ``alpha < 1`` — mass is pushed towards the corners; near-pure
      (one-class-dominant) prevalences become common.

    Parameters
    ----------
    n_dim : int
        Number of dimensions (classes).
    n_prev : int
        Number of prevalence vectors to generate.
    n_iter : int
        Number of repetitions for each generated vector.
    alpha : float or array-like of shape (n_dim,), default=1.0
        Dirichlet concentration parameter. A scalar is broadcast to a symmetric
        Dirichlet over all classes; an array sets a per-class concentration.
    min_val : float, default=0.0
        Minimum allowed prevalence for each class.
    max_val : float, default=1.0
        Maximum allowed prevalence for each class.
    max_tries : int, optional
        Maximum number of sampling iterations to reach the target count.
    random_state : int, RandomState instance or None, default=None
        Seed or generator controlling the sampling.

    Returns
    -------
    np.ndarray
        Array of shape (n_prev * n_iter, n_dim) with valid prevalence vectors.
    """
    if n_dim < 2:
        raise ValueError("n_dim must be >= 2.")
    if min_val * n_dim > 1 or max_val * n_dim < 1:
        raise ValueError("Invalid min_val/max_val for simplex constraints.")

    rng = check_random_state(random_state)

    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (n_dim,))
    if np.any(alpha <= 0):
        raise ValueError("alpha values must be strictly positive.")

    samples = []
    collected = 0
    tries = 0

    while collected < n_prev and tries < max_tries:
        tries += 1
        # Generate candidates via the Dirichlet distribution.
        x = rng.dirichlet(alpha, size=n_prev * 2)
        # Keep only those respecting the per-class bounds.
        mask = np.all((x >= min_val) & (x <= max_val), axis=1)
        valid = x[mask]
        if valid.size > 0:
            samples.append(valid)
            collected += len(valid)

    if not samples:
        raise RuntimeError(
            "No valid prevalences found with given constraints. "
            "Try adjusting min_val/max_val or alpha."
        )

    result = np.concatenate(samples, axis=0)[:n_prev]

    if n_iter > 1:
        result = np.repeat(result, n_iter, axis=0)

    return result


def simplex_uniform_sampling(
    n_dim: int,
    n_prev: int,
    n_iter: int,
    min_val: float = 0.0,
    max_val: float = 1.0,
    random_state: int = None
) -> np.ndarray:
    """
    Generates uniformly distributed prevalence vectors within the simplex,
    constrained by min_val ≤ p_i ≤ max_val.

    Thin wrapper around :func:`simplex_dirichlet_sampling` with a flat
    concentration (``alpha=1``), which is uniform over the simplex.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    n_prev : int
        Number of prevalence samples to generate.
    n_iter : int
        Number of repetitions.
    min_val : float
        Minimum allowed value for each prevalence component.
    max_val : float
        Maximum allowed value for each prevalence component.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_dim) with uniformly distributed prevalences.
    """
    return simplex_dirichlet_sampling(
        n_dim=n_dim,
        n_prev=n_prev,
        n_iter=n_iter,
        alpha=1.0,
        min_val=min_val,
        max_val=max_val,
        random_state=random_state,
    )


def bootstrap_sample_indices(
    n_samples: int,
    batch_size: int,
    n_bootstraps: int,
    random_state: int = None
):
    """
    Generate bootstrap sample indices for a dataset.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    batch_size : int
        Number of samples in each bootstrap sample.
    n_bootstraps : int
        Number of bootstrap samples to generate.
    random_state : int, optional
        Random seed for reproducibility.

    Yields
    ------
    np.ndarray
        Array containing indices for a bootstrap sample.
    """
    rng = check_random_state(random_state)

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=batch_size, replace=True)
        yield indices
