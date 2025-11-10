import numpy as np
import itertools


def get_indexes_with_prevalence(y, prevalence: list, sample_size:int):
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
        sampled_class_indexes = np.random.choice(class_indexes, size=num_samples, replace=True)
        
        sampled_indexes.extend(sampled_class_indexes)
        total_sampled += num_samples

    np.random.shuffle(sampled_indexes)  # Shuffle after collecting all indexes
        
    return sampled_indexes



def simplex_uniform_kraemer(n_dim: int, 
                            n_prev: int,
                            n_iter: int, 
                            min_val: float = 0.0, 
                            max_val: float = 1.0, 
                            max_tries: int = 1000) -> np.ndarray:
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

    effective_simplex_size = 1 - n_dim * min_val
    prevs = []

    # Amostragem em blocos até atingir n_prev válidos
    tries = 0
    batch_size = max(n_prev, 1000)  # Gera em blocos grandes para eficiência

    while len(prevs) < n_prev and tries < max_tries:
        tries += 1

        # Geração de pontos uniformes no simplex reduzido
        u = np.random.uniform(0, 1, (batch_size, n_dim - 1))
        u.sort(axis=1)
        simplex = np.diff(np.concatenate([np.zeros((batch_size, 1)), u, np.ones((batch_size, 1))], axis=1), axis=1)

        # Escala para [min_val, max_val]
        scaled = min_val + simplex * effective_simplex_size

        # Normaliza para garantir soma = 1
        scaled /= scaled.sum(axis=1, keepdims=True)

        # Filtra apenas vetores válidos
        mask = np.all((scaled >= min_val) & (scaled <= max_val), axis=1)
        valid = scaled[mask]

        if valid.size > 0:
            prevs.append(valid)

    if not prevs:
        raise RuntimeError("No valid prevalences found with given constraints. Try adjusting min_val/max_val.")
    
    if n_iter > 1:
        prevs = np.tile(prevs, (n_iter, 1))

    result = np.vstack(prevs)
    return result[:n_prev]
 
 
 
def simplex_grid_sampling(
    n_dim: int,
    n_prev: int,
    n_iter: int,
    min_val: float,
    max_val: float
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
        prevs = np.tile(prevs, (n_iter, 1))

    return prevs




def simplex_uniform_sampling(
    n_dim: int,
    n_prev: int,
    n_iter: int,
    min_val: float,
    max_val: float
) -> np.ndarray:
    """
    Generates uniformly distributed prevalence vectors within the simplex,
    constrained by min_val ≤ p_i ≤ max_val.

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
    if min_val * n_dim > 1 or max_val * n_dim < 1:
        raise ValueError("Invalid min_val/max_val for simplex constraints.")

    total_samples = n_prev * n_iter
    samples = []

    while len(samples) < total_samples:
        # Gera candidatos via Dirichlet (uniforme no simplex)
        x = np.random.dirichlet(np.ones(n_dim), size=total_samples * 2)
        # Filtra os que respeitam os limites
        mask = np.all((x >= min_val) & (x <= max_val), axis=1)
        valid = x[mask]
        if len(valid) > 0:
            samples.append(valid)
        
        if len(samples) > 0:
            all_samples = np.concatenate(samples, axis=0)
            if len(all_samples) >= total_samples:
                return all_samples[:total_samples]

    return np.concatenate(samples, axis=0)[:total_samples]


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
    if random_state is not None:
        np.random.seed(random_state)

    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, size=batch_size, replace=True)
        yield indices
