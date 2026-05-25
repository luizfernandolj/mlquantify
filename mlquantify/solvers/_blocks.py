import numpy as np

from mlquantify.solvers import minimize_prevalence


def minimize_prevalence_blocks(
    objective_factory,
    test_representation,
    train_representations,
    block_slices,
    n_classes,
    solver="grid",
    aggregate="median",
    grid_size=101,
):
    """Minimize a loss over multiple representation blocks and aggregate results.

    For each sub-vector (block) of the test and training representations,
    builds a block-specific objective via ``objective_factory`` and
    minimizes it independently.  The per-block prevalence estimates are
    then aggregated (median or mean) into a single prevalence vector.

    This is the core routine of the Histogram Distribution Matching (HDy)
    family of quantifiers, where each block corresponds to one histogram
    bin interval.

    Parameters
    ----------
    objective_factory : callable
        Factory that receives ``test_block`` and ``train_block`` keyword
        arguments and returns a scalar objective function suitable for
        :func:`~mlquantify.solvers.minimize_prevalence`.
    test_representation : ndarray of shape (n_components,)
        Full representation vector of the test sample.
    train_representations : array-like of shape (n_classes, n_components)
        Full representation vectors for each training class.
    block_slices : list of slice
        Ordered list of slices that partition ``n_components`` into blocks.
    n_classes : int
        Number of classes.
    solver : str, default='grid'
        Solver passed to :func:`~mlquantify.solvers.minimize_prevalence`
        for each block.
    aggregate : {'median', 'mean'}, default='median'
        How to combine the per-block prevalence estimates.
    grid_size : int, default=101
        Number of grid points used by the ``'grid'`` solver for binary
        problems.

    Returns
    -------
    prevalence : ndarray of shape (n_classes,)
        Aggregated prevalence vector summing to 1.
    loss : float
        Aggregated objective value across blocks.

    Raises
    ------
    ValueError
        If ``aggregate`` is not ``'median'`` or ``'mean'``.

    Examples
    --------
    >>> import numpy as np
    >>> from mlquantify.solvers._blocks import minimize_prevalence_blocks
    >>> # Toy binary example with two histogram blocks
    >>> test_rep = np.array([0.4, 0.6, 0.3, 0.7])
    >>> train_reps = np.array([[0.5, 0.5, 0.5, 0.5],
    ...                        [0.2, 0.8, 0.2, 0.8]])
    >>> slices = [slice(0, 2), slice(2, 4)]
    >>> factory = lambda test_block, train_block: (
    ...     lambda a: np.sum((test_block - ((1-a)*train_block[0] + a*train_block[1]))**2)
    ... )
    >>> prevalence, loss = minimize_prevalence_blocks(
    ...     factory, test_rep, train_reps, slices, n_classes=2
    ... )
    >>> prevalence.shape
    (2,)
    """
    prevalences = []
    losses = []

    train_representations = np.asarray(train_representations)

    if solver == "grid" and n_classes == 2:
        alphas = np.linspace(0.0, 1.0, int(grid_size))

        for block_slice in block_slices:
            test_block = test_representation[block_slice]
            train_block = train_representations[:, block_slice]

            objective = objective_factory(
                test_block=test_block,
                train_block=train_block,
            )

            block_losses = np.asarray([
                objective(alpha)
                for alpha in alphas
            ])

            best_idx = int(np.argmin(block_losses))

            alpha = float(alphas[best_idx])
            loss = float(block_losses[best_idx])

            prevalences.append(
                np.asarray([1.0 - alpha, alpha])
            )

            losses.append(loss)

    else:
        for block_slice in block_slices:
            test_block = test_representation[block_slice]
            train_block = train_representations[:, block_slice]

            objective = objective_factory(
                test_block=test_block,
                train_block=train_block,
            )

            prevalence, loss = minimize_prevalence(
                objective=objective,
                n_classes=n_classes,
                solver=solver,
            )

            prevalences.append(prevalence)
            losses.append(loss)

    prevalences = np.asarray(prevalences)
    losses = np.asarray(losses)

    if aggregate == "median":
        prevalence = np.median(prevalences, axis=0)
        loss = float(np.median(losses))

    elif aggregate == "mean":
        prevalence = np.mean(prevalences, axis=0)
        loss = float(np.mean(losses))

    else:
        raise ValueError(f"Unknown aggregate={aggregate!r}.")

    prevalence = np.clip(prevalence, 0.0, None)

    total = prevalence.sum()

    if total > 0:
        prevalence = prevalence / total
    else:
        prevalence = np.ones(n_classes) / n_classes

    return prevalence, loss