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