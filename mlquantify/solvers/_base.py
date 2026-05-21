from ._binary import solve_binary
from ._simplex import solve_simplex


def minimize_prevalence(
    objective,
    n_classes,
    solver="auto",
    grid_size=101,
    tol=1e-10,
    x0=None,
    random_state=None,
):
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")

    if solver == "auto":
        solver = "bounded" if n_classes == 2 else "slsqp"

    if n_classes == 2 and solver in {
        "grid",
        "ternary",
        "bounded",
    }:
        return solve_binary(
            objective=objective,
            solver=solver,
            grid_size=grid_size,
            tol=tol,
        )

    if solver == "slsqp":
        return solve_simplex(
            objective=objective,
            n_classes=n_classes,
            x0=x0,
            tol=tol,
            random_state=random_state,
        )

    raise ValueError(
        f"Solver {solver!r} is incompatible with "
        f"{n_classes} classes."
    )
